"""registry.py
Dataset-centric experiment registry with file locking.

Each entry records searchable metadata about one experiment; the raw dataset
(x_data, y_data) is persisted as a separate .npz file.

Registry location:  <AC_MCP_DIR>/registry.json
Dataset directory:  <AC_MCP_DIR>/experiments/<experiment_id>.npz
"""

from __future__ import annotations

import json
import numpy as np
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

def _storage_dir() -> Path:
    raw = os.environ.get("AC_MCP_DIR")
    if not raw:
        raise RuntimeError(
            "AC_MCP_DIR is not set.  Start the server with --storage-dir /path/to/app "
            "so each application keeps its own registry and experiment files."
        )
    d = Path(raw)
    d.mkdir(parents=True, exist_ok=True)
    (d / "datasets").mkdir(exist_ok=True)
    return d


def _registry_path() -> Path:
    return _storage_dir() / "registry.json"


def _npz_path(experiment_id: str) -> Path:
    return _storage_dir() / "datasets" / f"{experiment_id}.npz"


# ---------------------------------------------------------------------------
# Thread-safe registry access
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()


def _load_registry() -> dict:
    path = _registry_path()
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    # Back-fill run_status for entries created before this field was added.
    dirty = False
    for entry in data.values():
        if "run_status" not in entry:
            npz = Path(entry.get("data_path", ""))
            entry["run_status"] = "completed" if npz.exists() else "in_progress"
            dirty = True
    if dirty:
        _save_registry(data)
    return data


def _save_registry(data: dict) -> None:
    path = _registry_path()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)  # atomic rename — prevents partial-write corruption


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_experiment(
    *,
    name: str,
    description: str,
    param_specs: list[dict],
    fixed_context: dict,
    output_label: str,
    hpc_config_path: str,
    output_field_path: str = "y_data",
    experiment_type: str = "optimization",   # "optimization" | "evaluation"
) -> str:
    """Create a new registry entry and return its experiment_id."""
    experiment_id = str(uuid.uuid4())
    entry = {
        "id":                experiment_id,
        "name":              name,
        "description":       description,
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "run_status":        "in_progress",   # "in_progress" | "completed"
        "param_specs":       param_specs,
        "fixed_context":     fixed_context,
        "output_label":      output_label,
        "output_field_path": output_field_path,
        "hpc_config_path":   hpc_config_path,
        "experiment_type":   experiment_type,
        "output_labels":     [output_label],   # list for multi-output support
        "n_samples":         0,
        "best_x":            None,
        "best_y":            None,
        "data_path":         str(_npz_path(experiment_id)),
        "forked_from":       None,
    }
    with _registry_lock:
        reg = _load_registry()
        reg[experiment_id] = entry
        _save_registry(reg)
    return experiment_id


def get_entry(experiment_id: str) -> dict:
    with _registry_lock:
        reg = _load_registry()
    entry = reg.get(experiment_id)
    if entry is None:
        raise KeyError(f"Experiment not found: {experiment_id!r}")
    return entry


def update_entry(experiment_id: str, **kwargs: Any) -> None:
    """Overwrite specific fields in an existing registry entry."""
    with _registry_lock:
        reg = _load_registry()
        if experiment_id not in reg:
            raise KeyError(f"Experiment not found: {experiment_id!r}")
        reg[experiment_id].update(kwargs)
        _save_registry(reg)


def list_entries(name_filter: Optional[str] = None) -> list[dict]:
    """Return all registry entries, optionally filtered by substring in name."""
    with _registry_lock:
        reg = _load_registry()
    entries = list(reg.values())
    if name_filter:
        entries = [e for e in entries if name_filter.lower() in e["name"].lower()]
    return sorted(entries, key=lambda e: e["created_at"], reverse=True)


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------

def save_dataset(
    experiment_id: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_labels: Optional[list[str]] = None,
    *,
    set_completed: bool = True,
) -> None:
    """Save (x_data, y_data) as .npz and update registry stats.

    Parameters
    ----------
    x_data : (N, d) array of parameter vectors.
    y_data : (N, m) array of output values.  NaN rows are ignored for stats.
    output_labels : optional list of m label strings.
    set_completed : mark experiment as completed when True (default).
    """
    npz = _npz_path(experiment_id)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(npz), x_data=x_data, y_data=y_data)

    # Compute stats from first output column (primary objective)
    y2d = np.atleast_2d(y_data) if y_data.ndim == 1 else y_data
    valid_mask = ~np.isnan(y2d[:, 0])
    n_samples = int(valid_mask.sum())

    best_y, best_x = None, None
    if n_samples > 0:
        idx = int(np.argmax(y2d[valid_mask, 0]))
        best_y = float(y2d[valid_mask][idx, 0])
        best_x = x_data[valid_mask][idx].tolist()

    kwargs: dict[str, Any] = dict(
        n_samples=n_samples, best_x=best_x, best_y=best_y,
        data_path=str(npz),
    )
    if output_labels is not None:
        kwargs["output_labels"] = output_labels
    if set_completed:
        kwargs["run_status"] = "completed"
    update_entry(experiment_id, **kwargs)


def load_dataset(experiment_id: str) -> dict[str, np.ndarray]:
    """Load (x_data, y_data) arrays for an experiment.

    Accepts a full UUID or an 8-char prefix.  Returns a dict with keys
    'x_data' (N, d) and 'y_data' (N, m).
    """
    npz = _npz_path(experiment_id)
    if not npz.exists() and len(experiment_id) < 36:
        candidates = list((_storage_dir() / "datasets").glob(f"{experiment_id}*.npz"))
        if len(candidates) == 1:
            npz = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous experiment prefix {experiment_id!r}: "
                f"matches {[p.stem for p in candidates]}"
            )
    if not npz.exists():
        raise FileNotFoundError(f"No saved dataset for experiment {experiment_id!r}")
    data = np.load(str(npz))
    return {"x_data": data["x_data"], "y_data": data["y_data"]}


# ---------------------------------------------------------------------------
# Experiment matching
# ---------------------------------------------------------------------------

def find_matching_experiment(
    name: str,
    param_specs: list[dict],
    fixed_context: dict,
    experiment_type: str,
) -> Optional[dict]:
    """Return the most recent *completed* experiment that matches the given key fields.

    Identity is defined by experiment_type + fixed_context + param *names* and
    *types* (not bounds).  The experiment name is intentionally excluded from
    matching because the LLM may generate slightly different label strings across
    runs for the same underlying problem.  Bounds are treated as tuning metadata.
    Returns None if no completed match exists.
    """
    with _registry_lock:
        reg = _load_registry()

    # Extract just name+type from param_specs for matching (ignore min/max/categories)
    def _param_key(specs: list[dict]) -> list[tuple]:
        return sorted((s.get("name", ""), s.get("type", "")) for s in specs)

    target_key = _param_key(param_specs)

    candidates = [
        e for e in reg.values()
        if e.get("run_status") == "completed"
        and e["experiment_type"] == experiment_type
        and e["fixed_context"] == fixed_context
        and _param_key(e["param_specs"]) == target_key
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["created_at"])


def find_reusable_data(
    param_specs: list[dict],
    fixed_context: dict,
    experiment_type: str,
    exclude_id: Optional[str] = None,
) -> dict:
    """Search all completed experiments that share fixed_context and param names/types
    (regardless of bounds), load their datasets, and return only the data points
    that fall within the bounds defined in *this* param_specs.

    Parameters
    ----------
    param_specs : list[dict]
        The current experiment's param specs — used for both identity matching
        (name/type) and bounds filtering (min/max).
    fixed_context : dict
        Must match exactly.
    experiment_type : str
    exclude_id : str or None
        Experiment ID to exclude (typically the current experiment itself).

    Returns
    -------
    dict with keys:
        x_valid : (N, d) ndarray or None
        y_valid : (N, m) ndarray or None
        n_valid : int   — number of in-bounds, non-NaN points found
        source_ids : list[str]  — 8-char prefixes of experiments data came from
    """
    with _registry_lock:
        reg = _load_registry()

    def _param_key(specs):
        return sorted((s.get("name", ""), s.get("type", "")) for s in specs)

    target_key = _param_key(param_specs)

    # Build per-param bounds from current param_specs
    bounds = []  # list of (lo, hi) per continuous param, in param order
    for s in param_specs:
        if s.get("type") == "continuous":
            bounds.append((s.get("min"), s.get("max")))
        else:
            bounds.append((None, None))

    candidates = [
        e for e in reg.values()
        if e.get("run_status") == "completed"
        and e["experiment_type"] == experiment_type
        and e["fixed_context"] == fixed_context
        and _param_key(e["param_specs"]) == target_key
        and e["id"] != exclude_id
    ]

    all_x, all_y, source_ids = [], [], []
    for e in candidates:
        try:
            data = load_dataset(e["id"])
            x, y = data["x_data"], data["y_data"]
            y2d = np.atleast_2d(y) if y.ndim == 1 else y

            # Cherry-pick: must be within bounds AND have valid (non-NaN) primary output
            mask = ~np.isnan(y2d[:, 0])
            for j, (lo, hi) in enumerate(bounds):
                if lo is not None:
                    mask &= (x[:, j] >= lo)
                if hi is not None:
                    mask &= (x[:, j] <= hi)

            if mask.any():
                all_x.append(x[mask])
                all_y.append(y2d[mask])
                source_ids.append(e["id"][:8])
        except FileNotFoundError:
            pass

    if not all_x:
        return {"x_valid": None, "y_valid": None, "n_valid": 0, "source_ids": []}

    x_combined = np.vstack(all_x)
    y_combined = np.vstack(all_y)
    return {
        "x_valid": x_combined,
        "y_valid": y_combined,
        "n_valid": len(x_combined),
        "source_ids": source_ids,
    }
