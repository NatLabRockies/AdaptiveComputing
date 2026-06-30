"""
registry.py
===========
JSON-backed experiment registry with file locking.

Each entry records searchable metadata about one experiment; the full AC driver
state (dataset + trained surrogate) is persisted as a separate pickle file.

Registry location: ~/.ac_mcp/registry.json
Pickle directory:  ~/.ac_mcp/experiments/<experiment_id>.pkl
"""

from __future__ import annotations

import json
import os
import pickle
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
    (d / "experiments").mkdir(exist_ok=True)
    return d


def _registry_path() -> Path:
    return _storage_dir() / "registry.json"


def _pkl_path(experiment_id: str) -> Path:
    return _storage_dir() / "experiments" / f"{experiment_id}.pkl"


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
            pkl = Path(entry.get("pkl_path", ""))
            entry["run_status"] = "completed" if pkl.exists() else "in_progress"
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
        "n_samples":         0,
        "best_x":            None,
        "best_y":            None,
        "pkl_path":          str(_pkl_path(experiment_id)),
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
# Driver persistence
# ---------------------------------------------------------------------------

def save_driver(experiment_id: str, ac_driver: Any, *, set_completed: bool = True) -> None:
    """Pickle an AC driver to disk, then update n_samples/best in registry.

    Parameters
    ----------
    set_completed : bool
        When True (default) the registry entry is updated to run_status="completed".
        Pass False for intermediate checkpoints (e.g. after LHS warmup but before
        BO completes) so the experiment is not treated as reusable prematurely.
    """
    pkl = _pkl_path(experiment_id)
    with open(pkl, "wb") as f:
        pickle.dump(ac_driver, f)

    # Refresh summary stats in registry
    try:
        import numpy as np
        y_data = ac_driver.dataset.y_data[0]          # (N, 1)
        x_data = ac_driver.dataset.x_data[0]          # (N, d)
        valid_mask = ~np.isnan(y_data[:, 0])
        n_samples = int(valid_mask.sum())

        best_y, best_x = None, None
        if n_samples > 0:
            idx = int(np.argmax(y_data[valid_mask, 0]))
            best_y = float(y_data[valid_mask][idx, 0])
            best_x = x_data[valid_mask][idx].tolist()

        kwargs: dict = dict(n_samples=n_samples, best_x=best_x, best_y=best_y)
        if set_completed:
            kwargs["run_status"] = "completed"
        update_entry(experiment_id, **kwargs)
    except Exception:
        pass  # stats are best-effort; don't crash on partial data


def load_driver(experiment_id: str) -> Any:
    """Unpickle an AC driver from disk.

    Accepts either a full UUID or an 8-char prefix (as shown in the registry
    summary).  If the exact file is not found, a prefix glob is attempted.
    """
    pkl = _pkl_path(experiment_id)
    if not pkl.exists() and len(experiment_id) < 36:
        # Try prefix match: find any .pkl whose filename starts with this prefix
        candidates = list((_storage_dir() / "experiments").glob(f"{experiment_id}*.pkl"))
        if len(candidates) == 1:
            pkl = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous experiment prefix {experiment_id!r}: "
                f"matches {[p.stem for p in candidates]}"
            )
    if not pkl.exists():
        raise FileNotFoundError(f"No saved driver for experiment {experiment_id!r}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


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
