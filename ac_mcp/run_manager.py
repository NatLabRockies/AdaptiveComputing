"""
run_manager.py
==============
Background-thread execution of Hero-backed evaluation and optimisation runs.

Each call to `submit_evaluation_run` or `submit_optimization_run` starts a
daemon thread and immediately returns a `run_id`.  The caller polls
`get_status(run_id)` until status is "completed" or "error".

Thread safety: RunStatus fields are updated under a per-run lock.  The global
_runs dict itself is protected by a separate lock.

HPC serialisation: Only one HPC environment (run_remote_managers / cleanup) can
be active at a time.  A global lock enforces this.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# RunStatus
# ---------------------------------------------------------------------------

@dataclass
class RunStatus:
    run_id:        str
    experiment_id: str
    run_type:      str          # "evaluation" | "optimization"
    status:        str = "running"   # "running" | "completed" | "error"
    n_completed:   int = 0
    n_total:       int = 0
    message:       str = ""          # human-readable phase description
    results:       list = field(default_factory=list)
    best_x:        Optional[dict] = None
    best_y:        Optional[float] = None
    error:         Optional[str] = None
    _lock:         threading.Lock = field(default_factory=threading.Lock,
                                         repr=False)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_runs: dict[str, RunStatus] = {}
_runs_lock = threading.Lock()

# Only one thread may hold HPC resources at a time
_hpc_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_hpc_config(hpc_config_path: str) -> Any:
    """Import an hpc_config.py file from an arbitrary path."""
    spec = importlib.util.spec_from_file_location("_hpc_config", hpc_config_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_hpc(hpc_config_path: str):
    """Import HPC config, start remote managers via SSH, return (hpc, cleanup).

    The MCP server/agent can run anywhere (laptop, cloud, etc.).  The manager
    must run on the HPC login node(s) where sbatch is available.
    autonomous_managers.py handles SSH to those nodes; hpc_config.py specifies
    remote_hosts and remote_dirs.
    """
    import os
    import time

    hpc     = _load_hpc_config(hpc_config_path)
    hpc_dir = os.path.dirname(os.path.abspath(hpc_config_path))

    # autonomous_managers.py lives in the AC examples directory, not in the
    # application's hpc_config dir.  Locate it relative to this package.
    ac_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mgr_dirs = [
        hpc_dir,
        os.path.join(ac_root, "examples", "hero_HPC_managers"),
    ]
    for d in mgr_dirs:
        if d not in sys.path:
            sys.path.insert(0, d)

    from autonomous_managers import (
        run_remote_managers, cleanup_remote_managers,
        setup_remote_state, verify_remote_managers,
    )

    # setup_remote_state() registers a SIGINT handler, which Python only allows
    # from the main thread.  Worker threads must skip it.
    import signal
    import threading
    if threading.current_thread() is not threading.main_thread():
        _orig_signal = signal.signal
        signal.signal = lambda *a, **kw: None   # no-op in thread
        try:
            setup_remote_state(hpc.machine_names, hpc.remote_usernames,
                               hpc.remote_hosts, hpc.remote_dirs)
        finally:
            signal.signal = _orig_signal
    else:
        setup_remote_state(hpc.machine_names, hpc.remote_usernames,
                           hpc.remote_hosts, hpc.remote_dirs)

    run_remote_managers()
    print("Waiting 10 s for remote managers to start...")
    time.sleep(10)
    verify_remote_managers()
    return hpc, cleanup_remote_managers


def _update(rs: RunStatus, **kwargs):
    with rs._lock:
        for k, v in kwargs.items():
            setattr(rs, k, v)


def _extract_results(ac_driver, param_specs: list[dict],
                     fixed_context: dict) -> tuple[list, Optional[dict], Optional[float]]:
    """Extract result list and best (x, y) from a driver's dataset."""
    x_data = ac_driver.dataset.x_data[0]   # (N, d)
    y_data = ac_driver.dataset.y_data[0]   # (N, 1)
    results = []
    best_y, best_x_dict = None, None

    for i, (x_row, y_row) in enumerate(zip(x_data, y_data)):
        y_val = float(y_row[0])
        acc   = None if np.isnan(y_val) else y_val

        x_dict: dict = dict(fixed_context)
        for j, spec in enumerate(param_specs):
            raw   = float(x_row[j])
            ptype = spec["type"].lower()
            if ptype == "categorical":
                x_dict[spec["name"]] = spec["categories"][int(round(raw))]
            elif ptype == "ordered":
                x_dict[spec["name"]] = int(round(raw))
            else:
                x_dict[spec["name"]] = raw

        results.append({"x": x_dict, "y": acc})

        if acc is not None and (best_y is None or acc > best_y):
            best_y  = acc
            best_x_dict = x_dict

    return results, best_x_dict, best_y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_run(experiment_id: str, run_type: str, n_total: int) -> str:
    run_id = str(uuid.uuid4())
    status = RunStatus(run_id=run_id, experiment_id=experiment_id,
                       run_type=run_type, n_total=n_total)
    with _runs_lock:
        _runs[run_id] = status
    return run_id


def get_status(run_id: str) -> dict:
    with _runs_lock:
        status = _runs.get(run_id)
    if status is None:
        raise KeyError(f"Unknown run_id: {run_id!r}")
    with status._lock:
        return {
            "run_id":        status.run_id,
            "experiment_id": status.experiment_id,
            "run_type":      status.run_type,
            "status":        status.status,
            "n_completed":   status.n_completed,
            "n_total":       status.n_total,
            "message":       status.message,
            "results":       list(status.results),
            "best_x":        status.best_x,
            "best_y":        status.best_y,
            "error":         status.error,
        }


# ---------------------------------------------------------------------------
# Evaluation run (no surrogate)
# ---------------------------------------------------------------------------

def _eval_worker(run_id: str, entry: dict, jobs: list[dict]):
    from adaptive_computing.drivers import ActiveLoopDriverHero
    from adaptive_computing.datasets import OrderedVariable
    from ac_mcp.param_builder import build_evaluation_formatter
    from ac_mcp import registry

    rs = _runs[run_id]   # RunStatus object (named 'rs' to avoid collision with status= kwarg)
    n = len(jobs)
    _update(rs, n_total=n)

    with _hpc_lock:
        try:
            hpc, cleanup = _setup_hpc(entry["hpc_config_path"])
        except Exception as exc:
            _update(rs, status="error",
                    error=f"HPC setup failed: {exc}\n{traceback.format_exc()}")
            return

        try:
            formatter = build_evaluation_formatter(jobs, hpc.machine_names)
            driver = ActiveLoopDriverHero(
                simulations=[None],
                params=[OrderedVariable(min_val=0, max_val=max(n - 1, 1))],
                machine_names=hpc.machine_names,
                output_field_path=entry["output_field_path"],
                surrogate=None,
                blocking=False,
                task_formatter=formatter,
            )

            # Submit all jobs at once, x_data = [[0], [1], ..., [n-1]]
            x_all = np.array([[i] for i in range(n)], dtype=float)
            driver.dataset.add_samples(x_all, None, 0)
            driver.dataset.hero_wait_for_data()

            # Collect results
            y_data = driver.dataset.y_data[0]
            results = []
            best_y, best_x = None, None
            for i, (job, y_row) in enumerate(zip(jobs, y_data)):
                y_val = float(y_row[0])
                acc   = None if np.isnan(y_val) else y_val
                results.append({"x": dict(job), "y": acc})
                if acc is not None and (best_y is None or acc > best_y):
                    best_y = acc
                    best_x = dict(job)

            registry.save_driver(run_id, driver)   # save under run_id as a snapshot
            # Mark the *experiment* itself as completed so list_experiments shows the right status
            n_successful = sum(1 for r in results if r["y"] is not None)
            registry.update_entry(entry["id"], run_status="completed",
                                   n_samples=n_successful,
                                   best_x=best_x, best_y=best_y)
            _update(rs, status="completed", n_completed=n,
                    results=results, best_x=best_x, best_y=best_y)

        except Exception as exc:
            _update(rs, status="error",
                    error=f"{exc}\n{traceback.format_exc()}")
        finally:
            try:
                cleanup()
            except Exception:
                pass


def submit_evaluation_run(entry: dict, jobs: list[dict]) -> str:
    run_id = create_run(entry["id"], "evaluation", len(jobs))
    t = threading.Thread(target=_eval_worker, args=(run_id, entry, jobs),
                         daemon=True)
    t.start()
    return run_id


# ---------------------------------------------------------------------------
# Optimisation run (with surrogate + BO)
# ---------------------------------------------------------------------------

def _opt_worker(run_id: str, entry: dict,
                n_init: int, n_steps: int, acq_func: str, blocking: bool = False,
                fork_experiment_id: str = None):
    from adaptive_computing.drivers import ActiveLoopDriverHero
    from ac_mcp.param_builder import build_ac_params, build_task_formatter
    from ac_mcp import registry

    rs = _runs[run_id]   # RunStatus object (named 'rs' to avoid collision with status= kwarg)
    n_total = n_steps if fork_experiment_id else n_init + n_steps
    _update(rs, n_total=n_total)

    param_specs   = entry["param_specs"]
    fixed_context = entry["fixed_context"]

    with _hpc_lock:
        try:
            hpc, cleanup = _setup_hpc(entry["hpc_config_path"])
        except Exception as exc:
            _update(rs, status="error",
                    error=f"HPC setup failed: {exc}\n{traceback.format_exc()}")
            return

        try:
            ac_params = build_ac_params(param_specs)
            formatter = build_task_formatter(param_specs, fixed_context,
                                             hpc.machine_names)
            mode_str = "sequential (blocking)" if blocking else "parallel (non-blocking)"
            print(f"[run {run_id[:8]}] BO mode: {mode_str}")
            driver = ActiveLoopDriverHero(
                simulations=[None],
                params=ac_params,
                machine_names=hpc.machine_names,
                output_field_path=entry["output_field_path"],
                surrogate="SMT_GP",
                acq_func=acq_func,
                blocking=blocking,
                task_formatter=formatter,
            )

            if fork_experiment_id:
                # ── Fork path: seed from prior experiment, skip LHS warm-up ────
                # Strip any surrounding brackets the LLM may have included (e.g. "[61082afc]")
                fork_experiment_id = fork_experiment_id.strip().strip("[]")
                print(f"[run {run_id[:8]}] Forking from {fork_experiment_id[:8]} ...")
                _update(rs, message=f"loading prior data from {fork_experiment_id[:8]}")
                old_driver = registry.load_driver(fork_experiment_id)
                x_old = old_driver.dataset.x_data[0]   # (N, d)
                y_old = old_driver.dataset.y_data[0]   # (N, 1)
                valid = ~np.isnan(y_old[:, 0])
                x_valid, y_valid = x_old[valid], y_old[valid]
                n_prior = len(x_valid)
                print(f"[run {run_id[:8]}] Seeding {n_prior} prior points, then {n_steps} BO steps")
                _update(rs, n_total=n_prior + n_steps,
                        message=f"seeding {n_prior} prior points")
                if n_prior > 0:
                    # add_samples_nohero stores data without submitting Hero tasks
                    driver.dataset.add_samples_nohero(x_valid, y_valid, 0)
                    driver.surrogate.train(driver.dataset)
                results, best_x, best_y = _extract_results(driver, param_specs,
                                                            fixed_context)
                _update(rs, n_completed=n_prior, results=results,
                        best_x=best_x, best_y=best_y,
                        message=f"seeded {n_prior} points; starting {n_steps} BO steps")
                registry.save_driver(entry["id"], driver)
            else:
                # ── Normal path: LHS warm-up ─────────────────────────────────────
                print(f"[run {run_id[:8]}] Warm-up ({n_init} LHS points)...")
                _update(rs, message=f"warmup: waiting for {n_init} LHS jobs")
                driver.initialize(N_samples_init=n_init)
                driver.hero_wait_for_data_and_train()

                results, best_x, best_y = _extract_results(driver, param_specs,
                                                            fixed_context)
                _update(rs, n_completed=n_init, results=results,
                        best_x=best_x, best_y=best_y,
                        message=f"warmup done; starting {n_steps} BO steps")
                registry.save_driver(entry["id"], driver)

            # ── BO phase (same for both paths) ────────────────────────────────
            print(f"[run {run_id[:8]}] BO: {n_steps} steps ({mode_str})...")
            _update(rs, message=f"BO: running {n_steps} steps ({mode_str})")
            driver.run(N_steps=n_steps)

            _update(rs, message=f"BO: waiting for {n_steps} jobs")
            driver.hero_wait_for_data_and_train()

            results, best_x, best_y = _extract_results(driver, param_specs,
                                                        fixed_context)
            n_done = (n_prior if fork_experiment_id else n_init) + n_steps
            _update(rs, n_completed=n_done,
                    results=results, best_x=best_x, best_y=best_y,
                    message="BO complete")
            registry.save_driver(entry["id"], driver)
            print(f"[run {run_id[:8]}] BO done. best_y={best_y}")

            _update(rs, status="completed", message="optimization complete")

        except Exception as exc:
            _update(rs, status="error",
                    error=f"{exc}\n{traceback.format_exc()}")
        finally:
            try:
                cleanup()
            except Exception:
                pass


def submit_optimization_run(entry: dict, n_init: int,
                             n_steps: int, acq_func: str,
                             blocking: bool = False,
                             fork_experiment_id: str = None) -> str:
    n_total = n_steps if fork_experiment_id else n_init + n_steps
    run_id = create_run(entry["id"], "optimization", n_total)
    t = threading.Thread(target=_opt_worker,
                         args=(run_id, entry, n_init, n_steps, acq_func,
                               blocking, fork_experiment_id),
                         daemon=True)
    t.start()
    return run_id
