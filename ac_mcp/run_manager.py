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
import subprocess
import sys
import threading
import time
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
    run_id:          str
    experiment_id:   str
    run_type:        str             # "evaluation" | "optimization"
    experiment_name: str = ""        # human-readable name for display
    status:          str = "running" # "running" | "completed" | "error"
    hero_state:      str = "active"  # "active" | "waiting" | "completed" | "error"
    n_completed:     int = 0
    n_total:         int = 0
    n_warmup:        int = 0         # LHS init or prior seed points (before BO)
    message:         str = ""        # human-readable phase description
    results:         list = field(default_factory=list)
    best_x:          Optional[dict] = None
    best_y:          Optional[float] = None
    error:           Optional[str] = None
    _lock:           threading.Lock = field(default_factory=threading.Lock,
                                           repr=False)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_runs: dict[str, RunStatus] = {}
_runs_lock = threading.Lock()

# Only one thread may hold HPC resources at a time
_hpc_lock = threading.Lock()

# Emergency cleanup: called by atexit / signal handlers when the server is
# killed before a worker thread's finally block can run.
_active_cleanup_fn: Optional[callable] = None
_atexit_registered: bool = False


def _set_active_cleanup(fn: callable) -> None:
    """Register fn as the cleanup to run if the process exits mid-run."""
    import atexit
    global _active_cleanup_fn, _atexit_registered
    _active_cleanup_fn = fn
    if not _atexit_registered:
        atexit.register(_emergency_cleanup)
        _atexit_registered = True


def _clear_active_cleanup() -> None:
    global _active_cleanup_fn
    _active_cleanup_fn = None


def _emergency_cleanup() -> None:
    """Attempt cleanup of remote managers; safe to call more than once."""
    fn = _active_cleanup_fn
    if fn is not None:
        print("[ac_mcp] Server exiting — cleaning up remote HPC managers...")
        try:
            fn()
        except Exception as exc:
            print(f"[ac_mcp] Cleanup error (ignored): {exc}")
        _clear_active_cleanup()


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
    adaptive_computing.hpc.autonomous handles SSH to those nodes; hpc_config.py
    specifies remote_hosts and remote_dirs.
    """
    import os
    import signal
    import threading

    from adaptive_computing.hpc.autonomous import (
        cleanup_remote_managers,
        run_remote_managers,
        setup_remote_state,
        wait_for_managers,
    )

    hpc     = _load_hpc_config(hpc_config_path)
    python_paths = getattr(hpc, 'python_paths', {})

    # setup_remote_state() registers a SIGINT handler, which Python only allows
    # from the main thread.  Worker threads must skip it.
    if threading.current_thread() is not threading.main_thread():
        _orig_signal = signal.signal
        signal.signal = lambda *a, **kw: None   # no-op in thread
        try:
            setup_remote_state(hpc.machine_names, hpc.remote_usernames,
                               hpc.remote_hosts, hpc.remote_dirs,
                               python_paths)
        finally:
            signal.signal = _orig_signal
    else:
        setup_remote_state(hpc.machine_names, hpc.remote_usernames,
                           hpc.remote_hosts, hpc.remote_dirs,
                           python_paths)

    # Infinite-retry startup: keep trying until every manager is confirmed up.
    while True:
        run_remote_managers()
        try:
            wait_for_managers()
            break
        except RuntimeError as exc:
            print(f"[ac_mcp] Managers not ready yet ({exc}). Retrying in 15s...")
            time.sleep(15)

    _set_active_cleanup(cleanup_remote_managers)
    return hpc, cleanup_remote_managers


def _update(rs: RunStatus, **kwargs):
    with rs._lock:
        for k, v in kwargs.items():
            setattr(rs, k, v)


# ---------------------------------------------------------------------------
# Manager health watchdog
# ---------------------------------------------------------------------------

def _check_manager_alive(machine: str, username: str, host: str) -> bool | None:
    """SSH-ping the remote manager_session.  Returns True/False/None (unknown)."""
    from adaptive_computing.hpc.remote_manager import SESSION_NAME
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"{username}@{host}",
             (
                 "bash -l -c '"
                 "command -v tmux &>/dev/null || module load tmux 2>/dev/null; "
                 f"tmux has-session -t {SESSION_NAME} 2>/dev/null "
                 "&& echo ALIVE || echo DEAD'"
             )],
            capture_output=True, text=True, timeout=20,
        )
        if "ALIVE" in r.stdout:
            return True
        if "DEAD" in r.stdout:
            return False
        return None
    except Exception:
        return None


def _wait_with_watchdog(wait_fn, hpc, rs: RunStatus, phase: str,
                        check_interval: int = 30):
    """
    Call wait_fn() while a background watchdog polls each remote manager every
    `check_interval` seconds.  If a manager is found dead the watchdog signals
    hero_wait_for_data to exit cleanly, then the manager is restarted and
    wait_fn is re-called.  This loop continues until wait_fn completes
    normally, so transient manager crashes never abort a run.
    """
    from adaptive_computing.datasets import hero as _hero_ds
    from adaptive_computing.hpc.autonomous import run_remote_managers, wait_for_managers

    restart_count = 0

    while True:
        _hero_ds._abort_event.clear()

        done_event   = threading.Event()
        dead_machine = [None]   # filled by watchdog thread

        def _watchdog(done=done_event, dead=dead_machine):
            while not done.wait(timeout=check_interval):
                for machine in hpc.machine_names:
                    alive = _check_manager_alive(
                        machine,
                        hpc.remote_usernames[machine],
                        hpc.remote_hosts[machine],
                    )
                    if alive is False:
                        dead[0] = machine
                        _hero_ds._abort_event.set()   # break hero_wait_for_data loop
                        return

        wt = threading.Thread(target=_watchdog, daemon=True)
        wt.start()
        manager_died = False
        try:
            wait_fn()
        except RuntimeError as exc:
            # hero_wait_for_data raises RuntimeError when _abort_event fires
            if dead_machine[0] is not None or "manager died" in str(exc).lower():
                manager_died = True
            else:
                done_event.set()
                wt.join(timeout=5)
                raise
        finally:
            done_event.set()
            wt.join(timeout=5)

        if not manager_died:
            return   # normal completion

        # ── Manager died: notify, restart, loop ───────────────────────────
        restart_count += 1
        machine = dead_machine[0] or "unknown"
        msg = (
            f"Manager on {machine} has died during '{phase}' "
            f"(restart attempt {restart_count}). Restarting..."
        )
        print(f"[run {rs.run_id[:8]}] ⚠️  {msg}")
        _update(rs, message=f"[{phase}] WARNING: {msg}")

        _hero_ds._abort_event.clear()

        # Restart loop: keep trying until the manager is back up.
        while True:
            try:
                run_remote_managers()
                wait_for_managers()
                ok = (f"Manager restarted (attempt {restart_count}). "
                      "Resuming wait for HPC results...")
                print(f"[run {rs.run_id[:8]}] ✅ {ok}")
                _update(rs, message=f"[{phase}] {ok}")
                break
            except RuntimeError as exc:
                retry_msg = f"Manager restart failed ({exc}). Retrying in 15s..."
                print(f"[run {rs.run_id[:8]}] ❌ {retry_msg}")
                _update(rs, message=f"[{phase}] {retry_msg}")
                time.sleep(15)
        # outer while True: re-enter wait_fn with the restarted manager


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

def create_run(experiment_id: str, run_type: str, n_total: int,
               experiment_name: str = "") -> str:
    run_id = str(uuid.uuid4())
    status = RunStatus(run_id=run_id, experiment_id=experiment_id,
                       run_type=run_type, experiment_name=experiment_name,
                       n_total=n_total)
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
            "run_id":          status.run_id,
            "experiment_id":   status.experiment_id,
            "experiment_name": status.experiment_name,
            "run_type":        status.run_type,
            "status":          status.status,
            "hero_state":      status.hero_state,
            "n_completed":     status.n_completed,
            "n_total":         status.n_total,
            "n_warmup":        status.n_warmup,
            "message":         status.message,
            "results":         list(status.results),
            "best_x":          status.best_x,
            "best_y":          status.best_y,
            "error":           status.error,
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
    _update(rs, n_total=n, message="starting HPC managers...")

    with _hpc_lock:
        _update(rs, message="starting HPC managers...")
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
            driver.dataset.add_samples(x_all, 0)
            _update(rs, hero_state="waiting")
            _wait_with_watchdog(driver.dataset.hero_wait_for_data, hpc, rs,
                                "evaluation")
            _update(rs, hero_state="active")

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

            n_successful = sum(1 for r in results if r["y"] is not None)
            x_arr = np.array([[i] for i in range(n)], dtype=float)
            registry.save_dataset(entry["id"], x_arr, y_data)   # also sets run_status="completed"
            registry.update_entry(entry["id"], best_x=best_x, best_y=best_y,
                                   n_samples=n_successful)
            _update(rs, status="completed", n_completed=n, hero_state="completed",
                    results=results, best_x=best_x, best_y=best_y)

        except Exception as exc:
            _update(rs, status="error",
                    error=f"{exc}\n{traceback.format_exc()}")
        finally:
            try:
                cleanup()
            except Exception:
                pass
            _clear_active_cleanup()


def submit_evaluation_run(entry: dict, jobs: list[dict]) -> str:
    run_id = create_run(entry["id"], "evaluation", len(jobs),
                        experiment_name=entry.get("name", ""))
    t = threading.Thread(target=_eval_worker, args=(run_id, entry, jobs),
                         daemon=True)
    t.start()
    return run_id


# ---------------------------------------------------------------------------
# Optimisation run (with surrogate + BO)
# ---------------------------------------------------------------------------

def _opt_worker(run_id: str, entry: dict,
                n_init: int, n_steps: int, acq_func: str, blocking: bool = False,
                skip_warmstart: bool = False):
    from adaptive_computing.drivers import ActiveLoopDriverHero
    from ac_mcp.param_builder import build_ac_params, build_task_formatter
    from ac_mcp import registry

    rs = _runs[run_id]   # RunStatus object (named 'rs' to avoid collision with status= kwarg)
    param_specs   = entry["param_specs"]
    fixed_context = entry["fixed_context"]

    # ── Auto warm-start: look for prior in-bounds data from matching experiments ──
    prior     = registry.find_reusable_data(
        param_specs=param_specs,
        fixed_context=fixed_context,
        experiment_type=entry["experiment_type"],
        exclude_id=entry["id"],
    )
    n_prior   = prior["n_valid"]
    n_dupes   = prior.get("n_duplicates_removed", 0)
    use_prior = n_prior > 0 and not skip_warmstart
    if skip_warmstart and n_prior > 0:
        print(f"[run {run_id[:8]}] skip_warmstart=True: ignoring {n_prior} prior pts, running LHS from scratch")

    n_total = (n_prior + n_steps) if use_prior else (n_init + n_steps)
    _update(rs, n_total=n_total, message="starting HPC managers...")

    with _hpc_lock:
        _update(rs, message="starting HPC managers...")
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

            if use_prior:
                # ── Auto warm-start: seed from prior in-bounds data, skip LHS ──
                src = ", ".join(prior["source_ids"])
                print(f"[run {run_id[:8]}] Auto warm-start: {n_prior} pts from [{src}], then {n_steps} BO steps"
                      + (f" ({n_dupes} duplicates removed)" if n_dupes else ""))
                _update(rs, message=f"warm-start: seeding {n_prior} prior pts from [{src}]"
                        + (f" ({n_dupes} dupes removed)" if n_dupes else ""))
                driver.dataset.add_known_samples(prior["x_valid"], prior["y_valid"], 0)
                driver.surrogate.train(driver.dataset)
                results, best_x, best_y = _extract_results(driver, param_specs, fixed_context)
                _update(rs, n_completed=n_prior, n_warmup=n_prior, results=results,
                        best_x=best_x, best_y=best_y,
                        message=f"seeded {n_prior} pts; starting {n_steps} BO steps")
                registry.save_dataset(entry["id"],
                                      driver.dataset.x_data[0],
                                      driver.dataset.y_data[0],
                                      set_completed=False)
            else:
                # ── Normal path: LHS warm-up ──────────────────────────────────
                print(f"[run {run_id[:8]}] Warm-up ({n_init} LHS points)...")
                _update(rs, message=f"warmup: waiting for {n_init} LHS jobs")
                driver.initialize(N_samples_init=n_init)
                _update(rs, hero_state="waiting")
                _wait_with_watchdog(driver.hero_wait_for_data_and_train, hpc, rs,
                                    "LHS warmup")
                _update(rs, hero_state="active")
                results, best_x, best_y = _extract_results(driver, param_specs, fixed_context)
                _update(rs, n_completed=n_init, n_warmup=n_init, results=results,
                        best_x=best_x, best_y=best_y,
                        message=f"warmup done; starting {n_steps} BO steps")
                registry.save_dataset(entry["id"],
                                      driver.dataset.x_data[0],
                                      driver.dataset.y_data[0],
                                      set_completed=False)

            # ── BO phase (same for both paths) ────────────────────────────────
            print(f"[run {run_id[:8]}] BO: {n_steps} steps ({mode_str})...")
            _update(rs, message=f"BO: running {n_steps} steps ({mode_str})")
            driver.run(N_steps=n_steps)

            _update(rs, message=f"BO: waiting for {n_steps} jobs", hero_state="waiting")
            _wait_with_watchdog(driver.hero_wait_for_data_and_train, hpc, rs,
                                "BO")
            _update(rs, hero_state="active")

            results, best_x, best_y = _extract_results(driver, param_specs,
                                                        fixed_context)
            n_done = (n_prior if use_prior else n_init) + n_steps
            _update(rs, n_completed=n_done,
                    results=results, best_x=best_x, best_y=best_y,
                    message="BO complete")
            registry.save_dataset(entry["id"],
                                   driver.dataset.x_data[0],
                                   driver.dataset.y_data[0])
            print(f"[run {run_id[:8]}] BO done. best_y={best_y}")

            _update(rs, status="completed", hero_state="completed",
                    message="optimization complete")

        except Exception as exc:
            _update(rs, status="error",
                    error=f"{exc}\n{traceback.format_exc()}")
        finally:
            try:
                cleanup()
            except Exception:
                pass
            _clear_active_cleanup()


def submit_optimization_run(entry: dict, n_init: int,
                             n_steps: int, acq_func: str,
                             blocking: bool = False,
                             skip_warmstart: bool = False) -> str:
    run_id = create_run(entry["id"], "optimization", n_init + n_steps,
                        experiment_name=entry.get("name", ""))
    t = threading.Thread(target=_opt_worker,
                         args=(run_id, entry, n_init, n_steps, acq_func, blocking,
                               skip_warmstart),
                         daemon=True)
    t.start()
    return run_id
