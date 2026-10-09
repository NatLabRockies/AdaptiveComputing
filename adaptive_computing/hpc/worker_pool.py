"""
worker_pool.py — Auto-managed worker pool for LocalHeroClient-based
multi-fidelity optimization on a single multicore / GPU machine.

Launches worker subprocesses that poll a LocalHeroClient JSON database and
process tasks using either a Python callable (low-fidelity) or a command-line
script (high-fidelity).  No HPC scheduler (SLURM / PBS) is required.

GPU assignment
--------------
Each high-fidelity worker is assigned one GPU device.  Pass ``gpu_ids``
explicitly or set it to ``None`` to auto-detect all NVIDIA GPUs via
``nvidia-smi``.  The device index is injected into the subprocess environment
as ``CUDA_VISIBLE_DEVICES`` so the HF script sees exactly one GPU.

Usage::

    from adaptive_computing.hpc.worker_pool import LocalWorkerPool

    with LocalWorkerPool(
        hero_db_path   = "runs/exp1/local_hero.json",
        base_queue     = "local",           # fidelity 0 → "local", 1 → "local1"
        lf_module      = "simulation",      # importable module
        lf_fn_name     = "run_lf",          # function(x_data: list) -> float
        hf_script      = "/abs/path/to/hf_script.py",
        n_lf_workers   = 4,                 # None → cpu_count - 1
        n_hf_workers   = 1,                 # None → len(gpu_ids) or 1
        gpu_ids        = [0],               # None → auto-detect
    ) as pool:
        # ... run optimization ...

The pool is also usable without the context manager::

    pool = LocalWorkerPool(...)
    pool.start()
    try:
        ...
    finally:
        pool.stop()
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------

def _detect_gpu_ids() -> list[int]:
    """Return a list of available NVIDIA GPU device indices via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(x.strip()) for x in result.stdout.strip().splitlines()]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Worker process entry points  (top-level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------

def _lf_worker_main(
    hero_db_path: str,
    queue_name: str,
    lf_module: str,
    lf_fn_name: str,
    worker_id: str,
    stop_path: str,
    search_paths: list[str],
    poll_interval: float,
) -> None:
    """Entry point for a low-fidelity worker subprocess.

    Polls the LocalHeroClient queue, calls ``lf_fn(x_data) -> float``, and
    writes the result back.

    Args:
        hero_db_path:   Absolute path to the LocalHeroClient JSON database.
        queue_name:     Name of the Hero queue to poll (fidelity-0 queue).
        lf_module:      Importable module name that contains the LF function.
        lf_fn_name:     Name of the callable within that module.
        worker_id:      Human-readable worker identifier (for logging).
        stop_path:      When this file appears on disk the worker exits cleanly.
        search_paths:   Extra directories prepended to ``sys.path`` so that
                        ``lf_module`` can be imported.
        poll_interval:  Seconds between queue polls when the queue is empty.
    """
    for p in reversed(search_paths):
        if p not in sys.path:
            sys.path.insert(0, p)

    import importlib
    mod = importlib.import_module(lf_module)
    lf_fn = getattr(mod, lf_fn_name)

    from adaptive_computing.local_hero import LocalHeroClient
    task_engine = LocalHeroClient(hero_db_path).TaskEngine()

    print(f"[{worker_id}] LF worker started (queue={queue_name!r})", flush=True)

    while not os.path.exists(stop_path):
        try:
            queue = task_engine.read_queue_by_name(name=queue_name, state="active")
        except ValueError:
            time.sleep(poll_interval)
            continue

        ready = task_engine.read_tasks(queue_id=queue["id"], metatype="Task", state="ready")
        for task in ready:
            if not task_engine.claim_task(task["id"], worker_id):
                continue

            x_data = task["metadata"]["x_data"]
            try:
                result = lf_fn(x_data)
                task["metadata"]["y_data"] = [float(result)]
                task_engine.update_task(task["id"], "done", task["name"], task["metadata"])
                print(f"[{worker_id}] done  x={x_data} → y={result:.4f}", flush=True)
            except Exception as exc:
                task["metadata"]["error"] = str(exc)
                task_engine.update_task(task["id"], "error", task["name"], task["metadata"])
                print(f"[{worker_id}] error x={x_data}: {exc}", flush=True)

        time.sleep(poll_interval)

    print(f"[{worker_id}] LF worker stopped.", flush=True)


def _hf_worker_main(
    hero_db_path: str,
    queue_name: str,
    hf_script: str,
    worker_id: str,
    gpu_id: int,
    stop_path: str,
    poll_interval: float,
) -> None:
    """Entry point for a high-fidelity worker subprocess.

    Polls the LocalHeroClient queue, runs ``hf_script`` as a subprocess, reads
    the result from a temporary file the script writes, and updates the task.

    The HF script is called with::

        python hf_script.py --x_data '[1.5, 2.3]' --output /tmp/result.txt

    It must write a single float (the simulation result) to ``--output``.

    Args:
        hero_db_path:   Absolute path to the LocalHeroClient JSON database.
        queue_name:     Name of the Hero queue to poll (fidelity-1 queue).
        hf_script:      Absolute path to the HF simulation script.
        worker_id:      Human-readable worker identifier (for logging).
        gpu_id:         GPU device index injected as ``CUDA_VISIBLE_DEVICES``.
                        Pass ``-1`` for CPU-only HF workers.
        stop_path:      When this file appears on disk the worker exits cleanly.
        poll_interval:  Seconds between queue polls when the queue is empty.
    """
    from adaptive_computing.local_hero import LocalHeroClient
    task_engine = LocalHeroClient(hero_db_path).TaskEngine()

    env = os.environ.copy()
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_label = f"GPU:{gpu_id}"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        gpu_label = "CPU"

    print(f"[{worker_id}] HF worker started (queue={queue_name!r}, {gpu_label})", flush=True)

    while not os.path.exists(stop_path):
        try:
            queue = task_engine.read_queue_by_name(name=queue_name, state="active")
        except ValueError:
            time.sleep(poll_interval)
            continue

        ready = task_engine.read_tasks(queue_id=queue["id"], metatype="Task", state="ready")
        for task in ready:
            if not task_engine.claim_task(task["id"], worker_id):
                continue

            x_data = task["metadata"]["x_data"]
            x_json = json.dumps(x_data)

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
                out_path = tf.name

            try:
                proc = subprocess.run(
                    [sys.executable, hf_script,
                     "--x_data", x_json,
                     "--output", out_path],
                    env=env,
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"HF script exited {proc.returncode}:\n{proc.stderr.strip()}"
                    )
                result_str = Path(out_path).read_text().strip()
                result = float(result_str)
                task["metadata"]["y_data"] = [result]
                task_engine.update_task(task["id"], "done", task["name"], task["metadata"])
                print(f"[{worker_id}] done  x={x_data} → y={result:.4f}", flush=True)
            except Exception as exc:
                task["metadata"]["error"] = str(exc)
                task_engine.update_task(task["id"], "error", task["name"], task["metadata"])
                print(f"[{worker_id}] error x={x_data}: {exc}", flush=True)
            finally:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

        time.sleep(poll_interval)

    print(f"[{worker_id}] HF worker stopped.", flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LocalWorkerPool:
    """Auto-managed pool of LF (Python callable) and HF (script) workers.

    Each worker runs in a separate subprocess so that crashes are isolated and
    resources (CPU cores, GPU memory) are properly bounded.

    Queue naming follows the LocalHeroClient / HeroDataset convention:
      * fidelity 0 → ``base_queue``
      * fidelity 1 → ``base_queue + "1"``

    Args:
        hero_db_path:  Path to the LocalHeroClient JSON database shared by all
                       workers and the controller.
        base_queue:    Base queue name (= ``LocalHeroClient.queue_name``).
        lf_module:     Python module containing the LF function.  Must be
                       importable by the worker subprocess; use ``search_paths``
                       if it is not on the default ``sys.path``.
        lf_fn_name:    Name of the LF callable: ``fn(x_data: list) -> float``.
        hf_script:     Absolute path to the HF simulation script.  The script
                       must accept ``--x_data <json>`` and ``--output <path>``
                       and write a single float to the output file.
        n_lf_workers:  Number of LF worker processes.
                       Default: ``max(1, os.cpu_count() - 1)``.
        n_hf_workers:  Number of HF worker processes.
                       Default: ``len(gpu_ids)`` (one worker per GPU), or 1
                       when no GPUs are found.
        gpu_ids:       GPU device indices assigned round-robin to HF workers.
                       ``None`` → auto-detect via nvidia-smi.  Pass ``[]`` to
                       force CPU-only HF workers.
        search_paths:  Extra directories prepended to ``sys.path`` in each LF
                       worker subprocess so that ``lf_module`` can be imported.
                       Defaults to the directory containing ``hero_db_path``.
        poll_interval: Seconds between queue polls when the queue is empty.
    """

    def __init__(
        self,
        hero_db_path: str,
        base_queue: str,
        lf_module: str,
        lf_fn_name: str,
        hf_script: str,
        n_lf_workers: int | None = None,
        n_hf_workers: int | None = None,
        gpu_ids: list[int] | None = None,
        search_paths: list[str] | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.hero_db_path  = os.path.abspath(hero_db_path)
        self.base_queue    = base_queue
        self.lf_module     = lf_module
        self.lf_fn_name    = lf_fn_name
        self.hf_script     = os.path.abspath(hf_script)
        self.poll_interval = poll_interval

        # Resolve GPU IDs
        if gpu_ids is None:
            gpu_ids = _detect_gpu_ids()
        self._gpu_ids = gpu_ids

        # Worker counts — cap the LF default at 8 so we don't over-subscribe
        # shared login nodes (which can have 100+ cores).  On a personal
        # GPU workstation, set n_lf_workers explicitly to use more cores.
        cpu_count = os.cpu_count() or 1
        default_lf = max(1, min(cpu_count - 1, 8))
        self._n_lf = n_lf_workers if n_lf_workers is not None else default_lf
        self._n_hf = n_hf_workers if n_hf_workers is not None else (len(self._gpu_ids) or 1)

        # Search paths for the LF module import inside worker subprocesses
        default_sp = [os.path.dirname(self.hero_db_path)]
        self._search_paths = list(search_paths) if search_paths else default_sp

        # Shutdown signal: workers exit when this sentinel file appears
        self._stop_path = self.hero_db_path + ".stop"

        self._processes: list[mp.Process] = []
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch all worker subprocesses."""
        if self._started:
            raise RuntimeError("LocalWorkerPool.start() called twice")
        if os.path.exists(self._stop_path):
            os.remove(self._stop_path)

        lf_queue = self.base_queue
        hf_queue = self.base_queue + "1"

        for i in range(self._n_lf):
            wid = f"lf-{i}"
            p = mp.Process(
                target=_lf_worker_main,
                kwargs=dict(
                    hero_db_path  = self.hero_db_path,
                    queue_name    = lf_queue,
                    lf_module     = self.lf_module,
                    lf_fn_name    = self.lf_fn_name,
                    worker_id     = wid,
                    stop_path     = self._stop_path,
                    search_paths  = self._search_paths,
                    poll_interval = self.poll_interval,
                ),
                name=wid,
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        for i in range(self._n_hf):
            wid = f"hf-{i}"
            gpu_id = self._gpu_ids[i % len(self._gpu_ids)] if self._gpu_ids else -1
            p = mp.Process(
                target=_hf_worker_main,
                kwargs=dict(
                    hero_db_path  = self.hero_db_path,
                    queue_name    = hf_queue,
                    hf_script     = self.hf_script,
                    worker_id     = wid,
                    gpu_id        = gpu_id,
                    stop_path     = self._stop_path,
                    poll_interval = self.poll_interval,
                ),
                name=wid,
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        self._started = True
        gpu_msg = f", GPUs={self._gpu_ids}" if self._gpu_ids else ""
        print(
            f"[LocalWorkerPool] Started {self._n_lf} LF + {self._n_hf} HF workers"
            f"{gpu_msg}  (db={self.hero_db_path})"
        )

    def stop(self, timeout: float = 15.0) -> None:
        """Signal all workers to stop and wait for them to exit.

        Args:
            timeout: Seconds to wait for graceful shutdown before terminating.
        """
        if not self._started:
            return
        # Drop the sentinel file — workers check for it each poll cycle
        Path(self._stop_path).touch()
        for p in self._processes:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        self._processes.clear()
        try:
            os.remove(self._stop_path)
        except OSError:
            pass
        self._started = False
        print("[LocalWorkerPool] All workers stopped.")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LocalWorkerPool":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
