"""
manager_base.py — Abstract base class for Hero/HPC manager daemons.

Overview
--------
``HeroHPCManager`` implements the full event loop for a manager daemon that
runs on an HPC login node:

1. Authenticates with Hero.
2. Finds or creates the Hero task queue.
3. Runs startup reconciliation (resets stale job IDs, retries error tasks).
4. Enters the main polling loop:
   - **Pass 1**: For tasks that already have a scheduler job ID, check job
     status and call ``hero_initialize`` / ``hero_finalize`` as appropriate.
   - **Pass 2**: For tasks without a job ID, call :meth:`submit_job` to
     submit them to the scheduler.
   - **Running tasks**: Cancel duplicates; finalize jobs that completed while
     their task was in the ``running`` state.

Subclassing
-----------
Override two abstract methods::

    class MyAppManager(HeroHPCManager):

        def submit_job(self, task, machine_name, i_fidelity):
            # Extract parameters from task['metadata'], set up working dirs,
            # run sbatch/qsub, and return the scheduler job ID string.
            params = task['metadata']
            case_dir = Path(f"cases/{task['id']}")
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "config.json").write_text(json.dumps(params))
            cmd = f"sbatch run_sim.sh {task['id']}"
            return self._run_submit(cmd, self.get_scheduler_type(machine_name))

        def read_result(self, task_id):
            result_file = f"result_{task_id}.txt"
            if os.path.exists(result_file):
                value = open(result_file).read().strip()
                os.remove(result_file)
                return value
            return "-1"

    if __name__ == "__main__":
        import sys, hpc_config
        manager = MyAppManager(hpc_config)
        manager.run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)

Scheduler helpers
-----------------
:meth:`_run_submit` runs an ``sbatch`` / ``qsub`` command, parses the job ID,
and raises the appropriate exception on failure.

- :class:`JobLimitError` — raised when the per-user job limit is reached;
  the event loop catches this, skips the rest of Pass 2, and retries next cycle.
- :class:`TaskError` — raised for all other submission failures; the event
  loop marks the task as ``error``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod

from .scheduler import (
    cancel_all_user_jobs,
    cancel_job,
    get_job_status,
    is_job_limit_error,
    parse_job_id,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class JobLimitError(RuntimeError):
    """Raised by :meth:`HeroHPCManager._run_submit` when the scheduler's
    per-user job-submission limit has been reached.  The event loop catches
    this and stops Pass 2 for the current cycle; the task is *not* marked as
    error and will be retried in the next polling iteration.
    """


class TaskError(RuntimeError):
    """Raised by :meth:`HeroHPCManager.submit_job` when a task cannot be
    submitted due to invalid metadata or other task-specific issues.  The
    event loop marks the task as ``error``.
    """


# ---------------------------------------------------------------------------
# Hero subprocess helpers (login-node → Hero API)
# ---------------------------------------------------------------------------

def _call_hero_initialize(task_id: str, machine_name: str, i_fidelity: int) -> int:
    """Mark a task as running in Hero.  Returns exit code (0=success, 2=already claimed)."""
    result = subprocess.run(
        f"{sys.executable} -m adaptive_computing.hero_utils.hero_initialize "
        f"{task_id} {machine_name} {i_fidelity}",
        shell=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        print(f"  hero_initialize: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  hero_initialize stderr: {result.stderr.strip()}")
    return result.returncode


def _call_hero_finalize(result_value: str, task_id: str, machine_name: str, i_fidelity: int) -> bool:
    """Publish result to Hero and mark task done.  Returns True on success."""
    result = subprocess.run(
        f"{sys.executable} -m adaptive_computing.hero_utils.hero_finalize "
        f"{result_value} {task_id} {machine_name} {i_fidelity}",
        shell=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        print(f"  hero_finalize: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  hero_finalize stderr: {result.stderr.strip()}")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class HeroHPCManager(ABC):
    """Abstract manager daemon for Hero + HPC scheduler workflows.

    Attributes:
        hpc_config:      The imported ``hpc_config`` module (must have
                         ``machine_names``, ``remote_usernames``,
                         ``remote_hosts``, ``remote_dirs``,
                         ``python_paths``, ``batch_scripts``).
        poll_interval:   Seconds to sleep between polling cycles (default 5).
        simulation_dir:  Directory to ``chdir`` into before the event loop
                         (relative to the directory containing manager.py).
                         Set to ``None`` to skip the ``chdir``.
    """

    poll_interval: int = 5
    simulation_dir: str | None = "simulation_files"

    def __init__(self, hpc_config) -> None:
        self.hpc_config = hpc_config

    # ------------------------------------------------------------------
    # Abstract interface — implement these two methods in your subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def submit_job(self, task: dict, machine_name: str, i_fidelity: int) -> str:
        """Submit *task* to the scheduler and return the scheduler job ID.

        Implementations should:

        1. Extract simulation parameters from ``task['metadata']``.
        2. Validate them; raise :class:`TaskError` if invalid.
        3. Prepare any required case directories or config files.
        4. Build the ``sbatch`` / ``qsub`` command and call
           :meth:`_run_submit` to execute it.
        5. Return the scheduler job ID string returned by :meth:`_run_submit`.

        Args:
            task:         Hero task dict (keys: ``'id'``, ``'name'``,
                          ``'metadata'``, …).
            machine_name: Logical machine name from ``hpc_config``.
            i_fidelity:   Fidelity level index (0 for single-fidelity).

        Returns:
            Scheduler job ID string (e.g. ``"12345"``).

        Raises:
            :class:`JobLimitError`: Per-user job limit reached; retry next cycle.
            :class:`TaskError`:     Task-specific failure; mark task as error.
        """

    @abstractmethod
    def read_result(self, task_id: str) -> str:
        """Read the simulation output file for *task_id* and return its value.

        Return the result as a string (e.g. ``"3.14"``), or ``"-1"`` if the
        file is missing or unreadable.  Implementations should also delete the
        result file after reading to avoid stale data in future cycles.

        Args:
            task_id: Hero task ID string.

        Returns:
            Result string to pass to ``hero_finalize``.
        """

    # ------------------------------------------------------------------
    # Scheduler helpers available to subclasses
    # ------------------------------------------------------------------

    def get_scheduler_type(self, machine_name: str) -> str:
        """Return ``'slurm'`` or ``'pbs'`` for *machine_name*.

        Reads ``hpc_config.scheduler`` if present; defaults to ``'slurm'``.
        """
        return getattr(self.hpc_config, "scheduler", {}).get(machine_name, "slurm")

    def _run_submit(self, command: str, scheduler_type: str = "slurm") -> str:
        """Run an ``sbatch`` / ``qsub`` command and return the job ID.

        Args:
            command:        Full submission command string.
            scheduler_type: ``'slurm'`` or ``'pbs'``.

        Returns:
            Scheduler job ID string.

        Raises:
            :class:`JobLimitError`: Per-user job limit reached.
            :class:`TaskError`:     Any other non-zero exit code.
        """
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            if is_job_limit_error(result.stderr):
                raise JobLimitError(
                    f"Job limit reached — will retry next cycle.\n"
                    f"  STDERR: {result.stderr.strip()}"
                )
            raise TaskError(
                f"Job submission failed (rc={result.returncode}).\n"
                f"  STDOUT: {result.stdout.strip()}\n"
                f"  STDERR: {result.stderr.strip()}"
            )
        return parse_job_id(result.stdout)

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    def run(self, machine_name: str, i_fidelity: int = 0) -> None:
        """Start the manager event loop.

        Authenticates with Hero, reconciles stale tasks, then polls
        indefinitely until interrupted.

        Args:
            machine_name: Logical machine name matching ``hpc_config``.
            i_fidelity:   Fidelity level index (0 for single-fidelity).
        """
        from hero import HeroClient, get_env_variable

        from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
        set_hero_env_vars()

        try:
            hero_env     = get_env_variable("HERO_ENV", "dev")
            hero_project = get_env_variable("HERO_PROJECT")
            hero_queue   = get_env_variable("HERO_QUEUE")
        except EnvironmentError as e:
            print(e)
            sys.exit(1)

        application_id = f"{hero_env}-{hero_project}"
        queue_name = hero_queue if i_fidelity == 0 else hero_queue + str(i_fidelity)
        scheduler_type = self.get_scheduler_type(machine_name)

        # ---- Authenticate -----------------------------------------------
        hero = HeroClient()
        task_engine = hero.TaskEngine(application_id)
        try:
            hero.authenticate()
        except Exception as e:
            print(f"ERROR: Hero authentication failed: {e}")
            sys.exit(1)

        # ---- Find / create queue ----------------------------------------
        try:
            queue_record = task_engine.read_queue_by_name(name=queue_name, state="active")
            print(f"Found existing active queue: {queue_name}")
        except Exception:
            print(f"No active queue found, creating new queue: {queue_name}")
            queue_record = task_engine.add_queue(name=queue_name)

        print(f"Scheduler type for {machine_name}: {scheduler_type}")

        # ---- Optional chdir ---------------------------------------------
        if self.simulation_dir is not None:
            manager_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            target = os.path.join(manager_dir, self.simulation_dir)
            if os.path.isdir(target):
                os.chdir(target)
            else:
                print(
                    f"WARNING: simulation_dir '{self.simulation_dir}' not found "
                    f"at {target}; skipping chdir."
                )

        # ---- Startup reconciliation -------------------------------------
        print("Running startup reconciliation...")
        for state in ("ready", "error"):
            stale_tasks = task_engine.read_tasks(
                queue_id=queue_record["id"], metatype="Task", state=state
            )
            for task in stale_tasks:
                try:
                    meta   = task.get("metadata") or {}
                    job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
                    needs_reset = False

                    if state == "error":
                        print(f"  Resetting error task {task['id']} to ready for retry")
                        needs_reset = True
                    elif job_id != -1:
                        # Check if the job still exists in the scheduler
                        check_cmd = (
                            f"qstat -x {job_id}"
                            if scheduler_type == "pbs"
                            else f"squeue -j {job_id} --noheader"
                        )
                        check = subprocess.run(
                            check_cmd, shell=True, capture_output=True, text=True
                        )
                        if check.returncode != 0 or not check.stdout.strip():
                            print(
                                f"  Stale job ID {job_id} for task {task['id']} "
                                "not found in scheduler — resetting"
                            )
                            needs_reset = True

                    if needs_reset:
                        task["metadata"]["scheduler_job_id"][machine_name] = -1
                        task["metadata"]["running"][machine_name] = False
                        task_engine.update_task(
                            task_id=task["id"], state="ready",
                            name=task["name"], metadata=task["metadata"],
                        )
                except Exception as e:
                    print(f"  WARNING: reconciliation failed for task {task['id']}: {e}")
        print("Startup reconciliation complete.")

        # ---- Main event loop --------------------------------------------
        print("Continuously checking queue — polling every {self.poll_interval}s...")
        while True:
            self._poll_cycle(
                task_engine, queue_record, machine_name, i_fidelity, scheduler_type
            )
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Single poll cycle (split out for testability)
    # ------------------------------------------------------------------

    def _poll_cycle(self, task_engine, queue_record, machine_name, i_fidelity, scheduler_type):
        """Execute one pass of the event loop."""
        # Status summary
        for state in ("ready", "running", "error", "done"):
            n = len(task_engine.read_tasks(
                queue_id=queue_record["id"], metatype="Task", state=state
            ))
            print(f"  {n} task(s) in '{state}' state.")

        ready_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="ready"
        )
        pass1_processed: set[str] = set()

        # ----------------------------------------------------------
        # Pass 1: check status of already-submitted jobs
        # ----------------------------------------------------------
        for task in ready_tasks:
            meta   = task["metadata"]
            job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
            if job_id == -1:
                continue  # not yet submitted — Pass 2 handles this

            result_file = self.read_result.__func__.__doc__ and None  # peek-only; don't call yet
            result_file_path = f"result_{task['id']}.txt"
            status = get_job_status(job_id, scheduler_type, result_file=result_file_path)

            if status == "RUNNING" and not meta.get("running", {}).get(machine_name, False):
                rc = _call_hero_initialize(task["id"], machine_name, i_fidelity)
                if rc == 0:
                    meta["running"][machine_name] = True
                    task_engine.update_task(
                        task_id=task["id"], state="running",
                        name=task["name"], metadata=meta,
                    )
                    print(f"Task {task['id']}: claimed, state = running")
                elif rc == 2:
                    print(f"Task {task['id']}: already claimed by another machine. Canceling job {job_id}.")
                    cancel_job(job_id, scheduler_type)
                    meta["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task["id"], state="running",
                        name=task["name"], metadata=meta,
                    )
                    pass1_processed.add(task["id"])
                else:
                    print(f"hero_initialize failed (rc={rc}) for task {task['id']}. Canceling job.")
                    cancel_job(job_id, scheduler_type)
                    meta["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task["id"], state="error",
                        name=task["name"], metadata=meta,
                    )
                    pass1_processed.add(task["id"])

            elif status == "COMPLETED":
                result_value = self.read_result(task["id"])
                if not meta.get("running", {}).get(machine_name, False):
                    rc = _call_hero_initialize(task["id"], machine_name, i_fidelity)
                    if rc == 2:
                        print(f"Task {task['id']}: already claimed by another machine (job completed).")
                        meta["scheduler_job_id"][machine_name] = -1
                        task_engine.update_task(
                            task_id=task["id"], state="running",
                            name=task["name"], metadata=meta,
                        )
                        pass1_processed.add(task["id"])
                        continue
                    if rc != 0:
                        print(f"hero_initialize failed (rc={rc}) for completed job {job_id}. Marking error.")
                        meta["scheduler_job_id"][machine_name] = -1
                        task_engine.update_task(
                            task_id=task["id"], state="error",
                            name=task["name"], metadata=meta,
                        )
                        pass1_processed.add(task["id"])
                        continue
                print(
                    f"Job {job_id} completed for task {task['id']}, "
                    f"result={result_value}. Calling hero_finalize."
                )
                _call_hero_finalize(result_value, task["id"], machine_name, i_fidelity)
                pass1_processed.add(task["id"])

            elif status == "UNKNOWN":
                print(
                    f"Job {job_id} not found in scheduler for task {task['id']} "
                    "— stale, resetting to unsubmitted."
                )
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="ready",
                    name=task["name"], metadata=meta,
                )

            elif status in ("FAILED",):
                print(f"Job {job_id} failed for task {task['id']}.")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )
                pass1_processed.add(task["id"])

        # ----------------------------------------------------------
        # Pass 2: submit new jobs for tasks without a job ID
        # ----------------------------------------------------------
        for task in ready_tasks:
            if task["id"] in pass1_processed:
                continue
            meta   = task["metadata"]
            job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
            if job_id != -1:
                continue  # already submitted

            # Ensure bookkeeping fields exist
            needs_update = False
            if "scheduler_job_id" not in meta:
                meta["scheduler_job_id"] = {machine_name: -1}
                needs_update = True
            elif machine_name not in meta["scheduler_job_id"]:
                meta["scheduler_job_id"][machine_name] = -1
                needs_update = True
            if "running" not in meta:
                meta["running"] = {machine_name: False}
                needs_update = True
            elif machine_name not in meta["running"]:
                meta["running"][machine_name] = False
                needs_update = True
            if needs_update:
                task_engine.update_task(
                    task_id=task["id"], state="ready",
                    name=task["name"], metadata=meta,
                )

            try:
                new_job_id = self.submit_job(task, machine_name, i_fidelity)
            except JobLimitError as e:
                print(f"  {e}")
                break  # stop Pass 2 for this cycle; retry next time
            except TaskError as e:
                print(f"Task {task['id']}: submission error — marking as error.\n  {e}")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )
                continue

            meta["scheduler_job_id"][machine_name] = new_job_id
            try:
                task_engine.update_task(
                    task_id=task["id"], state="ready",
                    name=task["name"], metadata=meta,
                )
            except Exception as e:
                # Hero update failed after a successful submission — cancel the
                # job immediately to avoid an orphaned job.
                print(
                    f"WARNING: Hero update failed after submitting job {new_job_id}: {e}\n"
                    "  Canceling job to avoid orphan."
                )
                cancel_job(new_job_id, scheduler_type)
                continue

            print(f"Task {task['id']}: Slurm/PBS job {new_job_id} queued on {machine_name}")

        # ----------------------------------------------------------
        # Running tasks: cancel duplicates; finalize completed jobs
        # ----------------------------------------------------------
        running_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="running"
        )
        for task in running_tasks:
            meta   = task["metadata"]
            job_id = meta.get("scheduler_job_id", {}).get(machine_name, -1)
            meta.setdefault("scheduler_job_id", {}).setdefault(machine_name, -1)
            meta.setdefault("running", {}).setdefault(machine_name, False)

            if not meta["running"][machine_name]:
                if job_id != -1:
                    print(
                        f"Canceling job {job_id} for task {task['id']} "
                        "(task claimed by another machine)."
                    )
                    cancel_job(job_id, scheduler_type)
                    meta["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task["id"], state="running",
                        name=task["name"], metadata=meta,
                    )
                continue

            # Running on this machine — check for completion
            result_file_path = f"result_{task['id']}.txt"
            status = get_job_status(job_id, scheduler_type, result_file=result_file_path)

            if status == "COMPLETED":
                result_value = self.read_result(task["id"])
                print(
                    f"Job {job_id} completed for task {task['id']}, "
                    f"result={result_value}. Calling hero_finalize."
                )
                if not _call_hero_finalize(result_value, task["id"], machine_name, i_fidelity):
                    print(f"WARNING: hero_finalize failed for task {task['id']}")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False

            elif status in ("FAILED",):
                print(f"Job {job_id} failed for running task {task['id']}.")
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )

            elif status == "UNKNOWN":
                print(
                    f"Job {job_id} not found in scheduler for running task {task['id']} "
                    "— marking as error."
                )
                meta["scheduler_job_id"][machine_name] = -1
                meta["running"][machine_name] = False
                task_engine.update_task(
                    task_id=task["id"], state="error",
                    name=task["name"], metadata=meta,
                )
