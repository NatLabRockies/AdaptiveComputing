"""
cleanup.py — Emergency job cancellation for Hero/HPC workflows.

Connects to the Hero queue, iterates all ready/running/error tasks, cancels
any associated scheduler jobs, and also issues a bulk ``scancel`` / ``qdel``
as a failsafe.
"""

from __future__ import annotations

import subprocess
import sys

from .scheduler import cancel_all_user_jobs


def kill_all_scheduler_jobs(hpc_config, machine_name: str, hero_queue: str | None = None) -> None:
    """Cancel all scheduler jobs tracked by the Hero queue on *machine_name*.

    Reads ``HERO_ENV``, ``HERO_PROJECT``, and ``HERO_QUEUE`` from the
    environment (via ``set_hero_env_vars``) unless *hero_queue* is given.

    Args:
        hpc_config:   The imported ``hpc_config`` module.
        machine_name: Logical machine name from ``hpc_config``.
        hero_queue:   Override the ``HERO_QUEUE`` environment variable.

    Raises:
        SystemExit: If Hero authentication fails.
    """
    from hero import HeroClient, get_env_variable
    from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars

    set_hero_env_vars()

    try:
        hero_env     = get_env_variable("HERO_ENV", "dev")
        hero_project = get_env_variable("HERO_PROJECT")
        queue_name   = hero_queue or get_env_variable("HERO_QUEUE")
    except EnvironmentError as e:
        print(e)
        sys.exit(1)

    application_id = f"{hero_env}-{hero_project}"

    hero = HeroClient()
    task_engine = hero.TaskEngine(application_id)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: Hero authentication failed: {e}")
        sys.exit(1)

    try:
        queue_record = task_engine.read_queue_by_name(name=queue_name, state="active")
        print(f"Found existing active queue: {queue_name}")
    except Exception:
        print(f"No active queue found; creating queue: {queue_name}")
        queue_record = task_engine.add_queue(name=queue_name)

    scheduler_type = getattr(hpc_config, "scheduler", {}).get(machine_name, "slurm")
    print(f"Scheduler type for {machine_name}: {scheduler_type}")

    ready_tasks   = task_engine.read_tasks(queue_id=queue_record["id"], metatype="Task", state="ready")
    running_tasks = task_engine.read_tasks(queue_id=queue_record["id"], metatype="Task", state="running")
    error_tasks   = task_engine.read_tasks(queue_id=queue_record["id"], metatype="Task", state="error")

    for task in ready_tasks + running_tasks + error_tasks:
        job_id = (task.get("metadata") or {}).get("scheduler_job_id", {}).get(machine_name, -1)
        if job_id != -1:
            cmd = f"qdel {job_id}" if scheduler_type == "pbs" else f"scancel {job_id}"
            print(f"  Canceling job {job_id} for task {task['id']}: {cmd}")
            subprocess.run(cmd, shell=True, check=False)

    print("Canceling all scheduler jobs for current user as failsafe...")
    cancel_all_user_jobs(scheduler_type)
    print("Done.")
