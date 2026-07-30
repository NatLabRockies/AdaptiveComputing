"""
adaptive_computing.hpc
======================
Reusable HPC orchestration utilities for Hero-backed workflows.

Key components
--------------
HeroHPCManager
    Abstract base class for a manager daemon that runs on an HPC login node,
    polls the Hero task queue, submits scheduler jobs (SLURM or PBS), monitors
    job status, and finalizes completed tasks back to Hero.

    Subclass it and implement two abstract methods:

        submit_job(task, machine_name, i_fidelity) -> str
            Validate task metadata, prepare working directories / config files,
            submit the batch job, and return the scheduler job ID.

        read_result(task_id) -> str
            Read the simulation output file for the given task_id and return
            the result as a string (or "-1" on failure).

    Then run the manager with::

        manager = MyAppManager(hpc_config)
        manager.run(machine_name, i_fidelity)

LocalHPCManager
    In-process manager for controllers that run on any node with scheduler
    access and internet access to the Hero API (typically a login node, but
    also works from a compute node with internet access).
    No SSH, no tmux, no hpc_config.py required.  Call run_until_done() after
    queuing tasks; it submits and monitors scheduler jobs and returns when all
    tasks are done or in error.

        manager = MyAppManager(
            machine_name='local',
            batch_scripts=['/abs/path/to/script.sh'],
            scheduler_type='slurm',
            simulation_dir='/abs/path/to/simulation_files',
        )
        ac_driver.add_samples(x_data, i_fidelity=0)
        manager.run_until_done(i_fidelity=0)
        ac_driver.hero_wait_for_data_and_train()

autonomous_managers
    SSH-based remote manager lifecycle: start, poll, and stop manager tmux
    sessions on remote HPC login nodes.

cleanup
    kill_all_scheduler_jobs — cancel all Hero-tracked scheduler jobs on a machine.

remote_manager
    CLI for managing the manager tmux session on remote HPC login nodes
    (replaces the old run_manager.sh / run_kill_scheduler_jobs.sh shell scripts).
"""

from .manager_base import HeroHPCManager, JobLimitError, TaskError
from .local_manager import LocalHPCManager
from .autonomous import (
    setup_remote_state,
    run_remote_managers,
    wait_for_managers,
    cleanup_remote_managers,
)
from .cleanup import kill_all_scheduler_jobs

__all__ = [
    "HeroHPCManager",
    "LocalHPCManager",
    "JobLimitError",
    "TaskError",
    "setup_remote_state",
    "run_remote_managers",
    "wait_for_managers",
    "cleanup_remote_managers",
    "kill_all_scheduler_jobs",
]
