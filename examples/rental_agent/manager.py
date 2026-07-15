"""
manager.py — AC/Hero queue manager for rental car mock simulations.
Adapted from AdaptiveComputing/examples/hero_HPC_managers/manager.py
and /projects/newbridge/kgriffin/stdp-mnist/agent/manager.py.

Polls the Hero task queue and launches Slurm jobs on Kestrel.
Each task's metadata carries:
    utility_rate          str   "Moderate" | "Aggressive"
    storage              str   "Grid" | "Storage-025" | ...
    number_of_daily_evs   int
    return_soc            int   25 | 35 | 45 | 55
    scheduler_job_id          dict  {machine_name: job_id | -1}
    running               dict  {machine_name: bool}
"""

from hero import HeroClient, get_env_variable
import json as _json
import numpy as np
import os
import subprocess
import sys
import time

from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
set_hero_env_vars()

try:
    import hpc_config
except ImportError:
    print("ERROR: hpc_config.py not found in the agent directory.")
    sys.exit(1)

try:
    HERO_ENV     = get_env_variable('HERO_ENV', 'dev')
    HERO_PROJECT = get_env_variable('HERO_PROJECT')
    HERO_QUEUE   = get_env_variable('HERO_QUEUE')
except EnvironmentError as e:
    print(e)
    sys.exit(1)

APPLICATION_ID = f'{HERO_ENV}-{HERO_PROJECT}'


def hero_manager():
    if len(sys.argv) > 1:
        machine_name = sys.argv[1]
    else:
        print("Missing machine_name as a command-line argument.")
        sys.exit(1)

    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: HERO authentication failed: {e}")
        sys.exit(1)

    try:
        queue_record = task_engine.read_queue_by_name(name=HERO_QUEUE, state="active")
        print(f'Found existing active queue: {HERO_QUEUE}')
    except Exception:
        print(f'No active queue found, creating new queue: {HERO_QUEUE}')
        queue_record = task_engine.add_queue(name=HERO_QUEUE)

    print("Continuously checking queue — will claim ready tasks and launch Slurm jobs...")
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(agent_dir, "simulation_files"))

    while True:
        for state in ("ready", "running", "error", "done"):
            n = len(task_engine.read_tasks(
                queue_id=queue_record["id"], metatype="Task", state=state
            ))
            print(f"  {n} task(s) in \"{state}\" state.")

        # ----------------------------------------------------------------
        # Ready tasks: submit to Slurm if not yet queued on this machine
        # ----------------------------------------------------------------
        ready_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="ready"
        )
        for current_task in ready_tasks:
            if current_task["metadata"]["scheduler_job_id"][machine_name] == -1:
                meta     = current_task["metadata"]
                task_id  = current_task["id"]

                # Validate required fields — skip stale/incompatible tasks
                required = {"utility_rate", "storage", "number_of_daily_evs", "return_soc"}
                missing  = required - meta.keys()
                if missing:
                    print(f"Skipping task {task_id}: missing metadata fields {missing}")
                    print(f"  Available keys: {list(meta.keys())}")
                    task_engine.update_task(
                        task_id=task_id, state="error",
                        name=current_task["name"], metadata=meta,
                    )
                    continue

                # Build case directory and write config.json
                case_dir      = os.path.join(agent_dir, "cases_agent", task_id)
                case_logs_dir = os.path.join(case_dir, "logs")
                os.makedirs(case_logs_dir, exist_ok=True)

                config_data = {
                    "utility_rate":        meta["utility_rate"],
                    "storage":            meta["storage"],
                    "number_of_daily_evs": meta["number_of_daily_evs"],
                    "return_soc":          meta["return_soc"],
                    "description": (
                        f"utility_rate={meta['utility_rate']}, "
                        f"storage={meta['storage']}, "
                        f"number_of_daily_evs={meta['number_of_daily_evs']}, "
                        f"return_soc={meta['return_soc']}, "
                        f"task_id={task_id}"
                    ),
                }
                config_path = os.path.join(case_dir, "config.json")
                with open(config_path, "w") as f:
                    _json.dump(config_data, f, indent=4)
                print(f"Wrote config.json: {config_path}")

                if machine_name in hpc_config.slurm_scripts:
                    script_name = hpc_config.slurm_scripts[machine_name]
                    slurm_out   = os.path.join(case_logs_dir, "slurm_%j.out")
                    slurm_err   = os.path.join(case_logs_dir, "slurm_%j.err")

                    sbatch_flags = f"--output={slurm_out} --error={slurm_err} "
                    if getattr(hpc_config, 'debug_run', False):
                        debug_parts = getattr(hpc_config, 'debug_partitions', {})
                        partition = debug_parts.get(machine_name)
                        if partition:
                            sbatch_flags += f"--partition={partition} "
                    command = f"sbatch {sbatch_flags}{script_name} {task_id} {machine_name}"
                else:
                    raise RuntimeError(
                        f"Machine '{machine_name}' not in hpc_config.slurm_scripts. "
                        f"Available: {list(hpc_config.slurm_scripts.keys())}"
                    )

                print(f"Submitting: {command}")
                result = subprocess.run(
                    command, shell=True, check=False, capture_output=True, text=True
                )

                if result.returncode != 0:
                    print("sbatch error:")
                    print("  STDOUT:", result.stdout)
                    print("  STDERR:", result.stderr)
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    current_task["metadata"]["running"][machine_name] = False
                    task_engine.update_task(
                        task_id=current_task["id"], state="error",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )
                    continue

                job_id = result.stdout.strip().split()[-1]
                current_task["metadata"]["scheduler_job_id"][machine_name] = job_id
                task_engine.update_task(
                    task_id=current_task["id"], state="ready",
                    name=current_task["name"], metadata=current_task["metadata"],
                )
                print(f"Task {current_task['id']}: Slurm job {job_id} queued on {machine_name}")

            else:
                # Already queued — check for Slurm errors
                job_id       = current_task["metadata"]["scheduler_job_id"][machine_name]
                status_check = subprocess.run(
                    f"sacct -j {job_id} --format=State --noheader",
                    shell=True, capture_output=True, text=True,
                )
                status = status_check.stdout.strip()
                if any(s in status for s in ("FAILED", "CANCELLED", "TIMEOUT")):
                    print(f"Slurm job {job_id} in error state: {status}")
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    current_task["metadata"]["running"][machine_name] = False
                    task_engine.update_task(
                        task_id=current_task["id"], state="error",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )

        # ----------------------------------------------------------------
        # Running tasks: cancel if another machine claimed them
        # ----------------------------------------------------------------
        running_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="running"
        )
        for current_task in running_tasks:
            if not current_task["metadata"]["running"][machine_name]:
                job_id = current_task["metadata"]["scheduler_job_id"][machine_name]
                if job_id != -1:
                    print(f"Cancelling Slurm job {job_id} (task claimed by another machine)")
                    subprocess.run(f"scancel {job_id}", shell=True)
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=current_task["id"], state="running",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )
            else:
                job_id       = current_task["metadata"]["scheduler_job_id"][machine_name]
                status_check = subprocess.run(
                    f"sacct -j {job_id} --format=State --noheader",
                    shell=True, capture_output=True, text=True,
                )
                status = status_check.stdout.strip()
                if any(s in status for s in ("FAILED", "CANCELLED", "TIMEOUT")):
                    print(f"Slurm job {job_id} failed: {status}")
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    current_task["metadata"]["running"][machine_name] = False
                    task_engine.update_task(
                        task_id=current_task["id"], state="error",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )

        time.sleep(5)


if __name__ == "__main__":
    hero_manager()
