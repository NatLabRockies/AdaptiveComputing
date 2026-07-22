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
import signal
import subprocess
import sys
import time
import traceback

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


def _call_hero_initialize(task_id, machine_name):
    """Mark a task as running. Returns exit code: 0=success, 2=already claimed, other=error."""
    result = subprocess.run(
        f"{sys.executable} -m adaptive_computing.hero_utils.hero_initialize {task_id} {machine_name}",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode not in (0, 2):
        print(f"hero_initialize failed (rc={result.returncode}): {result.stderr.strip()}")
    return result.returncode


def _call_hero_finalize(result_value, task_id, machine_name):
    """Publish result back to Hero and mark task done."""
    result = subprocess.run(
        f"{sys.executable} -m adaptive_computing.hero_utils.hero_finalize {result_value} {task_id} {machine_name}",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"hero_finalize failed (rc={result.returncode}): {result.stderr.strip()}")
    return result.returncode == 0


def hero_manager():
    if len(sys.argv) > 1:
        machine_name = sys.argv[1]
    else:
        print("Missing machine_name as a command-line argument.")
        sys.exit(1)

    print(f"Manager PID: {os.getpid()}")

    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"Manager received signal {sig_name} ({signum}) — shutting down.", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGHUP, _handle_signal)

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

    _consecutive_errors = 0
    while True:
      try:
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
            # Ensure bookkeeping fields exist (tasks created externally may omit them)
            meta    = current_task["metadata"]
            task_id = current_task["id"]
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
                print(f"Task {task_id}: initializing missing bookkeeping fields.")
                task_engine.update_task(
                    task_id=task_id, state="ready",
                    name=current_task["name"], metadata=meta,
                )

            if meta["scheduler_job_id"][machine_name] == -1:

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

                if machine_name in hpc_config.batch_scripts:
                    scripts = hpc_config.batch_scripts[machine_name]
                    script_name = scripts[0] if isinstance(scripts, list) else scripts
                    slurm_out   = os.path.join(case_logs_dir, "slurm_%j.out")
                    slurm_err   = os.path.join(case_logs_dir, "slurm_%j.err")

                    sbatch_flags = f"--output={slurm_out} --error={slurm_err} "
                    if getattr(hpc_config, 'debug_run', False):
                        debug_parts = getattr(hpc_config, 'debug_partitions', {})
                        partition = debug_parts.get(machine_name)
                        if partition:
                            sbatch_flags += f"--partition={partition} "
                    command = f"sbatch {sbatch_flags}{script_name} {task_id}"
                else:
                    raise RuntimeError(
                        f"Machine '{machine_name}' not in hpc_config.batch_scripts. "
                        f"Available: {list(hpc_config.batch_scripts.keys())}"
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
                # Already submitted — check Slurm status
                task_id = current_task["id"]
                job_id  = current_task["metadata"]["scheduler_job_id"][machine_name]
                sacct   = subprocess.run(
                    f"sacct -j {job_id} --format=State --noheader",
                    shell=True, capture_output=True, text=True,
                )
                sacct_out = sacct.stdout.strip()

                if "COMPLETED" in sacct_out:
                    status = "COMPLETED"
                elif any(s in sacct_out for s in ("FAILED", "CANCELLED", "TIMEOUT")):
                    status = "FAILED"
                elif not sacct_out:
                    # sacct delay: job may still be running or just finished
                    squeue = subprocess.run(
                        f"squeue -j {job_id} --noheader",
                        shell=True, capture_output=True, text=True,
                    )
                    if squeue.stdout.strip():
                        status = "PENDING"  # still in SLURM
                    else:
                        # Not in squeue either — check for result file
                        result_file = os.path.join(agent_dir, "simulation_files", f"result_{task_id}.txt")
                        status = "COMPLETED" if os.path.exists(result_file) else "PENDING"
                else:
                    status = "PENDING"

                if status == "COMPLETED":
                    result_file = os.path.join(agent_dir, "simulation_files", f"result_{task_id}.txt")
                    result_value = "-1"
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result_value = f.read().strip()
                        os.remove(result_file)
                    rc = _call_hero_initialize(task_id, machine_name)
                    if rc == 2:
                        print(f"Task {task_id} already claimed by another machine — skipping.")
                        continue
                    if rc != 0:
                        print(f"hero_initialize failed for task {task_id}, marking error.")
                        task_engine.update_task(
                            task_id=task_id, state="error",
                            name=current_task["name"], metadata=current_task["metadata"],
                        )
                        continue
                    _call_hero_finalize(result_value, task_id, machine_name)
                    print(f"Task {task_id}: finalized with result={result_value}")

                elif status == "FAILED":
                    print(f"Slurm job {job_id} in error state: {sacct_out}")
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    current_task["metadata"]["running"][machine_name] = False
                    task_engine.update_task(
                        task_id=task_id, state="error",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )

        # ----------------------------------------------------------------
        # Running tasks: cancel if another machine claimed them; finalize
        # if job completed but hero_finalize wasn't called yet
        # ----------------------------------------------------------------
        running_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="running"
        )
        for current_task in running_tasks:
            task_id = current_task["id"]
            meta    = current_task["metadata"]
            meta.setdefault("scheduler_job_id", {}).setdefault(machine_name, -1)
            meta.setdefault("running", {}).setdefault(machine_name, False)
            if not meta["running"][machine_name]:
                job_id = current_task["metadata"]["scheduler_job_id"][machine_name]
                if job_id != -1:
                    print(f"Cancelling Slurm job {job_id} (task claimed by another machine)")
                    subprocess.run(f"scancel {job_id}", shell=True)
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task_id, state="running",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )
            else:
                job_id = current_task["metadata"]["scheduler_job_id"][machine_name]
                sacct  = subprocess.run(
                    f"sacct -j {job_id} --format=State --noheader",
                    shell=True, capture_output=True, text=True,
                )
                sacct_out = sacct.stdout.strip()

                if "COMPLETED" in sacct_out:
                    result_file = os.path.join(agent_dir, "simulation_files", f"result_{task_id}.txt")
                    result_value = "-1"
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result_value = f.read().strip()
                        os.remove(result_file)
                    _call_hero_finalize(result_value, task_id, machine_name)
                    print(f"Task {task_id}: finalized with result={result_value}")
                elif any(s in sacct_out for s in ("FAILED", "CANCELLED", "TIMEOUT")):
                    print(f"Slurm job {job_id} failed: {sacct_out}")
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    current_task["metadata"]["running"][machine_name] = False
                    task_engine.update_task(
                        task_id=task_id, state="error",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )

        _consecutive_errors = 0
      except Exception:
        _consecutive_errors += 1
        print(f"ERROR in manager loop (consecutive error #{_consecutive_errors}):", flush=True)
        traceback.print_exc()
        if _consecutive_errors >= 5:
            print("Too many consecutive errors — exiting.", flush=True)
            sys.exit(1)
      time.sleep(5)


if __name__ == "__main__":
    hero_manager()
