"""
manager.py — AC/Hero queue manager for rental car mock simulations.
Adapted from AdaptiveComputing/examples/hero_HPC_managers/manager.py
and /projects/newbridge/kgriffin/stdp-mnist/agent/manager.py.

Polls the Hero task queue and launches SLURM or PBS jobs depending on
the scheduler field in hpc_config.py.  Each task's metadata carries:
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


def _call_hero_initialize(task_id, machine_name, task_engine):
    """Mark a task as running. Returns 0 on success, 2 if already claimed, 1 on error."""
    from adaptive_computing.hero_utils.hero_initialize import hero_initialize, TaskAlreadyClaimed
    try:
        hero_initialize(task_id, machine_name, task_engine=task_engine)
        return 0
    except TaskAlreadyClaimed:
        print(f"  hero_initialize: task {task_id} already claimed by another machine.")
        return 2
    except Exception as e:
        print(f"  hero_initialize failed for task {task_id}: {e}")
        return 1


def _call_hero_finalize(result_value, task_id, machine_name, task_engine):
    """Publish result back to Hero and mark task done. Returns True on success."""
    from adaptive_computing.hero_utils.hero_finalize import hero_finalize
    try:
        hero_finalize(result_value, task_id, machine_name, task_engine=task_engine)
        return True
    except Exception as e:
        print(f"  hero_finalize failed for task {task_id}: {e}")
        return False


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
        ready_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="ready"
        )

        # Split 'ready' into unsubmitted vs submitted-but-pending in the scheduler.
        # Hero has no separate "queued" state, so both live under 'ready'.
        n_queued = sum(
            1 for t in ready_tasks
            if t.get("metadata", {}).get("scheduler_job_id", {}).get(machine_name, -1) != -1
        )
        n_unsubmitted = len(ready_tasks) - n_queued
        print(f"  {n_unsubmitted} task(s) in 'ready' state (not yet submitted to scheduler).")
        print(f"  {n_queued} task(s) in 'queued' state (submitted to scheduler, awaiting execution).")
        for state in ("running", "error", "done"):
            n = len(task_engine.read_tasks(
                queue_id=queue_record["id"], metatype="Task", state=state
            ))
            print(f"  {n} task(s) in '{state}' state.")

        # ----------------------------------------------------------------
        # Ready tasks: submit to Slurm if not yet queued on this machine
        # ----------------------------------------------------------------
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

            # ── Early-claim check ─────────────────────────────────────────────
            # When one machine's scheduler job transitions to RUNNING it writes
            # early_claim.  Any other machine that sees it cancels its own job
            # immediately rather than waiting for the claimer to finish.
            early_claimer = meta.get("early_claim")
            if early_claimer and early_claimer != machine_name:
                own_job = meta.get("scheduler_job_id", {}).get(machine_name, -1)
                sched_ec = getattr(hpc_config, 'scheduler', {}).get(machine_name, 'slurm')
                if own_job != -1:
                    cc = f"qdel {own_job}" if sched_ec == 'pbs' else f"scancel {own_job}"
                    print(f"Task {task_id}: {early_claimer} is RUNNING — cancelling our "
                          f"{sched_ec.upper()} job {own_job}")
                    subprocess.run(cc, shell=True)
                    # Do NOT reset job_id to -1 here: if qdel fails silently we'd
                    # lose track of the job.  The running_tasks and done_tasks
                    # sections will retry qdel and reset once confirmed gone.
                continue

            if meta["scheduler_job_id"][machine_name] == -1:

                # Skip if another machine claimed the task while we were queued.
                if meta.get("early_claim") and meta["early_claim"] != machine_name:
                    continue

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

                scheduler_type = getattr(hpc_config, 'scheduler', {}).get(machine_name, 'slurm')

                if machine_name in hpc_config.batch_scripts:
                    scripts = hpc_config.batch_scripts[machine_name]
                    script_name = scripts[0] if isinstance(scripts, list) else scripts

                    if scheduler_type == 'pbs':
                        pbs_out = os.path.join(case_logs_dir, "pbs.out")
                        pbs_err = os.path.join(case_logs_dir, "pbs.err")
                        python_path = (
                            getattr(hpc_config, 'python_paths', {})
                            .get(machine_name, '')
                        )
                        qsub_vars = f"task_id={task_id}"
                        if python_path:
                            qsub_vars += f",python_path={python_path}"
                        command = (
                            f"qsub -v \"{qsub_vars}\" "
                            f"-o {pbs_out} -e {pbs_err} "
                            f"{script_name}"
                        )
                    else:
                        slurm_out = os.path.join(case_logs_dir, "slurm_%j.out")
                        slurm_err = os.path.join(case_logs_dir, "slurm_%j.err")
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
                    print(f"{'qsub' if scheduler_type == 'pbs' else 'sbatch'} error:")
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
                # Re-read metadata before writing to capture any concurrent
                # updates from other managers (avoids overwriting their job_ids).
                try:
                    fresh = task_engine.read_tasks(
                        queue_id=queue_record["id"], metatype="Task", state="ready"
                    )
                    fresh_task = next((t for t in fresh if t["id"] == current_task["id"]), None)
                    write_meta = fresh_task["metadata"] if fresh_task else current_task["metadata"]
                except Exception:
                    write_meta = current_task["metadata"]
                write_meta.setdefault("scheduler_job_id", {})[machine_name] = job_id
                write_meta.setdefault("running", {})[machine_name] = False
                task_engine.update_task(
                    task_id=current_task["id"], state="ready",
                    name=current_task["name"], metadata=write_meta,
                )
                print(f"Task {current_task['id']}: {scheduler_type.upper()} job {job_id} queued on {machine_name}")

            else:
                # Already submitted — check scheduler status
                task_id        = current_task["id"]
                job_id         = current_task["metadata"]["scheduler_job_id"][machine_name]
                scheduler_type = getattr(hpc_config, 'scheduler', {}).get(machine_name, 'slurm')
                result_file    = os.path.join(agent_dir, "simulation_files", f"result_{task_id}.txt")

                if scheduler_type == 'pbs':
                    import re as _re
                    qstat = subprocess.run(
                        f"qstat -f -x {job_id}",
                        shell=True, capture_output=True, text=True,
                    )
                    if qstat.returncode != 0 or not qstat.stdout.strip():
                        # qstat failed — job not found in scheduler.  Could be:
                        #   (a) truly finished and left the queue, or
                        #   (b) transient error / job not yet visible.
                        # Use result file as ground truth to avoid false positives.
                        status = "COMPLETED" if os.path.exists(result_file) else "PENDING"
                    else:
                        state_match = _re.search(r'job_state\s*=\s*(\S+)', qstat.stdout)
                        state = state_match.group(1) if state_match else "?"
                        if state in ('F', 'C'):
                            exit_match = _re.search(r'exit_status\s*=\s*(\S+)', qstat.stdout)
                            exit_val = int(exit_match.group(1)) if exit_match else 0
                            status = "COMPLETED" if exit_val == 0 else "FAILED"
                        elif state in ('R', 'E'):
                            status = "RUNNING"   # R=running, E=exiting (still running)
                        else:
                            status = "PENDING"   # Q=queued, H=held, W=waiting, etc.
                else:
                    sacct = subprocess.run(
                        f"sacct -j {job_id} --format=State --noheader",
                        shell=True, capture_output=True, text=True,
                    )
                    sacct_out = sacct.stdout.strip()
                    if "COMPLETED" in sacct_out:
                        status = "COMPLETED"
                    elif any(s in sacct_out for s in ("FAILED", "CANCELLED", "TIMEOUT")):
                        status = "FAILED"
                    elif "RUNNING" in sacct_out or "COMPLETING" in sacct_out:
                        # sacct shows the job is actively executing
                        status = "RUNNING"
                    elif not sacct_out:
                        # sacct has no record yet — fall back to squeue
                        squeue = subprocess.run(
                            f"squeue -j {job_id} --format=%T --noheader",
                            shell=True, capture_output=True, text=True,
                        )
                        sq_state = squeue.stdout.strip()
                        if "RUNNING" in sq_state or "COMPLETING" in sq_state:
                            status = "RUNNING"
                        elif sq_state:
                            status = "PENDING"
                        else:
                            status = "COMPLETED" if os.path.exists(result_file) else "PENDING"
                    else:
                        status = "PENDING"

                if status == "RUNNING":
                    # Job started on this machine — write early_claim so other
                    # machines cancel their competing jobs immediately.
                    try:
                        fresh_r = task_engine.read_tasks(
                            queue_id=queue_record["id"], metatype="Task", state="ready"
                        )
                        ft = next((t for t in fresh_r if t["id"] == task_id), None)
                        fm = ft["metadata"] if ft else current_task["metadata"]
                    except Exception:
                        fm = current_task["metadata"]
                    existing = fm.get("early_claim")
                    if not existing:
                        fm["early_claim"] = machine_name
                        task_engine.update_task(task_id=task_id, state="ready",
                                                name=current_task["name"], metadata=fm)
                        print(f"Task {task_id}: job RUNNING on {machine_name} — early_claim set")
                    elif existing != machine_name:
                        # Another machine already claimed — cancel ours.
                        # Keep job_id in metadata so running_tasks/done_tasks can
                        # retry qdel if this one fails silently.
                        print(f"Task {task_id}: {existing} claimed first — cancelling our "
                              f"{scheduler_type.upper()} job {job_id}")
                        cc = f"qdel {job_id}" if scheduler_type == 'pbs' else f"scancel {job_id}"
                        subprocess.run(cc, shell=True)

                elif status == "COMPLETED":
                    result_value = "-1"
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result_value = f.read().strip()
                        os.remove(result_file)
                    rc = _call_hero_initialize(task_id, machine_name, task_engine)
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

                    # Sleep briefly so other managers' polling loops can see
                    # this task in "running" state and cancel their own scheduler
                    # jobs before we finalize.  Also gives us a window to detect
                    # the rare race where two machines both got through
                    # hero_initialize simultaneously.
                    time.sleep(5)

                    # Re-read running tasks to detect concurrent claims.
                    running_check = task_engine.read_tasks(
                        queue_id=queue_record["id"], metatype="Task", state="running"
                    )
                    task_now = next((t for t in running_check if t["id"] == task_id), None)
                    if task_now:
                        rivals = [
                            m for m, v in task_now["metadata"].get("running", {}).items()
                            if v and m != machine_name
                        ]
                        if rivals:
                            # Two machines both claimed — tiebreak by machine_names order.
                            all_claimants = [
                                m for m, v in task_now["metadata"]["running"].items() if v
                            ]
                            winner = min(
                                all_claimants,
                                key=lambda m: hpc_config.machine_names.index(m)
                                if m in hpc_config.machine_names else 999,
                            )
                            if winner != machine_name:
                                print(f"Task {task_id}: race detected — deferring to {winner}")
                                jid = current_task["metadata"]["scheduler_job_id"].get(machine_name, -1)
                                if jid != -1:
                                    cc = f"qdel {jid}" if scheduler_type == "pbs" else f"scancel {jid}"
                                    subprocess.run(cc, shell=True)
                                m2 = task_now["metadata"]
                                m2["running"][machine_name] = False
                                m2["scheduler_job_id"][machine_name] = -1
                                task_engine.update_task(
                                    task_id=task_id, state="running",
                                    name=current_task["name"], metadata=m2,
                                )
                                continue

                    _call_hero_finalize(result_value, task_id, machine_name, task_engine)
                    print(f"Task {task_id}: finalized with result={result_value}")

                elif status == "FAILED":
                    print(f"{scheduler_type.upper()} job {job_id} in error state.")
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
            sched = getattr(hpc_config, 'scheduler', {}).get(machine_name, 'slurm')
            if not meta["running"][machine_name]:
                job_id = current_task["metadata"]["scheduler_job_id"][machine_name]
                if job_id != -1:
                    cancel_cmd = f"qdel {job_id}" if sched == 'pbs' else f"scancel {job_id}"
                    print(f"Cancelling {sched.upper()} job {job_id} (task claimed by another machine)")
                    subprocess.run(cancel_cmd, shell=True)
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    task_engine.update_task(
                        task_id=task_id, state="running",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )
            else:
                job_id = current_task["metadata"]["scheduler_job_id"][machine_name]
                result_file = os.path.join(agent_dir, "simulation_files", f"result_{task_id}.txt")

                if sched == 'pbs':
                    import re as _re
                    qstat = subprocess.run(
                        f"qstat -f -x {job_id}",
                        shell=True, capture_output=True, text=True,
                    )
                    if qstat.returncode != 0 or not qstat.stdout.strip():
                        status = "COMPLETED"
                    else:
                        state_match = _re.search(r'job_state\s*=\s*(\S+)', qstat.stdout)
                        state = state_match.group(1) if state_match else "?"
                        if state in ('F', 'C'):
                            exit_match = _re.search(r'exit_status\s*=\s*(\S+)', qstat.stdout)
                            exit_val = int(exit_match.group(1)) if exit_match else 0
                            status = "COMPLETED" if exit_val == 0 else "FAILED"
                        elif state in ('R', 'E'):
                            status = "RUNNING"   # R=running, E=exiting (still running)
                        else:
                            status = "PENDING"   # Q=queued, H=held, W=waiting, etc.
                else:
                    sacct = subprocess.run(
                        f"sacct -j {job_id} --format=State --noheader",
                        shell=True, capture_output=True, text=True,
                    )
                    sacct_out = sacct.stdout.strip()
                    if "COMPLETED" in sacct_out:
                        status = "COMPLETED"
                    elif any(s in sacct_out for s in ("FAILED", "CANCELLED", "TIMEOUT")):
                        status = "FAILED"
                    else:
                        status = "PENDING"

                if status == "COMPLETED":
                    result_value = "-1"
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result_value = f.read().strip()
                        os.remove(result_file)
                    _call_hero_finalize(result_value, task_id, machine_name, task_engine)
                    print(f"Task {task_id}: finalized with result={result_value}")
                elif status == "FAILED":
                    print(f"{sched.upper()} job {job_id} failed.")
                    current_task["metadata"]["scheduler_job_id"][machine_name] = -1
                    current_task["metadata"]["running"][machine_name] = False
                    task_engine.update_task(
                        task_id=task_id, state="error",
                        name=current_task["name"], metadata=current_task["metadata"],
                    )

        # ----------------------------------------------------------------
        # Done tasks: cancel any lingering scheduler jobs this machine
        # submitted but that were claimed and finalized by another machine
        # before our running_tasks loop could cancel them.
        # ----------------------------------------------------------------
        done_tasks = task_engine.read_tasks(
            queue_id=queue_record["id"], metatype="Task", state="done"
        )
        sched_global = getattr(hpc_config, 'scheduler', {}).get(machine_name, 'slurm')
        for done_task in done_tasks:
            done_job_id = done_task.get("metadata", {}).get("scheduler_job_id", {}).get(machine_name, -1)
            if done_job_id != -1:
                cancel_cmd = f"qdel {done_job_id}" if sched_global == 'pbs' else f"scancel {done_job_id}"
                print(f"Cancelling lingering {sched_global.upper()} job {done_job_id} "
                      f"for done task {done_task['id'][:8]}")
                subprocess.run(cancel_cmd, shell=True)
                done_task["metadata"]["scheduler_job_id"][machine_name] = -1
                task_engine.update_task(
                    task_id=done_task["id"], state="done",
                    name=done_task["name"], metadata=done_task["metadata"],
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
