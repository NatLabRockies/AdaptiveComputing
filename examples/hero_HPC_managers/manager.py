from hero import HeroClient, get_env_variable
import numpy as np
import time
import subprocess
import os
import sys

# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
set_hero_env_vars()

def _get_pbs_status(stdout, returncode):
    """Parse qstat -f -x output into a scheduler-agnostic status string."""
    import re
    if returncode != 0 or not stdout.strip():
        # Job not found in scheduler — it has already left the queue; treat as finished.
        return 'COMPLETED'
    state_match = re.search(r'job_state\s*=\s*(\S+)', stdout)
    if not state_match:
        return 'UNKNOWN'
    state = state_match.group(1)
    if state in ('F', 'C'):  # Finished / Complete
        exit_match = re.search(r'exit_status\s*=\s*(\S+)', stdout)
        exit_status = int(exit_match.group(1)) if exit_match else 0
        return 'COMPLETED' if exit_status == 0 else 'FAILED'
    if state in ('R', 'E'):  # Actually running on a compute node
        return 'RUNNING'
    if state in ('Q', 'H', 'W', 'T', 'M', 'S', 'U'):  # Queued or waiting
        return 'QUEUED'
    return 'UNKNOWN'

def _call_hero_initialize(task_id, machine_name, i_fidelity):
    """Call hero_initialize from the login node (which has internet access).
    Returns the exit code: 0 = success, 2 = already claimed by another machine, other = error.
    """
    result = subprocess.run(
        f"{sys.executable} -m adaptive_computing.hero_utils.hero_initialize {task_id} {machine_name} {i_fidelity}",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"  hero_initialize: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  hero_initialize stderr: {result.stderr.strip()}")
    return result.returncode


def _call_hero_finalize(result_value, task_id, machine_name, i_fidelity):
    """Call hero_finalize from the login node. Returns True on success."""
    result = subprocess.run(
        f"{sys.executable} -m adaptive_computing.hero_utils.hero_finalize {result_value} {task_id} {machine_name} {i_fidelity}",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"  hero_finalize: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  hero_finalize stderr: {result.stderr.strip()}")
    return result.returncode == 0


# Import HPC configuration
try:
    import hpc_config
except ModuleNotFoundError:
    print("ERROR: hpc_config.py not found. Please copy and edit hpc_config_template.py to hpc_config.py with your HPC settings.")
    exit(1)
_required = ['machine_names', 'remote_usernames', 'remote_hosts', 'remote_dirs', 'batch_scripts']
_missing = [f for f in _required if not hasattr(hpc_config, f)]
if _missing:
    _defined = [a for a in dir(hpc_config) if not a.startswith('_')]
    print(f"ERROR: hpc_config.py is missing required field(s): {', '.join(_missing)}")
    print(f"Fields currently defined in hpc_config.py: {', '.join(_defined)}")
    print("Please check hpc_config.py against hpc_config_template.py (look for typos).")
    exit(1)

try:
    HERO_ENV = get_env_variable('HERO_ENV', 'dev')
    HERO_PROJECT = get_env_variable('HERO_PROJECT')
    HERO_QUEUE = get_env_variable('HERO_QUEUE')
except EnvironmentError as e:
    print(e)
    exit(1)

APPLICATION_ID = f'{HERO_ENV}-{HERO_PROJECT}'

def hero_manager():
    import sys
    if len(sys.argv) > 1:
        machine_name = sys.argv[1]
    else:
        print("Missing the machine_name as a command line argument when manager.py is run.")
    
    i_fidelity = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    queue_name = HERO_QUEUE if i_fidelity == 0 else HERO_QUEUE + str(i_fidelity)

    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: HERO authentication failed: {e}")
        sys.exit(1)

    # Try to find existing active queue first (like HeroDataset does)
    try:
        queue_record = task_engine.read_queue_by_name(name=queue_name, state="active")
        print(f'Found existing active queue: {queue_name}')
    except Exception:
        print(f'No active queue found, creating new queue: {queue_name}')
        queue_record = task_engine.add_queue(name=queue_name)

    scheduler_type = getattr(hpc_config, 'scheduler', {}).get(machine_name, 'slurm')
    print(f'Scheduler type for {machine_name}: {scheduler_type}')
    print('Continuously check the queue, claim a ready task, and launch a job...')
    os.chdir('simulation_files')

    # --- Startup reconciliation ---
    # Reset any stale job IDs for jobs that are no longer in the scheduler,
    # and reset error-state tasks to ready so they can be retried.
    print('Running startup reconciliation...')
    for state in ('ready', 'error'):
        stale_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state=state)
        for task in stale_tasks:
            try:
                job_id = (task.get('metadata') or {}).get('scheduler_job_id', {}).get(machine_name, -1)
                needs_reset = False
                if state == 'error':
                    print(f"  Resetting error task {task['id']} to ready for retry")
                    needs_reset = True
                elif job_id != -1:
                    # Check if the job still exists in the scheduler
                    if scheduler_type == 'pbs':
                        check = subprocess.run(f"qstat -x {job_id}", shell=True, capture_output=True, text=True)
                    else:
                        check = subprocess.run(f"squeue -j {job_id} --noheader", shell=True, capture_output=True, text=True)
                    if check.returncode != 0 or not check.stdout.strip():
                        print(f"  Stale job ID {job_id} for task {task['id']} not found in scheduler — resetting")
                        needs_reset = True
                if needs_reset:
                    task['metadata']['scheduler_job_id'][machine_name] = -1
                    task['metadata']['running'][machine_name] = False
                    task_engine.update_task(task_id=task['id'], state='ready', name=task['name'], metadata=task['metadata'])
            except Exception as e:
                print(f"  WARNING: reconciliation failed for task {task['id']}: {e}")
    print('Startup reconciliation complete.')
    
    while True:
        # Print the state of the queue
        task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
        print(f'There are {len(task_records)} in the "ready" state.')
        task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
        print(f'There are {len(task_records)} in the "running" state.')
        task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='error')
        print(f'There are {len(task_records)} in the "error" state.')
        task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
        print(f'There are {len(task_records)} in the "done" state.')

        # For all ready tasks, if it's not queued on my machine, queue it
        ready_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')

        # Pass 1: Check PBS/SLURM status for already-submitted tasks (job_id != -1).
        # This must run before Pass 2 so that a job-limit break in Pass 2 does not
        # skip status checks for jobs that are already running or completed.
        # Tasks handled here are recorded in pass1_processed so Pass 2 does not
        # resubmit them in the same poll cycle.
        pass1_processed = set()
        for current_task in ready_tasks:
            if current_task['metadata']['scheduler_job_id'][machine_name] == -1:
                continue  # not yet submitted — handled in Pass 2
            job_id = current_task['metadata']['scheduler_job_id'][machine_name]
            if scheduler_type == 'pbs':
                status_check = subprocess.run(f"qstat -f -x {job_id}", shell=True, capture_output=True, text=True)
                status = _get_pbs_status(status_check.stdout, status_check.returncode)
            else:
                status_check = subprocess.run(f"sacct -j {job_id} --format=State --noheader", shell=True, capture_output=True, text=True)
                status = status_check.stdout.strip()
                if not status:
                    # sacct has no record yet (job may still be pending). Check squeue before resetting.
                    squeue_check = subprocess.run(f"squeue -j {job_id} --noheader", shell=True, capture_output=True, text=True)
                    if squeue_check.stdout.strip():
                        # Job is queued/pending (e.g. QOSMaxJobsPerUserLimit) — nothing to do this cycle.
                        status = 'PENDING'
                    else:
                        # Not in sacct or squeue. Check result file before treating as stale —
                        # sacct can lag minutes behind on some systems (e.g. debug partition).
                        result_file = f"result_{current_task['id']}.txt"
                        if os.path.exists(result_file):
                            # Job finished and wrote result before sacct updated — treat as completed.
                            status = 'COMPLETED'
                        else:
                            # Truly stale (no result file, not in any scheduler record). Reset so it gets resubmitted.
                            print(f"Job {job_id} not found in sacct or squeue for task {current_task['id']} — stale, resetting to unsubmitted.")
                            current_task['metadata']['scheduler_job_id'][machine_name] = -1
                            current_task['metadata']['running'][machine_name] = False
                            task_engine.update_task(task_id=current_task['id'], state='ready', name=current_task['name'], metadata=current_task['metadata'])
                            continue
            if 'RUNNING' in status and not current_task['metadata']['running'][machine_name]:
                print(f"Job {job_id} is running for task {current_task['id']} — calling hero_initialize")
                rc = _call_hero_initialize(current_task['id'], machine_name, i_fidelity)
                if rc == 0:
                    current_task['metadata']['running'][machine_name] = True
                    task_engine.update_task(task_id=current_task['id'], state='running', name=current_task['name'], metadata=current_task['metadata'])
                    print(f"Task {current_task['id']}: claimed successfully, state = running")
                elif rc == 2:
                    print(f"Task {current_task['id']}: already claimed by another machine. Canceling job {job_id}.")
                    cancel_cmd = f"qdel {job_id}" if scheduler_type == 'pbs' else f"scancel {job_id}"
                    subprocess.run(cancel_cmd, shell=True)
                    current_task['metadata']['scheduler_job_id'][machine_name] = -1
                    task_engine.update_task(task_id=current_task['id'], state='running', name=current_task['name'], metadata=current_task['metadata'])
                    pass1_processed.add(current_task['id'])
                else:
                    print(f"hero_initialize failed with code {rc} for task {current_task['id']}. Canceling job and marking error.")
                    cancel_cmd = f"qdel {job_id}" if scheduler_type == 'pbs' else f"scancel {job_id}"
                    subprocess.run(cancel_cmd, shell=True)
                    current_task['metadata']['scheduler_job_id'][machine_name] = -1
                    task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                    pass1_processed.add(current_task['id'])
            elif 'COMPLETED' in status:
                result_file = f"result_{current_task['id']}.txt"
                result_value = "-1"
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        result_value = f.read().strip()
                    os.remove(result_file)
                else:
                    print(f"WARNING: result file not found for task {current_task['id']}, using -1")
                if not current_task['metadata']['running'][machine_name]:
                    rc = _call_hero_initialize(current_task['id'], machine_name, i_fidelity)
                    if rc == 2:
                        print(f"Task {current_task['id']}: already claimed by another machine (job completed).")
                        current_task['metadata']['scheduler_job_id'][machine_name] = -1
                        task_engine.update_task(task_id=current_task['id'], state='running', name=current_task['name'], metadata=current_task['metadata'])
                        pass1_processed.add(current_task['id'])
                        continue
                    elif rc != 0:
                        print(f"hero_initialize failed (code {rc}) on completed job {job_id}. Marking as error.")
                        current_task['metadata']['scheduler_job_id'][machine_name] = -1
                        task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                        pass1_processed.add(current_task['id'])
                        continue
                print(f"Job {job_id} completed for task {current_task['id']}, result={result_value}. Calling hero_finalize.")
                _call_hero_finalize(result_value, current_task['id'], machine_name, i_fidelity)
                # Task is now 'done' in Hero. Do NOT reset scheduler_job_id locally —
                # leaving it non-(-1) ensures Pass 2 skips this task this cycle.
                pass1_processed.add(current_task['id'])
            elif any(x in status for x in ['FAILED', 'CANCELLED', 'TIMEOUT']):
                print("Job error detected.")
                current_task['metadata']['scheduler_job_id'][machine_name] = -1
                current_task['metadata']['running'][machine_name] = False
                task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                pass1_processed.add(current_task['id'])

        # Pass 2: Submit new jobs for tasks that don't have a job yet.
        for current_task in ready_tasks:
            if current_task['id'] in pass1_processed:
                continue  # already handled in Pass 1 this cycle
            if current_task['metadata']['scheduler_job_id'][machine_name] != -1:
                continue  # already submitted — handled in Pass 1
            t = current_task['metadata']['x_data'][0]
            # Use configured script name for this machine
            if machine_name in hpc_config.batch_scripts:
                scripts_for_machine = hpc_config.batch_scripts[machine_name]
                if isinstance(scripts_for_machine, str):
                    raise Exception(f"hpc_config.batch_scripts['{machine_name}'] must be a list, not a string. "
                                    f"Change it to ['{scripts_for_machine}'] in hpc_config.py.")
                script_name = scripts_for_machine[i_fidelity]
                # Use absolute path for the script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                absolute_script_path = os.path.join(script_dir, script_name)
                if scheduler_type == 'pbs':
                    command = f"qsub -v \"temp={t},task_id={current_task['id']}\" {absolute_script_path}"
                else:
                    command = f"sbatch {absolute_script_path} {t} {current_task['id']}"
            else:
                raise Exception(f"The machine name '{machine_name}' is not configured in hpc_config.batch_scripts. Available machines: {list(hpc_config.batch_scripts.keys())}")
            print(f"Running command: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                # Job limit reached: don't mark as error, just stop submitting
                # this cycle and retry in the next polling iteration.
                slurm_limit = 'QOSMaxSubmitJobPerUserLimit' in result.stderr or 'MaxSubmitJobsPerUser' in result.stderr
                pbs_limit = 'Job exceeds queue' in result.stderr or 'PBS_MAXSELECTJOB' in result.stderr or 'would exceed' in result.stderr or 'violates queue' in result.stderr
                if slurm_limit or pbs_limit:
                    print(f"Job limit reached. Will retry task {current_task['id']} in next polling cycle.")
                    break
                print(f"Error occurred during job submission (return code {result.returncode}).")
                print(f"STDOUT: {result.stdout.strip()}")
                print(f"STDERR: {result.stderr.strip()}")
                current_task['metadata']['scheduler_job_id'][machine_name] = -1
                current_task['metadata']['running'][machine_name] = False
                task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                continue
            job_id = result.stdout.strip().split()[-1]  # parse the job_id from the string returned (works for both SLURM and PBS)
            current_task['metadata']['scheduler_job_id'][machine_name] = job_id
            try:
                task_engine.update_task(task_id=current_task['id'], state='ready', name=current_task['name'], metadata=current_task['metadata'])
            except Exception as e:
                # Hero update failed after a successful job submission — cancel the
                # job immediately so it doesn't become orphaned (PBS has it but Hero
                # doesn't know the ID, causing the manager to resubmit forever).
                print(f"WARNING: Hero update failed after submitting job {job_id}: {e}")
                cancel_cmd = f"qdel {job_id}" if scheduler_type == 'pbs' else f"scancel {job_id}"
                print(f"Canceling job to avoid orphan: {cancel_cmd}")
                subprocess.run(cancel_cmd, shell=True)
                continue
            print(f"Task {current_task['id']}: state = ready, metadata = {current_task['metadata']}")

        # For all running tasks
        running_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
        for current_task in running_tasks:
            # If not running on my machine, if it's queued on my machine, cancel it and mark it as unqueue on my machine.
            if not current_task['metadata']['running'][machine_name]:
                if current_task['metadata']['scheduler_job_id'][machine_name] != -1:
                    job_id = current_task['metadata']['scheduler_job_id'][machine_name]
                    command = f"qdel {job_id}" if scheduler_type == 'pbs' else f"scancel {job_id}"
                    print(f"Running command: {command}")
                    subprocess.run(command, shell=True, check=True)
                    current_task['metadata']['scheduler_job_id'][machine_name] = -1
                    task_engine.update_task(task_id=current_task['id'], state='running', name=current_task['name'], metadata=current_task['metadata'])
                    print(f"Task {current_task['id']}: state = running, metadata = {current_task['metadata']}")
                
            # If it is running on my machine, wait for completion then call hero_finalize
            if current_task['metadata']['running'][machine_name]:
                job_id = current_task['metadata']['scheduler_job_id'][machine_name]
                if scheduler_type == 'pbs':
                    status_check = subprocess.run(f"qstat -f -x {job_id}", shell=True, capture_output=True, text=True)
                    status = _get_pbs_status(status_check.stdout, status_check.returncode)
                else:
                    status_check = subprocess.run(f"sacct -j {job_id} --format=State --noheader", shell=True, capture_output=True, text=True)
                    status = status_check.stdout.strip()
                    if not status:
                        squeue_check = subprocess.run(f"squeue -j {job_id} --noheader", shell=True, capture_output=True, text=True)
                        if squeue_check.stdout.strip():
                            status = 'PENDING'  # still queued, nothing to do this cycle
                        else:
                            # Not in sacct or squeue. Check result file before marking as error.
                            result_file = f"result_{current_task['id']}.txt"
                            if os.path.exists(result_file):
                                status = 'COMPLETED'
                            else:
                                print(f"Job {job_id} not found in sacct or squeue for running task {current_task['id']} — marking as error.")
                                current_task['metadata']['scheduler_job_id'][machine_name] = -1
                                current_task['metadata']['running'][machine_name] = False
                                task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                                continue
                if 'COMPLETED' in status:
                    # Job finished — read result file written by the batch script and call
                    # hero_finalize from the login node (internet access guaranteed here).
                    result_file = f"result_{current_task['id']}.txt"
                    result_value = "-1"
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result_value = f.read().strip()
                        os.remove(result_file)
                    else:
                        print(f"WARNING: result file not found for task {current_task['id']}, using -1")
                    print(f"Job {job_id} completed for task {current_task['id']}, result={result_value}. Calling hero_finalize.")
                    if not _call_hero_finalize(result_value, current_task['id'], machine_name, i_fidelity):
                        print(f"WARNING: hero_finalize failed for task {current_task['id']}")
                    current_task['metadata']['scheduler_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                if any(x in status for x in ['FAILED', 'CANCELLED', 'TIMEOUT']):
                    print("Job error detected.")
                    current_task['metadata']['scheduler_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                    task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                    
        
        # For all jobs in an error state, could retry them on all machines, could try to run them on only different machines, or could unqueue them from other machines
        #XXX depends which strategy we want to pursue. Probably requeueing them on machines where they haven't failed yet would be good. Make sure the error is clearly printed.
        
        time.sleep(5)
        
if __name__ == "__main__":
    hero_manager()
