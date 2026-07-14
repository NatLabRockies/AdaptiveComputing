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

# Import HPC configuration
try:
    import hpc_config
except ImportError:
    print("ERROR: hpc_config.py not found. Please copy and edit hpc_config_template.py to hpc_config.py with your HPC settings.")
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

    print('Continuously check the queue, claim a ready task, and launch a slurm process...')
    os.chdir('simulation_files')
    
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
        for current_task in ready_tasks:
            if current_task['metadata']['slurm_job_id'][machine_name] == -1:
                t = current_task['metadata']['x_data'][0]
                # Use configured script name for this machine
                if machine_name in hpc_config.slurm_scripts:
                    scripts_for_machine = hpc_config.slurm_scripts[machine_name]
                    if isinstance(scripts_for_machine, str):
                        raise Exception(f"hpc_config.slurm_scripts['{machine_name}'] must be a list, not a string. "
                                        f"Change it to ['{scripts_for_machine}'] in hpc_config.py.")
                    script_name = scripts_for_machine[i_fidelity]
                    # Use absolute path for the script
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    absolute_script_path = os.path.join(script_dir, script_name)
                    command = f"sbatch {absolute_script_path} {t} {current_task['id']} {machine_name} {i_fidelity}"
                else:
                    raise Exception(f"The machine name '{machine_name}' is not configured in hpc_config.slurm_scripts. Available machines: {list(hpc_config.slurm_scripts.keys())}")
                print(f"Running command: {command}")
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    # QOS job limit reached: don't mark as error, just stop submitting
                    # this cycle and retry in the next polling iteration.
                    if 'QOSMaxSubmitJobPerUserLimit' in result.stderr or 'MaxSubmitJobsPerUser' in result.stderr:
                        print(f"SLURM job limit reached. Will retry task {current_task['id']} in next polling cycle.")
                        break
                    print("Error occurred during sbatch script.")
                    print("STDOUT:")
                    print(result.stdout)
                    print("STDERR:")
                    print(result.stderr)
                    print(f"Return code: {result.returncode}")
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                    task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                    continue
                job_id = result.stdout.strip().split()[-1] # parse the job_id from the string returned
                current_task['metadata']['slurm_job_id'][machine_name] = job_id
                task_engine.update_task(task_id=current_task['id'], state='ready', name=current_task['name'], metadata=current_task['metadata'])
                print(f"Task {current_task['id']}: state = ready, metadata = {current_task['metadata']}")
            else: # if the job is queued on this machine, check for errors
                job_id = current_task['metadata']['slurm_job_id'][machine_name]
                status_check = subprocess.run(f"sacct -j {job_id} --format=State --noheader", shell=True, capture_output=True, text=True)
                status = status_check.stdout.strip()
                if 'COMPLETED' in status:
                    # Job completed successfully - check if hero_finalize worked
                    # First re-read the task to get current state
                    updated_task = task_engine.read_task(task_id=current_task['id'])
                    if updated_task['state'] == 'done':
                        print(f"Job {job_id} completed successfully for task {current_task['id']} - hero_finalize succeeded")
                    else:
                        print(f"WARNING: Job {job_id} completed successfully for task {current_task['id']}, but Hero task is still in '{updated_task['state']}' state")
                        print(f"This likely indicates hero_finalize failed. Check SLURM output files for errors.")
                    
                    # Reset the slurm job id since this job is done
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                if any(x in status for x in ['FAILED', 'CANCELLED', 'TIMEOUT']):
                    print("Slurm job error detected.")
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                    task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])

        # For all running tasks
        running_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
        for current_task in running_tasks:
            # If not running on my machine, if it's queued on my machine, cancel it and mark it as unqueue on my machine.
            if not current_task['metadata']['running'][machine_name]:
                if current_task['metadata']['slurm_job_id'][machine_name] != -1:
                    job_id = current_task['metadata']['slurm_job_id'][machine_name]
                    command = f"scancel {job_id}"
                    print(f"Running command: {command}")
                    subprocess.run(command, shell=True, check=True)
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    task_engine.update_task(task_id=current_task['id'], state='running', name=current_task['name'], metadata=current_task['metadata'])
                    print(f"Task {current_task['id']}: state = running, metadata = {current_task['metadata']}")
                
            # If its running on my machine, check if it should be in an error state
            if current_task['metadata']['running'][machine_name]:
                job_id = current_task['metadata']['slurm_job_id'][machine_name]
                status_check = subprocess.run(f"sacct -j {job_id} --format=State --noheader", shell=True, capture_output=True, text=True)
                status = status_check.stdout.strip()
                if 'COMPLETED' in status:
                    # Job completed successfully - check if hero_finalize worked
                    # First re-read the task to get current state
                    updated_task = task_engine.read_task(task_id=current_task['id'])
                    if updated_task['state'] == 'done':
                        print(f"Job {job_id} completed successfully for task {current_task['id']} - hero_finalize succeeded")
                    else:
                        print(f"WARNING: Job {job_id} completed successfully for task {current_task['id']}, but Hero task is still in '{updated_task['state']}' state")
                        print(f"This likely indicates hero_finalize failed. Check SLURM output files for errors.")
                    
                    # Reset the slurm job id since this job is done  
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                if any(x in status for x in ['FAILED', 'CANCELLED', 'TIMEOUT']):
                    print("Slurm job error detected.")
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                    task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                    
        
        # For all jobs in an error state, could retry them on all machines, could try to run them on only different machines, or could unqueue them from other machines
        #XXX depends which strategy we want to pursue. Probably requeueing them on machines where they haven't failed yet would be good. Make sure the error is clearly printed.
        
        time.sleep(5)
        
if __name__ == "__main__":
    hero_manager()
