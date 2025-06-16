from hero import HeroClient, get_env_variable
import numpy as np
import time
import subprocess

import os
os.environ["HERO_ENV"] = "dev"
os.environ["HERO_PROJECT"] = "adaptive-computing-app"
os.environ["HERO_CLIENT_ID"] = "f4om7c738a1um7fgjao6msve7"
os.environ["HERO_CLIENT_SECRET"] = "mbk361rg0eedkd6k34t5cujukl19clbv50qnteqi829gnpufkde"
os.environ["HERO_QUEUE"] = "queue-degrees"
os.environ["HERO_QUEUE_VISIBILITY_TIMEOUT"] = "60"
os.environ["HERO_DATABASE_PASSWORD"] = "8fc2a2e2-ed9e-413d-996a-72da94e11c5c"

try:
    HERO_ENV = get_env_variable('HERO_ENV', 'dev')
    HERO_PROJECT = get_env_variable('HERO_PROJECT')
except EnvironmentError as e:
    print(e)
    exit(1)

APPLICATION_ID = f'{HERO_ENV}-{HERO_PROJECT}'

from adaptive_computing.hero_utils.get_machine_name import get_machine_name

def hero_manager():
    machine_name = get_machine_name()
    
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    hero.authenticate()

    # Use the queue corresponding to fidelity level zero
    queue_record = task_engine.add_queue(name='0')

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
                if machine_name == 'kestrel':
                    command = f"sbatch script_kestrel.sh {t} {current_task['id']}"
                elif machine_name == 'vermilion':
                    command = f"sbatch script_vermilion.sh {t} {current_task['id']}"
                else:
                    raise Exception(f"The machine name found is {machine_name}. A branch of the if statement must be written for how to run the job on this machine.")
                print(f"Running command: {command}")
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print("Error occurred during sbatch script.")
                    print("STDOUT:")
                    print(result.stdout)
                    print("STDERR:")
                    print(result.stderr)
                    current_task['metadata']['slurm_job_id'][machine_name] = -1
                    current_task['metadata']['running'][machine_name] = False
                    task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
                job_id = result.stdout.strip().split()[-1] # parse the job_id from the string returned
                current_task['metadata']['slurm_job_id'][machine_name] = job_id
                task_engine.update_task(task_id=current_task['id'], state='ready', name=current_task['name'], metadata=current_task['metadata'])
                print(f"Task {current_task['id']}: state = ready, metadata = {current_task['metadata']}")
            else: # if the job is queued on this machine, check for errors
                job_id = current_task['metadata']['slurm_job_id'][machine_name]
                status_check = subprocess.run(f"sacct -j {job_id} --format=State --noheader", shell=True, capture_output=True, text=True)
                status = status_check.stdout.strip()
                if 'COMPLETED' in status:
                    raise Exception("Slurm job succeeded, but this job was not marked as ready still.")
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
                    pass
                    # it is possible that the job completed after the task_engine.read_tasks was called so this isn't necessarily an issue.
                    #XXX alternate implementation could omit hero_finalize and perform those duties here (don't raise Exception in that case)
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
