from hero import HeroClient, get_env_variable
import subprocess
import os

from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
set_hero_env_vars()

try:
    HERO_ENV = get_env_variable('HERO_ENV', 'dev')
    HERO_PROJECT = get_env_variable('HERO_PROJECT')
except EnvironmentError as e:
    print(e)
    exit(1)

APPLICATION_ID = f'{HERO_ENV}-{HERO_PROJECT}'

def kill_slurm_jobs():
    import sys
    if len(sys.argv) > 1:
        machine_name = sys.argv[1]
    else:
        print("Missing the machine_name as a command line argument when kill_slurm_jobs.py is run.")
    
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    hero.authenticate()

    # Use the queue corresponding to fidelity level zero
    queue_record = task_engine.add_queue(name='0')

    print('Cancel all slurm jobs corresponding to queued or running Hero tasks...')
    os.chdir('simulation_files')
    
    # Print the state of the queue
    task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
    print(f'There are {len(task_records)} in the "ready" state.')
    task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
    print(f'There are {len(task_records)} in the "running" state.')
    task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='error')
    print(f'There are {len(task_records)} in the "error" state.')
    task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
    print(f'There are {len(task_records)} in the "done" state.')

    # For all tasks, cancel any associated slurm job
    ready_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
    running_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
    all_tasks = ready_tasks + running_tasks
    for current_task in all_tasks:
        if current_task['metadata']['slurm_job_id'][machine_name] != -1:
            command = f"scancel {current_task['metadata']['slurm_job_id'][machine_name]}"
            print(f"Running command: {command}")
            result = subprocess.run(command, shell=True, check=True)
            current_task['metadata']['slurm_job_id'][machine_name] = -1
            task_engine.update_task(task_id=current_task['id'], state='error', name=current_task['name'], metadata=current_task['metadata'])
        
if __name__ == "__main__":
    kill_slurm_jobs()
