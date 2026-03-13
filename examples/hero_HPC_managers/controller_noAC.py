from hero import HeroClient, get_env_variable
import json
import time
import numpy as np

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

def hero_controller():
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    hero.authenticate()

    # Use the queue corresponding to fidelity level zero
    queue_record = task_engine.add_queue(name=HERO_QUEUE+'0')

    # Clear out any existing tasks
    ready_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
    running_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
    error_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='error')
    done_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
    for task_record in ready_task_records:
        task_engine.delete_task(task_id=task_record['id'])
    for task_record in running_task_records:
        task_engine.delete_task(task_id=task_record['id'])
    for task_record in error_task_records:
        task_engine.delete_task(task_id=task_record['id'])
    for task_record in done_task_records:
        task_engine.delete_task(task_id=task_record['id'])

    # Print the state of the queue
    ready_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
    running_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
    done_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
    print('Initial state of the queue.')
    print(f'There are {len(ready_task_records)} in the "ready" state.')
    print(f'There are {len(running_task_records)} in the "running" state.')
    print(f'There are {len(done_task_records)} in the "done" state.')

    # Add degrees tasks
    temp = np.linspace(0.7, 2.0, 5)
    for t in temp:
        # Initialize slurm_job_id and running status for all configured machines
        slurm_job_id = {machine: -1 for machine in hpc_config.machine_names}
        running = {machine: False for machine in hpc_config.machine_names}
        new_task = task_engine.add_task(queue_id=queue_record['id'], name='Test from Python', metatype='Task', metadata={'x_data': [t], 'y_data': None, 'slurm_job_id': slurm_job_id, 'running': running})
    print('All tasks submitted to Hero queue. Waiting for all tasks to be done...')

    while True:
        # Print the state of the queue
        ready_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
        running_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
        done_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
        
        # Analyze ready tasks by machine
        print(f'There are {len(ready_task_records)} tasks in the "ready" state.')
        if ready_task_records:
            for machine in hpc_config.machine_names:
                queued_count = sum(1 for task in ready_task_records if task['metadata']['slurm_job_id'][machine] != -1)
                print(f'  {queued_count} queued on {machine}')
        
        # Analyze running tasks by machine
        print(f'There are {len(running_task_records)} tasks in the "running" state.')
        if running_task_records:
            for machine in hpc_config.machine_names:
                running_count = sum(1 for task in running_task_records if task['metadata']['running'][machine])
                print(f'  {running_count} running on {machine}')
            
            # Check for tasks running on multiple machines (potential issue)
            for task in running_task_records:
                running_machines = [m for m in hpc_config.machine_names if task['metadata']['running'][m]]
                if len(running_machines) > 1:
                    print(f"WARNING: Task {task['id']} running on multiple machines: {running_machines}")
        
        print(f'There are {len(done_task_records)} tasks in the "done" state.')
        
        # Check for done tasks still queued on machines (potential issue)
        if done_task_records:
            for task in done_task_records:
                queued_machines = [m for m in hpc_config.machine_names if task['metadata']['slurm_job_id'][m] != -1]
                if queued_machines:
                    print(f"WARNING: Task {task['id']} is done but still queued on: {queued_machines}")
                    print(f"  metadata: {task['metadata']}")
        print()
        
        task_record = task_engine.read_task(task_id=new_task['id'])
        if len(ready_task_records)+len(running_task_records) == 0:
            print('All tasks complete. Removing all done tasks...')
            break
        time.sleep(5)

    # Print the conductivities for the done tasks and delete them
    task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
    for task_record in task_records:
        print(f"The task with temperature={task_record['metadata']['x_data']}, computed conductivity={task_record['metadata']['y_data']}")
        task_engine.delete_task(task_id=task_record['id'])
        
    # Print the state of the queue
    ready_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
    running_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
    error_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='error')
    done_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
    print(f'There are {len(ready_task_records)} in the "ready" state.')
    print(f'There are {len(running_task_records)} in the "running" state.')
    print(f'There are {len(error_task_records)} in the "error" state.')
    print(f'There are {len(done_task_records)} in the "done" state.')
        
if __name__ == "__main__":
    hero_controller()
