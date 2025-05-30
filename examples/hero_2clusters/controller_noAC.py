from hero import HeroClient, get_env_variable
import json
import time
import numpy as np

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

def hero_controller():
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    hero.authenticate()

    # Use the queue corresponding to fidelity level zero
    queue_record = task_engine.add_queue(name='0')

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
        new_task = task_engine.add_task(queue_id=queue_record['id'], name='Test from Python', metatype='Task', metadata={'x_data': [t], 'y_data': None, 'slurm_job_id': {'kestrel': -1,'vermilion': -1}, 'running': {'kestrel': False,'vermilion': False}})
    print('All tasks submitted to Hero queue. Waiting for all tasks to be done...')

    while True:
        # Print the state of the queue
        ready_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
        running_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
        done_task_records = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
        #print(f'There are {len(ready_task_records)} tasks in the "ready" state.')
        queued_kestrel = np.array([task['metadata']['slurm_job_id']['kestrel'] for task in ready_task_records]) != -1
        queued_vermilion = np.array([task['metadata']['slurm_job_id']['vermilion'] for task in ready_task_records]) != -1
        not_queued = np.sum(np.logical_and(np.logical_not(queued_kestrel), np.logical_not(queued_vermilion)))
        k_only = np.sum(np.logical_and(queued_kestrel, np.logical_not(queued_vermilion)))
        v_only = np.sum(np.logical_and(np.logical_not(queued_kestrel), queued_vermilion))
        both = np.sum(np.logical_and(queued_kestrel, queued_vermilion))
        print(f'There are {len(ready_task_records)} tasks in the "ready" state. {not_queued} not queued, {k_only} queued on kestrel only, {v_only} queued on vermilion only, {both} queued on both.')
        running_kestrel = np.array([task['metadata']['running']['kestrel'] for task in running_task_records])
        running_vermilion = np.array([task['metadata']['running']['vermilion'] for task in running_task_records])
        k_only = np.sum(np.logical_and(running_kestrel, np.logical_not(running_vermilion)))
        v_only = np.sum(np.logical_and(np.logical_not(running_kestrel), running_vermilion))
        both = np.sum(np.logical_and(running_kestrel, running_vermilion))
        print(f'There are {len(running_task_records)} tasks in the "running" state. {k_only} running on kestrel, {v_only} running on vermilion, {both} running on both')
        if both > 0:
            print(f"WARNING!!!!: {both} tasks are running on both machines.")
        print(f'There are {len(done_task_records)} tasks in the "done" state.')
        queued_kestrel = np.array([task['metadata']['slurm_job_id']['kestrel'] for task in done_task_records]) != -1
        queued_vermilion = np.array([task['metadata']['slurm_job_id']['vermilion'] for task in done_task_records]) != -1
        done_queued_tasks = np.sum(np.logical_or(queued_kestrel, queued_vermilion))
        if done_queued_tasks > 0:
            for task in done_task_records:
                print(f"metadata: {task['metadata']}")
            print(f"WARNING!!!!: {done_queued_tasks} tasks are done but still queued on another machine.")
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
