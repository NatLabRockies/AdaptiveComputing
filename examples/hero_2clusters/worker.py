#!/usr/bin/env python3
"""
Worker script that processes Hero queue tasks locally without SLURM.
Takes tasks from the queue, computes conductivity = temperature^2/1000, 
and marks tasks as done.
"""

from hero import HeroClient, get_env_variable
import numpy as np
import time
import sys

from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
set_hero_env_vars()

try:
    HERO_ENV = get_env_variable('HERO_ENV', 'dev')
    HERO_PROJECT = get_env_variable('HERO_PROJECT')
    HERO_QUEUE = get_env_variable('HERO_QUEUE')
except EnvironmentError as e:
    print(e)
    exit(1)

APPLICATION_ID = f'{HERO_ENV}-{HERO_PROJECT}'

def compute_conductivity(temperature):
    """
    Compute conductivity from temperature using the formula:
    conductivity = temperature^2 / 1000
    """
    return temperature * temperature / 1000.0

def hero_worker(worker_name='local_worker'):
    """
    Local worker that processes Hero queue tasks without SLURM.
    
    Args:
        worker_name (str): Name identifier for this worker
    """
    
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: HERO authentication failed: {e}")
        sys.exit(1)

    # Use the queue corresponding to fidelity level zero
    queue_record = task_engine.add_queue(name=HERO_QUEUE+'0')
    
    print(f'Worker "{worker_name}" starting - processing tasks locally...')
    
    while True:
        # Get ready tasks from the queue
        ready_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
        
        # Print queue status
        running_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')
        done_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='done')
        error_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='error')
        
        print(f'Queue status: Ready={len(ready_tasks)}, Running={len(running_tasks)}, Done={len(done_tasks)}, Error={len(error_tasks)}')
        
        # Process ready tasks
        for task in ready_tasks:
            try:
                # Get the temperature from the task
                temperature = task['metadata']['x_data'][0]
                task_id = task['id']
                
                print(f"Processing task {task_id}: temperature = {temperature}")
                
                # Compute conductivity locally
                conductivity = compute_conductivity(temperature)
                
                # Update task metadata with the result
                task['metadata']['y_data'] = [conductivity]
                
                # Mark task as done
                task_engine.update_task(
                    task_id=task_id, 
                    state='done', 
                    name=task['name'], 
                    metadata=task['metadata']
                )
                
                print(f"Completed task {task_id}: temperature={temperature} -> conductivity={conductivity}")
                
            except Exception as e:
                print(f"Error processing task {task.get('id', 'unknown')}: {e}")
                # Mark task as error
                try:
                    task_engine.update_task(
                        task_id=task['id'], 
                        state='error', 
                        name=task['name'], 
                        metadata=task['metadata']
                    )
                except:
                    pass  # Ignore secondary errors
        
        # Wait before checking again
        time.sleep(2)

if __name__ == "__main__":
    worker_name = sys.argv[1] if len(sys.argv) > 1 else 'local_worker'
    try:
        hero_worker(worker_name)
    except KeyboardInterrupt:
        print(f"\nWorker '{worker_name}' stopped by user.")
    except Exception as e:
        print(f"Worker error: {e}")
        sys.exit(1)