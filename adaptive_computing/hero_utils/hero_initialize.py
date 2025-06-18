from hero import HeroClient, get_env_variable
import json
import numpy as np
import time
import sys

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

def hero_initialize(task_id, machine_name):
    
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: HERO authentication failed: {e}")
        sys.exit(1)

    # Use the queue corresponding to fidelity level zero
    queue_record = task_engine.add_queue(name='0')

    current_task = task_engine.read_task(task_id)
    if current_task['state'] == 'running':
        print("Task is already running on another machine. Skipping.")
        sys.exit(2) # terminate the slurm job successfully because the task has already started running on a different machine and don't want to run it twice
        
    # Update the task's metatdata and mark it as running
    current_task['metadata']['running'][machine_name] = True
    task_engine.update_task(task_id=task_id, state='running', name=current_task['name'], metadata=current_task['metadata'])
    print(f"Task {task_id}: state = running, metadata = {current_task['metadata']}")
            
if __name__ == "__main__":
    # Validate and parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python hero_initialize.py <task_id> <machine_name>")
        sys.exit(1)

    task_id = sys.argv[1]
    machine_name = sys.argv[2]

    hero_initialize(task_id, machine_name)
