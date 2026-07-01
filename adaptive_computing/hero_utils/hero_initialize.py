from hero import HeroClient, get_env_variable
import json
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

def hero_initialize(task_id, machine_name, i_fidelity=0):
    
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: HERO authentication failed: {e}")
        sys.exit(1)

    current_task = task_engine.read_task(task_id)
    if current_task['state'] == 'running':
        print("Task is already running on another machine. Skipping.")
        sys.exit(2)  # terminate the slurm job successfully; task already claimed by another machine

    # Update the task's metadata and mark it as running
    current_task['metadata']['running'][machine_name] = True
    task_engine.update_task(task_id=task_id, state='running', name=current_task['name'], metadata=current_task['metadata'])
    print(f"Task {task_id}: state = running, metadata = {current_task['metadata']}")
            
if __name__ == "__main__":
    # Validate and parse command-line arguments
    if len(sys.argv) not in (3, 4):
        print("Usage: python hero_initialize.py <task_id> <machine_name> [i_fidelity]")
        sys.exit(1)

    task_id = sys.argv[1]
    machine_name = sys.argv[2]
    i_fidelity = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    hero_initialize(task_id, machine_name, i_fidelity)
