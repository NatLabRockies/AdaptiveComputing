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

def hero_initialize(task_id):
    # Setup the HERO client and authenticate
    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    hero.authenticate()

    # Get an existing project, or create one if it doesn't exist.
    queue_record = task_engine.add_queue(name='Degrees queue')
    #print(json.dumps(queue_record, indent=2))

    # Update the task's metatdata and mark it as running
    current_task = task_engine.read_task(task_id)
    current_task['metadata']['slurm_job_id']['kestrel'] = -1
    current_task['metadata']['running']['kestrel'] = True
    task_engine.update_task(task_id=task_id, state='running', name=current_task['name'], metadata=current_task['metadata'])
    print(f"Task {task_id}: state = running, metadata = {current_task['metadata']}")

if __name__ == "__main__":
    # Validate and parse command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python hero_initialize.py <task_id>")
        sys.exit(1)

    # Extract the first argument as a string
    task_id = sys.argv[1]

    hero_initialize(task_id)
