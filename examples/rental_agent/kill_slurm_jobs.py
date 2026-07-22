import subprocess
import os
import sys

# --- path / env setup ---
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from adaptive_computing.utils import load_env_file
load_env_file(os.path.join(_AGENT_DIR, "env_vars.txt"))
# --- end path / env setup ---

from hero import HeroClient, get_env_variable
from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars
set_hero_env_vars()

try:
    HERO_ENV     = get_env_variable('HERO_ENV', 'dev')
    HERO_PROJECT = get_env_variable('HERO_PROJECT')
    HERO_QUEUE   = get_env_variable('HERO_QUEUE')
except EnvironmentError as e:
    print(e)
    sys.exit(1)

APPLICATION_ID = f'{HERO_ENV}-{HERO_PROJECT}'


def kill_slurm_jobs():
    if len(sys.argv) > 1:
        machine_name = sys.argv[1]
    else:
        print("Missing machine_name as a command-line argument.")
        sys.exit(1)

    hero = HeroClient()
    task_engine = hero.TaskEngine(APPLICATION_ID)
    try:
        hero.authenticate()
    except Exception as e:
        print(f"ERROR: HERO authentication failed: {e}")
        sys.exit(1)

    try:
        queue_record = task_engine.read_queue_by_name(name=HERO_QUEUE, state='active')
        print(f'Found existing active queue: {HERO_QUEUE}')
    except Exception:
        print(f'No active queue found, creating new queue: {HERO_QUEUE}')
        queue_record = task_engine.add_queue(name=HERO_QUEUE)

    print('Cancelling all Slurm jobs for queued/running Hero tasks...')

    for state in ('ready', 'running', 'error', 'done'):
        n = len(task_engine.read_tasks(
            queue_id=queue_record['id'], metatype='Task', state=state
        ))
        print(f'  {n} task(s) in "{state}" state.')

    ready_tasks   = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='ready')
    running_tasks = task_engine.read_tasks(queue_id=queue_record['id'], metatype='Task', state='running')

    for current_task in ready_tasks + running_tasks:
        job_id = current_task['metadata']['scheduler_job_id'].get(machine_name, -1)
        if job_id != -1:
            command = f"scancel {job_id}"
            print(f"Running: {command}")
            subprocess.run(command, shell=True, check=False)
            current_task['metadata']['scheduler_job_id'][machine_name] = -1
            task_engine.update_task(
                task_id=current_task['id'], state='error',
                name=current_task['name'], metadata=current_task['metadata'],
            )


if __name__ == "__main__":
    kill_slurm_jobs()
