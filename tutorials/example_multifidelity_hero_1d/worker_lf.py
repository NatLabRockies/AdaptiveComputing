import numpy as np
from decimal import Decimal
# Hero setup ...
# Set Hero environment variables
import os
os.environ["HERO_PROJECT"] = "adaptive-computing"
os.environ["HERO_QUEUE"] = "queue-1"
os.environ["HERO_CLIENT_ID"] = "f4om7c738a1um7fgjao6msve7"
os.environ["HERO_CLIENT_SECRET"] = "mbk361rg0eedkd6k34t5cujukl19clbv50qnteqi829gnpufkde"
os.environ["HERO_QUEUE_VISIBILITY_TIMEOUT"] = "60"
os.environ["HERO_DATABASE_PASSWORD"] = "8fc2a2e2-ed9e-413d-996a-72da94e11c5c"
# Add the path to hero
import sys
sys.path.insert(0, '/projects/acldrd/kgriffin/hero/')
from hero import Hero

# low fidelity model
def lf_simulation(x):
    import numpy as np
    return (
        0.5 * ((x[0] * 6 - 2) ** 2) * np.sin((x[0] * 6 - 2) * 2) + (x[0] - 0.5) * 10.0 - 5
    )

if __name__ == "__main__":
    fidelity_level = 0
    hero = Hero(queue=str(fidelity_level))
    while True:
        task = hero.pull_task(attempts=1)
        print('Task:', task)
        if task:
            hero.claim_task(task)
            print(f'task name = {task.data["inputs"]["name"]}')
            print(f'input args = {task.data["inputs"]["args"]}')
            args = task.data['inputs']['args']
            # Convert the args from strings to their native types
            for j in range(len(args)):
                if args[j].endswith("_categorical"):
                    args[j] = args[j][:-12]
                elif args[j].endswith("_continuous"):
                    args[j] = float(args[j][:-11])
                elif args[j].endswith("_ordered"):
                    args[j] = int(float(args[j][:-8]))
                else:
                    raise Exception('Unrecognized type for parameter '+str(j))
            y_eval = lf_simulation(args)
            print(f'function output = {y_eval}')
            hero.update_task(task, {"objective": str(y_eval)})
        else:
            wait_time_if_no_hero_tasks = 4
            print(f"No tasks. Waiting {wait_time_if_no_hero_tasks+1} seconds.")
            hero.wait(wait_time_if_no_hero_tasks)
        hero.wait(1)
