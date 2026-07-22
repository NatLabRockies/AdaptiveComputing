"""
manager_template.py — Minimal HeroHPCManager subclass skeleton.

Copy this file to your application directory as manager.py, rename the class,
and implement the two abstract methods.

Usage::

    python manager.py <machine_name> [<i_fidelity>]
"""

import os
import sys

from adaptive_computing.hpc import HeroHPCManager, JobLimitError, TaskError

try:
    import hpc_config
except ImportError:
    print("ERROR: hpc_config.py not found. Copy hpc_config_template.py and edit it.")
    sys.exit(1)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))


class MyAppManager(HeroHPCManager):
    """Replace 'MyApp' with your application name."""

    # Directory to chdir into before the event loop.
    # Set to None to skip the chdir.
    simulation_dir = "simulation_files"

    def submit_job(self, task: dict, machine_name: str, i_fidelity: int) -> str:
        """Extract parameters from task metadata, prepare inputs, and submit.

        Steps:
        1. Validate task['metadata'] — raise TaskError on invalid data.
        2. Create case directories and write any config files.
        3. Build the sbatch / qsub command.
        4. Call self._run_submit(cmd, scheduler_type) and return its result.

        Example (SLURM, single script)::

            meta = task['metadata']
            case_dir = os.path.join(_APP_DIR, "cases", task['id'])
            os.makedirs(case_dir, exist_ok=True)

            cmd = (
                f"sbatch "
                f"--output={case_dir}/slurm_%j.out "
                f"--error={case_dir}/slurm_%j.err "
                f"simulation_files/run_sim.sh {task['id']}"
            )
            return self._run_submit(cmd, self.get_scheduler_type(machine_name))
        """
        raise NotImplementedError

    def read_result(self, task_id: str) -> str:
        """Read and return the simulation result string for task_id.

        Return "-1" if the result file is missing or unreadable.
        Delete the result file after reading to avoid stale data.

        Example::

            result_file = f"result_{task_id}.txt"
            if os.path.exists(result_file):
                value = open(result_file).read().strip()
                os.remove(result_file)
                return value
            return "-1"
        """
        raise NotImplementedError


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manager.py <machine_name> [<i_fidelity>]")
        sys.exit(1)
    machine_name = sys.argv[1]
    i_fidelity   = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    MyAppManager(hpc_config).run(machine_name, i_fidelity)
