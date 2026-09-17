"""
manager.py — LocalHPCManager for the rental_agent_HPC_onsite example.

Submits one SLURM job per Hero task.  Each job receives the task ID as its
only argument; the manager writes a ``config.json`` (utility_rate, storage,
number_of_daily_evs, return_soc) into ``cases/<task_id>/`` before submission
so the batch script can read it without parsing command-line arguments.

The batch script (job.sh) runs mock_simulation.py, then writes the negated
cost to ``result_<task_id>.txt`` in ``$SLURM_SUBMIT_DIR`` (= this directory)
for the manager to pick up via read_result().

run_forever() keeps the daemon alive between controller restarts when this
script is run directly from the command line.  The controller calls
run_until_done() inline instead.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.hpc.local_manager import LocalHPCManager
from adaptive_computing.local_hero import LocalHeroClient

SCRIPT_DIR = Path(__file__).parent.resolve()
MANAGER_SCRIPT = Path(__file__).resolve()


class RentalAgentManager(LocalHPCManager):
    """Manager for the rental car electrification simulation.

    Writes a per-task ``config.json`` to ``cases/<task_id>/`` before
    submitting the SLURM job.  The batch script reads config.json, runs
    ``simulation_files/mock_simulation.py``, and writes the negated cost
    to ``result_<task_id>.txt``.
    """

    def __init__(self, *args, work_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._work_dir = Path(work_dir) if work_dir else SCRIPT_DIR

    def submit_job(self, task: dict, machine_name: str, i_fidelity: int) -> str:
        task_id = task["id"]
        meta    = task["metadata"]

        # Write config.json so the batch script can read simulation parameters.
        case_dir = self._work_dir / "cases" / task_id
        case_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "utility_rate":        meta.get("utility_rate",        "Moderate"),
            "storage":             float(meta.get("storage",        0.0)),
            "number_of_daily_evs": float(meta.get("number_of_daily_evs", 1000.0)),
            "return_soc":          float(meta.get("return_soc",    40.0)),
        }
        (case_dir / "config.json").write_text(json.dumps(config, indent=2))

        script = self.batch_scripts[i_fidelity]
        cmd    = f"sbatch {script} {task_id}"
        return self._run_submit(cmd)

    def read_result(self, task_id: str) -> str:
        result_file = self._work_dir / f"result_{task_id}.txt"
        if result_file.exists():
            value = result_file.read_text().strip()
            result_file.unlink()
            return value
        print(f"WARNING: result file not found for task {task_id}, using -1")
        return "-1"


def create_manager(
    machine_name: str = "local",
    hero_client: LocalHeroClient | None = None,
    work_dir: str | None = None,
) -> RentalAgentManager:
    """Return a configured RentalAgentManager.

    Args:
        machine_name: Logical name for this machine stored in task metadata.
        hero_client:  Shared LocalHeroClient instance (same DB as controller).
        work_dir:     Absolute path to the working directory where result files
                      and case directories live.  Defaults to this script's dir.
    """
    work_dir = work_dir or str(SCRIPT_DIR)
    batch_script  = str(SCRIPT_DIR / "simulation_files" / "job.sh")
    simulation_dir = str(SCRIPT_DIR / "simulation_files")

    return RentalAgentManager(
        machine_name=machine_name,
        batch_scripts=[batch_script],
        scheduler_type="slurm",
        simulation_dir=simulation_dir,
        poll_interval=10,
        hero_client=hero_client,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    work_dir     = sys.argv[1] if len(sys.argv) > 1 else str(SCRIPT_DIR)
    machine_name = sys.argv[2] if len(sys.argv) > 2 else "local"

    hero_client = LocalHeroClient(
        db_path=str(Path(work_dir) / "hero_db.json"),
        queue_name="jobs",
        application_id="rental_agent_HPC_onsite",
    )
    mgr = create_manager(
        machine_name=machine_name,
        hero_client=hero_client,
        work_dir=work_dir,
    )
    mgr.run_forever()
