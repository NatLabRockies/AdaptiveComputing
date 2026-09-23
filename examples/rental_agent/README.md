# rental_agent — Rental Car Electrification Co-Scientist

An interactive LangGraph agent that optimises vehicle fleet charging cost at an
airport rental car facility using Bayesian optimisation over four parameters
(utility rate, storage, daily vehicle demand, return state-of-charge).

## Quick start

```bash
cd examples/rental_agent
cp hpc_config.py.template hpc_config.py   # fill in your HPC details
python rental_agent.py
```

For multi-session management use the co-scientist portal:

```bash
python co_scientist.py
```

## Configuration

**`hpc_config.py`** (created from `hpc_config.py.template`) controls how the
remote manager connects to your HPC cluster.  Key fields:

| Field | Description |
|---|---|
| `machine_names` | List of logical machine names |
| `remote_usernames` | SSH username per machine |
| `remote_hosts` | **Pin to a specific login node** (not a load-balanced hostname) so the manager's tmux session persists |
| `remote_dirs` | Absolute path to this directory on the remote machine |
| `scheduler` | `'slurm'` (default) or `'pbs'` per machine |
| `batch_scripts` | Script to submit per machine — see *Scheduler support* below |
| `python_paths` | Full path to the AC conda Python on the remote machine |
| `debug_run` | Set `True` to use a short-walltime debug partition during testing |

## Scheduler support

Two batch scripts are provided in `simulation_files/`:

| Script | Scheduler | Submission |
|---|---|---|
| `script_slurm.sh` | SLURM (sbatch) | `sbatch script_slurm.sh <task_id>` |
| `script_pbs.sh` | PBS/Torque (qsub) | `qsub -v "task_id=<id>" script_pbs.sh` |

To switch schedulers, set `scheduler` and `batch_scripts` in `hpc_config.py`:

```python
# SLURM (default)
scheduler    = {'kestrel': 'slurm'}
batch_scripts = {'kestrel': ['script_slurm.sh']}

# PBS / Torque
scheduler    = {'vermilion': 'pbs'}
batch_scripts = {'vermilion': ['script_pbs.sh']}
```

Remember to update the `#SBATCH` / `#PBS` directives in the script itself
(account, queue/partition, walltime) to match your site.

## Files

```
rental_agent/
  rental_agent.py          Main agent (MCP-server pattern)
  co_scientist.py          Multi-session portal
  manager.py               Remote manager daemon (runs on HPC login node)
  hpc_config.py.template   Template — copy to hpc_config.py
  hpc_config.py            Your site config (git-ignored)
  simulation_files/
    mock_simulation.py     Analytic cost model (replace with real simulation)
    script_slurm.sh        SLURM batch script
    script_pbs.sh          PBS/Torque batch script
```
