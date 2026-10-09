# onsite_mf — Multi-Fidelity BO on a Local Workstation

Run multi-fidelity Bayesian optimisation entirely on one machine — no HPC
scheduler (SLURM / PBS), no AWS Hero, no SSH.

## What it demonstrates

| Aspect | Detail |
|---|---|
| **Queue backend** | `LocalHeroClient` — file-backed JSON, no network |
| **Fidelities** | 2 — LF (cheap) and HF (accurate) |
| **LF worker** | Python callable (`simulation.run_lf`) — runs in CPU subprocesses |
| **HF worker** | Command-line script (`hf_script.py`) — launched as subprocess, GPU-aware |
| **Scheduling** | `LocalWorkerPool` auto-launches and manages all workers |
| **Optimiser** | `ActiveLoopDriverHeroCostRatio` — cost-ratio MF-BO |

## Quick start

```bash
cd examples/onsite_mf
python controller.py
```

Results are printed to the terminal and a plot is saved to
`bayesian_1d_mf_result.png`.

## Adapting for your own simulation

1. **`simulation.py`** — replace `run_lf(x_data)` with your cheap model
   (analytic formula, fast surrogate, reduced-order model, etc.).

2. **`hf_script.py`** — replace the `simulate()` function with your
   expensive model.  The script receives parameters via `--x_data '[...]'`
   and must write a float to `--output <path>`.  On a GPU workstation the
   `CUDA_VISIBLE_DEVICES` environment variable is set automatically by the
   HF worker before the script is launched.

3. **`config.py`** — set the parameter bounds, fidelity cost ratio, and
   worker counts for your hardware.

## Worker pool configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `N_LF_WORKERS` | `cpu_count − 1` | Parallel LF evaluation threads |
| `N_HF_WORKERS` | `len(GPUs)` or 1 | Parallel HF workers |
| `GPU_IDS` | auto-detect | Which GPU devices to assign to HF workers |
| `POLL_INTERVAL` | 0.5 s | Queue polling frequency |

## Running multiple investigations (Marc's scenario)

Each controller instance gets its own `local_hero_db.json` path — just change
`HERO_DB_PATH` in `config.py` or pass a unique path per run.  Multiple
`onsite_mf` controllers on the same workstation never share queues, so they
cooperate automatically: each pool claims the cores / GPUs it needs without
interfering with others.

## Comparison with related examples

| Example | Queue | Scheduler | Fidelities |
|---|---|---|---|
| `hero` | Cloud Hero (AWS) | None | 1 |
| `HPC_onsite` | LocalHero | SLURM / PBS | 1 |
| `HPC_onsite_mf` | LocalHero | SLURM / PBS | 2+ |
| **`onsite_mf`** | **LocalHero** | **None** | **2+** |
