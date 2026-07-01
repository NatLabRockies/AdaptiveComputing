# Hero Introduction - Getting Started

This directory provides a simple introduction to the Hero framework for task queue management. **Start here** before exploring more complex HPC workflows.

## What You'll Learn

- How Hero controllers add tasks to queues
- How Hero workers process tasks from queues  
- Basic Hero authentication and queue operations
- Simple local task processing without HPC complexity

## Files

- **controller_noAC.py**: Basic Hero task management without adaptive computing - manually adds tasks and monitors queue status (direct `HeroClient`/`TaskEngine` API)
- **controller_manual.py**: Mid-level Hero integration using the `HeroDataset` API - demonstrates different ways to add samples (queued, local-only, with initial guesses)
- **controller.py**: Full Bayesian optimization loop using `ActiveLoopDriverHero` - runs active learning with a surrogate model and acquisition function
- **worker.py**: Processes tasks from the Hero queue using a simple conductivity calculation (`conductivity = temperature²/1000`)

## Prerequisites

1. Complete the [main repository setup](../../README.md)
2. Configure your Hero credentials in `adaptive_computing/hero_utils/set_hero_env_vars.py`

## Quick Start

**Terminal 1** - Start the worker:
```bash
cd examples/hero
mamba activate AC
python worker.py
```

**Terminal 2** - Run the controller:
```bash
cd examples/hero
mamba activate AC  
python controller_noAC.py  # Direct Hero API example
# OR
python controller_manual.py # HeroDataset API example
# OR
python controller.py        # Full Bayesian optimization example
```

## What Happens

1. **Controller**: Creates tasks with different temperature values and adds them to a Hero queue
2. **Worker**: Continuously monitors the queue, processes tasks locally, and marks them complete
3. **Controller**: Retrieves completed results and displays the temperature/conductivity pairs

**Three controllers in order of increasing abstraction**:
- **controller_noAC.py**: Learn the **direct Hero API** - shows low-level `HeroClient`/`TaskEngine` queue operations with detailed status monitoring
- **controller_manual.py**: Learn the **HeroDataset API** - demonstrates different ways to add samples (queued, local-only, with initial guesses)
- **controller.py**: Learn the **ActiveLoopDriverHero API** - runs a full Bayesian optimization loop with a surrogate model and acquisition function

All three controllers work with the same `worker.py` and represent a natural learning progression from raw API to full adaptive computing.

## Expected Output

**Worker output**:
```
Processing task: temperature=1.1, calculated conductivity=0.00121
Processing task: temperature=1.5, calculated conductivity=0.00225
...
```

**Controller output**:
```
Task temperature=[1.1], computed conductivity=[0.00121]
Task temperature=[1.5], computed conductivity=[0.00225]
...
```

## Benefits of This Approach

- ✅ **No HPC required**: Runs entirely on your local machine
- ✅ **Fast setup**: No SSH, SLURM, or cluster configuration needed
- ✅ **Clear separation**: See exactly how controllers and workers interact
- ✅ **Easy debugging**: Simple calculation makes it easy to verify results
- ✅ **Perfect for testing**: Validate Hero setup and credentials

## Next Steps

Once you're comfortable with these concepts, explore the [production HPC workflows](../hero_HPC_managers/) that use:
- Automated SSH deployment to multiple HPC clusters
- Real SLURM job submission and molecular dynamics simulations
- Multi-machine task distribution and error handling

## Troubleshooting

**"Authentication failed"**: Check your Hero credentials in `adaptive_computing/hero_utils/set_hero_env_vars.py`

**"Queue not found"**: Make sure your HERO_QUEUE environment variable is set correctly

**Import errors**: Ensure you've activated the `AC` conda environment and run `pip install -e .`