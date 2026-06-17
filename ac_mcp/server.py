"""
server.py
=========
FastMCP server exposing AdaptiveComputing surrogate modelling capabilities
as reusable, application-agnostic MCP tools.

Tools
-----
create_experiment   — register a new experiment in the registry
run_evaluations     — submit specific jobs (no surrogate), async
run_optimization    — LHS warm-up + EI-guided BO loop, async
get_run_status      — poll a running or completed run
predict             — query the trained surrogate (mean + variance)
fork_experiment     — copy dataset to a new experiment, immediately re-train
list_experiments    — search the registry
get_experiment      — retrieve one registry entry in full

Start the server
----------------
    # SNN agent application:
    python -m ac_mcp.server --storage-dir /projects/newbridge/kgriffin/stdp-mnist/agent

    # Override host/port via env vars:
    AC_MCP_HOST=0.0.0.0 AC_MCP_PORT=9000 python -m ac_mcp.server --storage-dir /path/to/app
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any, Optional

from fastmcp import FastMCP

# Ensure the AC package is importable
_AC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AC_ROOT not in sys.path:
    sys.path.insert(0, _AC_ROOT)

mcp = FastMCP(
    name="AdaptiveComputing",
    instructions=(
        "Surrogate modelling and Bayesian optimisation tools powered by "
        "AdaptiveComputing.  Use create_experiment first, then "
        "run_evaluations or run_optimization, then poll get_run_status."
    ),
)


# ===========================================================================
# Tool: create_experiment
# ===========================================================================

@mcp.tool()
def create_experiment(
    name: str,
    description: str,
    param_specs: list[dict],
    fixed_context: dict,
    output_label: str,
    hpc_config_path: str,
    output_field_path: str = "y_data",
    experiment_type: str = "optimization",
    force_new: bool = False,
) -> dict:
    """
    Register a new experiment and return its experiment_id.

    Before creating a new entry, checks the registry for an existing *completed*
    experiment with the same name, experiment_type, fixed_context, and param_specs.
    If one is found it is returned immediately (no new simulations needed).
    Pass force_new=True to skip the cache check and always create a fresh experiment.

    Parameters
    ----------
    name : str
        Short human-readable name, e.g. "g_b_val sweep BE-LIF".
    description : str
        Longer description of what this experiment investigates.
    param_specs : list[dict]
        Free variables the surrogate/optimiser will explore.
        Each dict must have "name" and "type" plus type-specific keys:
          {"name": "g_b_val",     "type": "continuous", "min": 0.0, "max": 10.0}
          {"name": "n_layers",    "type": "ordered",    "min": 1,   "max": 8}
          {"name": "neuron_type", "type": "categorical", "categories": ["LIF","BE-LIF"]}
        Pass an empty list [] for evaluation-only experiments.
    fixed_context : dict
        Parameters constant across all jobs, e.g. {"n_e": 100, "neuron_type": "BE-LIF"}.
        These are written verbatim into every job's metadata.
    output_label : str
        Semantic description of y_data, e.g. "MNIST prediction accuracy (%)".
        Stored in the registry for co-scientist reasoning.
    hpc_config_path : str
        Absolute path to the application's hpc_config.py (contains machine_names,
        remote_usernames, remote_hosts, remote_dirs).
    output_field_path : str
        Field in the Hero task metadata that holds the simulation result.
        Defaults to "y_data".
    experiment_type : str
        "optimization" or "evaluation".
    force_new : bool
        If True, skip cache check and always register a new experiment.

    Returns
    -------
    dict with "experiment_id" (str) and "reused" (bool, True if an existing
    completed experiment was returned instead of creating a new one).
    """
    from ac_mcp import registry

    if not force_new:
        existing = registry.find_matching_experiment(
            name=name,
            param_specs=param_specs,
            fixed_context=fixed_context,
            experiment_type=experiment_type,
        )
        if existing is not None:
            return {"experiment_id": existing["id"], "reused": True,
                    "n_samples": existing["n_samples"],
                    "best_x": existing["best_x"],
                    "best_y": existing["best_y"]}

    exp_id = registry.register_experiment(
        name=name,
        description=description,
        param_specs=param_specs,
        fixed_context=fixed_context,
        output_label=output_label,
        hpc_config_path=hpc_config_path,
        output_field_path=output_field_path,
        experiment_type=experiment_type,
    )
    return {"experiment_id": exp_id, "reused": False}


# ===========================================================================
# Tool: run_evaluations
# ===========================================================================

@mcp.tool()
def run_evaluations(
    experiment_id: str,
    jobs: list[dict],
) -> dict:
    """
    Submit a batch of evaluation jobs (no surrogate training) and return
    a run_id immediately.  Poll get_run_status for results.

    Use this for simulate / compare workflows where you want specific
    parameter combinations evaluated without optimisation.

    Parameters
    ----------
    experiment_id : str
        ID returned by create_experiment.
    jobs : list[dict]
        One dict per job.  Each dict should contain all fields the Hero
        manager needs (e.g. neuron_type, STDP_type, n_e, g_b_val).
        Any field not present is filled from fixed_context at submission time.

    Returns
    -------
    dict with "run_id" (str).
    """
    from ac_mcp import registry, run_manager

    entry = registry.get_entry(experiment_id)

    # Merge fixed_context into each job so the manager sees the full picture
    merged_jobs = []
    for job in jobs:
        merged = dict(entry["fixed_context"])
        merged.update(job)
        merged_jobs.append(merged)

    run_id = run_manager.submit_evaluation_run(entry, merged_jobs)
    return {"run_id": run_id}


# ===========================================================================
# Tool: run_optimization
# ===========================================================================

@mcp.tool()
def run_optimization(
    experiment_id: str,
    n_init_samples: int = 3,
    n_steps: int = 5,
    acq_func: str = "expected_improvement",
    blocking: bool = False,
) -> dict:
    """
    Run a full Bayesian optimisation loop (LHS warm-up + EI-guided steps)
    and return a run_id immediately.  Poll get_run_status for progress and results.

    Both modes use driver.run(N_steps) then hero_wait_for_data_and_train().
    The difference is how ActiveLoopDriverHero is instantiated:

    blocking=False (default, parallel / "Kriging Believer" batch BO):
        All n_steps Hero tasks are queued immediately using surrogate placeholder
        predictions between steps.  Simulations run in parallel on HPC.  The
        surrogate is retrained once on all real results.  Faster wall-clock time.

    blocking=True (sequential BO):
        Each step waits for the simulation to complete and retrains the surrogate
        on real data before proposing the next point.  Higher sample efficiency
        (each EI decision uses real results), but n_steps times slower.

    Parameters
    ----------
    experiment_id : str
        Must have been created with at least one param_spec (the optimisation space).
    n_init_samples : int
        Number of LHS warm-up evaluations before the surrogate is first trained.
    n_steps : int
        Number of acquisition-function-guided evaluations after warm-up.
    acq_func : str
        Acquisition function: "expected_improvement" or "maximum_variance".
    blocking : bool
        False (default): parallel batch BO — all n_steps jobs run simultaneously.
        True: sequential BO — each job informs the next before it is submitted.

    Returns
    -------
    dict with "run_id" (str).
    """
    from ac_mcp import registry, run_manager

    entry = registry.get_entry(experiment_id)
    if not entry["param_specs"]:
        raise ValueError(
            "run_optimization requires at least one param_spec. "
            "Use run_evaluations for fixed-parameter jobs."
        )
    run_id = run_manager.submit_optimization_run(
        entry, n_init_samples, n_steps, acq_func, blocking
    )
    return {"run_id": run_id}


# ===========================================================================
# Tool: get_run_status
# ===========================================================================

@mcp.tool()
def get_run_status(run_id: str) -> dict:
    """
    Return the current status of a run started by run_evaluations or
    run_optimization.

    Returned fields
    ---------------
    status       : "running" | "completed" | "error"
    n_completed  : jobs finished so far
    n_total      : total jobs in this run
    results      : list of {x: {param_name: value, ...}, y: float|null}
    best_x       : parameter dict for the best result seen so far
    best_y       : best y value seen so far
    error        : error message if status == "error", else null
    """
    from ac_mcp import run_manager
    return run_manager.get_status(run_id)


# ===========================================================================
# Tool: predict
# ===========================================================================

@mcp.tool()
def predict(experiment_id: str, x_points: list[list]) -> dict:
    """
    Query the trained surrogate model for mean and variance predictions at
    arbitrary input points.  No simulation jobs are submitted.

    Requires the experiment to have a saved driver with a trained surrogate
    (i.e. run_optimization must have completed at least the warm-up phase).

    Parameters
    ----------
    experiment_id : str
    x_points : list[list[float]]
        Each inner list is one point in the same order as param_specs.
        E.g. [[1.5], [3.0]] for a 1-D g_b_val experiment.

    Returns
    -------
    dict with "predictions": list of {x, mean, variance}.
    """
    import numpy as np
    from ac_mcp import registry

    entry  = registry.get_entry(experiment_id)
    driver = registry.load_driver(experiment_id)

    x_arr   = np.array(x_points, dtype=float)
    means   = driver.surrogate.predict_values(x_arr)
    variances = driver.surrogate.predict_variances(x_arr)

    preds = []
    for i, x_row in enumerate(x_points):
        preds.append({
            "x":        x_row,
            "mean":     float(np.array(means).flat[i]),
            "variance": float(np.array(variances).flat[i]),
        })
    return {"experiment_id": experiment_id, "predictions": preds}


# ===========================================================================
# Tool: fork_experiment
# ===========================================================================

@mcp.tool()
def fork_experiment(
    source_id: str,
    new_name: str,
    new_description: str,
    new_param_specs: Optional[list[dict]] = None,
    new_output_label: str = "y_data",
    new_output_field_path: str = "y_data",
) -> dict:
    """
    Create a new experiment by copying the dataset from an existing one.

    This is useful when you want to:
    - Optimise a different objective over the same data
    - Explore a different subspace after observing initial results
    - Build a secondary surrogate on a different output column

    The forked experiment's surrogate is immediately trained on the copied data.

    Parameters
    ----------
    source_id : str
        Experiment to fork from.
    new_name : str
    new_description : str
    new_param_specs : list[dict] or None
        If None, inherits param_specs from the source.
    new_output_label : str
        Semantic label for the new objective.
    new_output_field_path : str
        Field in task metadata to use as y.

    Returns
    -------
    dict with "experiment_id" (str) for the new experiment.
    """
    import pickle
    import numpy as np
    from ac_mcp import registry

    source_entry = registry.get_entry(source_id)
    source_driver = registry.load_driver(source_id)

    param_specs = new_param_specs or source_entry["param_specs"]

    new_id = registry.register_experiment(
        name=new_name,
        description=new_description,
        param_specs=param_specs,
        fixed_context=source_entry["fixed_context"],
        output_label=new_output_label,
        hpc_config_path=source_entry["hpc_config_path"],
        output_field_path=new_output_field_path,
        experiment_type=source_entry["experiment_type"],
    )
    # Record provenance
    registry.update_entry(new_id, forked_from=source_id)

    # Deep-copy driver, train surrogate immediately on copied data
    new_driver = pickle.loads(pickle.dumps(source_driver))
    try:
        new_driver.surrogate.train(new_driver.dataset)
    except Exception as exc:
        # Not enough data yet — that's OK, will train when more data arrives
        print(f"fork_experiment: surrogate training skipped ({exc})")

    registry.save_driver(new_id, new_driver)
    return {"experiment_id": new_id}


# ===========================================================================
# Tool: list_experiments
# ===========================================================================

@mcp.tool()
def list_experiments(name_filter: Optional[str] = None) -> list:
    """
    Return a list of experiments from the registry, newest first.

    Each entry contains: id, name, description, created_at, output_label,
    n_samples, best_x, best_y, experiment_type, forked_from.

    Parameters
    ----------
    name_filter : str or None
        If provided, only return experiments whose name contains this substring
        (case-insensitive).
    """
    from ac_mcp import registry
    entries = registry.list_entries(name_filter=name_filter)
    # Return a concise summary rather than the full entry (param_specs can be large)
    return [
        {
            "id":              e["id"],
            "name":            e["name"],
            "description":     e["description"],
            "created_at":      e["created_at"],
            "output_label":    e["output_label"],
            "experiment_type": e["experiment_type"],
            "n_samples":       e["n_samples"],
            "best_x":          e["best_x"],
            "best_y":          e["best_y"],
            "forked_from":     e.get("forked_from"),
        }
        for e in entries
    ]


# ===========================================================================
# Tool: get_experiment
# ===========================================================================

@mcp.tool()
def get_experiment(experiment_id: str) -> dict:
    """
    Return the full registry entry for one experiment, including param_specs,
    fixed_context, hpc_config_path, and current summary statistics.
    """
    from ac_mcp import registry
    return registry.get_entry(experiment_id)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AC MCP Server")
    parser.add_argument(
        "--storage-dir",
        required=True,
        help="Directory for registry.json and experiment pickles (application-specific).",
    )
    parser.add_argument("--host", default=os.environ.get("AC_MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AC_MCP_PORT", "8765")))
    args = parser.parse_args()

    # Set before any registry call so _storage_dir() picks it up
    os.environ["AC_MCP_DIR"] = os.path.abspath(args.storage_dir)

    print(f"Starting AC MCP server on http://{args.host}:{args.port}")
    print(f"Storage dir: {os.environ['AC_MCP_DIR']}")
    mcp.run(transport="http", host=args.host, port=args.port)
