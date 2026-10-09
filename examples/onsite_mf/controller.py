"""
controller.py — Multi-fidelity Bayesian optimization on a local workstation.

Demonstrates onsite_mf: LocalHeroClient + LocalWorkerPool, no HPC scheduler.

  Low-fidelity  (fidelity 0): fast Python callable run in CPU subprocesses
  High-fidelity (fidelity 1): accurate command-line script run in HF workers
                               (with optional GPU assignment)

Usage::

    python controller.py

The script prints progress as LF and HF evaluations complete and saves a
scatter-plot of the results to ``bayesian_1d_mf_result.png``.

Adapting for your own simulation
---------------------------------
1. Edit ``simulation.py`` — replace ``run_lf`` with your cheap model.
2. Edit ``hf_script.py`` — replace ``simulate()`` with your expensive model.
3. Edit ``config.py`` — set param bounds, costs, and worker counts.
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure the example directory is importable
# ---------------------------------------------------------------------------

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_AC_ROOT     = os.path.dirname(os.path.dirname(_EXAMPLE_DIR))
for _p in [_EXAMPLE_DIR, _AC_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import config
import simulation

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHeroCostRatio
from adaptive_computing.hpc.worker_pool import LocalWorkerPool
from adaptive_computing.local_hero import LocalHeroClient


# ---------------------------------------------------------------------------
# Optional: matplotlib for result plot
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Clean up any leftover database from a previous run ────────────────
    if os.path.exists(config.HERO_DB_PATH):
        os.remove(config.HERO_DB_PATH)
    for ext in (".lock", ".stop"):
        p = config.HERO_DB_PATH + ext
        if os.path.exists(p):
            os.remove(p)

    # ── Shared LocalHeroClient (file-backed, no AWS/network required) ──────
    local_hero = LocalHeroClient(
        db_path    = config.HERO_DB_PATH,
        queue_name = config.BASE_QUEUE,
    )

    # ── Worker pool — auto-launches LF and HF worker subprocesses ──────────
    pool = LocalWorkerPool(
        hero_db_path  = config.HERO_DB_PATH,
        base_queue    = config.BASE_QUEUE,
        lf_module     = "simulation",
        lf_fn_name    = "run_lf",
        hf_script     = simulation.HF_SCRIPT,
        n_lf_workers  = config.N_LF_WORKERS,
        n_hf_workers  = config.N_HF_WORKERS,
        gpu_ids       = config.GPU_IDS,
        search_paths  = [_EXAMPLE_DIR],
        poll_interval = config.POLL_INTERVAL,
    )

    params = [ContinuousVariable(min=config.PARAM_MIN, max=config.PARAM_MAX)]

    # ── Multifidelity driver (no inline_manager — workers run in background) ─
    # With inline_manager=None the driver queues tasks and returns immediately;
    # hero_wait_for_data_and_train() blocks until workers deliver all results.
    ac_driver = ActiveLoopDriverHeroCostRatio(
        simulations    = [None, None],      # workers handle evaluation
        fidelity_costs = config.FIDELITY_COSTS,
        params         = params,
        machine_names  = ["local"],
        output_field_path = "y_data",
        surrogate      = "SMT_GP",
        acq_func       = "maximum_variance",
        blocking       = False,
        inline_manager = None,              # background workers, not blocking manager
        hero_client    = local_hero,
    )

    with pool:
        # ── Initialisation: LHS warm-up at both fidelities ──────────────────
        x_init_lf = np.linspace(config.PARAM_MIN, config.PARAM_MAX,
                                 config.N_INIT_LF).reshape(-1, 1)
        x_init_hf = np.linspace(config.PARAM_MIN, config.PARAM_MAX,
                                 config.N_INIT_HF).reshape(-1, 1)

        print(f"\nQueuing {config.N_INIT_LF} LF + {config.N_INIT_HF} HF init samples...")
        ac_driver.add_samples(x_init_lf, i_fidelity=0)
        ac_driver.add_samples(x_init_hf, i_fidelity=1)

        print("Waiting for init evaluations...")
        ac_driver.hero_wait_for_data_and_train()

        print("\nInit complete.")
        print(f"  LF data: {ac_driver.dataset.x_data[0].flatten()} → "
              f"{ac_driver.dataset.y_data[0].flatten()}")
        print(f"  HF data: {ac_driver.dataset.x_data[1].flatten()} → "
              f"{ac_driver.dataset.y_data[1].flatten()}")

        # ── BO loop ──────────────────────────────────────────────────────────
        print(f"\nRunning {config.N_STEPS} multifidelity BO steps...")
        ac_driver.run(N_steps=config.N_STEPS)
        print("Waiting for BO evaluations...")
        ac_driver.hero_wait_for_data_and_train()

    # ── Results ──────────────────────────────────────────────────────────────
    lf_x = ac_driver.dataset.x_data[0].flatten()
    lf_y = ac_driver.dataset.y_data[0].flatten()
    hf_x = ac_driver.dataset.x_data[1].flatten()
    hf_y = ac_driver.dataset.y_data[1].flatten()

    best_hf_idx = np.nanargmax(hf_y)
    print(f"\nBest HF result: x={hf_x[best_hf_idx]:.3f}, y={hf_y[best_hf_idx]:.6f}")
    print(f"Total evaluations: {len(lf_x)} LF + {len(hf_x)} HF")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        plt = _setup_matplotlib()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(lf_x, lf_y, marker="o", color="steelblue",
                   label=f"Low-fidelity (n={len(lf_x)})", zorder=3)
        ax.scatter(hf_x, hf_y, marker="s", color="tomato", s=80,
                   label=f"High-fidelity (n={len(hf_x)})", zorder=4)
        ax.set_xlabel("Temperature T (°C)")
        ax.set_ylabel("Thermal conductance k")
        ax.set_title("onsite_mf — Multi-Fidelity Bayesian Optimisation")
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(_EXAMPLE_DIR, "bayesian_1d_mf_result.png")
        fig.savefig(out_png, dpi=150)
        print(f"Plot saved to {out_png}")
        try:
            plt.show()
        except Exception:
            pass
    except ImportError:
        print("(matplotlib not available — skipping plot)")

    return ac_driver


if __name__ == "__main__":
    main()
