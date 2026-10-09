"""
simulation.py — Simulation functions for the onsite_mf example.

This file demonstrates the two supported worker patterns:

Low-fidelity (LF) — Python callable
-------------------------------------
``run_lf(x_data)`` is called directly inside the LF worker subprocess.
It should be fast (milliseconds to seconds) and purely Python so that many
instances can run in parallel on CPU cores.  Here it uses a closed-form
analytic approximation.

High-fidelity (HF) — command-line script
-----------------------------------------
``hf_script.py`` (in the same directory) is the HF simulation.  It is
launched as a subprocess by the HF worker with::

    python hf_script.py --x_data '[1.5]' --output /tmp/result_xxx.txt

It should write a single float to the output file.  This pattern works for
compiled binaries, external tools, or any simulation that benefits from
subprocess isolation (separate memory space, GPU ownership, etc.).

Physical model
--------------
Both fidelities model 1-D steady-state heat conductance through a material
slab as a function of temperature ``T`` (°C, range 0–10):

    LF:  k_lf(T) = T² / 1000               — cheap analytic approximation
    HF:  k_hf(T) = T² / 1000 + 0.05 * T   — more accurate (linear correction)

The goal is to find the temperature that maximises conductance.  The true
optimum is at T = 10 (boundary of the domain for this simplified model).
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the HF script — passed to LocalWorkerPool as hf_script.
HF_SCRIPT = os.path.join(EXAMPLE_DIR, "hf_script.py")


# ---------------------------------------------------------------------------
# Low-fidelity function  (Python callable)
# ---------------------------------------------------------------------------

def run_lf(x_data: list) -> float:
    """Cheap analytic approximation of thermal conductance.

    Args:
        x_data: List with a single element — temperature T in [0, 10].

    Returns:
        Approximate conductance k_lf = T² / 1000.
    """
    t = float(x_data[0])
    return t * t / 1000.0
