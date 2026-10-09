"""
config.py — Configuration for the onsite_mf example.

Edit this file to match your workstation hardware and simulation requirements.
"""

import os

# ---------------------------------------------------------------------------
# Optimization problem
# ---------------------------------------------------------------------------

# Search space: temperature in [0, 10] °C
PARAM_MIN = 0.0
PARAM_MAX = 10.0

# Relative cost of one HF evaluation vs one LF evaluation.
# A cost ratio of 100 means one HF sim is 100x more expensive than one LF sim.
# The multifidelity driver uses this to decide when to spend budget on HF.
FIDELITY_COSTS = [1, 100]

# LHS initialisation samples per fidelity level
N_INIT_LF = 5
N_INIT_HF = 3

# Number of multifidelity BO steps (each step queries one fidelity level)
N_STEPS = 10

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

# Number of LF workers (Python callable, CPU).
# None → min(cpu_count - 1, 8)  — capped at 8 to avoid overloading shared
# login nodes.  On a personal workstation, set explicitly (e.g. 16 or 32).
N_LF_WORKERS = None

# Number of HF workers (command-line script, optionally GPU).
# None → one worker per detected GPU, or 1 if no GPU is found.
N_HF_WORKERS = None

# GPU device IDs assigned round-robin to HF workers.
# None → auto-detect via nvidia-smi.
# []   → force CPU-only HF workers (set CUDA_VISIBLE_DEVICES="").
GPU_IDS = None   # e.g. [0, 1] to use the first two GPUs

# Seconds between queue polls when the queue is empty.
POLL_INTERVAL = 0.5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
HERO_DB_PATH = os.path.join(EXAMPLE_DIR, "local_hero_db.json")
BASE_QUEUE   = "local"   # fidelity 0 → "local", fidelity 1 → "local1"
