#!/usr/bin/env python3
"""
hf_script.py — High-fidelity simulation script for the onsite_mf example.

This script is launched as a subprocess by LocalWorkerPool's HF workers.
It receives parameters via command-line arguments and writes a single float
result to the specified output file.

Interface (called by the HF worker)::

    python hf_script.py --x_data '[1.5]' --output /tmp/result_xxx.txt

The script must:
  1. Parse --x_data (JSON-encoded list of floats)
  2. Run the simulation
  3. Write a single float to --output  (one line, no trailing whitespace)

GPU usage
---------
The HF worker sets CUDA_VISIBLE_DEVICES before launching this script, so
``import torch; torch.cuda.current_device()`` will see exactly the assigned
GPU.  For CPU-only runs CUDA_VISIBLE_DEVICES is set to the empty string.

Adapting for a real simulation
--------------------------------
Replace the body of the ``simulate()`` function with your actual workload:
    - Call an external binary via subprocess
    - Run a PyTorch / JAX model on the assigned GPU
    - Parse output from a domain-specific tool (OpenFOAM, LAMMPS, etc.)
"""

import argparse
import json
import sys
import time


def simulate(x_data: list) -> float:
    """More-accurate thermal conductance model (HF).

    In a real workflow this would call an expensive solver, a GPU neural
    network, or a compiled simulation binary.  Here we add a linear correction
    term and a short sleep to mimic non-trivial runtime.

    k_hf(T) = T² / 1000 + 0.05 * T
    """
    t = float(x_data[0])
    time.sleep(0.5)          # simulate computational cost
    return t * t / 1000.0 + 0.05 * t


def main() -> None:
    parser = argparse.ArgumentParser(description="HF simulation script")
    parser.add_argument("--x_data", required=True,
                        help="JSON-encoded list of input parameters, e.g. '[1.5]'")
    parser.add_argument("--output", required=True,
                        help="Path to write the float result")
    args = parser.parse_args()

    try:
        x_data = json.loads(args.x_data)
    except json.JSONDecodeError as exc:
        print(f"ERROR: could not parse --x_data: {exc}", file=sys.stderr)
        sys.exit(1)

    result = simulate(x_data)

    with open(args.output, "w") as f:
        f.write(f"{result}\n")

    print(f"[hf_script] x={x_data} → y={result:.6f}")


if __name__ == "__main__":
    main()
