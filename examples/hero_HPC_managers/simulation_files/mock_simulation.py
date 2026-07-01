#!/usr/bin/env python3
"""
Mock simulation for the generic HPC tutorial.

Computes conductivity = temperature^2 / 1000 to stand in for a real
simulation (e.g., LAMMPS molecular dynamics). The result is printed
in the format 'conductivity=<value>' so that script_generic.sh can
parse it with grep/awk — mirroring how script_kestrel.sh parses the
LAMMPS output file.

Usage: python mock_simulation.py <temperature>
"""
import sys
import time


def main():
    if len(sys.argv) != 2:
        print("Usage: python mock_simulation.py <temperature>")
        sys.exit(1)

    try:
        temp = float(sys.argv[1])
    except ValueError:
        print(f"Error: temperature must be a number, got '{sys.argv[1]}'")
        sys.exit(1)

    print(f"Starting mock simulation: temperature={temp}")

    # Simulate computation time. Replace this block with a real simulation
    # call (e.g., subprocess to run a code, file I/O, etc.).
    time.sleep(2)

    # Compute result: same formula used in examples/hero/worker.py
    conductivity = temp * temp / 1000.0

    # Print result in parseable format — script_generic.sh greps this line
    print(f"conductivity={conductivity}")
    print(f"Mock simulation complete: temp={temp:.4f} -> conductivity={conductivity:.6f}")


if __name__ == "__main__":
    main()
