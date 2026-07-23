"""
CLI entry point for canceling all scheduler jobs tracked by the Hero queue.

Run from the application directory (where hpc_config.py lives)::

    python -m adaptive_computing.hpc.kill_scheduler_jobs <machine_name>

"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m adaptive_computing.hpc.kill_scheduler_jobs <machine_name>")
        sys.exit(1)

    machine_name = sys.argv[1]

    # Import hpc_config from the current working directory.
    try:
        import hpc_config
    except ImportError:
        print(
            "ERROR: hpc_config.py not found in the current directory.\n"
            f"  cwd: {os.getcwd()}"
        )
        sys.exit(1)

    from adaptive_computing.hpc.cleanup import kill_all_scheduler_jobs
    kill_all_scheduler_jobs(hpc_config, machine_name)


if __name__ == "__main__":
    main()
