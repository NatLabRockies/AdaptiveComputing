"""
CLI entry point for scaffolding a new Hero/HPC application directory.

Run once when starting a new application::

    python -m adaptive_computing.hpc.install_scripts [target_dir]
    # or via the console entry point:
    ac-install-scripts [target_dir]

If *target_dir* is omitted the current directory is used.

Copies ``manager_template.py`` and ``hpc_config_template.py`` from the package
into *target_dir* as starting points.  No shell scripts need to be copied —
the manager infrastructure (tmux session management, job cancellation) runs
entirely through ``python -m adaptive_computing.hpc.remote_manager`` and
``python -m adaptive_computing.hpc.kill_scheduler_jobs``, both of which are
available on the remote machine through the installed AC environment.
"""

import os
import shutil
import sys


_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def main():
    target_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")

    if not os.path.isdir(target_dir):
        print(f"ERROR: target directory does not exist: {target_dir}")
        sys.exit(1)

    files = [
        ("manager_template.py",    "manager.py"),
        ("hpc_config_template.py", "hpc_config.py"),
    ]

    for src_name, dst_name in files:
        src = os.path.join(_TEMPLATES_DIR, src_name)
        dst = os.path.join(target_dir, dst_name)
        if not os.path.exists(src):
            print(f"WARNING: template not found, skipping: {src}")
            continue
        if os.path.exists(dst):
            print(f"Skipping (already exists): {dst}")
            continue
        shutil.copy2(src, dst)
        print(f"Created: {dst}")

    print(
        "\nNext steps:\n"
        "  1. Edit hpc_config.py — fill in machine names, SSH hosts, remote dirs,\n"
        "     python_paths, and batch script paths.\n"
        "  2. Edit manager.py — implement submit_job() and read_result() for your\n"
        "     simulation.\n"
        "  3. Copy manager.py and hpc_config.py to each machine's remote_dirs path.\n"
        "     (No shell scripts needed — the AC environment provides everything else.)"
    )


if __name__ == "__main__":
    main()
