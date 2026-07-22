"""
scheduler.py — SLURM / PBS abstraction for job status, submission, and cancellation.

All public functions return normalised status strings:
    'RUNNING'   — job is executing on a compute node
    'PENDING'   — job is queued or otherwise waiting
    'COMPLETED' — job finished with exit code 0
    'FAILED'    — job finished with non-zero exit code, was cancelled, or timed out
    'UNKNOWN'   — status could not be determined (stale reference)
"""

from __future__ import annotations

import re
import subprocess


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

def get_slurm_status(job_id: str, result_file: str | None = None) -> str:
    """Return normalised SLURM job status.

    Checks ``sacct`` first; falls back to ``squeue`` for jobs not yet in
    accounting.  If neither scheduler command shows the job and *result_file*
    is provided, its existence is used as a proxy for completion (handles
    ``sacct`` lag on debug partitions).

    Args:
        job_id:      SLURM job ID string.
        result_file: Optional path to the simulation output file.  When given,
                     a missing ``sacct``/``squeue`` record combined with an
                     existing result file is treated as COMPLETED.

    Returns:
        One of 'RUNNING', 'PENDING', 'COMPLETED', 'FAILED', 'UNKNOWN'.
    """
    import os
    sacct = subprocess.run(
        f"sacct -j {job_id} --format=State --noheader",
        shell=True, capture_output=True, text=True,
    )
    status = sacct.stdout.strip()

    if "COMPLETED" in status:
        return "COMPLETED"
    if any(s in status for s in ("FAILED", "CANCELLED", "TIMEOUT")):
        return "FAILED"
    if "RUNNING" in status:
        return "RUNNING"
    if status:
        return "PENDING"

    # sacct has no record yet — check squeue
    squeue = subprocess.run(
        f"squeue -j {job_id} --noheader",
        shell=True, capture_output=True, text=True,
    )
    if squeue.stdout.strip():
        return "PENDING"

    # Not in sacct or squeue
    if result_file is not None and os.path.exists(result_file):
        return "COMPLETED"

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# PBS
# ---------------------------------------------------------------------------

def get_pbs_status(stdout: str, returncode: int) -> str:
    """Parse ``qstat -f -x`` output into a normalised status string.

    Args:
        stdout:     The captured stdout of ``qstat -f -x <job_id>``.
        returncode: The exit code of that command.

    Returns:
        One of 'RUNNING', 'PENDING', 'COMPLETED', 'FAILED', 'UNKNOWN'.
    """
    if returncode != 0 or not stdout.strip():
        # Job not found — it has already left the PBS queue; treat as finished.
        return "COMPLETED"

    state_match = re.search(r"job_state\s*=\s*(\S+)", stdout)
    if not state_match:
        return "UNKNOWN"

    state = state_match.group(1)
    if state in ("F", "C"):          # Finished / Complete
        exit_match = re.search(r"exit_status\s*=\s*(\S+)", stdout)
        exit_status = int(exit_match.group(1)) if exit_match else 0
        return "COMPLETED" if exit_status == 0 else "FAILED"
    if state in ("R", "E"):          # Running / Exiting
        return "RUNNING"
    if state in ("Q", "H", "W", "T", "M", "S", "U"):   # Queued / Waiting
        return "PENDING"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_job_status(
    job_id: str,
    scheduler_type: str,
    result_file: str | None = None,
) -> str:
    """Return normalised job status for either SLURM or PBS.

    Args:
        job_id:         Scheduler job ID string.
        scheduler_type: ``'slurm'`` or ``'pbs'``.
        result_file:    Optional path to the simulation output file (used only
                        for SLURM to detect ``sacct`` lag).

    Returns:
        One of 'RUNNING', 'PENDING', 'COMPLETED', 'FAILED', 'UNKNOWN'.
    """
    if scheduler_type == "pbs":
        result = subprocess.run(
            f"qstat -f -x {job_id}", shell=True, capture_output=True, text=True
        )
        return get_pbs_status(result.stdout, result.returncode)
    return get_slurm_status(job_id, result_file=result_file)


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def is_job_limit_error(stderr: str) -> bool:
    """Return True if *stderr* indicates a per-user scheduler job limit."""
    slurm_patterns = ("QOSMaxSubmitJobPerUserLimit", "MaxSubmitJobsPerUser")
    pbs_patterns = (
        "Job exceeds queue", "PBS_MAXSELECTJOB", "would exceed", "violates queue"
    )
    return any(p in stderr for p in slurm_patterns + pbs_patterns)


def parse_job_id(stdout: str) -> str:
    """Extract the job ID from sbatch / qsub stdout (last whitespace-delimited token)."""
    return stdout.strip().split()[-1]


def cancel_job(job_id: str, scheduler_type: str = "slurm") -> None:
    """Cancel a scheduler job (best-effort; errors are printed but not re-raised)."""
    cmd = f"qdel {job_id}" if scheduler_type == "pbs" else f"scancel {job_id}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: job cancellation failed for {job_id}: {result.stderr.strip()}")


def cancel_all_user_jobs(scheduler_type: str = "slurm") -> None:
    """Cancel all scheduler jobs belonging to the current user (failsafe cleanup)."""
    if scheduler_type == "pbs":
        subprocess.run("qselect -u $(whoami) | xargs qdel", shell=True)
    else:
        subprocess.run("scancel -u $(whoami)", shell=True)
