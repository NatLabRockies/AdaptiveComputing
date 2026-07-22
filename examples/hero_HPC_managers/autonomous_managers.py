# This file is a backward-compatibility shim.
# All functionality has moved to adaptive_computing.hpc.autonomous.
# Import from there in new code.
from adaptive_computing.hpc.autonomous import (  # noqa: F401
    setup_remote_state,
    run_remote_managers,
    wait_for_managers,
    cleanup_remote_managers,
)
