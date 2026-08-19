"""
OpenStackWorker — HeroWorker variant for persistent daemons on OpenStack VMs.

Adds two things on top of HeroWorker:

1. **Machine-name auto-detection** — resolved from (in priority order):
   a. ``WORKER_MACHINE_NAME`` environment variable.
   b. Nova metadata API (``http://169.254.169.254/openstack/…``).
   c. System hostname (``socket.gethostname()``).

2. **Graceful SIGTERM handling** — forwards SIGTERM to :meth:`~HeroWorker.stop`
   so systemd's ``TimeoutStopSec`` window is used cleanly rather than forcibly
   killing the process mid-task.

Usage
-----
Subclass :class:`OpenStackWorker` and implement :meth:`process_task`::

    class GatesWorker(OpenStackWorker):
        def process_task(self, task):
            msg   = task["metadata"]["Task"]["inputs"]["message"]
            state = task["metadata"]["Task"]["state_file_id"]
            graph = create_gates_graph(self._task_engine)
            result = process_query(graph, state, task["id"], self._data_repo, msg)
            return {"Task": {"response": result["response"][-1]}}

    if __name__ == "__main__":
        GatesWorker().run()

The worker's machine name is resolved automatically; pass ``machine_name`` to
the constructor to override all auto-detection.

Systemd setup
-------------
Use :func:`adaptive_computing.worker.systemd.generate_unit` to produce the
unit file, then::

    sudo cp gates-worker.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now gates-worker
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import urllib.request
from abc import abstractmethod

from .base import HeroWorker, TaskError  # noqa: F401 — re-exported for convenience

logger = logging.getLogger(__name__)

_NOVA_METADATA_URL = "http://169.254.169.254/openstack/latest/meta_data.json"
_NOVA_METADATA_TIMEOUT = 2  # seconds


class OpenStackWorker(HeroWorker):
    """HeroWorker for persistent daemon processes running on OpenStack VMs.

    Suitable for environments like Gila where workers run as long-lived
    systemd services on VM instances rather than as ephemeral Lambda functions.

    Args:
        machine_name: Override auto-detected machine name.  When omitted
                      :meth:`get_machine_name` is called at construction time.
    """

    def __init__(self, machine_name: str | None = None) -> None:
        super().__init__(machine_name or self.get_machine_name())

    # ------------------------------------------------------------------
    # Machine name resolution
    # ------------------------------------------------------------------

    @classmethod
    def get_machine_name(cls) -> str:
        """Return a stable identifier for this OpenStack VM.

        Resolution order:
        1. ``WORKER_MACHINE_NAME`` environment variable.
        2. Nova metadata API ``name`` field (``uuid`` as fallback).
        3. ``socket.gethostname()``.
        """
        env_name = os.environ.get("WORKER_MACHINE_NAME")
        if env_name:
            logger.debug("Machine name from env: %s", env_name)
            return env_name

        try:
            with urllib.request.urlopen(
                _NOVA_METADATA_URL, timeout=_NOVA_METADATA_TIMEOUT
            ) as resp:
                data = json.loads(resp.read())
            name = data.get("name") or data.get("uuid")
            if name:
                logger.debug("Machine name from Nova metadata: %s", name)
                return name
        except Exception:
            logger.debug("Nova metadata API unavailable — falling back to hostname.")

        hostname = socket.gethostname()
        logger.debug("Machine name from hostname: %s", hostname)
        return hostname

    # ------------------------------------------------------------------
    # SIGTERM → graceful stop
    # ------------------------------------------------------------------

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        logger.info(
            "SIGTERM handler installed — worker '%s' will stop cleanly on SIGTERM.",
            self.machine_name,
        )
        super().run()

    def _handle_sigterm(self, signum, frame) -> None:  # noqa: ARG002
        logger.info("SIGTERM received — stopping after current poll cycle.")
        self.stop()

    # ------------------------------------------------------------------
    # Abstract interface (re-declared for documentation clarity)
    # ------------------------------------------------------------------

    @abstractmethod
    def process_task(self, task: dict) -> dict:
        """See :meth:`HeroWorker.process_task`."""
