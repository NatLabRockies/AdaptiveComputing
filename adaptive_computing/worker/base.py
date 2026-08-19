"""
HeroWorker — abstract base class for persistent HERO-queue worker daemons.

Unlike HeroHPCManager (which submits batch jobs to an external scheduler and
polls their status across many cycles), HeroWorker processes tasks
synchronously in-process.  It handles the full claim → heartbeat → process →
finalize lifecycle; subclasses only need to implement :meth:`process_task`.

Metadata convention
-------------------
While a task is in flight the base class writes a ``_hero_worker`` key into
``task["metadata"]``::

    task["metadata"]["_hero_worker"] = {
        "machine":    <machine_name>,
        "pid":        <int>,
        "started_at": <ISO-8601>,
        "heartbeat":  <ISO-8601>,   # updated every heartbeat_interval seconds
    }

This key is reserved; subclasses must not write to it.  Startup reconciliation
uses it to detect tasks left in the ``running`` state by a crashed worker and
resets them to ``ready``.

A Lambda watchdog (CloudWatch schedule) can mark tasks with a stale
``_hero_worker.heartbeat`` as ``error`` without racing with this worker,
because the heartbeat thread only writes when the task is still genuinely
in-flight.

Result metadata
---------------
:meth:`process_task` returns a dict that is shallow-merged into
``task["metadata"]`` before the task is marked ``done``.  For gates-style
tasks this is typically ``{"Task": {"response": "..."}}``; for AC-style tasks
it is ``{"y_data": [value]}``.

Example subclass
----------------
::

    class MyWorker(HeroWorker):
        def process_task(self, task):
            inputs = task["metadata"]["Task"]["inputs"]
            result = run_my_simulation(inputs)
            return {"Task": {"response": result}}

    if __name__ == "__main__":
        MyWorker("gila-vm-1").run()
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TaskError(RuntimeError):
    """Raise from :meth:`HeroWorker.process_task` to mark the task as ``error``."""


class ParkTask(Exception):
    """Raise from :meth:`HeroWorker.process_task` to leave the task in ``running``.

    Use when the task is waiting on sub-tasks that will complete asynchronously.
    The base class writes ``metadata_update`` into the task and leaves it in
    the ``running`` state; it does NOT call :meth:`_finalize`.  A future poll
    cycle will pick up the continuation task that the sub-worker posts.

    Args:
        waiting_on:      List of sub-task IDs the caller is blocked on.
        metadata_update: Dict shallow-merged into ``task["metadata"]`` before
                         the park update (use this to write ``parked=True``,
                         ``waiting_on``, etc. into the app-specific metadata).
    """

    def __init__(self, waiting_on: list, metadata_update: dict = None) -> None:
        self.waiting_on = waiting_on
        self.metadata_update = metadata_update or {}


class HeroWorker(ABC):
    """Abstract base class for HERO-queue-driven in-process worker daemons.

    Attributes:
        poll_interval:      Seconds between poll cycles (default 5).
        heartbeat_interval: Seconds between heartbeat writes while a task is
                            in flight (default 300, i.e. 5 minutes).
        simulation_dir:     If set, ``chdir`` into this path (relative to the
                            working directory at startup) before the poll loop.
    """

    poll_interval: float = 5.0
    heartbeat_interval: float = 300.0
    simulation_dir: str | None = None

    def __init__(self, machine_name: str) -> None:
        self.machine_name = machine_name
        self._task_engine = None
        self._data_repo = None
        self._queue_record = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process_task(self, task: dict) -> dict:
        """Process a claimed task and return metadata updates.

        Called with the task already in ``state="running"``.  Subclasses may
        access ``self._task_engine`` and ``self._data_repo`` directly.

        Args:
            task: Full Hero task dict with keys ``id``, ``name``, ``metatype``,
                  ``metadata``, ``state``.

        Returns:
            A dict shallow-merged into ``task["metadata"]`` before the task is
            marked ``done``.

        Raises:
            :class:`TaskError`: Marks the task ``error``.  The error message is
                                 stored in ``_hero_worker.error``.
        """

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Authenticate, reconcile stale tasks, then poll until stopped."""
        self._authenticate()
        self._find_or_create_queue()
        self._startup_reconciliation()

        if self.simulation_dir is not None:
            target = os.path.join(os.getcwd(), self.simulation_dir)
            if os.path.isdir(target):
                os.chdir(target)
            else:
                logger.warning(
                    "simulation_dir '%s' not found at %s — skipping chdir.",
                    self.simulation_dir, target,
                )

        logger.info(
            "Worker '%s' starting poll loop (interval=%.1fs, heartbeat=%.1fs).",
            self.machine_name, self.poll_interval, self.heartbeat_interval,
        )
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception:
                logger.exception("Unexpected error in poll cycle — continuing.")
            self._stop_event.wait(timeout=self.poll_interval)
        logger.info("Worker '%s' stopped.", self.machine_name)

    def stop(self) -> None:
        """Signal :meth:`run` to exit cleanly after the current poll cycle."""
        logger.info("Stop requested for worker '%s'.", self.machine_name)
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Poll cycle
    # ------------------------------------------------------------------

    def _poll_cycle(self) -> None:
        ready_tasks = self._task_engine.read_tasks(
            queue_id=self._queue_record["id"], metatype="Task", state="ready"
        )
        if not ready_tasks:
            return

        logger.debug("%d ready task(s) in queue.", len(ready_tasks))
        self._try_claim_and_process(ready_tasks[0])

    def _try_claim_and_process(self, task: dict) -> None:
        from adaptive_computing.hero_utils.hero_initialize import TaskAlreadyClaimed

        task_id = task["id"]
        try:
            self._claim_task(task)
        except TaskAlreadyClaimed:
            logger.debug("Task %s already claimed — skipping.", task_id)
            return
        except Exception:
            logger.exception("Failed to claim task %s.", task_id)
            return

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(task_id, heartbeat_stop),
            daemon=True,
            name=f"heartbeat-{task_id[:8]}",
        )
        heartbeat_thread.start()

        try:
            result_meta = self.process_task(task)
            self._finalize(task, result_meta, error=False)
        except ParkTask as park:
            self._park_task(task, park.waiting_on, park.metadata_update)
        except TaskError as exc:
            logger.error("Task %s raised TaskError: %s", task_id, exc)
            self._finalize(task, {}, error=True, error_message=str(exc))
        except Exception:
            logger.exception("Unhandled exception while processing task %s.", task_id)
            self._finalize(task, {}, error=True, error_message="Unhandled exception in process_task")
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Claim / finalize
    # ------------------------------------------------------------------

    def _claim_task(self, task: dict) -> None:
        """Atomically set the task to ``running`` and write worker bookkeeping.

        Re-reads the task immediately before the update to guard against the
        race condition where two workers both see ``state="ready"`` and attempt
        to claim the same task.  If the re-read shows ``state="running"`` the
        task was already claimed and :class:`TaskAlreadyClaimed` is raised.

        The caller's ``task`` dict is updated in-place so subsequent code sees
        the metadata written here.
        """
        from adaptive_computing.hero_utils.hero_initialize import TaskAlreadyClaimed

        fresh = self._task_engine.read_task(task["id"])
        if fresh["state"] == "running":
            raise TaskAlreadyClaimed(task["id"])

        fresh["metadata"]["_hero_worker"] = {
            "machine": self.machine_name,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
        }
        self._task_engine.update_task(
            task_id=task["id"],
            state="running",
            name=task["name"],
            metadata=fresh["metadata"],
        )
        task["metadata"] = fresh["metadata"]
        logger.info("Claimed task %s on worker '%s'.", task["id"], self.machine_name)

    def _finalize(
        self,
        task: dict,
        result_meta: dict,
        error: bool,
        error_message: str = "",
    ) -> None:
        """Merge result_meta into task metadata and set the terminal state."""
        # Re-read to pick up any heartbeat writes that happened concurrently.
        fresh = self._task_engine.read_task(task["id"])
        merged = {**fresh["metadata"], **result_meta}
        merged.setdefault("_hero_worker", {})["heartbeat"] = (
            datetime.now(timezone.utc).isoformat()
        )
        if error:
            merged["_hero_worker"]["error"] = error_message

        state = "error" if error else "done"
        self._task_engine.update_task(
            task_id=task["id"],
            state=state,
            name=task["name"],
            metadata=merged,
        )
        logger.info(
            "Task %s finalized as '%s' on worker '%s'.",
            task["id"], state, self.machine_name,
        )

    def _park_task(self, task: dict, waiting_on: list, metadata_update: dict) -> None:
        """Keep the task in ``running`` while it waits for sub-tasks."""
        fresh = self._task_engine.read_task(task["id"])
        merged = {**fresh["metadata"], **metadata_update}
        merged.setdefault("_hero_worker", {})["heartbeat"] = (
            datetime.now(timezone.utc).isoformat()
        )
        self._task_engine.update_task(
            task_id=task["id"],
            state="running",
            name=task["name"],
            metadata=merged,
        )
        logger.info(
            "Task %s parked on worker '%s', waiting on %d sub-task(s): %s",
            task["id"], self.machine_name, len(waiting_on), waiting_on,
        )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _heartbeat_loop(self, task_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(timeout=self.heartbeat_interval):
            self._write_heartbeat(task_id)

    def _write_heartbeat(self, task_id: str) -> None:
        try:
            task = self._task_engine.read_task(task_id)
            task["metadata"].setdefault("_hero_worker", {})["heartbeat"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._task_engine.update_task(
                task_id=task_id,
                state="running",
                name=task["name"],
                metadata=task["metadata"],
            )
            logger.debug("Heartbeat written for task %s.", task_id)
        except Exception:
            logger.warning(
                "Heartbeat write failed for task %s.", task_id, exc_info=True
            )

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def _startup_reconciliation(self) -> None:
        """Reset tasks left in 'running' by a crashed previous worker instance."""
        running = self._task_engine.read_tasks(
            queue_id=self._queue_record["id"], metatype="Task", state="running"
        )
        reset_count = 0
        for task in running:
            worker_meta = task.get("metadata", {}).get("_hero_worker", {})
            if worker_meta.get("machine") == self.machine_name:
                logger.info(
                    "Startup reconciliation: resetting stale task %s to ready.",
                    task["id"],
                )
                task["metadata"]["_hero_worker"]["machine"] = None
                self._task_engine.update_task(
                    task_id=task["id"],
                    state="ready",
                    name=task["name"],
                    metadata=task["metadata"],
                )
                reset_count += 1

        if reset_count:
            logger.info("Reset %d stale task(s) to ready.", reset_count)
        else:
            logger.info("Startup reconciliation: no stale tasks found.")

    # ------------------------------------------------------------------
    # Auth / queue
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        from hero import HeroClient, get_env_variable
        from adaptive_computing.hero_utils.set_hero_env_vars import set_hero_env_vars

        set_hero_env_vars()

        hero_env = get_env_variable("HERO_ENV", "dev")
        hero_project = get_env_variable("HERO_PROJECT")
        application_id = f"{hero_env}-{hero_project}"

        hero = HeroClient()
        try:
            hero.authenticate()
        except Exception as exc:
            raise RuntimeError(f"Hero authentication failed: {exc}") from exc

        self._task_engine = hero.TaskEngine(application_id)
        self._data_repo = hero.DataRepo(application_id)
        logger.info("Authenticated with Hero (application_id=%s).", application_id)

    def _find_or_create_queue(self) -> None:
        from hero import get_env_variable

        queue_name = get_env_variable("HERO_QUEUE")
        try:
            self._queue_record = self._task_engine.read_queue_by_name(
                name=queue_name, state="active"
            )
            logger.info("Found existing active queue: %s.", queue_name)
        except Exception:
            logger.info("No active queue found — creating: %s.", queue_name)
            self._queue_record = self._task_engine.add_queue(name=queue_name)
