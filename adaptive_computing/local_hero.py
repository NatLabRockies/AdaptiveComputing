"""
local_hero — file-backed task engine that duck-types the Hero client interface.

Intended for single-machine workflows (e.g. HPC_onsite) where there is no
need for a distributed queue.  The API exactly matches the real Hero client so
that HeroDataset, LocalHPCManager, and hero_initialize/hero_finalize all work
unchanged.

Usage
-----
Create one LocalHeroClient and pass it to every component that touches Hero:

    from adaptive_computing.local_hero import LocalHeroClient

    local_hero = LocalHeroClient("local_hero_db.json")

    manager   = create_manager(..., hero_client=local_hero)
    ac_driver = ActiveLoopDriverHero(..., hero_client=local_hero)

Both the manager and the dataset read and write the same JSON file, giving them
a consistent shared view of all tasks without any network calls.

The LocalHeroClient instance is picklable (it carries only the db_path string),
so it survives round-trips through pickle.  After loading a saved driver, set
the hero_client on the dataset again before calling hero_authenticate:

    ac_driver = pickle.load(f)
    local_hero = LocalHeroClient("local_hero_db.json")
    ac_driver.dataset.hero_authenticate(machine_names=[...], hero_client=local_hero)
"""

import contextlib
import json
import os
import sys
import uuid as _uuid_module

# ---------------------------------------------------------------------------
# Cross-platform file lock — ensures atomic read-modify-write across processes
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import msvcrt as _msvcrt

    @contextlib.contextmanager
    def _lock_file(path: str):
        lf = open(path + ".lock", "a")
        try:
            while True:
                try:
                    _msvcrt.locking(lf.fileno(), _msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    import time as _t
                    _t.sleep(0.005)
            try:
                yield
            finally:
                _msvcrt.locking(lf.fileno(), _msvcrt.LK_UNLCK, 1)
        finally:
            lf.close()
else:
    import fcntl as _fcntl

    @contextlib.contextmanager
    def _lock_file(path: str):
        """Acquire an exclusive advisory lock on ``path + ".lock"``."""
        lf = open(path + ".lock", "a")
        try:
            _fcntl.flock(lf.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(lf.fileno(), _fcntl.LOCK_UN)
        finally:
            lf.close()


def get_env_variable(name, default=None):
    """Mirror of hero.get_env_variable — reads from os.environ."""
    val = os.environ.get(name, default)
    if val is None:
        raise EnvironmentError(f"Environment variable '{name}' is not set.")
    return val


class LocalTaskEngine:
    """File-backed task engine implementing the Hero TaskEngine interface.

    All state is stored in a JSON file at db_path.  Multiple instances pointing
    at the same path share state — this is how the dataset and the manager see
    each other's updates without any network calls.
    """

    def __init__(self, db_path: str, application_id: str = "local") -> None:
        self.db_path = db_path
        self.application_id = application_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict:
        """Read the database without acquiring the file lock (caller holds it)."""
        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0:
            with open(self.db_path) as f:
                return json.load(f)
        return {"queues": {}, "tasks": {}}

    def _save_raw(self, db: dict) -> None:
        """Write the database without acquiring the file lock (caller holds it)."""
        with open(self.db_path, "w") as f:
            json.dump(db, f, indent=2)

    def _load(self) -> dict:
        with _lock_file(self.db_path):
            return self._load_raw()

    def _save(self, db: dict) -> None:
        with _lock_file(self.db_path):
            self._save_raw(db)

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------

    def add_queue(self, name: str) -> dict:
        db = self._load()
        queue_id = str(_uuid_module.uuid4())
        record = {"id": queue_id, "name": name, "state": "active"}
        db["queues"][queue_id] = record
        self._save(db)
        return record

    def read_queue_by_name(self, name: str, state: str) -> dict:
        db = self._load()
        for q in db["queues"].values():
            if q["name"] == name and q["state"] == state:
                return q
        raise ValueError(f"No {state!r} queue named {name!r}")

    # ------------------------------------------------------------------
    # Task operations
    # ------------------------------------------------------------------

    def add_task(self, queue_id: str, name: str, metatype: str,
                 metadata: dict, state: str = "ready") -> dict:
        db = self._load()
        task_id = str(_uuid_module.uuid4())
        task = {
            "id": task_id,
            "queue_id": queue_id,
            "name": name,
            "metatype": metatype,
            "metadata": metadata,
            "state": state,
        }
        db["tasks"][task_id] = task
        self._save(db)
        return task

    def read_tasks(self, queue_id: str, metatype: str, state: str) -> list:
        db = self._load()
        return [
            t for t in db["tasks"].values()
            if t["queue_id"] == queue_id
            and t.get("metatype") == metatype
            and t["state"] == state
        ]

    def read_task(self, task_id: str) -> dict:
        db = self._load()
        if task_id not in db["tasks"]:
            raise KeyError(f"Task {task_id!r} not found")
        return db["tasks"][task_id]

    def update_task(self, task_id: str, state: str, name: str,
                    metadata: dict) -> dict:
        with _lock_file(self.db_path):
            db = self._load_raw()
            if task_id not in db["tasks"]:
                raise KeyError(f"Task {task_id!r} not found")
            db["tasks"][task_id].update({"state": state, "name": name,
                                         "metadata": metadata})
            self._save_raw(db)
            return db["tasks"][task_id]

    def claim_task(self, task_id: str, claimer: str = "worker") -> bool:
        """Atomically transition a task from ``"ready"`` to ``"running"``.

        Safe to call concurrently from multiple processes.  Returns ``True``
        if this caller claimed the task, ``False`` if another process already
        claimed it (task is no longer ``"ready"``).

        Args:
            task_id: ID of the task to claim.
            claimer: Identifier for the claiming worker (stored in metadata).
        """
        with _lock_file(self.db_path):
            db = self._load_raw()
            task = db["tasks"].get(task_id)
            if task is None or task["state"] != "ready":
                return False
            task["state"] = "running"
            task["metadata"]["_claimed_by"] = claimer
            self._save_raw(db)
        return True

    def delete_task(self, task_id: str) -> None:
        db = self._load()
        db["tasks"].pop(task_id, None)
        self._save(db)


class LocalHeroClient:
    """Duck-typed replacement for HeroClient backed by a local JSON file.

    Attributes:
        db_path:        Absolute path to the JSON task database.  Created
                        automatically on first write.
        queue_name:     Base queue name used by HeroDataset and LocalHPCManager
                        in place of the HERO_QUEUE environment variable.
        application_id: Application identifier passed to TaskEngine (has no
                        functional effect locally, kept for interface parity).
    """

    def __init__(self, db_path: str = "local_hero_db.json",
                 queue_name: str = "local",
                 application_id: str = "local") -> None:
        self.db_path = os.path.abspath(db_path)
        self.queue_name = queue_name
        self.application_id = application_id

    def authenticate(self) -> None:
        """No-op — local storage requires no authentication."""

    def TaskEngine(self, application_id: str = "local") -> LocalTaskEngine:
        return LocalTaskEngine(self.db_path, application_id)
