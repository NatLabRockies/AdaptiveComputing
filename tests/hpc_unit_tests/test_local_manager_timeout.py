"""LocalHPCManager.run_until_done timeout: raise TimeoutError instead of polling
forever when tasks never finish (e.g. a batch script that writes its result file
somewhere the manager does not look).  The scheduler is never touched: the
reconcile and poll steps are replaced with no-ops, so a task left in the
"running" state never completes.
"""
import math
import os
import time

import pytest

from adaptive_computing.hpc.local_manager import LocalHPCManager
from adaptive_computing.local_hero import LocalHeroClient


class _StuckManager(LocalHPCManager):
    def submit_job(self, task, machine_name, i_fidelity):
        raise AssertionError("submit_job should not be called")

    def read_result(self, task_id):
        raise AssertionError("read_result should not be called")

    def _reconcile(self, *args, **kwargs):
        pass

    def _poll_cycle(self, *args, **kwargs):
        pass


def _make(tmp_path, n_running=1, **kwargs):
    hero = LocalHeroClient(db_path=str(tmp_path / "db.json"))
    engine = hero.TaskEngine()
    queue = engine.add_queue(name=hero.queue_name)
    ids = [engine.add_task(queue_id=queue["id"], name=f"t{i}", metatype="Task",
                           metadata={}, state="running")["id"] for i in range(n_running)]
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    manager = _StuckManager(machine_name="local", batch_scripts=["unused.sh"],
                            simulation_dir=str(sim_dir), poll_interval=0,
                            hero_client=hero, **kwargs)
    return manager, engine, ids


def test_timeout_raises_and_leaves_state(tmp_path):
    manager, engine, ids = _make(tmp_path)
    cwd = os.getcwd()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as exc:
        manager.run_until_done(timeout=0.2)
    assert time.monotonic() - t0 < 5
    assert ids[0] in str(exc.value)
    assert "1 running" in str(exc.value)
    assert os.getcwd() == cwd, "working directory must be restored on timeout"
    assert engine.read_task(ids[0])["state"] == "running", "task state must be left for a later resume"


def test_instance_default_timeout(tmp_path):
    manager, _, _ = _make(tmp_path, run_timeout=0.2)
    with pytest.raises(TimeoutError):
        manager.run_until_done()


def test_explicit_timeout_overrides_default(tmp_path):
    manager, _, _ = _make(tmp_path, run_timeout=math.inf)
    with pytest.raises(TimeoutError):
        manager.run_until_done(timeout=0.2)


def test_empty_queue_returns_without_timeout_error(tmp_path):
    manager, _, _ = _make(tmp_path, n_running=0)
    manager.run_until_done(timeout=0)
