"""MP executor lifecycle tests."""

import tempfile
import uuid
from pathlib import Path

from shared.tasks import TaskType
from shared.tasks.specs import EchoSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import (
    make_live_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors.base_executor import Executor
from worker.executors.mp_executor import MPExecutor


class _SimpleMPExecutor(Executor):
    name = "simple_mp"

    def run(self, task, out_dir: Path) -> dict:
        return {"ok": True, "task_id": task.task_id}

    def cleanup_after_run(self) -> None:
        return None


def _simple_task_message() -> WorkerTaskMessage:
    return make_worker_task_message(
        EchoSpecStrict(taskType=TaskType.ECHO, data={"items": ["test"]}),
        api_version="mloc/v1",
        kind="EchoTask",
        task_id=str(uuid.uuid4()),
        workflow_id="test-workflow",
        owner_id="test-owner",
        assigned_worker="test-worker",
        dispatched_at="2026-03-01T00:00:00Z",
    )


def test_mp_executor_does_not_start_subprocess_until_first_run(tmp_path: Path) -> None:
    mp = MPExecutor(
        _SimpleMPExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    assert mp._shutdown is True
    assert mp._proc is None
    assert mp._cmd_q is None
    assert mp._res_q is None

    with tempfile.TemporaryDirectory() as out_dir:
        result = mp.run(_simple_task_message(), Path(out_dir))

    assert result["ok"] is True
    assert mp._shutdown is False
    assert mp._proc is not None
    assert mp._proc.is_alive()

    mp.cleanup_after_run()

    if mp._proc is not None:
        assert not mp._proc.is_alive()


def test_mp_executor_cleanup_before_run_is_noop(tmp_path: Path) -> None:
    mp = MPExecutor(
        _SimpleMPExecutor,
        config=make_live_worker_config(tmp_path),
        hardware=make_worker_hardware(),
    )

    mp.cleanup_after_run()

    assert mp._shutdown is True
    assert mp._proc is None
    assert mp._cmd_q is None
    assert mp._res_q is None
