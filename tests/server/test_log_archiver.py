import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from server.clients.redis import task_log_stream_key
from server.services.log_archiver import TaskLogArchiver
from server.task.models import TaskStatus


def test_terminal_task_flushes_logs_below_periodic_threshold(tmp_path: Path) -> None:
    runtime = Mock()
    runtime.list_tasks.return_value = [
        SimpleNamespace(task_id="tsk-leaf", status=TaskStatus.DONE)
    ]
    runtime.get_record.return_value = None
    redis = Mock()
    redis.get.return_value = None
    payload = json.dumps({"message": "worker-only leaf completed"})
    redis.xread_telemetry.return_value = [
        (task_log_stream_key("tsk-leaf"), [("1-0", {"payload": payload})])
    ]
    redis.xrange_telemetry.return_value = []
    archiver = TaskLogArchiver(
        redis, runtime, tmp_path, logging.getLogger("test-log-archiver")
    )
    archiver._tick()
    archive = tmp_path / "tsk-leaf" / "logs" / "logs.jsonl"
    assert archive.read_text().splitlines() == [payload]
    redis.set_value.assert_called_once()
    assert "tsk-leaf" not in archiver._states
