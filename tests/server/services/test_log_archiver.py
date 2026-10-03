"""Tests for the task log archiver's retry and drain behavior."""

import errno
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from server.clients.redis import task_log_stream_key
from server.services import log_archiver
from server.services.log_archiver import TaskLogArchiver
from server.task.models import TaskStatus


class _Streams:
    """Redis log streams for the archiver: each ``publish`` lands in the next tick's
    read."""

    def __init__(self) -> None:
        self.redis = MagicMock()
        self.redis.get.return_value = None
        self.redis.xrange_telemetry.return_value = []
        self._pending: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._seq = 0
        self.redis.xread_telemetry.side_effect = self._read

    def publish(self, task_id: str, message: str) -> None:
        self._seq += 1
        entry = (f"{self._seq}-0", {"payload": f'{{"m": "{message}"}}'})
        self._pending.setdefault(task_id, []).append(entry)

    def _read(self, *args: Any, **kwargs: Any) -> list[Any]:
        rows = [(task_log_stream_key(t), batch) for t, batch in self._pending.items()]
        self._pending = {}
        return rows


def _streaming_archiver(
    tmp_path: Path, statuses: dict[str, str], flush_max_entries: int = 1
) -> tuple[TaskLogArchiver, _Streams]:
    streams = _Streams()
    runtime = MagicMock()
    runtime.get_record.return_value = None
    runtime.task_statuses.return_value = statuses
    archiver = TaskLogArchiver(
        streams.redis,
        runtime,
        tmp_path,
        logging.getLogger("test"),
        flush_max_entries=flush_max_entries,
    )
    return archiver, streams


def _lines(archiver: TaskLogArchiver, task_id: str) -> list[str]:
    path = archiver._logs_path(task_id)
    return path.read_text().splitlines() if path.exists() else []


def _failing_logs_path(archiver: TaskLogArchiver, task_id: str, failures: list[int]):
    real = archiver._logs_path

    def _logs_path(tid: str) -> Path:
        path = real(tid)
        if tid == task_id and failures[0] != 0:
            failures[0] -= 1
            raise OSError(errno.ENOSPC, "No space left on device")
        return path

    return _logs_path


def test_a_transient_write_error_keeps_every_line_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    monkeypatch.setattr(
        archiver,
        "_logs_path",
        _failing_logs_path(archiver, "tsk-1", [2]),
    )

    with patch.object(log_archiver.time, "sleep"):
        for line in ("one", "two", "three"):
            streams.publish("tsk-1", line)
            archiver._tick()

    assert _lines(archiver, "tsk-1") == [
        '{"m": "one"}',
        '{"m": "two"}',
        '{"m": "three"}',
    ]
    streams.redis.set_value.assert_called_once()


def test_a_persistent_write_error_drops_after_its_bound_and_never_stalls_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-a": TaskStatus.DISPATCHED, "tsk-b": TaskStatus.DISPATCHED}
    )
    monkeypatch.setattr(
        archiver,
        "_logs_path",
        _failing_logs_path(archiver, "tsk-a", [-1]),
    )
    ticks = log_archiver._MAX_FLUSH_FAILURES + 2

    with patch.object(log_archiver.time, "sleep"):
        for tick in range(ticks):
            streams.publish("tsk-a", f"a{tick}")
            streams.publish("tsk-b", f"b{tick}")
            archiver._tick()
            assert len(_lines(archiver, "tsk-b")) == tick + 1

    assert len(archiver._buffers["tsk-a"]) < log_archiver._MAX_FLUSH_FAILURES


def test_a_finished_tasks_last_lines_are_archived(tmp_path: Path) -> None:
    # A buffer short of a full flush, read in the tick that finds the task finished.
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, flush_max_entries=100
    )
    streams.publish("tsk-1", "last")

    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert _lines(archiver, "tsk-1") == ['{"m": "last"}']
