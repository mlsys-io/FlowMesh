"""Tests for the task log archiver's retry and drain behavior."""

import errno
import logging
import os
import resource
import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from server.clients.redis import (
    task_log_archive_last_id_key,
    task_log_archived_key,
    task_log_stream_key,
)
from server.services import log_archiver
from server.services.log_archiver import TaskLogArchiver
from server.task.models import TaskStatus


class _Streams:
    """Redis log streams for the archiver: every published entry stays readable, and
    a read returns the entries past the id each requested stream names."""

    def __init__(self) -> None:
        self.redis = MagicMock()
        self.checkpoints: dict[str, str] = {}
        self.redis.get.side_effect = lambda key: self.checkpoints.get(key)
        self.redis.set_value.side_effect = self.checkpoints.__setitem__
        self.redis.delete.side_effect = lambda key: self.checkpoints.pop(key, None)
        self.redis.xread_telemetry.side_effect = self._read
        self.redis.xrange_telemetry.side_effect = self._range
        self._log: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._seq = 0

    def publish(self, task_id: str, message: str) -> None:
        self._seq += 1
        entry = (f"{self._seq}-0", {"payload": f'{{"m": "{message}"}}'})
        self._log.setdefault(task_log_stream_key(task_id), []).append(entry)

    def _after(self, key: str, last_id: str) -> list[tuple[str, dict[str, Any]]]:
        seq = int(last_id.split("-")[0])
        return [e for e in self._log.get(key, []) if int(e[0].split("-")[0]) > seq]

    def _read(self, streams: dict[str, str], **kwargs: Any) -> list[Any]:
        rows = [(key, self._after(key, last)) for key, last in streams.items()]
        return [(key, batch) for key, batch in rows if batch]

    def _range(self, key: str, min_id: str, count: int) -> list[Any]:
        return self._after(key, min_id.lstrip("("))[:count]


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def time(self) -> float:
        return self.now


def _streaming_archiver(
    tmp_path: Path,
    statuses: dict[str, str],
    flush_max_entries: int = 1,
    streams: _Streams | None = None,
) -> tuple[TaskLogArchiver, _Streams]:
    streams = streams or _Streams()
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


class _Failing:
    """``_logs_path`` failing with ENOSPC for one task while ``left`` is nonzero
    (negative: always), counting the attempts."""

    def __init__(self, archiver: TaskLogArchiver, task_id: str, left: int) -> None:
        self.task_id = task_id
        self.left = left
        self.attempts = 0
        self._logs_path = archiver._logs_path

    def __call__(self, tid: str) -> Path:
        if tid == self.task_id:
            self.attempts += 1
            if self.left != 0:
                self.left -= 1
                raise OSError(errno.ENOSPC, "No space left on device")
        return self._logs_path(tid)


def _ticks(archiver: TaskLogArchiver, clock: _Clock, count: int, step: float) -> None:
    with patch.object(log_archiver.time, "sleep"):
        for _ in range(count):
            archiver._tick()
            clock.now += step


def test_a_transient_write_error_keeps_every_line_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    failing = _Failing(archiver, "tsk-1", 2)
    monkeypatch.setattr(archiver, "_logs_path", failing)

    for line in ("one", "two", "three"):
        streams.publish("tsk-1", line)
        _ticks(archiver, clock, 1, 0.5)
    _ticks(archiver, clock, 20, 0.5)

    assert _lines(archiver, "tsk-1") == [
        '{"m": "one"}',
        '{"m": "two"}',
        '{"m": "three"}',
    ]


def test_a_persistent_write_error_backs_off_and_drops_only_after_its_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-a": TaskStatus.DISPATCHED, "tsk-b": TaskStatus.DISPATCHED}
    )
    failing = _Failing(archiver, "tsk-a", -1)
    monkeypatch.setattr(archiver, "_logs_path", failing)
    window = int(log_archiver._GIVE_UP_SEC)

    streams.publish("tsk-a", "lost")
    for tick in range(window * 10 - 10):
        streams.publish("tsk-b", f"b{tick}")
        _ticks(archiver, clock, 1, 0.1)
        assert len(_lines(archiver, "tsk-b")) == tick + 1
    assert archiver._buffers["tsk-a"]

    _ticks(archiver, clock, 200, 0.1)

    assert not archiver._buffers["tsk-a"]
    assert failing.attempts <= window / archiver._flush_interval_sec + 10


def test_a_failed_write_is_truncated_so_a_retry_writes_each_line_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DISPATCHED}, flush_max_entries=2
    )
    logs = archiver._base_dir("tsk-1") / "logs"
    logs.mkdir(parents=True)
    limit = 1 << 20
    earlier = '{"m": "' + "e" * (limit - 20) + '"}\n'
    (logs / "logs.jsonl").write_text(earlier)
    streams.publish("tsk-1", "alpha")
    streams.publish("tsk-1", "beta")
    # The file may grow only a few bytes: the write stops partway with EFBIG.
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
    try:
        _ticks(archiver, clock, 1, 1.0)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, handler)

    _ticks(archiver, clock, 10, 1.0)

    assert _lines(archiver, "tsk-1") == [
        earlier.rstrip("\n"),
        '{"m": "alpha"}',
        '{"m": "beta"}',
    ]


def test_a_restart_while_retrying_still_archives_the_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    write = os.write

    def _full(fd: int, data: Any) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(log_archiver.os, "write", _full)
    streams.publish("tsk-1", "kept")
    _ticks(archiver, clock, 1, 1.0)
    assert (archiver._base_dir("tsk-1") / "logs" / "logs.jsonl").exists()

    monkeypatch.setattr(log_archiver.os, "write", write)
    restarted, _ = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, streams=streams
    )
    _ticks(restarted, clock, 3, 1.0)

    assert _lines(restarted, "tsk-1") == ['{"m": "kept"}']
    assert streams.checkpoints[task_log_archived_key("tsk-1")]


def test_finalizing_records_the_task_archived_apart_from_its_stream_checkpoint(
    tmp_path: Path,
) -> None:
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    streams.publish("tsk-1", "one")
    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()
    assert streams.checkpoints[task_log_archive_last_id_key("tsk-1")] == "1-0"

    archiver._runtime.task_statuses.return_value = {  # type: ignore[attr-defined]
        "tsk-1": TaskStatus.DONE
    }
    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    # Older code reads the stream checkpoint as a stream id.
    assert task_log_archive_last_id_key("tsk-1") not in streams.checkpoints
    assert streams.checkpoints[task_log_archived_key("tsk-1")]
    restarted, _ = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, streams=streams
    )
    assert restarted._archived("tsk-1")


def test_a_finished_tasks_last_lines_are_archived(tmp_path: Path) -> None:
    # A buffer short of a full flush, read in the tick that finds the task finished.
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DONE}, flush_max_entries=100
    )
    streams.publish("tsk-1", "last")

    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert _lines(archiver, "tsk-1") == ['{"m": "last"}']


def test_a_tick_with_only_retrying_tasks_waits_for_the_earliest_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(log_archiver.time, "time", clock.time)
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(log_archiver.time, "sleep", _sleep)
    archiver, streams = _streaming_archiver(
        tmp_path, {"tsk-1": TaskStatus.DISPATCHED, "tsk-old": TaskStatus.DONE}
    )
    streams.checkpoints[task_log_archived_key("tsk-old")] = "1"
    read = streams.redis.xread_telemetry.side_effect

    def _blocking_read(requested: dict[str, str], **kwargs: Any) -> list[Any]:
        if not (rows := read(requested, **kwargs)):
            clock.now += kwargs["block_ms"] / 1000
        return rows

    streams.redis.xread_telemetry.side_effect = _blocking_read
    failing = _Failing(archiver, "tsk-1", -1)
    monkeypatch.setattr(archiver, "_logs_path", failing)
    streams.publish("tsk-1", "held")

    window = 60.0
    end = clock.now + window
    ticks = 0
    while clock.now < end and ticks < 10_000:
        archiver._tick()
        ticks += 1

    assert sleeps and all(0 < seconds <= 1.0 for seconds in sleeps)
    assert ticks <= 2 * window
    assert streams.redis.get.call_count <= 2 * ticks + 10
    assert failing.attempts <= window / archiver._flush_interval_sec + 5


def test_a_failed_write_never_cuts_another_writers_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, streams = _streaming_archiver(tmp_path, {"tsk-1": TaskStatus.DISPATCHED})
    logs = archiver._base_dir("tsk-1") / "logs"
    logs.mkdir(parents=True)
    path = logs / "logs.jsonl"
    path.write_text('{"m": "before"}\n')
    write = os.write
    calls = 0

    def _partial_then_full(fd: int, data: Any) -> int:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        with path.open("ab") as other:
            other.write(b'{"m": "theirs"}\n')
        return write(fd, bytes(data[:4]))

    monkeypatch.setattr(log_archiver.os, "write", _partial_then_full)
    streams.publish("tsk-1", "ours")
    with patch.object(log_archiver.time, "sleep"):
        archiver._tick()

    assert path.read_text().splitlines()[:2] == ['{"m": "before"}', '{"m": "theirs"}']
    assert not archiver._buffers["tsk-1"]
