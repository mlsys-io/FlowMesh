"""Stream-id comparison, trim detection, and batch consumption for the
task-event consumer."""

import logging
import threading
from typing import Any, cast

from server.services.monitoring import (
    TASK_EVENT_HANDLER_MAX_ATTEMPTS,
    EventMonitor,
    _stream_id_tuple,
)
from shared.schemas.event import Event, TaskEvent


def test_stream_id_tuple_orders_by_ms_then_seq() -> None:
    assert _stream_id_tuple("100-0") < _stream_id_tuple("100-1")
    assert _stream_id_tuple("100-5") < _stream_id_tuple("101-0")
    assert _stream_id_tuple("100") == (100, 0)
    assert _stream_id_tuple("bogus") == (0, 0)


class _FakeRedis:
    def __init__(self, first_id: str | None) -> None:
        self._first_id = first_id

    def xrange_telemetry(self, key: str, count: int | None = None) -> list[Any]:
        if self._first_id is None:
            return []
        return [(self._first_id, {})]


def _monitor(first_id: str | None) -> tuple[EventMonitor, list[str]]:
    monitor = EventMonitor.__new__(EventMonitor)
    monitor._redis_client = cast(Any, _FakeRedis(first_id))
    logger = logging.getLogger("trim-test")
    warnings: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    logger.addHandler(_Capture())
    logger.setLevel(logging.WARNING)
    monitor._logger = logger
    return monitor, warnings


def test_no_warning_when_cursor_is_sentinel() -> None:
    monitor, warnings = _monitor("500-0")
    monitor._warn_if_cursor_trimmed("$")
    assert warnings == []


def test_no_warning_when_cursor_keeps_up_with_head() -> None:
    # Stream head is at or behind the cursor: nothing past it was trimmed.
    monitor, warnings = _monitor("400-0")
    monitor._warn_if_cursor_trimmed("400-0")
    assert warnings == []


def test_warns_when_cursor_fell_behind_trim_horizon() -> None:
    # Stream head is newer than the cursor: entries in between were trimmed.
    monitor, warnings = _monitor("900-0")
    monitor._warn_if_cursor_trimmed("100-0")
    assert len(warnings) == 1
    assert "trimmed" in warnings[0]


class _RecordingRedis:
    def __init__(self) -> None:
        self.cursors: list[str] = []

    def set_value(self, key: str, value: str) -> None:
        self.cursors.append(value)


class _ConsumerMonitor(EventMonitor):
    """EventMonitor double with parse/handle overridden for batch-consume tests."""

    def __init__(
        self,
        redis: _RecordingRedis,
        fail_times: dict[str, int] | None = None,
        parse_raises_on: set[str] | None = None,
    ) -> None:
        self._redis_client = cast(Any, redis)
        self._stop_event = threading.Event()
        self._logger = logging.getLogger("consumer-test")
        self._event_handler_attempts: dict[str, int] = {}
        self.handled: list[str] = []
        self._fail_times = dict(fail_times or {})
        self._parse_raises_on = parse_raises_on or set()
        self._handle_calls: dict[str, int] = {}

    def _parse_stream_event(self, fields: dict[str, Any]) -> Event | None:
        task_id = fields["task_id"]
        if task_id in self._parse_raises_on:
            raise ValueError(f"malformed {task_id}")
        return TaskEvent(type="TASK_STARTED", task_id=task_id)

    def _handle_task_event(self, event: TaskEvent) -> None:
        seen = self._handle_calls.get(event.task_id, 0) + 1
        self._handle_calls[event.task_id] = seen
        if seen <= self._fail_times.get(event.task_id, 0):
            raise RuntimeError(f"boom {event.task_id}")
        self.handled.append(event.task_id)


def _consumer(
    fail_times: dict[str, int] | None = None,
    parse_raises_on: set[str] | None = None,
) -> tuple[_ConsumerMonitor, _RecordingRedis, list[str]]:
    redis = _RecordingRedis()
    monitor = _ConsumerMonitor(redis, fail_times, parse_raises_on)
    return monitor, redis, monitor.handled


def test_batch_advances_cursor_per_entry() -> None:
    monitor, redis, handled = _consumer()
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    cursor = monitor._consume_stream_batch(entries, "$")
    assert cursor == "3-0"
    assert redis.cursors == ["1-0", "2-0", "3-0"]
    assert handled == ["tsk-1", "tsk-2", "tsk-3"]


def test_batch_skips_malformed_entry_immediately() -> None:
    # A parse failure is deterministic poison: skipped without retry.
    monitor, redis, handled = _consumer(parse_raises_on={"tsk-2"})
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    cursor = monitor._consume_stream_batch(entries, "$")
    assert cursor == "3-0"
    assert redis.cursors == ["1-0", "2-0", "3-0"]
    assert handled == ["tsk-1", "tsk-3"]


def test_batch_retries_transient_handler_failure_without_loss() -> None:
    # tsk-2's handler fails once then succeeds: the entry must be retried, not lost.
    monitor, redis, handled = _consumer(fail_times={"tsk-2": 1})
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    # First pass advances past tsk-1, then stops at tsk-2 without advancing.
    cursor = monitor._consume_stream_batch(entries, "$")
    assert cursor == "1-0"
    assert handled == ["tsk-1"]
    # The stream re-delivers from after the cursor; the retry now succeeds.
    cursor = monitor._consume_stream_batch(entries[1:], cursor)
    assert cursor == "3-0"
    assert handled == ["tsk-1", "tsk-2", "tsk-3"]


def test_batch_dead_letters_poison_handler_after_max_attempts() -> None:
    # A handler that always raises is dropped once the attempt budget is spent,
    # so a persistent poison event cannot stall the stream forever.
    monitor, redis, handled = _consumer(fail_times={"tsk-2": 999})
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    cursor = monitor._consume_stream_batch(entries, "$")
    assert cursor == "1-0"  # stuck on tsk-2, not advanced
    remaining = entries[1:]
    for _ in range(TASK_EVENT_HANDLER_MAX_ATTEMPTS - 2):
        cursor = monitor._consume_stream_batch(remaining, cursor)
        assert cursor == "1-0"
    # Final attempt exhausts the budget: tsk-2 is dropped and tsk-3 proceeds.
    cursor = monitor._consume_stream_batch(remaining, cursor)
    assert cursor == "3-0"
    assert handled == ["tsk-1", "tsk-3"]


def test_batch_stops_mid_batch_when_stop_event_set() -> None:
    monitor, redis, handled = _consumer()
    monitor._stop_event.set()
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    cursor = monitor._consume_stream_batch(entries, "$")
    # First entry is processed, then the stop is observed.
    assert cursor == "1-0"
    assert redis.cursors == ["1-0"]
    assert handled == ["tsk-1"]
