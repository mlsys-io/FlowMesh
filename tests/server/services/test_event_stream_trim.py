"""Stream-id comparison, trim detection, and batch consumption for the
task-event consumer."""

import logging
import threading
from typing import Any, cast

from server.services.monitoring import EventMonitor, _stream_id_tuple
from shared.schemas.event import TaskEvent


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


def _consumer(
    handle_raises_on: set[str] | None = None,
) -> tuple[EventMonitor, _RecordingRedis, list[str]]:
    monitor = EventMonitor.__new__(EventMonitor)
    redis = _RecordingRedis()
    monitor._redis_client = cast(Any, redis)
    monitor._stop_event = threading.Event()
    monitor._logger = logging.getLogger("consumer-test")
    handled: list[str] = []
    raises_on = handle_raises_on or set()

    def parse(fields: dict[str, Any]) -> TaskEvent:
        return TaskEvent(type="TASK_STARTED", task_id=fields["task_id"])

    def handle(event: TaskEvent) -> None:
        if event.task_id in raises_on:
            raise RuntimeError(f"boom {event.task_id}")
        handled.append(event.task_id)

    monitor._parse_stream_event = parse  # type: ignore[method-assign]
    monitor._handle_task_event = handle  # type: ignore[method-assign]
    return monitor, redis, handled


def test_batch_advances_cursor_per_entry() -> None:
    monitor, redis, handled = _consumer()
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    cursor = monitor._consume_stream_batch(entries, "$")
    assert cursor == "3-0"
    assert redis.cursors == ["1-0", "2-0", "3-0"]
    assert handled == ["tsk-1", "tsk-2", "tsk-3"]


def test_batch_survives_handler_exception_and_skips_poison_entry() -> None:
    monitor, redis, handled = _consumer(handle_raises_on={"tsk-2"})
    entries = [(f"{i}-0", {"task_id": f"tsk-{i}"}) for i in range(1, 4)]
    # A raising handler must not propagate out of the consumer.
    cursor = monitor._consume_stream_batch(entries, "$")
    # Cursor advances past the poison entry so the stream keeps flowing.
    assert cursor == "3-0"
    assert redis.cursors == ["1-0", "2-0", "3-0"]
    # The poison entry is skipped; the others are still handled.
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
