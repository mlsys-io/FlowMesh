"""Stream-id comparison and trim-horizon detection for the task-event consumer."""

import logging
from typing import Any, cast

from server.services.monitoring import EventMonitor, _stream_id_tuple


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
