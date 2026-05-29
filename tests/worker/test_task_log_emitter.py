"""Tests for TaskLogEmitter payload construction."""

import logging
import sys
from typing import Any, cast
from unittest import mock

from worker.utils.logging import TaskLogEmitter


class _CapturingStream:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def send(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)

    def close(self) -> None:
        pass


def _make_emitter() -> tuple[TaskLogEmitter, _CapturingStream]:
    with mock.patch("worker.utils.logging._GrpcLogStream"):
        emitter = TaskLogEmitter(
            stub=mock.Mock(),
            metadata=(),
            struct_from_payload=lambda payload: payload,
            logger=logging.getLogger("test_task_log_emitter"),
            task_id="tsk-1",
            workflow_id="wfl-1",
            owner_id="own-1",
            worker_id="wrk-1",
        )
    capture = _CapturingStream()
    emitter._stream = cast(Any, capture)
    return emitter, capture


def _record(msg: str, args: tuple[Any, ...], exc_info: Any = None) -> logging.LogRecord:
    return logging.LogRecord(
        name="task",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


def test_emit_includes_traceback_when_exc_info_present() -> None:
    emitter, capture = _make_emitter()
    try:
        raise ValueError("spec.api.url is required")
    except ValueError:
        record = _record("Task %s failed", ("tsk-1",), exc_info=sys.exc_info())

    emitter.emit(record)

    assert len(capture.payloads) == 1
    message = capture.payloads[0]["message"]
    assert "Task tsk-1 failed" in message
    assert "Traceback (most recent call last)" in message
    assert "ValueError: spec.api.url is required" in message


def test_emit_plain_message_without_exc_info() -> None:
    emitter, capture = _make_emitter()

    emitter.emit(_record("Task %s failed: %s", ("tsk-1", "bad spec")))

    assert len(capture.payloads) == 1
    assert capture.payloads[0]["message"] == "Task tsk-1 failed: bad spec"
