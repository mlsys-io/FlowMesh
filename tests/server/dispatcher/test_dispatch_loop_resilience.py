"""The dispatch loop must survive a control-Redis outage.

A dropped Redis connection surfaces as a redis-py error inside `dispatch_once`
and, worse, inside the `except`-branch requeue. `_safe_requeue` swallows the
Redis error so the dispatcher thread keeps running (the connection pool
reconnects on the next command and the watchdog re-surfaces the task).
"""

from typing import Any
from unittest import mock

import pytest
import redis.exceptions

from tests.server.dispatcher.helpers import make_capturing_dispatcher


def test_safe_requeue_reenqueues_in_memory_when_persist_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = mock.Mock()
    dispatcher = make_capturing_dispatcher(runtime=runtime)

    def _boom(task_id: str, **kwargs: Any) -> None:
        raise redis.exceptions.ConnectionError("control redis down")

    monkeypatch.setattr(dispatcher, "requeue_task", _boom)
    # Must not raise (a requeue that also hits the outage cannot kill the loop),
    # AND the task must still be re-enqueued in memory so it isn't orphaned:
    # nothing else re-surfaces a PENDING task while the server stays up.
    dispatcher._safe_requeue("tsk-1")
    runtime.requeue.assert_called_once_with("tsk-1", front=True)


def test_safe_requeue_propagates_non_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = make_capturing_dispatcher()

    def _boom(task_id: str, **kwargs: Any) -> None:
        raise ValueError("a real bug, not an outage")

    monkeypatch.setattr(dispatcher, "requeue_task", _boom)
    # A non-Redis error is a genuine defect and must not be silently swallowed.
    with pytest.raises(ValueError, match="real bug"):
        dispatcher._safe_requeue("tsk-1")
