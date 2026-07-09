"""iter_pubsub_messages polls (get_message) instead of blocking (listen()).

Polling drives redis-py's periodic health-check PING that keeps an idle
SUBSCRIBE connection from being culled. Iteration ends on redis-py's own
ConnectionError (not the builtin), matching production.
"""

import json
from typing import Any, cast

import redis.exceptions
from redis.client import PubSub

from server.clients.redis import iter_pubsub_messages


class _FakePubSub:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.get_message_timeouts: list[float | None] = []
        self.listen_called = False

    def get_message(self, timeout: float | None = None) -> Any:
        self.get_message_timeouts.append(timeout)
        if not self._script:
            raise redis.exceptions.ConnectionError("drained")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def listen(self) -> Any:
        self.listen_called = True
        raise AssertionError("iter_pubsub_messages must not use blocking listen()")


def test_polls_with_timeout_and_yields_messages() -> None:
    ps = _FakePubSub(
        [
            None,  # idle tick, not terminal
            {"type": "subscribe", "data": 1},  # non-message, skipped
            {"type": "message", "data": json.dumps({"a": 1})},  # yielded
            {"type": "message", "data": b"not-json"},  # bad payload, skipped
            {"type": "message", "data": json.dumps({"b": 2})},  # yielded
            redis.exceptions.ConnectionError("culled"),  # clean stop
        ]
    )
    out = list(iter_pubsub_messages(cast(PubSub, ps), poll_timeout=0.01))

    assert out == [{"a": 1}, {"b": 2}]
    assert ps.listen_called is False
    # Reads polled get_message with the timeout (which drives check_health).
    assert ps.get_message_timeouts and all(t == 0.01 for t in ps.get_message_timeouts)


def test_idle_only_does_not_terminate_until_disconnect() -> None:
    # Idle ticks keep iterating; only a disconnect ends it.
    ps = _FakePubSub([None] * 5 + [redis.exceptions.ConnectionError("bye")])
    out = list(iter_pubsub_messages(cast(PubSub, ps), poll_timeout=0.01))
    assert out == []
    assert len(ps.get_message_timeouts) >= 5
