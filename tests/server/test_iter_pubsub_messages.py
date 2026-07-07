"""iter_pubsub_messages must poll (get_message) rather than block (listen()).

A blocking listen() never re-drives redis-py's health_check_interval on an idle
subscription, so the SUBSCRIBE connection sends no traffic and gets culled by an
idle proxy/LB timeout. Polling with get_message(timeout=...) runs check_health()
each iteration, emitting the periodic PING that keeps the connection warm. The
disconnect that ends iteration is redis-py's own ConnectionError, which does not
subclass the builtin, so the stop path is exercised with the production type.
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

    def listen(self) -> Any:  # pragma: no cover - must never be used
        self.listen_called = True
        raise AssertionError("iter_pubsub_messages must not use blocking listen()")


def test_polls_with_timeout_and_yields_messages() -> None:
    ps = _FakePubSub(
        [
            None,  # idle tick — must NOT end iteration
            {"type": "subscribe", "data": 1},  # non-message — skipped
            {"type": "message", "data": json.dumps({"a": 1})},  # yielded
            {"type": "message", "data": b"not-json"},  # bad payload — skipped
            {"type": "message", "data": json.dumps({"b": 2})},  # yielded
            redis.exceptions.ConnectionError("culled"),  # clean stop
        ]
    )
    out = list(iter_pubsub_messages(cast(PubSub, ps), poll_timeout=0.01))

    assert out == [{"a": 1}, {"b": 2}]
    assert ps.listen_called is False
    # Every read went through get_message with the poll timeout (drives check_health).
    assert ps.get_message_timeouts and all(t == 0.01 for t in ps.get_message_timeouts)


def test_idle_only_does_not_terminate_until_disconnect() -> None:
    # A long stretch of idle ticks keeps iterating (each tick emits the PING);
    # only a connection error ends it.
    ps = _FakePubSub([None] * 5 + [redis.exceptions.ConnectionError("bye")])
    out = list(iter_pubsub_messages(cast(PubSub, ps), poll_timeout=0.01))
    assert out == []
    assert len(ps.get_message_timeouts) >= 5
