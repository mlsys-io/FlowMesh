"""The node registry's response reader must survive a control-Redis restart.

If the connection drops, `_resubscribe` closes the dead pubsub and re-subscribes
to NODE_RESPONSE_CHANNEL with backoff so command-response futures keep resolving
(otherwise `node worker ...` round-trips would hang forever).
"""

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
import redis.exceptions
from redis.client import PubSub

import server.registries.node as node_module
from server.clients.redis import NODE_RESPONSE_CHANNEL
from server.registries.node import NodeRegistry

_LOGGER = logging.getLogger("test.node_registry")


class _FakePubSub:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.closed = False

    def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)

    def close(self) -> None:
        self.closed = True


class _FlakySync:
    """subscribe_control raises ConnectionError `fail_times` times, then returns
    a fresh _FakePubSub subscribed to the requested channel."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls: list[str] = []
        self.last: _FakePubSub | None = None

    def subscribe_control(self, channel: str) -> PubSub:
        self.calls.append(channel)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise redis.exceptions.ConnectionError("control redis down")
        pubsub = _FakePubSub()
        pubsub.subscribe(channel)
        self.last = pubsub
        return cast(PubSub, pubsub)


def _build_registry(sync: _FlakySync) -> NodeRegistry:
    reg = NodeRegistry.__new__(NodeRegistry)
    reg.logger = _LOGGER
    reg._rds = cast(Any, SimpleNamespace(sync=sync))
    reg._running = True
    return reg


def test_resubscribe_retries_until_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    sync = _FlakySync(fail_times=2)
    reg = _build_registry(sync)
    dead = _FakePubSub()

    result = reg._resubscribe(cast(PubSub, dead))

    assert result is not None
    assert dead.closed  # the dropped pubsub is closed before reconnecting
    # failed twice then succeeded -> three subscribe attempts on the channel
    assert sync.calls == [NODE_RESPONSE_CHANNEL] * 3
    assert sync.last is not None
    assert NODE_RESPONSE_CHANNEL in sync.last.subscribed


def test_resubscribe_gives_up_when_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    sync = _FlakySync(fail_times=1000)
    reg = _build_registry(sync)
    reg._running = False
    dead = _FakePubSub()

    assert reg._resubscribe(cast(PubSub, dead)) is None
    assert dead.closed
    assert sync.calls == []  # a stopped registry makes no reconnect attempts
