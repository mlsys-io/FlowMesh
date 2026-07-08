"""Re-subscribe and re-home behavior on node re-registration.

When the root registry loses a node, ``Lifecycle`` re-registers it under a new
node id. These tests prove the node also (a) moves its dispatch/command
subscriptions to the new channel and (b) re-homes its already-registered
workers under the new id, so the dispatcher can reach them again.

The rebind is split in two: the heartbeat thread only records the target id
(``rebind``), and the pubsub reader thread applies the actual subscribe /
unsubscribe (``_apply_pending_rebind``). A redis-py ``PubSub`` must not be
mutated from a second thread while its owner is reading it, so these tests
assert ``rebind`` touches nothing and the reader-side apply does the switch.
"""

import logging
from collections.abc import Callable
from threading import Lock
from typing import Any, cast

import pytest
from redis.client import PubSub

import server.supervisor.services.command_listener as command_listener_module
import server.supervisor.services.task_listener as task_listener_module
from server.clients.redis import (
    SyncRedisClient,
    node_cmd_channel,
    node_dispatch_channel,
    worker_key,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.command_listener import CommandListener, _CommandStream
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.task_listener import TaskListener
from tests.server.supervisor_helpers import StubLifecycle, StubRegistry

_LOGGER = logging.getLogger("test.reregister")
_ANY_OBJECT: Any = None


class _FlakyRedis:
    """subscribe_control raises ConnectionError `fail_times` times, then returns
    a fresh _FakePubSub subscribed to the requested channel."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls: list[str] = []
        self.last_pubsub: _FakePubSub | None = None

    def subscribe_control(self, channel: str) -> PubSub:
        self.calls.append(channel)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("redis down")
        pubsub = _FakePubSub()
        pubsub.subscribe(channel)
        self.last_pubsub = pubsub
        return _as_pubsub(pubsub)


class _FakePubSub:
    """Records subscribe/unsubscribe calls made on a live pubsub."""

    def __init__(self) -> None:
        self.subscribed: set[str] = set()
        self.subscribe_log: list[str] = []
        self.unsubscribe_log: list[str] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def subscribe(self, channel: str) -> None:
        self.subscribed.add(channel)
        self.subscribe_log.append(channel)

    def unsubscribe(self, channel: str) -> None:
        self.subscribed.discard(channel)
        self.unsubscribe_log.append(channel)


def _as_pubsub(fake: _FakePubSub) -> PubSub:
    return cast(PubSub, fake)


class _RaisingSubscribePubSub(_FakePubSub):
    """Raises ConnectionError the first time it is asked to subscribe to
    ``raise_on`` (the rebind target), then behaves normally."""

    def __init__(self, raise_on: str) -> None:
        super().__init__()
        self.raise_on = raise_on

    def subscribe(self, channel: str) -> None:
        if channel == self.raise_on:
            self.raise_on = ""
            raise ConnectionError("dropped mid-switch")
        super().subscribe(channel)


class _ScriptedPubSub(_FakePubSub):
    """PubSub whose get_message runs a supplied callback, so a test can raise or
    stop the reader loop deterministically."""

    def __init__(self, on_get: "Callable[[_ScriptedPubSub], Any]") -> None:
        super().__init__()
        self._on_get = on_get

    def get_message(self, timeout: float) -> Any:
        return self._on_get(self)


# --------------------------------------------------------------------------- #
# TaskListener.rebind (heartbeat thread) + _apply_pending_rebind (reader thread)
# --------------------------------------------------------------------------- #


def _build_task_listener(node_id: str) -> tuple[TaskListener, _FakePubSub]:
    listener = TaskListener(_ANY_OBJECT, node_id, _LOGGER)
    pubsub = _FakePubSub()
    pubsub.subscribe(node_dispatch_channel(node_id))
    return listener, pubsub


def test_task_listener_rebind_records_target_without_touching_pubsub() -> None:
    # Regression guard for the cross-thread race: rebind runs on the heartbeat
    # thread and must NOT mutate the live pubsub.
    listener, pubsub = _build_task_listener("nde-1")

    listener.rebind("nde-2")

    assert listener._pending_node_id == "nde-2"
    assert pubsub.subscribe_log == [node_dispatch_channel("nde-1")]
    assert pubsub.unsubscribe_log == []
    assert not listener._rebind_applied.is_set()


def test_task_listener_apply_pending_moves_subscription() -> None:
    listener, pubsub = _build_task_listener("nde-1")
    listener.add_worker("wkr-1")
    listener.rebind("nde-2")

    live = listener._apply_pending_rebind(_as_pubsub(pubsub), "nde-1")

    assert live == "nde-2"
    assert listener._node_id == "nde-2"
    assert node_dispatch_channel("nde-2") in pubsub.subscribed
    assert node_dispatch_channel("nde-1") not in pubsub.subscribed
    assert listener._rebind_applied.is_set()
    # worker queues survive the rebind so in-flight streams keep working
    assert "wkr-1" in listener._qs


def test_task_listener_rebind_same_id_is_noop() -> None:
    listener, pubsub = _build_task_listener("nde-1")
    listener.rebind("nde-1")
    # same id resolves immediately, nothing pending, no pubsub mutation on apply
    assert listener._rebind_applied.is_set()
    assert listener._apply_pending_rebind(_as_pubsub(pubsub), "nde-1") == "nde-1"
    assert pubsub.unsubscribe_log == []


# --------------------------------------------------------------------------- #
# _CommandStream.rebind + _apply_pending_rebind
# --------------------------------------------------------------------------- #


def _build_command_stream(node_id: str) -> tuple[_CommandStream, _FakePubSub]:
    stream = _CommandStream(node_id, _ANY_OBJECT, None, _LOGGER)
    pubsub = _FakePubSub()
    pubsub.subscribe(node_cmd_channel(node_id))
    return stream, pubsub


def test_command_listener_rebind_records_target_without_touching_pubsub() -> None:
    stream, pubsub = _build_command_stream("nde-1")
    listener = CommandListener(_ANY_OBJECT, "nde-1", _ANY_OBJECT, _LOGGER)
    listener._cmd_stream = stream

    listener.rebind("nde-2")

    assert listener._node_id == "nde-2"
    assert stream._pending_node_id == "nde-2"
    assert pubsub.unsubscribe_log == []


def test_command_stream_apply_pending_moves_subscription() -> None:
    stream, pubsub = _build_command_stream("nde-1")
    stream.rebind("nde-2")

    live = stream._apply_pending_rebind(_as_pubsub(pubsub), "nde-1")

    assert live == "nde-2"
    assert stream.node_id == "nde-2"
    assert node_cmd_channel("nde-2") in pubsub.subscribed
    assert node_cmd_channel("nde-1") not in pubsub.subscribed
    assert stream._rebind_applied.is_set()


# --------------------------------------------------------------------------- #
# Reader self-healing: _resubscribe reconnects instead of letting the thread die
# --------------------------------------------------------------------------- #


def test_task_listener_resubscribe_retries_until_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_listener_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    redis = _FlakyRedis(fail_times=2)
    listener = TaskListener(cast(SyncRedisClient, redis), "nde-1", _LOGGER)
    stale = _FakePubSub()
    stale.subscribe(node_dispatch_channel("nde-1"))
    listener._pubsub = _as_pubsub(stale)
    listener._running = True

    result = listener._resubscribe("nde-1")

    assert result is not None
    # failed twice then succeeded -> three subscribe attempts on the live channel
    assert redis.calls == [node_dispatch_channel("nde-1")] * 3
    assert redis.last_pubsub is not None
    assert node_dispatch_channel("nde-1") in redis.last_pubsub.subscribed
    assert stale.closed  # the dead pubsub is closed before reconnecting


def test_task_listener_resubscribe_gives_up_when_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_listener_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    redis = _FlakyRedis(fail_times=1000)
    listener = TaskListener(cast(SyncRedisClient, redis), "nde-1", _LOGGER)
    listener._running = False
    # a stopped listener must not retry forever; it returns without reconnecting
    assert listener._resubscribe("nde-1") is None
    assert redis.calls == []


def test_command_stream_resubscribe_retries_until_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(command_listener_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    redis = _FlakyRedis(fail_times=1)
    stream = _CommandStream("nde-1", cast(SyncRedisClient, redis), None, _LOGGER)
    stream._pubsub_running = True

    result = stream._resubscribe("nde-1")

    assert result is not None
    assert redis.calls == [node_cmd_channel("nde-1")] * 2
    assert redis.last_pubsub is not None
    assert node_cmd_channel("nde-1") in redis.last_pubsub.subscribed


def test_command_stream_resubscribe_gives_up_when_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(command_listener_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    redis = _FlakyRedis(fail_times=1000)
    stream = _CommandStream("nde-1", cast(SyncRedisClient, redis), None, _LOGGER)
    stream._pubsub_running = False
    assert stream._resubscribe("nde-1") is None
    assert redis.calls == []


def test_task_listener_apply_rearms_pending_when_connection_drops() -> None:
    # Regression guard for the lost-rebind race: if the subscribe fails mid-switch
    # the target must be re-armed, not silently dropped.
    listener, _ = _build_task_listener("nde-1")
    pubsub = _RaisingSubscribePubSub(raise_on=node_dispatch_channel("nde-2"))
    pubsub.subscribe(node_dispatch_channel("nde-1"))
    listener.rebind("nde-2")

    with pytest.raises(ConnectionError):
        listener._apply_pending_rebind(_as_pubsub(pubsub), "nde-1")

    assert listener._pending_node_id == "nde-2"  # re-armed
    assert not listener._rebind_applied.is_set()  # not marked applied
    assert listener._node_id == "nde-1"  # unchanged until it lands

    # a retry on the recovered connection completes the move
    assert listener._apply_pending_rebind(_as_pubsub(pubsub), "nde-1") == "nde-2"
    assert node_dispatch_channel("nde-2") in pubsub.subscribed
    assert node_dispatch_channel("nde-1") not in pubsub.subscribed
    assert listener._rebind_applied.is_set()


def test_command_stream_apply_rearms_pending_when_connection_drops() -> None:
    stream, _ = _build_command_stream("nde-1")
    pubsub = _RaisingSubscribePubSub(raise_on=node_cmd_channel("nde-2"))
    pubsub.subscribe(node_cmd_channel("nde-1"))
    stream.rebind("nde-2")

    with pytest.raises(ConnectionError):
        stream._apply_pending_rebind(_as_pubsub(pubsub), "nde-1")

    assert stream._pending_node_id == "nde-2"
    assert not stream._rebind_applied.is_set()
    assert stream.node_id == "nde-1"

    assert stream._apply_pending_rebind(_as_pubsub(pubsub), "nde-1") == "nde-2"
    assert node_cmd_channel("nde-2") in pubsub.subscribed
    assert stream._rebind_applied.is_set()


def test_task_listener_run_reconnects_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Full loop: a dropped read triggers reconnect and the loop keeps running on
    # the fresh pubsub instead of the thread dying.
    monkeypatch.setattr(task_listener_module, "_RECONNECT_BACKOFF_SEC", 0.0)
    events: list[str] = []

    def _stop_after_reconnect(ps: _ScriptedPubSub) -> Any:
        events.append("read-after-reconnect")
        listener._running = False
        return None

    reconnected = _ScriptedPubSub(_stop_after_reconnect)

    class _Redis:
        def subscribe_control(self, channel: str) -> PubSub:
            reconnected.subscribe(channel)
            return _as_pubsub(reconnected)

    listener = TaskListener(cast(SyncRedisClient, _Redis()), "nde-1", _LOGGER)
    listener._loop = cast(Any, object())  # never used: no message is processed

    def _drop_once(ps: _ScriptedPubSub) -> Any:
        raise ConnectionError("connection dropped")

    original = _ScriptedPubSub(_drop_once)
    original.subscribe(node_dispatch_channel("nde-1"))
    listener._pubsub = _as_pubsub(original)
    listener._running = True

    listener._run()

    assert events == ["read-after-reconnect"]  # resumed reading after reconnect
    assert original.closed  # old pubsub torn down
    assert node_dispatch_channel("nde-1") in reconnected.subscribed


# --------------------------------------------------------------------------- #
# SupervisorServicer.rebind_node — worker re-homing
# --------------------------------------------------------------------------- #


class _FakeWorker:
    def __init__(self, token: str) -> None:
        self.token = token


class _FakeWorkerRegistry:
    def __init__(self, token_to_id: dict[str, str | None]) -> None:
        self._token_to_id = token_to_id

    def all_workers(self) -> list[_FakeWorker]:
        return [_FakeWorker(t) for t in self._token_to_id]

    def get_worker_id(self, token: str) -> str | None:
        return self._token_to_id.get(token)


class _FakeRedis:
    """Simulates the re-home Lua: HSET node_id only for keys that exist."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = existing if existing is not None else set()
        self.hash_writes: list[tuple[str, dict[str, Any]]] = []

    def eval(self, script: str, numkeys: int, *args: str) -> int:
        keys = args[:numkeys]
        node_id = args[numkeys]
        rehomed = 0
        for key in keys:
            if key in self.existing:
                self.hash_writes.append((key, {"node_id": node_id}))
                rehomed += 1
        return rehomed


def _build_servicer(
    node_id: str,
    token_to_id: dict[str, str | None],
    existing: set[str] | None = None,
) -> tuple[SupervisorServicer, _FakeRedis]:
    if existing is None:
        existing = {worker_key(w) for w in token_to_id.values() if w is not None}
    redis = _FakeRedis(existing)
    servicer = SupervisorServicer.__new__(SupervisorServicer)
    servicer._registry = cast(WorkerRegistry, _FakeWorkerRegistry(token_to_id))
    servicer._redis = cast(SyncRedisClient, redis)
    servicer._node_id = node_id
    servicer._node_alias = "worker-box"
    servicer._logger = _LOGGER
    servicer._lock = Lock()
    return servicer, redis


def test_rebind_node_rehomes_all_registered_workers() -> None:
    servicer, redis = _build_servicer(
        "nde-1", {"tok-a": "wkr-1", "tok-b": "wkr-2", "tok-c": None}
    )

    servicer.rebind_node("nde-2")

    assert servicer._node_id == "nde-2"
    # both registered workers rewritten to the new node; the unregistered one skipped
    assert (worker_key("wkr-1"), {"node_id": "nde-2"}) in redis.hash_writes
    assert (worker_key("wkr-2"), {"node_id": "nde-2"}) in redis.hash_writes
    assert len(redis.hash_writes) == 2


def test_rebind_node_skips_workers_without_a_record() -> None:
    # Full-wipe scenario: the worker key was evicted; re-home must NOT resurrect
    # a partial record.
    servicer, redis = _build_servicer(
        "nde-1", {"tok-a": "wkr-1", "tok-b": "wkr-2"}, existing={worker_key("wkr-1")}
    )

    servicer.rebind_node("nde-2")

    assert redis.hash_writes == [(worker_key("wkr-1"), {"node_id": "nde-2"})]


def test_rebind_node_stamps_future_registrations() -> None:
    servicer, _ = _build_servicer("nde-1", {})
    servicer.rebind_node("nde-9")
    # a subsequent RegisterWorker would stamp worker_meta["node_id"] with this
    assert servicer._node_id == "nde-9"


def test_rebind_node_same_id_is_noop() -> None:
    servicer, redis = _build_servicer("nde-1", {"tok-a": "wkr-1"})
    servicer.rebind_node("nde-1")
    assert redis.hash_writes == []


# --------------------------------------------------------------------------- #
# End-to-end: registry loss -> re-register-under-new-id -> dispatchable again
# --------------------------------------------------------------------------- #


def test_full_reregister_rebinds_and_rehomes() -> None:
    """After a simulated registry loss and re-register under a new id, the node
    is subscribed on the NEW dispatch/command channels and its workers are
    homed under the NEW id (so the host routes tasks to them). The reader-side
    apply is driven inline to stand in for the pubsub reader thread."""
    task_listener, task_pubsub = _build_task_listener("nde-1")
    task_listener.add_worker("wkr-1")

    cmd_stream, cmd_pubsub = _build_command_stream("nde-1")
    cmd_listener = CommandListener(_ANY_OBJECT, "nde-1", _ANY_OBJECT, _LOGGER)
    cmd_listener._cmd_stream = cmd_stream

    servicer, redis = _build_servicer("nde-1", {"tok-a": "wkr-1"})

    def _on_reregister(new_node_id: str) -> None:
        task_listener.rebind(new_node_id)
        cmd_listener.rebind(new_node_id)
        # stand in for the reader threads applying the pending rebind
        task_listener._apply_pending_rebind(_as_pubsub(task_pubsub), "nde-1")
        cmd_stream._apply_pending_rebind(_as_pubsub(cmd_pubsub), "nde-1")
        assert task_listener.wait_rebound(1.0)
        assert cmd_listener.wait_rebound(1.0)
        servicer.rebind_node(new_node_id)

    lifecycle = StubLifecycle(StubRegistry(exists=False), "nde-1")
    lifecycle.set_reregister_callback(_on_reregister)

    lifecycle._reregister_if_lost()

    # dispatch subscription moved to the new node's channel
    assert node_dispatch_channel("nde-2") in task_pubsub.subscribed
    assert node_dispatch_channel("nde-1") not in task_pubsub.subscribed
    # command subscription moved too
    assert node_cmd_channel("nde-2") in cmd_pubsub.subscribed
    assert node_cmd_channel("nde-1") not in cmd_pubsub.subscribed
    # the worker is now homed under the new node id -> dispatchable
    assert (worker_key("wkr-1"), {"node_id": "nde-2"}) in redis.hash_writes
    assert servicer._node_id == "nde-2"
