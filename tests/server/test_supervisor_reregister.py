"""Re-subscribe and re-home behavior on node re-registration.

When the root registry loses a node, ``Lifecycle`` re-registers it under a new
node id. These tests prove the node also (a) moves its dispatch/command
subscriptions to the new channel and (b) re-homes its already-registered
workers under the new id, so the dispatcher can reach them again.
"""

import logging
from typing import Any

from server.clients.redis import (
    node_cmd_channel,
    node_dispatch_channel,
    worker_key,
)
from server.supervisor.services.command_listener import CommandListener, _CommandStream
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.lifecycle import Lifecycle
from server.supervisor.services.task_listener import TaskListener
from shared.schemas.node import NodeInfo

_LOGGER = logging.getLogger("test.reregister")


class _FakePubSub:
    """Records subscribe/unsubscribe calls made on a live pubsub."""

    def __init__(self) -> None:
        self.subscribed: set[str] = set()
        self.subscribe_log: list[str] = []
        self.unsubscribe_log: list[str] = []

    def subscribe(self, channel: str) -> None:
        self.subscribed.add(channel)
        self.subscribe_log.append(channel)

    def unsubscribe(self, channel: str) -> None:
        self.subscribed.discard(channel)
        self.unsubscribe_log.append(channel)


# --------------------------------------------------------------------------- #
# TaskListener.rebind
# --------------------------------------------------------------------------- #


def _build_task_listener(node_id: str) -> tuple[TaskListener, _FakePubSub]:
    listener = TaskListener.__new__(TaskListener)
    listener.logger = _LOGGER
    listener._node_id = node_id
    listener._qs = {}
    pubsub = _FakePubSub()
    pubsub.subscribe(node_dispatch_channel(node_id))
    listener._pubsub = pubsub  # type: ignore[assignment]
    listener._thread = object()  # type: ignore[assignment]
    listener._loop = None
    listener._running = True
    return listener, pubsub


def test_task_listener_rebind_moves_subscription() -> None:
    listener, pubsub = _build_task_listener("nde-1")
    listener.add_worker("wkr-1")

    listener.rebind("nde-2")

    assert listener._node_id == "nde-2"
    assert node_dispatch_channel("nde-2") in pubsub.subscribed
    assert node_dispatch_channel("nde-1") not in pubsub.subscribed
    # worker queues survive the rebind so in-flight streams keep working
    assert "wkr-1" in listener._qs


def test_task_listener_rebind_same_id_is_noop() -> None:
    listener, pubsub = _build_task_listener("nde-1")
    listener.rebind("nde-1")
    assert pubsub.subscribe_log == [node_dispatch_channel("nde-1")]
    assert pubsub.unsubscribe_log == []


def test_task_listener_rebind_before_start_only_updates_id() -> None:
    listener = TaskListener.__new__(TaskListener)
    listener.logger = _LOGGER
    listener._node_id = "nde-1"
    listener._qs = {}
    listener._pubsub = None
    listener.rebind("nde-2")
    assert listener._node_id == "nde-2"


# --------------------------------------------------------------------------- #
# CommandListener.rebind
# --------------------------------------------------------------------------- #


def test_command_listener_rebind_moves_subscription() -> None:
    stream = _CommandStream.__new__(_CommandStream)
    stream.node_id = "nde-1"
    stream.logger = _LOGGER
    pubsub = _FakePubSub()
    pubsub.subscribe(node_cmd_channel("nde-1"))
    stream._pubsub = pubsub  # type: ignore[assignment]

    listener = CommandListener.__new__(CommandListener)
    listener.logger = _LOGGER
    listener._node_id = "nde-1"
    listener._cmd_stream = stream

    listener.rebind("nde-2")

    assert listener._node_id == "nde-2"
    assert stream.node_id == "nde-2"
    assert node_cmd_channel("nde-2") in pubsub.subscribed
    assert node_cmd_channel("nde-1") not in pubsub.subscribed


def test_command_listener_rebind_before_start_only_updates_id() -> None:
    listener = CommandListener.__new__(CommandListener)
    listener.logger = _LOGGER
    listener._node_id = "nde-1"
    listener._cmd_stream = None
    listener.rebind("nde-2")
    assert listener._node_id == "nde-2"


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
    def __init__(self) -> None:
        self.hash_writes: list[tuple[str, dict[str, Any]]] = []

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        self.hash_writes.append((key, mapping))


def _build_servicer(
    node_id: str, token_to_id: dict[str, str | None]
) -> tuple[SupervisorServicer, _FakeRedis]:
    servicer = SupervisorServicer.__new__(SupervisorServicer)
    servicer._registry = _FakeWorkerRegistry(token_to_id)  # type: ignore[assignment]
    redis = _FakeRedis()
    servicer._redis = redis  # type: ignore[assignment]
    servicer._node_id = node_id
    servicer._node_alias = "worker-box"
    servicer._logger = _LOGGER
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
# Lifecycle wiring — callback fires on re-register, restoring dispatch
# --------------------------------------------------------------------------- #


def _build_lifecycle() -> Lifecycle:
    instance = Lifecycle.__new__(Lifecycle)
    instance._base_url = "http://root:8000"
    instance._node_info = NodeInfo(
        namespace="ns",
        cluster="cl",
        alias="worker-1",
        version="0.1.0",
        started_at="2026-05-13T00:00:00Z",
        tags=[],
        last_seen="2026-05-13T00:00:00Z",
        max_gpu_count=0,
    )
    instance.logger = _LOGGER
    instance._on_reregister = None
    return instance


class _StubRegistry:
    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def node_exists(self, node_id: str) -> bool:
        return self._exists


def test_reregister_invokes_callback_with_new_id() -> None:
    seen: list[str] = []
    instance = _build_lifecycle()
    instance._node_registry = _StubRegistry(exists=False)  # type: ignore[assignment]
    instance._node_id = "nde-1"
    instance._unregister_published = True
    instance._register = lambda: "nde-2"  # type: ignore[method-assign]
    instance._publish_event = lambda *a, **k: None  # type: ignore[method-assign]
    instance.set_reregister_callback(seen.append)

    instance._reregister_if_lost()

    assert instance._node_id == "nde-2"
    assert seen == ["nde-2"]


def test_reregister_present_does_not_invoke_callback() -> None:
    seen: list[str] = []
    instance = _build_lifecycle()
    instance._node_registry = _StubRegistry(exists=True)  # type: ignore[assignment]
    instance._node_id = "nde-1"
    instance.set_reregister_callback(seen.append)

    instance._reregister_if_lost()

    assert seen == []


def test_reregister_callback_error_does_not_break_heartbeat() -> None:
    instance = _build_lifecycle()
    instance._node_registry = _StubRegistry(exists=False)  # type: ignore[assignment]
    instance._node_id = "nde-1"
    instance._unregister_published = True
    instance._register = lambda: "nde-2"  # type: ignore[method-assign]
    instance._publish_event = lambda *a, **k: None  # type: ignore[method-assign]

    def _boom(_new_id: str) -> None:
        raise RuntimeError("rebind failed")

    instance.set_reregister_callback(_boom)

    # must not raise — a failed rebind should be logged, not crash the hb loop
    instance._reregister_if_lost()
    assert instance._node_id == "nde-2"


# --------------------------------------------------------------------------- #
# End-to-end: registry loss -> re-register-under-new-id -> dispatchable again
# --------------------------------------------------------------------------- #


def test_full_reregister_rebinds_and_rehomes() -> None:
    """After a simulated registry loss and re-register under a new id, the node
    is subscribed on the NEW dispatch/command channels and its workers are
    homed under the NEW id (so the host routes tasks to them)."""
    task_listener, task_pubsub = _build_task_listener("nde-1")
    task_listener.add_worker("wkr-1")

    cmd_stream = _CommandStream.__new__(_CommandStream)
    cmd_stream.node_id = "nde-1"
    cmd_stream.logger = _LOGGER
    cmd_pubsub = _FakePubSub()
    cmd_pubsub.subscribe(node_cmd_channel("nde-1"))
    cmd_stream._pubsub = cmd_pubsub  # type: ignore[assignment]
    cmd_listener = CommandListener.__new__(CommandListener)
    cmd_listener.logger = _LOGGER
    cmd_listener._node_id = "nde-1"
    cmd_listener._cmd_stream = cmd_stream

    servicer, redis = _build_servicer("nde-1", {"tok-a": "wkr-1"})

    def _on_reregister(new_node_id: str) -> None:
        servicer.rebind_node(new_node_id)
        task_listener.rebind(new_node_id)
        cmd_listener.rebind(new_node_id)

    lifecycle = _build_lifecycle()
    lifecycle._node_registry = _StubRegistry(exists=False)  # type: ignore[assignment]
    lifecycle._node_id = "nde-1"
    lifecycle._unregister_published = True
    lifecycle._register = lambda: "nde-2"  # type: ignore[method-assign]
    lifecycle._publish_event = lambda *a, **k: None  # type: ignore[method-assign]
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
