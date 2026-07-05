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
from threading import Event, Lock
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
# TaskListener.rebind (heartbeat thread) + _apply_pending_rebind (reader thread)
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
    listener._rebind_lock = Lock()
    listener._pending_node_id = None
    listener._rebind_applied = Event()
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

    live = listener._apply_pending_rebind(pubsub, "nde-1")  # type: ignore[arg-type]

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
    assert listener._apply_pending_rebind(pubsub, "nde-1") == "nde-1"  # type: ignore[arg-type]
    assert pubsub.unsubscribe_log == []


# --------------------------------------------------------------------------- #
# _CommandStream.rebind + _apply_pending_rebind
# --------------------------------------------------------------------------- #


def _build_command_stream(node_id: str) -> tuple[_CommandStream, _FakePubSub]:
    stream = _CommandStream.__new__(_CommandStream)
    stream.node_id = node_id
    stream.logger = _LOGGER
    stream._rebind_lock = Lock()
    stream._pending_node_id = None
    stream._rebind_applied = Event()
    pubsub = _FakePubSub()
    pubsub.subscribe(node_cmd_channel(node_id))
    return stream, pubsub


def test_command_listener_rebind_records_target_without_touching_pubsub() -> None:
    stream, pubsub = _build_command_stream("nde-1")
    listener = CommandListener.__new__(CommandListener)
    listener.logger = _LOGGER
    listener._node_id = "nde-1"
    listener._cmd_stream = stream

    listener.rebind("nde-2")

    assert listener._node_id == "nde-2"
    assert stream._pending_node_id == "nde-2"
    assert pubsub.unsubscribe_log == []


def test_command_stream_apply_pending_moves_subscription() -> None:
    stream, pubsub = _build_command_stream("nde-1")
    stream.rebind("nde-2")

    live = stream._apply_pending_rebind(pubsub, "nde-1")  # type: ignore[arg-type]

    assert live == "nde-2"
    assert stream.node_id == "nde-2"
    assert node_cmd_channel("nde-2") in pubsub.subscribed
    assert node_cmd_channel("nde-1") not in pubsub.subscribed
    assert stream._rebind_applied.is_set()


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


class _FakePipe:
    def __init__(self, redis: "_FakeRedis") -> None:
        self._redis = redis
        self._ops: list[tuple[str, str, dict[str, Any] | None]] = []

    def exists(self, key: str) -> None:
        self._ops.append(("exists", key, None))

    def hset(self, key: str, mapping: dict[str, Any] | None = None) -> None:
        self._ops.append(("hset", key, mapping))

    def execute(self) -> list[int]:
        results: list[int] = []
        for kind, key, mapping in self._ops:
            if kind == "exists":
                results.append(1 if key in self._redis.existing else 0)
            else:
                self._redis.hash_writes.append((key, mapping or {}))
                results.append(1)
        return results

    def __enter__(self) -> "_FakePipe":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeRedis:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = existing if existing is not None else set()
        self.hash_writes: list[tuple[str, dict[str, Any]]] = []

    def control_pipeline(self) -> _FakePipe:
        return _FakePipe(self)


def _build_servicer(
    node_id: str,
    token_to_id: dict[str, str | None],
    existing: set[str] | None = None,
) -> tuple[SupervisorServicer, _FakeRedis]:
    servicer = SupervisorServicer.__new__(SupervisorServicer)
    servicer._registry = _FakeWorkerRegistry(token_to_id)  # type: ignore[assignment]
    if existing is None:
        existing = {worker_key(w) for w in token_to_id.values() if w is not None}
    redis = _FakeRedis(existing)
    servicer._redis = redis  # type: ignore[assignment]
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
    homed under the NEW id (so the host routes tasks to them). The reader-side
    apply is driven inline to stand in for the pubsub reader thread."""
    task_listener, task_pubsub = _build_task_listener("nde-1")
    task_listener.add_worker("wkr-1")

    cmd_stream, cmd_pubsub = _build_command_stream("nde-1")
    cmd_listener = CommandListener.__new__(CommandListener)
    cmd_listener.logger = _LOGGER
    cmd_listener._node_id = "nde-1"
    cmd_listener._cmd_stream = cmd_stream

    servicer, redis = _build_servicer("nde-1", {"tok-a": "wkr-1"})

    def _on_reregister(new_node_id: str) -> None:
        task_listener.rebind(new_node_id)
        cmd_listener.rebind(new_node_id)
        # stand in for the reader threads applying the pending rebind
        task_listener._apply_pending_rebind(task_pubsub, "nde-1")  # type: ignore[arg-type]
        cmd_stream._apply_pending_rebind(cmd_pubsub, "nde-1")  # type: ignore[arg-type]
        assert task_listener.wait_rebound(1.0)
        assert cmd_listener.wait_rebound(1.0)
        servicer.rebind_node(new_node_id)

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
