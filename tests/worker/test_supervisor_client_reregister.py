"""Worker-side re-registration on supervisor restart (UNAUTHENTICATED)."""

import logging
import threading
from typing import Any
from unittest import mock

import grpc
from google.protobuf.json_format import MessageToDict

from shared.schemas.event import WorkerEvent
from shared.tasks.worker_message import WorkerStatus
from worker.supervisor_client import SupervisorClient


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


class _NoWaitEvent(threading.Event):
    """An Event whose wait() never blocks — returns the current flag at once."""

    def wait(self, timeout: float | None = None) -> bool:
        return self.is_set()


class _RegisterResponse:
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id


def _make_client() -> SupervisorClient:
    client = SupervisorClient(
        worker_token="fm-worker-0.deadbeef",
        owner_principal=None,
        grpc_target="localhost:50051",
        worker_namespace="ns",
        worker_cluster="cl",
        worker_alias="al",
        logger=logging.getLogger("test_reregister"),
    )
    client._worker_id = "wrk-old"
    client._register_meta = {"alias": "al", "status": "STARTING"}
    client._worker_register_event = WorkerEvent(
        type="REGISTER",
        worker_id="wrk-old",
        status=WorkerStatus.STARTING,
        ts="2026-01-01T00:00:00Z",
        tags=[],
        payload={"env": {}, "cost_per_hour": 1.0},
        actor=None,
    )
    client._shutdown.clear()
    client._stop.clear()
    return client


def test_reregister_swaps_worker_id_and_bumps_generation() -> None:
    client = _make_client()
    client._stub = mock.Mock()
    client._stub.RegisterWorker.return_value = _RegisterResponse("wrk-new")

    gen = client._reregister(seen_gen=0)

    assert gen == 1
    assert client._register_generation == 1
    assert client._worker_id == "wrk-new"
    client._stub.RegisterWorker.assert_called_once()


def test_reregister_is_performed_once_under_concurrency() -> None:
    client = _make_client()

    def slow_register(request: Any, metadata: Any) -> _RegisterResponse:
        threading.Event().wait(0.05)
        return _RegisterResponse("wrk-new")

    client._stub = mock.Mock()
    client._stub.RegisterWorker.side_effect = slow_register

    threads = [threading.Thread(target=client._reregister, args=(0,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert client._stub.RegisterWorker.call_count == 1
    assert client._register_generation == 1
    assert client._worker_id == "wrk-new"


def test_reregister_meta_carries_current_status() -> None:
    client = _make_client()
    client._last_status = WorkerStatus.BUSY
    captured: dict[str, Any] = {}

    def capture(request: Any, metadata: Any) -> _RegisterResponse:
        captured["meta"] = MessageToDict(request.meta, preserving_proto_field_name=True)
        return _RegisterResponse("wrk-new")

    client._stub = mock.Mock()
    client._stub.RegisterWorker.side_effect = capture

    client._reregister(seen_gen=0)

    assert captured["meta"]["status"] == WorkerStatus.BUSY.value


def test_retry_register_grpc_backs_off_and_never_exits() -> None:
    client = _make_client()
    client._shutdown = _NoWaitEvent()
    client._shutdown.clear()
    calls = {"n": 0}

    def fail_then_stop(request: Any, metadata: Any) -> _RegisterResponse:
        calls["n"] += 1
        if calls["n"] >= 2:
            client._stop.set()
        raise _FakeRpcError(grpc.StatusCode.UNAUTHENTICATED)

    client._stub = mock.Mock()
    client._stub.RegisterWorker.side_effect = fail_then_stop

    result = client._retry_register_grpc()

    assert result is None
    assert calls["n"] == 2  # retried after the first failure, no SystemExit


def test_rearm_register_event_stamps_new_id_and_generation() -> None:
    client = _make_client()
    client._worker_id = "wrk-new"
    client._register_generation = 3
    client._last_status = WorkerStatus.IDLE

    client._rearm_register_event()

    item = client._event_queue.get_nowait()
    assert isinstance(item, tuple)
    gen, payload = item
    assert gen == 3
    assert payload["type"] == "REGISTER"
    assert payload["worker_id"] == "wrk-new"


def test_send_event_stamps_current_identity() -> None:
    client = _make_client()
    client._stub = mock.Mock()
    client._worker_id = "wrk-new"
    client._register_generation = 5
    client._event_ready.set()

    client.set_status(WorkerStatus.IDLE)

    item = client._event_queue.get_nowait()
    assert isinstance(item, tuple)
    gen, payload = item
    assert gen == 5
    assert payload["worker_id"] == "wrk-new"
    assert client._last_status is WorkerStatus.IDLE


def test_event_messages_drops_superseded_generations() -> None:
    client = _make_client()
    client._register_generation = 1
    client._event_queue.put((0, {"type": "HEARTBEAT", "worker_id": "wrk-old"}))
    client._event_queue.put((1, {"type": "HEARTBEAT", "worker_id": "wrk-new"}))
    client._event_queue.put(client._EVENT_SENTINEL)

    messages = list(client._event_messages())

    assert len(messages) == 1
    payload = MessageToDict(messages[0].payload, preserving_proto_field_name=True)
    assert payload["worker_id"] == "wrk-new"


def _ready_future() -> mock.Mock:
    ready_future = mock.Mock()
    ready_future.result.return_value = None
    return ready_future


def test_task_stream_does_not_reregister_on_non_unauthenticated() -> None:
    client = _make_client()
    client._channel = mock.Mock()
    client._stub = mock.Mock()
    client._stub.StreamTasks.side_effect = _FakeRpcError(grpc.StatusCode.UNAVAILABLE)

    # Reach the code-discrimination branch (the loop's `_stop` check precedes
    # it), then stop the loop from the backoff so it exits after one pass.
    with (
        mock.patch("grpc.channel_ready_future", return_value=_ready_future()),
        mock.patch(
            "worker.supervisor_client.time.sleep",
            side_effect=lambda _s: client._stop.set(),
        ),
    ):
        client._run_task_stream()

    client._stub.RegisterWorker.assert_not_called()
    assert client._register_generation == 0


def test_task_stream_reregisters_and_reconnects_on_unauthenticated() -> None:
    client = _make_client()
    client._channel = mock.Mock()
    client._stub = mock.Mock()
    client._stub.RegisterWorker.return_value = _RegisterResponse("wrk-new")
    calls = {"n": 0}

    def stream_tasks(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeRpcError(grpc.StatusCode.UNAUTHENTICATED)
        client._stop.set()  # re-entered after re-registration; end the loop
        return iter(())

    client._stub.StreamTasks.side_effect = stream_tasks

    with mock.patch("grpc.channel_ready_future", return_value=_ready_future()):
        client._run_task_stream()

    client._stub.RegisterWorker.assert_called_once()
    assert client._register_generation == 1
    assert client._worker_id == "wrk-new"
    assert calls["n"] == 2  # stream re-entered after the fast-continue
