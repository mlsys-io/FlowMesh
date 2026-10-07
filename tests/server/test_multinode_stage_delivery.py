"""A dependent's upstream result, end to end, with and without a shared results volume.

Drives the real dispatcher, results router and worker delivery code in process:
the producer's worker writes into its own results directory, the dispatcher reads
the server's, and a consumer worker hydrates into a third. On a multi-node site
the three differ; on a single host the producer's directory is the server's.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app_state import get_event_monitor, get_results_dir, get_runtime
from server.routers.v1.results import router
from server.task.models import TaskStatus
from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult
from shared.tasks.worker_message import WorkerTaskMessage
from tests.server.dispatcher.test_stage_result_delivery import (
    _dispatcher,
    _payload,
    register,
)
from tests.server.task.merge_harness import build_runtime
from tests.worker.factories import make_worker_hardware
from worker import result_delivery
from worker.runner import Runner


class _Server:
    """The results router over ``results_dir``, reachable at http://server."""

    def __init__(self, runtime: Any, results_dir: Path) -> None:
        app = FastAPI()
        app.state.logger = logging.getLogger("test-multinode")
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_results_dir] = lambda: results_dir
        app.dependency_overrides[get_runtime] = lambda: runtime
        app.dependency_overrides[get_event_monitor] = lambda: mock.MagicMock()
        self.client = TestClient(app, base_url="http://server")
        self.requests: list[tuple[str, str]] = []
        self.client.event_hooks["request"].append(
            lambda request: self.requests.append((request.method, request.url.path))
        )

    def posts(self) -> list[str]:
        return [path for method, path in self.requests if method == "POST"]


class _Streamed:
    """What ``requests.get(..., stream=True)`` returns, over a TestClient reply."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def __enter__(self) -> "_Streamed":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        yield self._response.content


@pytest.fixture
def server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server")
    monkeypatch.setenv("FLOWMESH_API_KEY", "test-key")
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "0")


def _route_workers_to(server: _Server, monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``with`` on the returned client must not exit the shared one.
    monkeypatch.setattr(
        result_delivery.httpx, "Client", lambda **_: nullcontext(server.client)
    )
    monkeypatch.setattr(
        result_delivery.requests,
        "get",
        lambda url, headers, stream, timeout: _Streamed(
            server.client.get(url, headers=headers)
        ),
    )


def _runner(results_dir: Path) -> Runner:
    return Runner(
        cast(Any, mock.Mock(worker_id="wrk-producer")),
        [],
        results_dir,
        make_worker_hardware(),
        {},
        cast(Any, mock.Mock()),
        logging.getLogger("test-multinode-runner"),
    )


def _produce(
    runtime: Any, disp: Any, registry: Any, prep: str, worker_dir: Path
) -> None:
    """Dispatch ``prep`` and run it on a worker writing into ``worker_dir``."""
    assert disp.dispatch_once(prep) is True
    message: WorkerTaskMessage = registry.publish_task.call_args.args[1]
    request = message.result_delivery[prep]
    out_dir = worker_dir / prep
    (out_dir / "artifacts" / "model").mkdir(parents=True)
    (out_dir / "artifacts" / "model" / "weights").write_bytes(b"trained")
    _runner(worker_dir)._write_single_result(
        prep,
        message.spec,
        out_dir,
        BaseExecutorResult.model_validate(
            {
                "model": {"path": "model"},
                "_artifacts": ArtifactContext(base_dir=out_dir.as_posix()),
            }
        ),
        request,
        message.result_dispatch,
    )
    record = runtime._tasks[prep]
    record.status = TaskStatus.DONE
    record.finished_ts = time.time()
    registry.reset_mock()


def test_multinode_dependent_reads_the_delivered_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_env: None
) -> None:
    runtime, _ = build_runtime("multinode-delivery")
    _, nodes = register(runtime, _payload("python"))
    server_dir, producer_dir, consumer_dir = (
        tmp_path / "server",
        tmp_path / "producer",
        tmp_path / "consumer",
    )
    server = _Server(runtime, server_dir)
    _route_workers_to(server, monkeypatch)
    disp, registry = _dispatcher(runtime, server_dir)

    _produce(runtime, disp, registry, nodes["prep"], producer_dir)

    # The producer's worker uploaded the envelope and its artifacts once.
    assert server.posts() == [f"/api/v1/results/{nodes['prep']}/delivery"]
    assert (server_dir / nodes["prep"] / "results.json").is_file()
    assert disp.dispatch_once(nodes["score"]) is True
    assert disp.failed == []
    message: WorkerTaskMessage = registry.publish_task.call_args.args[1]
    assert message.upstream_task_ids == {"prep": nodes["prep"]}

    result_delivery.hydrate_task(message, consumer_dir)

    assert (consumer_dir / nodes["prep"] / "results.json").is_file()
    weights = consumer_dir / nodes["prep"] / "artifacts" / "model" / "weights"
    assert weights.read_bytes() == b"trained"


def test_multinode_without_delivery_stalls_then_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_env: None
) -> None:
    # The server's results directory never receives results.json: the dependent
    # waits out the grace, then fails with an error naming the missing delivery.
    runtime, _ = build_runtime("multinode-lost")
    _, nodes = register(runtime, _payload("python"))
    server_dir = tmp_path / "server"
    monkeypatch.delenv("FLOWMESH_BASE_URL")
    disp, registry = _dispatcher(runtime, server_dir, grace_sec=120)

    _produce(runtime, disp, registry, nodes["prep"], tmp_path / "producer")

    assert not (server_dir / nodes["prep"] / "results.json").exists()
    assert disp.dispatch_once(nodes["score"]) is False
    assert [task_id for task_id, _ in disp.requeued] == [nodes["score"]]
    runtime._tasks[nodes["prep"]].finished_ts = time.time() - 300
    assert disp.dispatch_once(nodes["score"]) is True
    registry.publish_task.assert_not_called()
    [(task_id, error, _)] = disp.failed
    assert task_id == nodes["score"]
    assert f"Result of task {nodes['prep']} has not reached the server" in error


@pytest.mark.parametrize("server_reachable", [True, False])
def test_single_host_keeps_the_shared_result_without_uploading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_env: None,
    server_reachable: bool,
) -> None:
    runtime, _ = build_runtime("single-host-delivery")
    _, nodes = register(runtime, _payload("python"))
    shared = tmp_path / "results"
    server = _Server(runtime, shared)
    _route_workers_to(server, monkeypatch)
    if not server_reachable:
        # A server restart while the producer finishes.
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        def unreachable(**_: Any) -> httpx.Client:
            return httpx.Client(transport=httpx.MockTransport(refuse))

        monkeypatch.setattr(result_delivery.httpx, "Client", unreachable)
    disp, registry = _dispatcher(runtime, shared)

    _produce(runtime, disp, registry, nodes["prep"], shared)

    # Nothing is uploaded: the server already holds the snapshot on the shared
    # volume, or cannot be reached and the producer still succeeds.
    assert server.posts() == []
    assert disp.dispatch_once(nodes["score"]) is True
    assert disp.failed == []
    message: WorkerTaskMessage = registry.publish_task.call_args.args[1]
    hydrate = mock.patch.object(
        result_delivery.requests, "get", side_effect=AssertionError("downloaded")
    )
    with hydrate:
        result_delivery.hydrate_task(message, shared)
    weights = shared / nodes["prep"] / "artifacts" / "model" / "weights"
    assert weights.read_bytes() == b"trained"


def test_external_destination_still_delivers_to_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_env: None
) -> None:
    runtime, _ = build_runtime("multinode-webhook")
    webhook = (
        "output:\n          destination:\n            type: http\n"
        "            url: http://hooks.example/sink"
    )
    _, nodes = register(runtime, _payload("python", prep_output=webhook))
    server_dir = tmp_path / "server"
    server = _Server(runtime, server_dir)
    _route_workers_to(server, monkeypatch)
    sent = mock.Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr("worker.runner.requests.request", sent)
    disp, registry = _dispatcher(runtime, server_dir)

    _produce(runtime, disp, registry, nodes["prep"], tmp_path / "producer")

    assert [call.args[1] for call in sent.call_args_list] == [
        "http://hooks.example/sink"
    ]
    assert server.posts() == [f"/api/v1/results/{nodes['prep']}/delivery"]
    assert disp.dispatch_once(nodes["score"]) is True
    assert disp.failed == []
