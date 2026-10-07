import logging
import tarfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest

from shared.schemas.result import BaseExecutorResult
from shared.schemas.result_delivery import ArtifactInput, ResultDeliveryRequest
from shared.tasks import TaskType
from shared.tasks.components.output import OutputSpec
from shared.tasks.specs import EchoSpecStrict, PythonSpecStrict, SFTSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.result_delivery import make_receipt, read_receipt, write_receipt
from tests.shared.test_result_delivery import populate
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker import result_delivery
from worker.executors.mixins.training import TrainingMixin
from worker.executors.utils.checkpoints import resolve_checkpoint_load
from worker.runner import Runner


@pytest.mark.parametrize("upload_all", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_system_publication_is_independent_and_streams_selected_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_all: bool, declared: bool
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", str(int(upload_all)))
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server")
    monkeypatch.setenv("FLOWMESH_API_KEY", "test-key")
    output = (
        {"destination": {"type": "http", "url": "http://external/sink"}}
        if declared
        else None
    )
    spec = EchoSpecStrict(
        taskType=TaskType.ECHO,
        data={"type": "list", "items": ["x"]},
        output=OutputSpec.model_validate(output) if output else None,
    )
    sent: list[bytes] = []

    def accept(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # The server does not hold the snapshot yet.
            return httpx.Response(404)
        assert request.url == "http://server/api/v1/results/tsk-up/delivery"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert isinstance(request.stream, httpx.SyncByteStream)
        chunks = list(request.stream)
        assert max(map(len, chunks)) <= 64 * 1024
        sent.append(b"".join(chunks))
        return httpx.Response(200)

    class StreamingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return accept(request)

    client_class = httpx.Client
    monkeypatch.setattr(
        result_delivery.httpx,
        "Client",
        lambda **kwargs: client_class(transport=StreamingTransport(), **kwargs),
    )
    external = Mock(return_value=Mock(status_code=200))
    monkeypatch.setattr("worker.runner.requests.request", external)
    runner = Runner(
        cast(Any, Mock(worker_id="wrk-test")),
        [],
        tmp_path,
        make_worker_hardware(),
        {},
        cast(Any, Mock()),
        logging.getLogger("test-system-upload"),
    )
    base = populate(tmp_path, b"weights" * 30000)
    runner._write_single_result(
        "tsk-up",
        spec,
        base,
        BaseExecutorResult.model_validate({"model": {"path": "model"}}),
        ResultDeliveryRequest(artifact_fields=["model"]),
    )
    assert len(sent) == 1
    assert (b"unused-checkpoint" in sent[0]) is upload_all
    assert external.call_count == int(declared)
    receipt = read_receipt(base)
    assert receipt is not None and receipt.all_artifacts
    assert spec.output == (
        EchoSpecStrict(
            taskType=TaskType.ECHO,
            output=OutputSpec.model_validate(output) if output else None,
        ).output
    )


def test_system_delivery_failure_keeps_producer_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server")
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "0")
    client_class = httpx.Client
    monkeypatch.setattr(
        result_delivery.httpx,
        "Client",
        lambda **kwargs: client_class(
            transport=httpx.MockTransport(lambda _: httpx.Response(503)), **kwargs
        ),
    )
    message = make_worker_task_message(
        EchoSpecStrict(taskType=TaskType.ECHO, data={"type": "list", "items": ["x"]}),
        result_delivery={"tsk-test": ResultDeliveryRequest()},
        result_dispatch="current-dispatch",
    )
    lifecycle = Mock(worker_id="wrk-test", cost_per_hour=0.0)
    executor = Mock()
    executor.run.return_value = BaseExecutorResult.model_validate({"value": "computed"})
    runner = Runner(
        cast(Any, lifecycle),
        [message],
        tmp_path,
        make_worker_hardware(),
        {"echo": executor},
        executor,
        logging.getLogger("test-failed-publication"),
    )
    monkeypatch.setattr(runner, "_start_interrupt_monitor", lambda: None)
    monkeypatch.setattr(
        runner, "_create_task_logger", lambda *args: (None, False, None)
    )
    runner.start()
    lifecycle.set_succeeded.assert_called_once()
    lifecycle.set_failed.assert_not_called()
    assert (tmp_path / "tsk-test" / "results.json").is_file()
    assert "current-dispatch" in (tmp_path / "tsk-test" / "results.json").read_text()


def test_named_directory_hydrates_over_partial_cache_and_keeps_code_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server")
    producer = populate(tmp_path / "producer")
    receipt = read_receipt(producer)
    assert receipt is not None
    from_server = result_delivery.create_delivery_bundle(producer, "tsk-up", ["model"])
    try:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [from_server.read_bytes()]
        download = Mock(return_value=response)
        monkeypatch.setattr(result_delivery.requests, "get", download)
        results = tmp_path / "consumer"
        (results / "tsk-up" / "artifacts" / "model").mkdir(parents=True)
        source = "/producer/results/tsk-up/artifacts/model"
        spec = PythonSpecStrict(
            taskType=TaskType.PYTHON,
            inputs=[],
            code=f"def main():\n    return '{source}'",
            env={"MODEL": source},
        )
        task = make_worker_task_message(
            spec,
            artifact_inputs={
                "tsk-test": [
                    ArtifactInput(
                        task_id="tsk-up",
                        path="model",
                        source=source,
                        generation=receipt.generation,
                    )
                ]
            },
        )
        hydrated = result_delivery.hydrate_task(task, results)
        assert isinstance(hydrated.spec, PythonSpecStrict)
        assert hydrated.spec.code == spec.code
        assert hydrated.spec.env is not None
        local = resolve_checkpoint_load(
            {"type": "local", "path": hydrated.spec.env["MODEL"]}, results
        )
        assert (local / "weights").read_bytes() == b"model"
        assert (local / "empty").is_dir()
        assert not (results / "tsk-up" / "artifacts" / "unused-checkpoint").exists()
        result_delivery.hydrate_task(task, results)
        download.assert_called_once()
    finally:
        from_server.unlink()


def test_training_cleanup_retains_dependency_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_CLEANUP_AFTER_UPLOAD", "1")
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "0")
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    archive = tmp_path / "model.tar"
    archive.write_bytes(b"archive")
    task = make_worker_task_message(
        EchoSpecStrict(
            taskType=TaskType.ECHO,
            output=OutputSpec.model_validate(
                {"destination": {"type": "http", "url": "http://external"}}
            ),
        ),
        result_delivery={"tsk-test": ResultDeliveryRequest(artifact_fields=["model"])},
    )
    TrainingMixin()._cleanup_local_artifacts(task, checkpoint, model, archive)
    assert all(path.exists() for path in (checkpoint, model, archive))
    task.result_delivery = {}
    TrainingMixin()._cleanup_local_artifacts(task, checkpoint, model, archive)
    assert all(not path.exists() for path in (checkpoint, model, archive))


@pytest.mark.parametrize(
    "mode", ["blanket", "archive", "explicit", "directory", "envelope", "none"]
)
def test_training_archive_generation_matches_delivery_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", str(int(mode == "blanket")))
    monkeypatch.setenv("MODEL_ARCHIVE_USE_PIGZ", "0")
    model = tmp_path / "final_model"
    model.mkdir()
    (model / "weights").write_bytes(b"model weights")
    output = (
        OutputSpec.model_validate(
            {"destination": {"type": "http", "url": "http://external"}}
        )
        if mode == "explicit"
        else None
    )
    requests = {
        "archive": ResultDeliveryRequest(artifact_fields=["final_model_archive"]),
        "directory": ResultDeliveryRequest(all_artifacts=True),
        "envelope": ResultDeliveryRequest(),
    }
    task = make_worker_task_message(
        SFTSpecStrict(taskType=TaskType.SFT, output=output),
        result_delivery={"tsk-test": requests[mode]} if mode in requests else {},
    )
    archive = TrainingMixin()._archive_model(task, model)
    if mode in {"blanket", "archive", "explicit"}:
        assert archive is not None and archive.is_file()
        with tarfile.open(archive) as contents:
            weights = contents.extractfile("final_model/weights")
            assert weights is not None and weights.read() == b"model weights"
    else:
        assert archive is None
        assert not model.with_suffix(".tar.gz").exists()


def test_hydrated_http_checkpoint_archive_unpacks_without_second_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_ARCHIVE_USE_PIGZ", "0")
    producer = populate(tmp_path / "producer")
    (producer / "artifacts" / "model" / "config.json").write_text("{}")
    monkeypatch.setenv("WORKER_UPLOAD_RESULTS", "1")
    archive = TrainingMixin()._archive_model(
        make_worker_task_message(SFTSpecStrict(taskType=TaskType.SFT)),
        producer / "artifacts" / "model",
    )
    assert archive is not None
    write_receipt(producer, make_receipt(producer, "tsk-up", None))
    url = "http://server/api/v1/results/tsk-up/files/model.tar.gz"
    task = make_worker_task_message(
        SFTSpecStrict(
            taskType=TaskType.SFT,
            checkpoint={"load": {"type": "http", "url": url}},
        ),
        artifact_inputs={
            "tsk-test": [ArtifactInput(task_id="tsk-up", path=archive.name, source=url)]
        },
    )
    hydrated = result_delivery.hydrate_task(task, tmp_path / "producer")
    download = Mock(
        side_effect=AssertionError("Cached archive must not download again")
    )
    monkeypatch.setattr("worker.executors.utils.checkpoints.requests.get", download)
    assert isinstance(hydrated.spec, SFTSpecStrict)
    assert hydrated.spec.checkpoint is not None
    model = resolve_checkpoint_load(
        hydrated.spec.checkpoint["load"], tmp_path / "consumer"
    )
    assert (model / "weights").read_bytes() == b"model"
    assert (model / "config.json").read_text() == "{}"
    download.assert_not_called()


def test_delivery_wire_round_trip() -> None:
    message = make_worker_task_message(
        EchoSpecStrict(taskType=TaskType.ECHO),
        result_delivery={"tsk-test": ResultDeliveryRequest(all_artifacts=True)},
        artifact_inputs={
            "tsk-test": [
                ArtifactInput(
                    task_id="tsk-up",
                    path="model",
                    source="/old/model",
                    generation="generation",
                )
            ]
        },
    )
    restored = WorkerTaskMessage.model_validate_json(message.model_dump_json())
    assert restored.result_delivery == message.result_delivery
    assert restored.artifact_inputs == message.artifact_inputs


def test_transfers_wait_as_long_as_the_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server")
    monkeypatch.setenv("WORKER_RESULT_TRANSFER_TIMEOUT_SEC", "4321")
    producer = populate(tmp_path / "producer")
    timeouts: list[Any] = []
    client_class = httpx.Client

    def capture_client(**kwargs: Any) -> httpx.Client:
        timeouts.append(kwargs["timeout"])
        return client_class(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        )

    monkeypatch.setattr(result_delivery.httpx, "Client", capture_client)
    assert result_delivery.publish_result(
        producer, "tsk-up", ResultDeliveryRequest(), logging.getLogger("test")
    )

    def capture_get(url: str, **kwargs: Any) -> Any:
        timeouts.append(kwargs["timeout"])
        raise result_delivery.requests.ConnectionError("stop")

    monkeypatch.setattr(result_delivery.requests, "get", capture_get)
    with pytest.raises(result_delivery.ExecutionError):
        result_delivery.hydrate_result("tsk-up", tmp_path / "consumer")

    assert timeouts == [4321.0, 4321.0]
