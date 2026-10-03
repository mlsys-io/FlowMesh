import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from server.app_state import get_results_dir, get_runtime
from server.routers.v1.results import _create_result_bundle_archive, router
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.utils.result_delivery import (
    artifacts_ready,
    create_delivery_bundle,
    extract_delivery_bundle,
)
from tests.shared.test_result_delivery import populate
from worker.result_delivery import hydrate_result


def test_delivery_route_streams_and_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    producer = populate(tmp_path / "producer", b"large" * 200000)
    server = tmp_path / "server"
    app = FastAPI()
    app.state.logger = logging.getLogger("test-delivery")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_results_dir] = lambda: server
    app.dependency_overrides[get_runtime] = lambda: SimpleNamespace(
        get_record=lambda _: None
    )
    bundle = create_delivery_bundle(producer, "tsk-up", ["model"])
    try:
        with TestClient(app) as client, bundle.open("rb") as source:
            response = client.post(
                "/api/v1/results/tsk-up/delivery",
                files={"file": ("bundle.tar", source)},
            )
            assert response.status_code == 200, response.text
            assert artifacts_ready(server / "tsk-up", "tsk-up", ["model"])
            assert client.get("/api/v1/results/tsk-up").status_code == 200
            assert client.get("/api/v1/results/tsk-up/bundle").status_code == 404
            selected = client.get("/api/v1/results/tsk-up/bundle?artifact_path=model")
            assert selected.status_code == 200, selected.text
            downloaded = tmp_path / "downloaded.tar.gz"
            downloaded.write_bytes(selected.content)
            cache = extract_delivery_bundle(downloaded, tmp_path / "cache", "tsk-up")
            assert artifacts_ready(cache, "tsk-up", ["model"])
            assert not artifacts_ready(cache, "tsk-up")
            broken = client.post(
                "/api/v1/results/tsk-up/delivery",
                files={"file": ("bad.tar", b"truncated")},
            )
            assert broken.status_code == 400
            assert artifacts_ready(server / "tsk-up", "tsk-up", ["model"])
            replacement = client.post(
                "/api/v1/results/tsk-up/files",
                files={"file": ("model/weights", b"changed")},
            )
            assert replacement.status_code == 200
            assert (
                client.get(
                    "/api/v1/results/tsk-up/bundle?artifact_path=model"
                ).status_code
                == 404
            )
    finally:
        bundle.unlink()


def test_missing_requested_section_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    with pytest.raises(HTTPException) as error:
        _create_result_bundle_archive("tsk-up", tmp_path)
    assert error.value.status_code == 404


def test_server_generated_skipped_result_can_hydrate_remotely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = tmp_path / "server" / "tsk-skipped"
    server.mkdir(parents=True)
    (server / "artifacts").mkdir()
    envelope = ResultEnvelope(
        task_id="tsk-skipped", result=BaseExecutorResult(), metadata={"skipped": True}
    )
    (server / "results.json").write_text(envelope.model_dump_json())
    bundle = _create_result_bundle_archive(
        "tsk-skipped", server, ("results", "artifacts")
    )
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [bundle.read_bytes()]
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://server")
    monkeypatch.setattr(
        "worker.result_delivery.requests.get", Mock(return_value=response)
    )
    try:
        hydrate_result("tsk-skipped", tmp_path / "consumer")
        cache = tmp_path / "consumer" / "tsk-skipped"
        assert artifacts_ready(cache, "tsk-skipped")
        delivered = ResultEnvelope.model_validate_json(
            (cache / "results.json").read_text()
        )
        assert delivered.metadata == {"skipped": True}
    finally:
        bundle.unlink()


def test_stale_dispatch_delivery_cannot_replace_or_expose_results(
    tmp_path: Path,
) -> None:
    producer = populate(tmp_path / "producer", dispatch_id="previous")
    server = tmp_path / "server"
    populate(server, dispatch_id="previous")
    runtime = Mock(spec=TaskRuntime)
    runtime.get_record.return_value = SimpleNamespace(
        result_dispatch="current", status=TaskStatus.DONE
    )
    app = FastAPI()
    app.state.logger = logging.getLogger("test-stale-delivery")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_results_dir] = lambda: server
    app.dependency_overrides[get_runtime] = lambda: runtime
    bundle = create_delivery_bundle(producer, "tsk-up", ["model"])
    try:
        with TestClient(app) as client, bundle.open("rb") as source:
            response = client.post(
                "/api/v1/results/tsk-up/delivery",
                files={"file": ("bundle.tar", source)},
            )
            assert response.status_code == 404, response.text
            for suffix in ["", "/files/model/weights", "/bundle?artifact_path=model"]:
                response = client.get(f"/api/v1/results/tsk-up{suffix}")
                assert response.status_code == 404, response.text
    finally:
        bundle.unlink()


def test_independent_missing_child_never_inherits_parent_result(tmp_path: Path) -> None:
    parent = populate(tmp_path, task_id="tsk-parent")
    monitor: Any = EventMonitor.__new__(EventMonitor)
    monitor._results_dir = tmp_path
    monitor._pending_lock = threading.Lock()
    monitor._pending_result_clones = {}
    monitor._runtime = Mock()
    monitor._logger = logging.getLogger("test-delivery-mirror")
    monitor.mirror_task_results("tsk-parent", ["tsk-child"])
    assert (parent / "results.json").is_file()
    assert not (tmp_path / "tsk-child" / "results.json").exists()
