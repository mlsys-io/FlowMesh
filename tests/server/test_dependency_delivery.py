import logging
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from server.app_state import get_event_monitor, get_results_dir, get_runtime
from server.routers.v1.results import _create_result_bundle_archive, router
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.utils.manifest import sync_manifest
from shared.utils.result_delivery import (
    artifacts_ready,
    create_delivery_bundle,
    extract_delivery_bundle,
    make_receipt,
    result_generation,
    write_receipt,
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


def test_result_published_through_its_own_destination_bundles_completely(
    tmp_path: Path,
) -> None:
    server = tmp_path / "server"
    monitor = Mock()
    monitor.pop_pending_clones.return_value = []
    app = FastAPI()
    app.state.logger = logging.getLogger("test-delivery")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_results_dir] = lambda: server
    app.dependency_overrides[get_runtime] = lambda: SimpleNamespace(
        get_record=lambda _: None
    )
    app.dependency_overrides[get_event_monitor] = lambda: monitor
    envelope = ResultEnvelope(
        task_id="tsk-leaf",
        result=BaseExecutorResult(),
        metadata={"independent_results": True, "result_dispatch": "d1"},
    )
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/v1/results/tsk-leaf/files",
            files={"file": ("out/answer.txt", b"42")},
        )
        assert uploaded.status_code == 200, uploaded.text
        posted = client.post(
            "/api/v1/results", json=envelope.model_dump(mode="json", by_alias=True)
        )
        assert posted.status_code == 200, posted.text
        bundle = client.get("/api/v1/results/tsk-leaf/bundle")
    assert bundle.status_code == 200, bundle.text
    assert not artifacts_ready(server / "tsk-leaf", "tsk-leaf")


def test_manifest_does_not_describe_a_link_target(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("hunter2")
    base = tmp_path / "tsk-up"
    base.mkdir()
    (base / "leak").symlink_to(secret)
    manifest = sync_manifest(base, "tsk-up", ["leak"])
    [entry] = [item for item in manifest["entries"] if item["path"] == "leak"]
    assert entry["status"] == "present"
    assert "sha256" not in entry and "size" not in entry


def test_selected_bundle_carries_the_target_of_a_selected_link(tmp_path: Path) -> None:
    base = populate(tmp_path / "server")
    (base / "artifacts" / "latest").symlink_to("model")
    write_receipt(base, make_receipt(base, "tsk-up", None))
    bundle = _create_result_bundle_archive(
        "tsk-up", base, ("results", "artifacts"), ["latest"], None, None
    )
    try:
        cache = extract_delivery_bundle(bundle, tmp_path / "cache", "tsk-up")
    finally:
        bundle.unlink()
    assert artifacts_ready(cache, "tsk-up", ["latest"])
    assert (cache / "artifacts" / "latest" / "weights").read_bytes() == b"model"


def test_server_bundle_makes_links_into_own_artifacts_portable(tmp_path: Path) -> None:
    base = populate(tmp_path / "server")
    (base / "artifacts" / "latest").symlink_to("/worker-results/tsk-up/artifacts/model")
    write_receipt(base, make_receipt(base, "tsk-up", None))
    bundle = _create_result_bundle_archive(
        "tsk-up", base, ("results", "artifacts"), ["latest"], None, None
    )
    try:
        cache = extract_delivery_bundle(bundle, tmp_path / "cache", "tsk-up")
    finally:
        bundle.unlink()
    assert (cache / "artifacts" / "latest").readlink() == Path("model")
    assert (cache / "artifacts" / "latest" / "weights").read_bytes() == b"model"


def test_server_bundle_drops_links_outside_artifacts(tmp_path: Path) -> None:
    base = populate(tmp_path / "server")
    (base / "logs" / "planted").symlink_to(tmp_path)
    bundle = _create_result_bundle_archive("tsk-up", base, ("logs",))
    try:
        with tarfile.open(bundle) as archive:
            assert "tsk-up/logs/planted" not in archive.getnames()
    finally:
        bundle.unlink()


def _check_client(results_dir: Path) -> TestClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test-delivery-check")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_results_dir] = lambda: results_dir
    return TestClient(app)


def test_delivery_check_reports_a_held_snapshot_without_creating_one(
    tmp_path: Path,
) -> None:
    base = populate(tmp_path / "shared", task_id="tsk-up")
    generation = result_generation(base)
    with _check_client(tmp_path / "shared") as client:
        url = "/api/v1/results/tsk-up/delivery"
        held = client.get(url, params={"generation": generation, "all_artifacts": 1})
        assert held.status_code == 204, held.text
        selected = client.get(
            url, params={"generation": generation, "artifact_path": "model"}
        )
        assert selected.status_code == 204
        stale = client.get(url, params={"generation": "0" * 64, "all_artifacts": 1})
        assert stale.status_code == 404
        absent = client.get(
            url, params={"generation": generation, "artifact_path": "nothing"}
        )
        assert absent.status_code == 404
        assert client.get(url).status_code == 422
        bad = client.get(
            url, params={"generation": generation, "artifact_path": "../x"}
        )
        assert bad.status_code == 400
        both = client.get(
            url,
            params={
                "generation": generation,
                "artifact_path": "model",
                "all_artifacts": 1,
            },
        )
        assert both.status_code == 400
    with _check_client(tmp_path / "empty") as client:
        missing = client.get(
            "/api/v1/results/tsk-up/delivery",
            params={"generation": generation, "all_artifacts": 1},
        )
        assert missing.status_code == 404
    assert not (tmp_path / "empty" / "tsk-up").exists()
