"""Tests for SDK resource helper behavior that is not simple request wiring."""

import asyncio
import io
import json
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest
from flowmesh.exceptions import FlowMeshError
from flowmesh.models.common import TaskStatus
from flowmesh.models.tasks import TaskInfo
from flowmesh.resources.results import AsyncResults, Results
from flowmesh.ssh import (
    build_ssh_task_yaml,
    detect_public_key,
    ssh_connection_commands,
    ssh_proxy_url,
    wait_for_ssh_info,
    wait_for_ssh_info_async,
)


def _build_bundle(
    task_id: str,
    result_payload: dict | None = None,
    artifacts: dict[str, bytes] | None = None,
    logs: dict[str, bytes] | None = None,
    *,
    gzip: bool = True,
) -> bytes:
    """Build a tar (optionally gzipped) that mirrors the host bundle shape."""
    buf = io.BytesIO()
    archive = (
        tarfile.open(fileobj=buf, mode="w:gz")
        if gzip
        else tarfile.open(fileobj=buf, mode="w")
    )
    with archive:
        if result_payload is not None:
            data = json.dumps(result_payload, indent=2).encode()
            info = tarfile.TarInfo(name=f"{task_id}/results.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for rel, payload in (artifacts or {}).items():
            info = tarfile.TarInfo(name=f"{task_id}/artifacts/{rel}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for rel, payload in (logs or {}).items():
            info = tarfile.TarInfo(name=f"{task_id}/logs/{rel}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _SyncResultClient:
    def __init__(
        self, payload: dict | None = None, bundle: bytes | None = None
    ) -> None:
        self.payload = payload or {}
        self.bundle = bundle
        self.download_paths: list[str] = []

    def _request(self, method: str, path: str):
        assert method == "GET"
        assert path.startswith("/results/")
        return self.payload

    def _download(self, path: str, output_path: Path) -> None:
        self.download_paths.append(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.bundle is not None and "/bundle" in path:
            output_path.write_bytes(self.bundle)
        else:
            output_path.write_text(path)


class _AsyncResultClient:
    def __init__(
        self,
        payload: dict | None = None,
        bundle: bytes | None = None,
    ) -> None:
        self.payload = payload or {}
        self.bundle = bundle
        self.download_paths: list[str] = []

    async def _request(self, method: str, path: str):
        assert method == "GET"
        assert path.startswith("/results/")
        return self.payload

    async def _download(self, path: str, output_path: Path) -> None:
        self.download_paths.append(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.bundle is not None and "/bundle" in path:
            output_path.write_bytes(self.bundle)
        else:
            output_path.write_text(path)


def _task_info(status: TaskStatus, latest_update: dict | None = None) -> TaskInfo:
    return TaskInfo(
        task_id="t-1",
        workflow_id="wf-1",
        owner_id="u-1",
        org_id="o-1",
        supplier_id="s-1",
        raw_yaml="kind: Task",
        task={},
        status=status,
        submitted_at="2025-01-01T00:00:00Z",
        submitted_ts=1.0,
        usages=[],
        attempts=1,
        max_attempts=1,
        load=1,
        depends_on=[],
        pending_dependencies=[],
        dependents=[],
        completed=status == TaskStatus.DONE,
        failed=status == TaskStatus.FAILED,
        latest_update=latest_update,
        error="boom" if status == TaskStatus.FAILED else None,
    )


class TestResultHelpers:
    def test_sync_materialize_extracts_bundle_and_rewrites_base_dir(
        self, tmp_path: Path
    ) -> None:
        result_payload = {
            "task_id": "task-1",
            "result": {
                "_artifacts": {
                    "base_dir": "/worker/results/task-1",
                    "base_url": "http://host:8010",
                },
                "images": [
                    {"path": "images/a.png"},
                    {"path": "images/b.png"},
                ],
            },
        }
        bundle = _build_bundle(
            "task-1",
            result_payload=result_payload,
            artifacts={"images/a.png": b"aaa", "images/b.png": b"bbb"},
        )
        client = _SyncResultClient(bundle=bundle)
        results = Results(cast(Any, client))

        materialized, json_path, extracted = results.materialize("task-1", tmp_path)

        # SDK requested the bundle endpoint with the default include.
        assert client.download_paths == [
            "/results/task-1/bundle?include=results&include=artifacts"
        ]
        # Extracted layout: <tmp>/task-1/{results.json, artifacts/images/*}
        task_root = tmp_path / "task-1"
        assert (task_root / "results.json").is_file()
        assert (task_root / "artifacts" / "images" / "a.png").read_bytes() == b"aaa"
        assert (task_root / "artifacts" / "images" / "b.png").read_bytes() == b"bbb"
        assert json_path == task_root / "results.json"
        assert {p.name for p in extracted} == {
            "results.json",
            "a.png",
            "b.png",
        }
        # base_dir rewritten to the local extracted dir; base_url dropped.
        assert materialized["result"]["_artifacts"] == {
            "base_dir": task_root.resolve().as_posix()
        }
        # Per-ref `path` stays relative.
        assert materialized["result"]["images"] == [
            {"path": "images/a.png"},
            {"path": "images/b.png"},
        ]
        # Persisted results.json mirrors the mutated payload.
        on_disk = json.loads(json_path.read_text())
        assert (
            on_disk["result"]["_artifacts"]["base_dir"]
            == task_root.resolve().as_posix()
        )

    def test_sync_materialize_result_only_does_not_rewrite(
        self, tmp_path: Path
    ) -> None:
        result_payload = {
            "task_id": "task-1",
            "result": {
                "_artifacts": {
                    "base_dir": "/worker/results/task-1",
                    "base_url": "http://host:8010",
                },
                "ok": True,
            },
        }
        bundle = _build_bundle("task-1", result_payload=result_payload)
        client = _SyncResultClient(bundle=bundle)
        results = Results(cast(Any, client))

        materialized, json_path, extracted = results.materialize(
            "task-1", tmp_path, include=["results"]
        )

        assert client.download_paths == ["/results/task-1/bundle?include=results"]
        assert json_path.is_file()
        assert {p.name for p in extracted} == {"results.json"}
        # No artifacts extracted → ctx is left intact.
        assert materialized["result"]["_artifacts"] == {
            "base_dir": "/worker/results/task-1",
            "base_url": "http://host:8010",
        }

    def test_sync_download_files_uses_basename_only(self, tmp_path: Path) -> None:
        results = Results(cast(Any, _SyncResultClient()))

        output_paths = list(
            results.download_files("task-1", ["nested/a.txt", "b.txt"], tmp_path)
        )

        assert [path.name for path in output_paths] == ["a.txt", "b.txt"]

    @pytest.mark.anyio
    async def test_async_results_materialize_matches_sync(self, tmp_path: Path) -> None:
        result_payload = {
            "task_id": "task-1",
            "result": {
                "_artifacts": {
                    "base_dir": "/worker/results/task-1",
                    "base_url": "http://host:8010",
                },
                "images": [{"path": "images/a.png"}],
            },
        }
        bundle = _build_bundle(
            "task-1",
            result_payload=result_payload,
            artifacts={"images/a.png": b"aaa"},
        )
        client = _AsyncResultClient(bundle=bundle)
        results = AsyncResults(cast(Any, client))

        materialized, json_path, extracted = await results.materialize(
            "task-1", tmp_path
        )
        output_paths: list[Path] = []
        async for path in results.download_files("task-1", ["nested/a.txt"], tmp_path):
            output_paths.append(path)

        task_root = tmp_path / "task-1"
        assert json_path == task_root / "results.json"
        assert {p.name for p in extracted} == {"results.json", "a.png"}
        assert materialized["result"]["_artifacts"] == {
            "base_dir": task_root.resolve().as_posix()
        }
        assert [path.name for path in output_paths] == ["a.txt"]


class TestSSHHelpers:
    def test_build_task_yaml(self) -> None:
        expected_yaml = build_ssh_task_yaml(
            name="ssh-demo",
            public_key="ssh-ed25519 AAAA",
            interactive=False,
            user="flowmesh",
            mode="proxy",
            ttl=600,
            idle_timeout=60,
            gpu=1,
            gpu_memory="24Gi",
            cpu=4,
            memory="16Gi",
            image="python:3.12",
            worker="worker-1",
            env_pairs=["A=1", "B=2"],
            command=["python", "script.py"],
            entrypoint=["/bin/sh", "-c"],
        )
        assert "kind: SSHTask" in expected_yaml
        assert "selected_worker: worker-1" in expected_yaml
        assert "env:" in expected_yaml
        assert "A:" in expected_yaml
        assert "B:" in expected_yaml

        interactive_yaml = build_ssh_task_yaml(
            name="ssh-interactive",
            public_key="ssh-ed25519 AAAA",
            interactive=True,
            user="flowmesh",
            mode="proxy",
            ttl=600,
            idle_timeout=60,
            gpu=None,
            gpu_memory=None,
            cpu=None,
            memory=None,
            image=None,
            worker=None,
            env_pairs=None,
            command=None,
            entrypoint=None,
        )
        assert "interactive: true" in interactive_yaml
        assert "authorizedKeys:" in interactive_yaml

    def test_proxy_url_and_connection_commands(self) -> None:
        base_url = "https://example.com"
        proxy = ssh_proxy_url(base_url, "task-1")
        assert proxy  # non-empty

        ssh_info = {
            "mode": "forward",
            "username": "flowmesh",
            "host": "proxy.example.com",
            "port": 32000,
            "directHost": "worker.example.com",
            "directPort": 22,
        }
        cmds = ssh_connection_commands("task-1", ssh_info, base_url=base_url)
        assert isinstance(cmds, list)

    def test_detect_public_key_prefers_standard_keys(self, tmp_path: Path) -> None:
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_rsa.pub").write_text("ssh-rsa BBBB")
        (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA")

        assert detect_public_key(tmp_path) == "ssh-ed25519 AAAA"

    @pytest.mark.anyio
    async def test_wait_for_ssh_info_sync_and_async(self) -> None:
        sync_tasks = iter(
            [
                _task_info(TaskStatus.DISPATCHED),
                _task_info(
                    TaskStatus.DISPATCHED,
                    latest_update={"ssh": {"mode": "proxy", "port": 2222}},
                ),
            ]
        )

        ssh_info = wait_for_ssh_info(
            retrieve_task=lambda task_id: next(sync_tasks),
            task_id="t-1",
            interval=0.0,
            sleep=lambda interval: None,
        )
        assert ssh_info == {"mode": "proxy", "port": 2222}

        async_tasks = iter(
            [
                _task_info(TaskStatus.DISPATCHED),
                _task_info(TaskStatus.DONE),
            ]
        )

        async def retrieve_task(task_id: str) -> TaskInfo:
            return next(async_tasks)

        with pytest.raises(FlowMeshError, match="completed without providing SSH info"):
            await wait_for_ssh_info_async(
                retrieve_task=retrieve_task,
                task_id="t-1",
                interval=0.0,
                sleep=lambda interval: asyncio.sleep(0),
            )
