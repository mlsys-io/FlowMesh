"""SSH executor result mounting tests."""

import tarfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.config import WorkerConfig
from worker.executors.ssh_executor import ResolvedSSHInput, SSHConfig, SSHExecutor


def _worker_config(
    tmp_path: Path,
    results_mount_source: str | None = "flowmesh-results",
    network_mode: str | None = None,
) -> WorkerConfig:
    return make_live_worker_config(
        tmp_path,
        results_mount_source=results_mount_source,
        network_mode=network_mode,
    )


def _task_message(**spec_updates: object) -> WorkerTaskMessage:
    payload = {
        "task_id": "task-ssh",
        "workflow_id": "wf-1",
        "owner_id": "owner",
        "assigned_worker": "worker-1",
        "dispatched_at": "2026-03-22T00:00:00Z",
        "upstream_task_ids": {"preprocess": "task-pre"},
        "task": {
            "apiVersion": "mloc/v1",
            "kind": "Task",
            "metadata": {"name": "wf:annotate"},
            "spec": {
                "taskType": "ssh",
                "accessMode": "direct",
                "inputs": [{"stage": "preprocess"}],
                "sshOutput": {"mountPath": "/mnt/flowmesh/output"},
                **spec_updates,
            },
        },
    }
    return WorkerTaskMessage.model_validate(payload)


def test_build_mount_plan_uses_worker_volume_view_in_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    (results_dir / "task-pre").mkdir(parents=True)
    out_dir = tmp_path / "task-ssh"

    monkeypatch.setenv("RESULTS_DIR", str(results_dir))
    task = _task_message()
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    executor = SSHExecutor(
        _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")
    )
    monkeypatch.setattr(
        executor,
        "_stage_inputs_in_volume",
        lambda client, resolved_inputs, results_source, session_id: "staged-inputs-vol",
    )
    # _build_mount_plan only hands the client to _stage_inputs_in_volume,
    # which we mock above — no need to hit docker.from_env() for this.
    fake_docker = MagicMock()
    monkeypatch.setattr(executor, "_get_docker_client", lambda: fake_docker)

    resolved_inputs = executor._resolve_inputs(task, cfg)  # noqa: SLF001
    plan = executor._build_mount_plan(
        executor._get_docker_client(), out_dir, resolved_inputs, cfg, "session-1234"
    )  # noqa: SLF001

    assert plan.volumes == ["staged-inputs-vol:/root/.flowmesh/results-source:ro"]
    assert plan.staged_input_specs == [
        (
            "/mnt/flowmesh/inputs/preprocess",
            "/root/.flowmesh/results-source/task-pre",
        )
    ]
    assert plan.copy_output_path == "/mnt/flowmesh/output"
    assert plan.direct_output_path is None
    assert plan.staged_inputs_volume == "staged-inputs-vol"
    assert plan.staged_inputs_dir is None


def test_build_mount_plan_uses_direct_binds_outside_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    (results_dir / "task-pre").mkdir(parents=True)
    out_dir = tmp_path / "task-ssh"
    artifacts_dir = out_dir / "artifacts"

    monkeypatch.setenv("RESULTS_DIR", str(results_dir))
    task = _task_message()
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    executor = SSHExecutor(_worker_config(tmp_path, results_mount_source=None))
    staged_inputs_dir = tmp_path / "staged-inputs"
    (staged_inputs_dir / "task-pre").mkdir(parents=True)
    monkeypatch.setattr(
        executor,
        "_stage_inputs_locally",
        lambda resolved_inputs, session_id: staged_inputs_dir,
    )
    # The results_mount_source=None path never calls into the client
    # either (see _build_mount_plan), so a mock is sufficient.
    fake_docker = MagicMock()
    monkeypatch.setattr(executor, "_get_docker_client", lambda: fake_docker)

    resolved_inputs = executor._resolve_inputs(task, cfg)  # noqa: SLF001
    plan = executor._build_mount_plan(
        executor._get_docker_client(), out_dir, resolved_inputs, cfg, "session-1234"
    )  # noqa: SLF001

    assert plan.staged_input_specs == []
    assert plan.direct_output_path == artifacts_dir
    assert plan.copy_output_path is None
    assert (
        f"{staged_inputs_dir / 'task-pre'}:/mnt/flowmesh/inputs/preprocess:ro"
        in plan.volumes
    )
    assert f"{artifacts_dir}:/mnt/flowmesh/output:rw" in plan.volumes
    assert plan.staged_inputs_dir == staged_inputs_dir
    assert plan.staged_inputs_volume is None


def test_resolve_inputs_rejects_unsafe_mount_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    (results_dir / "task-pre").mkdir(parents=True)

    monkeypatch.setenv("RESULTS_DIR", str(results_dir))
    task = _task_message(inputs=[{"stage": "preprocess", "mountPath": "/tmp/unsafe"}])
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    executor = SSHExecutor(
        _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")
    )

    with pytest.raises(Exception, match="must be under /mnt/flowmesh"):
        executor._resolve_inputs(task, cfg)  # noqa: SLF001


def test_stage_inputs_locally_downloads_missing_upstream_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task_message()
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    executor = SSHExecutor(_worker_config(tmp_path, results_mount_source=None))

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    resolved_inputs = executor._resolve_inputs(task, cfg)  # noqa: SLF001

    def _download(task_id: str, destination_dir: Path) -> None:
        staged = destination_dir / task_id
        staged.mkdir(parents=True)
        (staged / "results.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(executor, "_download_result_bundle", _download)

    staging_dir = executor._stage_inputs_locally(  # noqa: SLF001
        resolved_inputs, "session-remote"
    )

    assert (staging_dir / "task-pre" / "results.json").exists()


def test_stage_inputs_in_volume_downloads_missing_upstream_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://flowmesh.example")
    monkeypatch.setenv("FLOWMESH_API_KEY", "secret-token")

    executor = SSHExecutor(
        _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")
    )
    local_source = tmp_path / "results" / "task-local"
    local_source.mkdir(parents=True)
    resolved_inputs = [
        ResolvedSSHInput(
            stage="local",
            task_id="task-local",
            source_path=local_source,
            mount_path="/mnt/flowmesh/inputs/local",
        ),
        ResolvedSSHInput(
            stage="remote",
            task_id="task-remote",
            source_path=tmp_path / "results" / "task-remote",
            mount_path="/mnt/flowmesh/inputs/remote",
        ),
    ]

    class _FakeVolume:
        def remove(self, force: bool = False) -> None:
            return None

    class _FakeVolumes:
        def create(
            self, name: str, labels: dict[str, str] | None = None
        ) -> _FakeVolume:
            self.name = name
            self.labels = labels
            return _FakeVolume()

    class _FakeContainers:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def run(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _FakeClient:
        def __init__(self) -> None:
            self.volumes = _FakeVolumes()
            self.containers = _FakeContainers()

    fake_client = _FakeClient()
    monkeypatch.setattr(executor, "_get_docker_client", lambda: fake_client)

    volume_name = executor._stage_inputs_in_volume(
        executor._get_docker_client(),
        resolved_inputs,
        "flowmesh-results",
        "session-remote",
    )

    assert fake_client.containers.kwargs is not None
    command = cast(list[str], fake_client.containers.kwargs["command"])[2]
    assert volume_name == "flowmesh_ssh_inputs_session-remote"
    assert (
        fake_client.containers.kwargs["network_mode"] == "container:flowmesh-worker-1"
    )
    assert "cp -a /src/task-local/. /dst/task-local/" in command
    assert (
        "wget -qO- -T 300 -t 1 --header 'Authorization: Bearer secret-token' "
        "'http://flowmesh.example/api/v1/results/task-remote/bundle"
        "?include=results&include=artifacts' | tar -xz -C /dst" in command
    )


def test_extract_result_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.tar"
    with tarfile.open(bundle, mode="w") as archive:
        payload = tmp_path / "payload.txt"
        payload.write_text("x", encoding="utf-8")
        archive.add(payload, arcname="../escape.txt")

    with pytest.raises(Exception, match="Unsafe path"):
        SSHExecutor._extract_result_bundle(bundle, tmp_path / "dest")  # noqa: SLF001
