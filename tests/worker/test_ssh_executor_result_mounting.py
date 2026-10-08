"""SSH session input staging and result mounting tests."""

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from shared.schemas.result import BaseExecutorResult, ResultEnvelope, write_result
from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_live_worker_config
from worker.config import WorkerConfig
from worker.executors.ssh_session import ResolvedSSHInput, SSHConfig
from worker.executors.ssh_session import inputs as inputs_module
from worker.executors.ssh_session.backends.docker import DockerSessionBackend

if TYPE_CHECKING:
    from docker import DockerClient


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
            "apiVersion": "flowmesh/v1",
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
    worker_cfg = _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")
    backend = DockerSessionBackend(worker_cfg)
    monkeypatch.setattr(
        backend,
        "_stage_inputs_in_volume",
        lambda *args: "staged-inputs-vol",
    )
    # _build_mount_plan only hands the client to _stage_inputs_in_volume,
    # which we mock above — no need to hit docker.from_env() for this.
    fake_docker = MagicMock()

    resolved_inputs = inputs_module.resolve_inputs(task, cfg, worker_cfg.results_dir)
    plan = backend._build_mount_plan(
        fake_docker, out_dir, resolved_inputs, cfg, "session-1234", "worker-1"
    )

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
    worker_cfg = _worker_config(tmp_path, results_mount_source=None)
    backend = DockerSessionBackend(worker_cfg)
    staged_inputs_dir = tmp_path / "staged-inputs"
    (staged_inputs_dir / "task-pre").mkdir(parents=True)
    monkeypatch.setattr(
        "worker.executors.ssh_session.backends.docker.stage_inputs_locally",
        lambda resolved_inputs, session_id: staged_inputs_dir,
    )
    # The results_mount_source=None path never calls into the client
    # either (see _build_mount_plan), so a mock is sufficient.
    fake_docker = MagicMock()

    resolved_inputs = inputs_module.resolve_inputs(task, cfg, worker_cfg.results_dir)
    plan = backend._build_mount_plan(
        fake_docker, out_dir, resolved_inputs, cfg, "session-1234", "worker-1"
    )

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
    worker_cfg = _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")

    with pytest.raises(Exception, match="must be under /mnt/flowmesh"):
        inputs_module.resolve_inputs(task, cfg, worker_cfg.results_dir)


def test_stage_inputs_locally_downloads_missing_upstream_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task_message()
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    worker_cfg = _worker_config(tmp_path, results_mount_source=None)

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    resolved_inputs = inputs_module.resolve_inputs(task, cfg, worker_cfg.results_dir)

    def _hydrate(
        task_id: str,
        root: Path,
        paths: list[str] | None = None,
        generation: str | None = None,
    ) -> None:
        write_result(root, ResultEnvelope(task_id=task_id, result=BaseExecutorResult()))

    monkeypatch.setattr(inputs_module, "hydrate_result", _hydrate)

    staging_dir = inputs_module.stage_inputs_locally(resolved_inputs, "session-remote")

    assert (staging_dir / "task-pre" / "results.json").exists()


def test_stage_inputs_in_volume_downloads_missing_upstream_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWMESH_BASE_URL", "http://flowmesh.example")
    monkeypatch.setenv("FLOWMESH_API_KEY", "secret-token")

    backend = DockerSessionBackend(
        _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")
    )
    local_source = tmp_path / "results" / "task-local"
    write_result(
        local_source.parent,
        ResultEnvelope(task_id="task-local", result=BaseExecutorResult()),
    )

    def hydrate(
        task_id: str,
        root: Path,
        paths: list[str] | None = None,
        generation: str | None = None,
    ) -> None:
        write_result(root, ResultEnvelope(task_id=task_id, result=BaseExecutorResult()))

    monkeypatch.setattr(
        "worker.executors.ssh_session.backends.docker.hydrate_result", hydrate
    )
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

    volume_name = backend._stage_inputs_in_volume(
        cast("DockerClient", fake_client),
        resolved_inputs,
        "flowmesh-results",
        "session-remote",
        "worker-1",
    )

    assert fake_client.containers.kwargs is not None
    command = cast(list[str], fake_client.containers.kwargs["command"])[2]
    assert volume_name == "flowmesh_ssh_inputs_session-remote"
    assert (
        fake_client.containers.kwargs["network_mode"] == "container:flowmesh-worker-1"
    )
    assert "cp -a /src/task-local/. /dst/task-local/" in command
    assert "cp -a /src/task-remote/. /dst/task-remote/" in command


def test_local_input_staging_keeps_links_as_links(tmp_path: Path) -> None:
    secret = tmp_path / "worker-secret"
    secret.write_text("hunter2")
    source = tmp_path / "results" / "t-up"
    (source / "artifacts").mkdir(parents=True)
    write_result(
        source.parent, ResultEnvelope(task_id="t-up", result=BaseExecutorResult())
    )
    (source / "artifacts" / "leak").symlink_to(secret)
    staged = inputs_module.stage_inputs_locally(
        [ResolvedSSHInput("up", "t-up", source, "/mnt/flowmesh/inputs/up")], "ssn-1"
    )
    link = staged / "t-up" / "artifacts" / "leak"
    assert link.is_symlink()
    assert link.readlink() == secret


@pytest.mark.parametrize("selection", ["leak", "nested"])
def test_selected_input_staging_keeps_links_as_links(
    tmp_path: Path, selection: str
) -> None:
    secret = tmp_path / "worker-secret"
    secret.write_text("hunter2")
    source = tmp_path / "results" / "t-up"
    (source / "artifacts" / "nested").mkdir(parents=True)
    write_result(
        source.parent, ResultEnvelope(task_id="t-up", result=BaseExecutorResult())
    )
    (source / "artifacts" / "leak").symlink_to(secret)
    (source / "artifacts" / "nested" / "leak").symlink_to(secret)
    staged = inputs_module.stage_inputs_locally(
        [
            ResolvedSSHInput(
                "up",
                "t-up",
                source,
                "/mnt/flowmesh/references/t-up",
                artifact_paths=[selection],
            )
        ],
        "ssn-1",
    )
    link = staged / "t-up" / "artifacts" / selection
    if selection == "nested":
        link = link / "leak"
    assert link.is_symlink()
    assert link.readlink() == secret


def test_selected_input_staging_brings_in_artifact_link_targets(tmp_path: Path) -> None:
    source = tmp_path / "results" / "t-up"
    (source / "artifacts" / "checkpoint-500").mkdir(parents=True)
    (source / "artifacts" / "checkpoint-500" / "weights").write_text("trained")
    (source / "artifacts" / "final_model").symlink_to("checkpoint-500")
    write_result(
        source.parent, ResultEnvelope(task_id="t-up", result=BaseExecutorResult())
    )
    staged = inputs_module.stage_inputs_locally(
        [
            ResolvedSSHInput(
                "up",
                "t-up",
                source,
                "/mnt/flowmesh/references/t-up",
                artifact_paths=["final_model"],
            )
        ],
        "ssn-1",
    )
    model = staged / "t-up" / "artifacts" / "final_model"
    assert model.is_symlink()
    assert (model / "weights").read_text() == "trained"


@pytest.mark.parametrize("selection", [None, ["latest"]])
def test_local_input_staging_makes_links_into_own_artifacts_portable(
    tmp_path: Path, selection: list[str] | None
) -> None:
    source = tmp_path / "results" / "t-up"
    (source / "artifacts" / "model").mkdir(parents=True)
    (source / "artifacts" / "model" / "weights").write_text("trained")
    (source / "artifacts" / "latest").symlink_to("/worker-results/t-up/artifacts/model")
    write_result(
        source.parent, ResultEnvelope(task_id="t-up", result=BaseExecutorResult())
    )
    staged = inputs_module.stage_inputs_locally(
        [
            ResolvedSSHInput(
                "up",
                "t-up",
                source,
                "/mnt/flowmesh/inputs/up",
                artifact_paths=selection,
            )
        ],
        "ssn-1",
    )
    link = staged / "t-up" / "artifacts" / "latest"
    assert link.readlink() == Path("model")
    assert (link / "weights").read_text() == "trained"


def test_volume_staging_makes_links_into_own_artifacts_portable(
    tmp_path: Path,
) -> None:
    backend = DockerSessionBackend(
        _worker_config(tmp_path, network_mode="container:flowmesh-worker-1")
    )
    source = tmp_path / "results" / "t-up"
    (source / "artifacts" / "model").mkdir(parents=True)
    (source / "artifacts" / "latest").symlink_to("/worker-results/t-up/artifacts/model")
    write_result(
        source.parent, ResultEnvelope(task_id="t-up", result=BaseExecutorResult())
    )
    client = MagicMock()
    backend._stage_inputs_in_volume(
        client,
        [ResolvedSSHInput("up", "t-up", source, "/mnt/flowmesh/inputs/up")],
        "flowmesh-results",
        "session-1",
        "worker-1",
    )
    command = client.containers.run.call_args.kwargs["command"][2]
    assert "ln -sfn model /dst/t-up/artifacts/latest" in command


def test_staged_size_counts_files_once_without_following_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results" / "t-up"
    model = source / "artifacts" / "model"
    model.mkdir(parents=True)
    (model / "weights.bin").write_bytes(b"x" * 1000)
    (model / "weights.alias").hardlink_to(model / "weights.bin")
    (source / "artifacts" / "notes.txt").write_bytes(b"y" * 10)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 5000)
    (source / "artifacts" / "escape").symlink_to(outside)
    (source / "results.json").write_bytes(b"{}")

    every = ResolvedSSHInput("up", "t-up", source, "/mnt/in")
    assert inputs_module.staged_size_bytes(every) == 1000 + 10 + len(str(outside)) + 2
    selected = ResolvedSSHInput(
        "up", "t-up", source, "/mnt/in", artifact_paths=["notes.txt"]
    )
    assert inputs_module.staged_size_bytes(selected) == 2 + 10


def test_build_mount_plan_records_the_staged_input_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task_message()
    cfg = SSHConfig.from_spec(cast(SSHSpecStrict, task.spec), DEFAULT_WORKER_CONFIG)
    worker_cfg = _worker_config(tmp_path)
    upstream = worker_cfg.results_dir / "task-pre"
    upstream.mkdir(parents=True)
    (upstream / "results.json").write_bytes(b"x" * 300)
    backend = DockerSessionBackend(worker_cfg)
    monkeypatch.setattr(backend, "_stage_inputs_in_volume", lambda *args: "vol")

    resolved_inputs = inputs_module.resolve_inputs(task, cfg, worker_cfg.results_dir)
    plan = backend._build_mount_plan(
        MagicMock(), tmp_path / "out", resolved_inputs, cfg, "session-1", "worker-1"
    )

    assert plan.staged_input_bytes == 300
