"""Shared dummy constructors for test modules."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from shared.tasks import TaskType
from shared.tasks.worker_message import TaskEnvelopeStrict, WorkerTaskMessage
from worker.config import WorkerConfig

DEFAULT_WORKER_CONFIG: Final[WorkerConfig] = WorkerConfig(
    worker_token="test",
    supervisor_grpc_target="localhost:50051",
    supervisor_grpc_tls_ca_b64=None,
    results_dir=Path("/tmp/test-results"),
    results_mount_source=None,
    hb_interval_sec=30,
    hb_ttl_sec=120,
    hb_file=Path("/tmp/test-hb"),
    namespace="test",
    cluster="test",
    alias="test-worker",
    tags=[],
    log_level="WARNING",
    cost_per_hour=0.0,
    network_bandwidth_bytes_per_sec=None,
    executor_idle_cleanup_sec=None,
    enable_mp_executors=False,
    docker_gpu_runtime=None,
)


def make_worker_config(**overrides: Any) -> WorkerConfig:
    """Build a WorkerConfig with sensible test defaults.

    Override any field via kwargs, e.g.:
        make_worker_config(results_dir=tmp_path / "out", enable_mp_executors=True)
    """
    return replace(DEFAULT_WORKER_CONFIG, **overrides)


def make_live_worker_config(tmp_path: Path, **overrides: Any) -> WorkerConfig:
    """Build a WorkerConfig configured like a live worker — paths under
    ``tmp_path``, MP executors enabled, INFO logs, fixed alias.

    Use this for tests that exercise executor lifecycle (running, heartbeats,
    artifact persistence). For lightweight construction-only tests, call
    ``make_worker_config()`` directly.
    """
    return make_worker_config(
        results_dir=tmp_path / "worker-results",
        hb_file=tmp_path / "worker.hb",
        alias="worker-1",
        log_level="INFO",
        cost_per_hour=1.0,
        executor_idle_cleanup_sec=60.0,
        enable_mp_executors=True,
        **overrides,
    )


def make_worker_task_message(
    spec: Any,
    task_type: TaskType | None = None,
    task_id: str = "tsk-test",
    workflow_id: str = "wfl-test",
    owner_id: str = "usr-test",
    assigned_worker: str = "wrk-test",
    dispatched_at: str = "2026-04-28T00:00:00Z",
    api_version: str = "flowmesh/v1",
    kind: str = "Task",
    **overrides: Any,
) -> WorkerTaskMessage:
    """Wrap a spec in a WorkerTaskMessage with sensible test defaults."""
    return WorkerTaskMessage(
        task_id=task_id,
        workflow_id=workflow_id,
        owner_id=owner_id,
        assigned_worker=assigned_worker,
        dispatched_at=dispatched_at,
        task_type=task_type,
        task=TaskEnvelopeStrict(apiVersion=api_version, kind=kind, spec=spec),
        **overrides,
    )
