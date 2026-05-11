"""Model validation and round-trip tests for SDK Pydantic models."""

from datetime import UTC, datetime

import pytest
from flowmesh.models import (
    ActiveWaitBreakdown,
    AssetSummary,
    CriticalPathSummary,
    E2EBreakdown,
    EventSummary,
    LineageEdge,
    LogQueryResponse,
    Node,
    NodeWorkerInfo,
    OkResponse,
    ProfileSummary,
    TaskInfo,
    TaskTiming,
    TaskUsage,
    WorkerHardware,
    WorkerInfo,
    Workflow,
    WorkflowSubmitResponse,
    WorkflowValidateResponse,
)
from flowmesh.models.ssh import SSHConnectionInfo
from pydantic import BaseModel

from server.governance.analyzer import ActiveWaitBreakdown as SrvActiveWaitBreakdown
from server.governance.analyzer import AssetSummary as SrvAssetSummary
from server.governance.analyzer import CriticalPathSummary as SrvCriticalPathSummary
from server.governance.analyzer import E2EBreakdown as SrvE2EBreakdown
from server.governance.analyzer import EventSummary as SrvEventSummary
from server.governance.analyzer import LineageEdge as SrvLineageEdge
from server.governance.analyzer import ProfileSummary as SrvProfileSummary
from server.governance.analyzer import TaskTiming as SrvTaskTiming
from server.registries.node import Node as SrvNode
from server.registries.worker import Worker as SrvWorker
from server.registries.worker import WorkerInfo as SrvWorkerInfo
from server.registries.workflow import Workflow as SrvWorkflow
from server.registries.workflow import WorkflowStatus as SrvWorkflowStatus
from server.schemas.common import OkResponse as SrvOkResponse
from server.schemas.logs import LogEntry as SrvLogEntry
from server.schemas.logs import LogEvent as SrvLogEvent
from server.schemas.logs import LogQueryResponse as SrvLogQueryResponse
from server.schemas.node import CPUInfo as SrvCPUInfo
from server.schemas.node import GpuInfo as SrvGpuInfo
from server.schemas.node import GpuPlatformInfo as SrvGpuPlatformInfo
from server.schemas.node import HostInfo as SrvHostInfo
from server.schemas.node import MemoryInfo as SrvMemoryInfo
from server.schemas.node import NetworkInfo as SrvNetworkInfo
from server.schemas.node import NodeWorkerInfo as SrvNodeWorkerInfo
from server.schemas.node import NodeWorkerStatus as SrvNodeWorkerStatus
from server.schemas.node import StorageInfo as SrvStorageInfo
from server.schemas.node import WorkerHardware as SrvWorkerHardware
from server.schemas.ssh import SSHConnectionInfo as SrvSSHConnectionInfo
from server.schemas.workflow import WorkflowSubmitResponse as SrvWorkflowSubmitResponse
from server.schemas.workflow import (
    WorkflowSubmitTaskEntry as SrvWorkflowSubmitTaskEntry,
)
from server.schemas.workflow import (
    WorkflowValidateResponse as SrvWorkflowValidateResponse,
)
from server.schemas.workflow import (
    WorkflowValidateTaskEntry as SrvWorkflowValidateTaskEntry,
)
from server.task.models import TaskInfo as SrvTaskInfo
from server.task.models import TaskUsage as SrvTaskUsage
from shared.schemas.worker import WorkerStatus as SharedWorkerStatus
from shared.tasks.envelope import TaskEnvelopeTemplate
from shared.tasks.specs.misc import EchoSpecTemplate
from shared.tasks.task_type import TaskType as SharedTaskType
from shared.tasks.worker_message import CPUInfo as SharedCPUInfo
from shared.tasks.worker_message import GpuInfo as SharedGpuInfo
from shared.tasks.worker_message import GpuPlatformInfo as SharedGpuPlatformInfo
from shared.tasks.worker_message import HardwareUsage as SharedHardwareUsage
from shared.tasks.worker_message import MemoryInfo as SharedMemoryInfo
from shared.tasks.worker_message import NetworkInfo as SharedNetworkInfo
from shared.tasks.worker_message import WorkerHardware as SharedWorkerHardware

# ------------------------------------------------------------------ #
# Server-side model instances — single source of truth for test payloads
# ------------------------------------------------------------------ #

_SRV_HARDWARE = SrvWorkerHardware(
    cpu=SrvCPUInfo(logical_cores=16, model="AMD EPYC", arch="x86_64"),
    memory=SrvMemoryInfo(total_bytes=68719476736),
    gpu=SrvGpuPlatformInfo(
        devices=[
            SrvGpuInfo(
                index=0, name="A100", memory_total_bytes=85899345920, uuid="GPU-0"
            )
        ],
    ),
    network=SrvNetworkInfo(ip="10.0.0.1"),
    storage=SrvStorageInfo(disk_space=500.0),
    host=SrvHostInfo(os_version="Ubuntu 22.04"),
)

_SRV_WORKFLOW = SrvWorkflow(
    workflow_id="wf-abc123",
    task_ids=["t-1", "t-2"],
    submitted_at="2025-01-15T10:30:00Z",
    updated_at="2025-01-15T10:31:00Z",
    status=SrvWorkflowStatus.DONE,
    dispatched_tasks=["t-1", "t-2"],
    completed_tasks=["t-1", "t-2"],
    failed_tasks=[],
    cancelled_tasks=[],
)

_SRV_TASK_USAGE = SrvTaskUsage(
    started_at="2025-01-15T10:30:00Z",
    finished_at="2025-01-15T10:31:00Z",
    runtime_sec=60.0,
    hardware=SharedHardwareUsage(
        gpu=SharedGpuPlatformInfo(driver_version=None, cuda_version=None, devices=[])
    ),
    cost_per_hour=2.5,
    total_cost=0.042,
    status="DONE",
)

_ECHO_SPEC = EchoSpecTemplate(taskType=SharedTaskType.ECHO)
_TASK_ENVELOPE = TaskEnvelopeTemplate(
    apiVersion="flowmesh/v1", kind="Task", spec=_ECHO_SPEC
)

_SRV_TASK_INFO = SrvTaskInfo(
    task_id="t-abc",
    workflow_id="wf-abc",
    owner_id="usr-1",
    raw_yaml="apiVersion: flowmesh/v1\nkind: Task",
    task=_TASK_ENVELOPE,
    status="DONE",
    task_type="echo",
    submitted_at="2025-01-15T10:30:00Z",
    submitted_ts=1705312200.0,
    usages=[_SRV_TASK_USAGE],
    attempts=1,
    max_attempts=3,
    load=1,
    depends_on=[],
    pending_dependencies=[],
    dependents=["t-def"],
    completed=True,
    failed=False,
)

_SHARED_HARDWARE = SharedWorkerHardware(
    cpu=SharedCPUInfo(logical_cores=16, model="AMD EPYC"),
    memory=SharedMemoryInfo(total_bytes=68719476736),
    gpu=SharedGpuPlatformInfo(
        driver_version=None,
        cuda_version=None,
        devices=[
            SharedGpuInfo(
                index=0, name="A100", uuid="GPU-0", memory_total_bytes=85899345920
            )
        ],
    ),
    network=SharedNetworkInfo(ip="10.0.0.1", bandwidth_bytes_per_sec=None),
)

_SRV_WORKER = SrvWorker(
    id="w-1",
    alias="gpu-a100-01",
    namespace="default",
    cluster="us-west",
    node_id="g-1",
    node_alias="node-01",
    status=SharedWorkerStatus.IDLE,
    hardware=_SHARED_HARDWARE,
    tags=["gpu", "a100"],
)

_SRV_WORKER_INFO = SrvWorkerInfo(
    **_SRV_WORKER.model_dump(),
    stale=False,
)

_SRV_EVENT_SUMMARY = SrvEventSummary(
    event_type=["model load", "generation"],
    count=[1, 2],
    total_seconds=[53.39, 1.23],
    avg_seconds=[53.39, 0.62],
    min_seconds=[53.39, 0.39],
    max_seconds=[53.39, 0.84],
)

_SRV_NETWORK_SUMMARY = SrvEventSummary(
    event_type=["dump to storage"],
    count=[2],
    total_seconds=[0.001],
    avg_seconds=[0.001],
    min_seconds=[0.000],
    max_seconds=[0.001],
)

_SRV_TASK_TIMING = SrvTaskTiming(
    data_id="tsk-a",
    start_time=datetime(2026, 4, 30, 14, 0, 1, tzinfo=UTC),
    end_time=datetime(2026, 4, 30, 14, 0, 55, tzinfo=UTC),
    duration_seconds=54.0,
    queuing_delay_seconds=0.5,
    parent_data_ids=["tsk-up-a"],
    blocking_parent_data_id="tsk-up-a",
)

_SRV_ACTIVE_WAIT = SrvActiveWaitBreakdown(
    data_id=["tsk-a", "tsk-b"],
    active_seconds=[54.0, 0.84],
    wait_seconds=[0.0, 0.5],
)

_SRV_E2E = SrvE2EBreakdown(
    hardware_summary=_SRV_EVENT_SUMMARY,
    network_summary=_SRV_NETWORK_SUMMARY,
    workflow_duration_seconds=55.05,
    total_network_seconds=0.001,
)

_SRV_CP = SrvCriticalPathSummary(
    path=["tsk-a", "tsk-b"],
    critical_path_seconds=55.05,
    active_wait_breakdown=_SRV_ACTIVE_WAIT,
    hardware_summary=_SRV_EVENT_SUMMARY,
    network_summary=_SRV_NETWORK_SUMMARY,
    total_network_seconds=0.001,
)

_SRV_PROFILE = SrvProfileSummary(
    workflow_id="wfl-abc",
    event_count=18,
    data_ids=["tsk-a", "tsk-b"],
    assets=[
        SrvAssetSummary(
            asset_guid="g-1",
            latest_data_id="tsk-a",
            latest_version=1,
            user_id="alice",
            versions=1,
            created_at="2026-04-30T14:00:55Z",
        )
    ],
    lineage=[
        SrvLineageEdge(
            data_id="tsk-b",
            source_data_id="tsk-a",
            created_at="2026-04-30T14:00:55Z",
        )
    ],
    e2e_breakdown=_SRV_E2E,
    per_data_id=[_SRV_TASK_TIMING],
    critical_path=_SRV_CP,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _dump(server_obj: BaseModel) -> dict:
    """Dump a server Pydantic model to a JSON-compatible dict."""
    return server_obj.model_dump(mode="json")


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestWorkflowModels:
    def test_workflow_validate(self) -> None:
        wf = Workflow.model_validate(_dump(_SRV_WORKFLOW))
        assert wf.workflow_id == "wf-abc123"
        assert wf.status == "DONE"

    def test_workflow_submit_response(self) -> None:
        server = SrvWorkflowSubmitResponse(
            ok=True,
            workflow_id="wf-1",
            count=2,
            tasks=[
                SrvWorkflowSubmitTaskEntry(task_id="t-1", status="PENDING"),
                SrvWorkflowSubmitTaskEntry(task_id="t-2", depends_on=["t-1"]),
            ],
        )
        resp = WorkflowSubmitResponse.model_validate(_dump(server))
        assert resp.ok is True
        assert resp.count == 2

    def test_workflow_validate_response(self) -> None:
        server = SrvWorkflowValidateResponse(
            ok=True,
            count=1,
            tasks=[
                SrvWorkflowValidateTaskEntry(
                    task_id="t-mock", graph_node_name="step1", depends_on=[]
                )
            ],
        )
        resp = WorkflowValidateResponse.model_validate(_dump(server))
        assert resp.ok is True


class TestTaskModels:
    def test_task_info_validate(self) -> None:
        task = TaskInfo.model_validate(_dump(_SRV_TASK_INFO))
        assert task.task_id == "t-abc"
        assert task.completed is True

    def test_task_usage(self) -> None:
        usage = TaskUsage.model_validate(_dump(_SRV_TASK_USAGE))
        assert usage.runtime_sec == 60.0
        assert usage.total_cost == pytest.approx(0.042)


class TestWorkerModels:
    def test_worker_info_validate(self) -> None:
        w = WorkerInfo.model_validate(_dump(_SRV_WORKER_INFO))
        assert w.id == "w-1"
        assert w.status == "IDLE"
        assert w.hardware is not None
        assert w.hardware.cpu is not None
        assert w.hardware.cpu.logical_cores == 16

    def test_worker_info_accepts_string_tags(self) -> None:
        payload = _dump(_SRV_WORKER_INFO)
        payload["tags"] = "gpu,a100"
        w = WorkerInfo.model_validate(payload)
        assert w.tags == ["gpu", "a100"]

    def test_worker_hardware_roundtrip(self) -> None:
        hw = WorkerHardware.model_validate(_dump(_SRV_HARDWARE))
        dumped = hw.model_dump(mode="json")
        hw2 = WorkerHardware.model_validate(dumped)
        assert hw2.cpu is not None and hw.cpu is not None
        assert hw2.cpu.logical_cores == hw.cpu.logical_cores


class TestNodeModels:
    def test_node_validate(self) -> None:
        node = SrvNode(
            id="g-1",
            namespace="default",
            cluster="us-west",
            alias="node-01",
            tags=["gpu"],
        )
        payload = {k: getattr(node, k) for k in SrvNode.model_fields}
        g = Node.model_validate(payload)
        assert g.id == "g-1"

    def test_node_worker_info(self) -> None:
        node = SrvNodeWorkerInfo(
            id="w-1",
            name="worker-a100",
            namespace="default",
            cluster="us-west",
            node_id="g-1",
            node_alias="node-01",
            provider="docker",
            status=SrvNodeWorkerStatus.IDLE,
        )
        w = NodeWorkerInfo.model_validate(_dump(node))
        assert w.name == "worker-a100"
        assert w.node_id == "g-1"


class TestMiscModels:
    def test_ok_response(self) -> None:
        server = SrvOkResponse(ok=True)
        r = OkResponse.model_validate(_dump(server))
        assert r.ok is True

    def test_log_query_response(self) -> None:
        server = SrvLogQueryResponse(
            entries=[
                SrvLogEntry(
                    cursor="1705312200000-0",
                    event=SrvLogEvent(
                        ts="2025-01-15T10:30:00Z",
                        message="Starting task",
                        level="INFO",
                    ),
                )
            ],
            next_cursor="1705312200001-0",
        )
        r = LogQueryResponse.model_validate(_dump(server))
        assert len(r.entries) == 1
        assert r.entries[0].event.message == "Starting task"

    def test_ssh_connection_info(self) -> None:
        server = SrvSSHConnectionInfo(
            connection_id="conn-1",
            access_mode="proxy",
            task_id="t-1",
            connected_at="2025-01-15T10:30:00Z",
        )
        r = SSHConnectionInfo.model_validate(_dump(server))
        assert r.access_mode == "proxy"


class TestTraceModels:
    def test_asset_summary(self) -> None:
        server = _SRV_PROFILE.assets[0]
        r = AssetSummary.model_validate(_dump(server))
        assert r.asset_guid == "g-1"
        assert r.latest_data_id == "tsk-a"
        assert r.latest_version == 1
        assert r.user_id == "alice"
        assert r.versions == 1

    def test_lineage_edge(self) -> None:
        server = _SRV_PROFILE.lineage[0]
        r = LineageEdge.model_validate(_dump(server))
        assert r.data_id == "tsk-b"
        assert r.source_data_id == "tsk-a"

    def test_event_summary_parallel_lists_align(self) -> None:
        r = EventSummary.model_validate(_dump(_SRV_EVENT_SUMMARY))
        n = len(r.event_type)
        assert n == 2
        assert all(
            len(field) == n
            for field in (
                r.count,
                r.total_seconds,
                r.avg_seconds,
                r.min_seconds,
                r.max_seconds,
            )
        )
        assert r.event_type[0] == "model load"
        assert r.total_seconds[0] == pytest.approx(53.39)

    def test_e2e_breakdown(self) -> None:
        r = E2EBreakdown.model_validate(_dump(_SRV_E2E))
        assert r.workflow_duration_seconds == pytest.approx(55.05)
        assert "model load" in r.hardware_summary.event_type
        assert "dump to storage" in r.network_summary.event_type

    def test_active_wait_breakdown(self) -> None:
        r = ActiveWaitBreakdown.model_validate(_dump(_SRV_ACTIVE_WAIT))
        assert r.data_id == ["tsk-a", "tsk-b"]
        assert r.wait_seconds[1] == pytest.approx(0.5)

    def test_task_timing_datetime_round_trip(self) -> None:
        r = TaskTiming.model_validate(_dump(_SRV_TASK_TIMING))
        assert r.data_id == "tsk-a"
        assert r.start_time == _SRV_TASK_TIMING.start_time
        assert r.end_time == _SRV_TASK_TIMING.end_time
        assert r.queuing_delay_seconds == pytest.approx(0.5)
        assert r.blocking_parent_data_id == "tsk-up-a"

    def test_critical_path_summary(self) -> None:
        r = CriticalPathSummary.model_validate(_dump(_SRV_CP))
        assert r.path == ["tsk-a", "tsk-b"]
        assert r.critical_path_seconds == pytest.approx(55.05)
        assert r.active_wait_breakdown.data_id == ["tsk-a", "tsk-b"]

    def test_profile_summary(self) -> None:
        r = ProfileSummary.model_validate(_dump(_SRV_PROFILE))
        assert r.workflow_id == "wfl-abc"
        assert r.event_count == 18
        assert r.data_ids == ["tsk-a", "tsk-b"]
        assert len(r.assets) == 1
        assert len(r.lineage) == 1
        assert r.e2e_breakdown.workflow_duration_seconds == pytest.approx(55.05)
        assert r.critical_path is not None
        assert r.critical_path.path == ["tsk-a", "tsk-b"]

    def test_profile_summary_critical_path_optional(self) -> None:
        server = _SRV_PROFILE.model_copy(update={"critical_path": None})
        r = ProfileSummary.model_validate(_dump(server))
        assert r.critical_path is None

    def test_profile_summary_rejects_extra_fields(self) -> None:
        payload = _dump(_SRV_PROFILE)
        payload["unexpected"] = 1
        with pytest.raises(Exception):
            ProfileSummary.model_validate(payload)
