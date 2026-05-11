"""Schema compatibility tests: ensure SDK models match server-side schemas.

Imports both server Pydantic models (via conftest stubs) and SDK models,
then compares ``model_fields`` to detect drift.
"""

import pytest

# SDK-side imports
from flowmesh.models import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    HostInfo,
    LogEntry,
    LogEvent,
    LogLevel,
    LogQueryResponse,
    LogStream,
    MemoryInfo,
    NetworkInfo,
    Node,
    NodeRegisterResponse,
    NodeWorkerInfo,
    OkResponse,
    ResultEnvelope,
    StorageInfo,
    TaskInfo,
    TaskType,
    TaskUsage,
    Worker,
    WorkerHardware,
    WorkerInfo,
    WorkerRegisterResponse,
    WorkerStatus,
    Workflow,
    WorkflowStatus,
    WorkflowSubmitResponse,
    WorkflowSubmitTaskEntry,
    WorkflowValidateResponse,
    WorkflowValidateTaskEntry,
)

# Server-side imports (stubs installed by conftest.py)
from server.registries.node import Node as SrvNode
from server.registries.worker import Worker as SrvWorker
from server.registries.worker import WorkerInfo as SrvWorkerInfo
from server.registries.workflow import Workflow as SrvWorkflow
from server.registries.workflow import WorkflowStatus as SrvWorkflowStatus
from server.schemas.common import OkResponse as SrvOkResponse
from server.schemas.logs import LogEntry as SrvLogEntry
from server.schemas.logs import LogEvent as SrvLogEvent
from server.schemas.logs import LogLevel as SrvLogLevel
from server.schemas.logs import LogQueryResponse as SrvLogQueryResponse
from server.schemas.logs import LogStream as SrvLogStream
from server.schemas.node import CPUInfo as SrvCPUInfo
from server.schemas.node import GpuInfo as SrvGpuInfo
from server.schemas.node import GpuPlatformInfo as SrvGpuPlatformInfo
from server.schemas.node import HostInfo as SrvHostInfo
from server.schemas.node import MemoryInfo as SrvMemoryInfo
from server.schemas.node import NetworkInfo as SrvNetworkInfo
from server.schemas.node import NodeRegisterResponse as SrvNodeRegisterResponse
from server.schemas.node import NodeWorkerInfo as SrvNodeWorkerInfo
from server.schemas.node import StorageInfo as SrvStorageInfo
from server.schemas.node import WorkerHardware as SrvWorkerHardware
from server.schemas.node import WorkerRegisterResponse as SrvWorkerRegisterResponse
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
from shared.schemas.result import ResultEnvelope as SrvResultEnvelope
from shared.tasks.task_type import TaskType as SrvTaskType

from .helpers import assert_enum_members_match, assert_fields_match

# ------------------------------------------------------------------ #
# Parametrized model-pair tests
# ------------------------------------------------------------------ #

MODEL_PAIRS = [
    # Workflow schemas
    (SrvWorkflowSubmitResponse, WorkflowSubmitResponse),
    (SrvWorkflowSubmitTaskEntry, WorkflowSubmitTaskEntry),
    (SrvWorkflowValidateResponse, WorkflowValidateResponse),
    (SrvWorkflowValidateTaskEntry, WorkflowValidateTaskEntry),
    # Workflow registry
    (SrvWorkflow, Workflow),
    # Task models (last_queue_ts is server-internal scheduling field)
    (SrvTaskUsage, TaskUsage),
    # Worker models
    (SrvWorker, Worker),
    (SrvWorkerInfo, WorkerInfo),
    # Node schemas
    (SrvNode, Node),
    (SrvNodeRegisterResponse, NodeRegisterResponse),
    (SrvNodeWorkerInfo, NodeWorkerInfo),
    # Hardware sub-models
    (SrvCPUInfo, CPUInfo),
    (SrvGpuInfo, GpuInfo),
    (SrvGpuPlatformInfo, GpuPlatformInfo),
    (SrvMemoryInfo, MemoryInfo),
    (SrvNetworkInfo, NetworkInfo),
    (SrvStorageInfo, StorageInfo),
    (SrvHostInfo, HostInfo),
    (SrvWorkerHardware, WorkerHardware),
    # Logs
    (SrvLogEvent, LogEvent),
    (SrvLogEntry, LogEntry),
    (SrvLogQueryResponse, LogQueryResponse),
    # Common
    (SrvOkResponse, OkResponse),
    # Results
    (SrvResultEnvelope, ResultEnvelope),
]


@pytest.mark.parametrize(
    "server_model,sdk_model",
    MODEL_PAIRS,
    ids=[f"{h.__name__}->{s.__name__}" for h, s in MODEL_PAIRS],
)
def test_model_fields_match(server_model: type, sdk_model: type) -> None:
    """SDK model must contain all fields from the server model."""
    assert_fields_match(server_model, sdk_model)


def test_task_info_fields() -> None:
    """TaskInfo: last_queue_ts is server-internal, skip it."""
    assert_fields_match(SrvTaskInfo, TaskInfo, skip_server_fields={"last_queue_ts"})


def test_worker_register_response_fields() -> None:
    assert_fields_match(SrvWorkerRegisterResponse, WorkerRegisterResponse)


# ------------------------------------------------------------------ #
# Enum compatibility tests
# ------------------------------------------------------------------ #

ENUM_PAIRS = [
    (SrvWorkflowStatus, WorkflowStatus),
    (SrvTaskType, TaskType),
    (SrvLogLevel, LogLevel),
    (SrvLogStream, LogStream),
]


@pytest.mark.parametrize(
    "server_enum,sdk_enum",
    ENUM_PAIRS,
    ids=[f"{h.__name__}->{s.__name__}" for h, s in ENUM_PAIRS],
)
def test_enum_members_match(server_enum: type, sdk_enum: type) -> None:
    """SDK enum must have the same members as the server enum."""
    assert_enum_members_match(server_enum, sdk_enum)


def test_worker_status_superset() -> None:
    """SDK WorkerStatus covers all expected values."""
    expected = {"STARTING", "IDLE", "BUSY", "STOPPING", "STOPPED", "UNKNOWN"}
    sdk_values = {m.value for m in WorkerStatus}
    assert expected <= sdk_values
