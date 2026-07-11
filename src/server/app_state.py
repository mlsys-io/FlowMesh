import logging
from pathlib import Path

from starlette.requests import HTTPConnection

from .clients import RedisClient
from .dispatcher import Dispatcher
from .hooks import PrincipalContext
from .registries import NodeRegistry, WorkerRegistry, WorkflowRegistry
from .services.metrics import MetricsRecorder
from .services.monitoring import EventMonitor
from .services.port_forward import PortForwardService
from .services.ssh_audit import SshAuditService
from .services.watchdog import WorkerWatchdog
from .supervisor.supervisor import WorkerSupervisor
from .task.runtime import TaskRuntime


def get_logger(conn: HTTPConnection) -> logging.Logger:
    return conn.app.state.logger


def get_runtime(conn: HTTPConnection) -> TaskRuntime:
    return conn.app.state.runtime


def get_dispatcher(conn: HTTPConnection) -> Dispatcher:
    return conn.app.state.dispatcher


def get_workflow_registry(conn: HTTPConnection) -> WorkflowRegistry:
    return conn.app.state.workflow_registry


def get_worker_registry(conn: HTTPConnection) -> WorkerRegistry:
    return conn.app.state.worker_registry


def get_node_registry(conn: HTTPConnection) -> NodeRegistry:
    return conn.app.state.node_registry


def get_metrics(conn: HTTPConnection) -> MetricsRecorder:
    return conn.app.state.metrics_recorder


def get_watchdog(conn: HTTPConnection) -> WorkerWatchdog:
    return conn.app.state.watchdog


def get_event_monitor(conn: HTTPConnection) -> EventMonitor:
    return conn.app.state.event_monitor


def get_results_dir(conn: HTTPConnection) -> Path:
    return conn.app.state.results_dir


def get_redis_client(conn: HTTPConnection) -> RedisClient:
    return conn.app.state.redis_client


def get_supervisor(conn: HTTPConnection) -> WorkerSupervisor:
    supervisor = conn.app.state.supervisor
    if supervisor is None:
        raise RuntimeError("Supervisor not initialized; worker management unavailable")
    return supervisor


def get_node_id(conn: HTTPConnection) -> str:
    node_id = conn.app.state.node_id
    if not isinstance(node_id, str) or not node_id:
        raise RuntimeError("node_id not assigned; supervisor not started")
    return node_id


def get_system_principal(conn: HTTPConnection) -> PrincipalContext:
    if conn.app.state.system_principal is None:
        raise RuntimeError("System principal not initialized")
    return conn.app.state.system_principal


def get_port_forward(conn: HTTPConnection) -> PortForwardService | None:
    return conn.app.state.port_forward


def get_ssh_audit(conn: HTTPConnection) -> SshAuditService | None:
    return conn.app.state.ssh_audit


def get_ssh_proxy_enabled(conn: HTTPConnection) -> bool:
    return conn.app.state.ssh_proxy_enabled
