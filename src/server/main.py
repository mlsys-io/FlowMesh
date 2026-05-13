import argparse
import atexit
import importlib
import inspect
import os
import threading
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from fastapi import FastAPI
from lumid_hooks import HookBindings

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "server"
    sys.modules.setdefault("server.main", sys.modules[__name__])

from shared._version import FLOWMESH_VERSION

from .auth import resolve_system_principal
from .clients import RedisClient
from .config import NodeRole, ServerConfig
from .dispatcher.factory import create_dispatcher
from .hooks import register
from .registries import WorkerRegistry, WorkflowRegistry
from .registries.node import NodeRegistry
from .routers import docs, health, v1
from .services.cleanup import clear_redis_state
from .services.log_archiver import TaskLogArchiver
from .services.metrics import MetricsRecorder
from .services.monitoring import EventMonitor
from .services.ssh_audit import SshAuditService
from .services.ssh_forward import SshForwardService
from .services.watchdog import WorkerWatchdog
from .supervisor import WorkerSupervisor
from .task.runtime import TaskRuntime
from .utils.logging import get_logger

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

config = ServerConfig.from_env()
NODE_ROLE = config.identity.role
IS_ROOT_NODE = NODE_ROLE is NodeRole.ROOT

if NODE_ROLE is NodeRole.WORKER and not config.worker_management.enabled:
    raise SystemExit("Worker node role requires ENABLE_SUPERVISOR=true")

# --------------------------------------------------------------------------- #
# Shared services (all node roles)
# --------------------------------------------------------------------------- #

logger = get_logger(
    name="server",
    log_file=config.logging.file,
    max_bytes=config.logging.max_bytes,
    backup_count=config.logging.backup_count,
    level=config.logging.level,
)

# Result & metrics directories
RESULTS_DIR = config.results_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

assert config.metrics.dir is not None
METRICS_DIR = config.metrics.dir

# Redis connections
REDIS_CLIENT = RedisClient(
    control_url=config.redis.control_url,
    telemetry_url=config.redis.telemetry_url,
    logger=logger,
    acl_enabled=config.redis.acl_enabled,
    username=config.redis.username,
    password=config.redis.password,
    tls_ca_file=config.redis.tls_ca_file,
)

NODE_REGISTRY = NodeRegistry(REDIS_CLIENT, logger)

METRICS_RECORDER = MetricsRecorder(
    METRICS_DIR,
    logger,
    enable_density_plot=config.metrics.enable_density_plot,
    density_bucket_seconds=config.metrics.density_bucket_sec,
)

SUPERVISOR: WorkerSupervisor | None = None
if config.worker_management.enabled:
    SUPERVISOR = WorkerSupervisor(
        identity=config.identity,
        redis=config.redis,
        grpc=config.grpc,
        worker_management=config.worker_management,
        logging_config=config.logging,
        logger=logger,
    )

# --------------------------------------------------------------------------- #
# Root node services (orchestrator)
# --------------------------------------------------------------------------- #

WORKFLOW_REGISTRY = None
WORKER_REGISTRY = None
RUNTIME = None
DISPATCHER = None
SSH_AUDIT_SERVICE = None
SSH_FORWARD_SERVICE = None
WATCHDOG = None
EVENT_MONITOR = None
LOG_ARCHIVER = None

if IS_ROOT_NODE:
    WORKFLOW_REGISTRY = WorkflowRegistry(REDIS_CLIENT)
    WORKER_REGISTRY = WorkerRegistry(REDIS_CLIENT)
    RUNTIME = TaskRuntime(WORKFLOW_REGISTRY, WORKER_REGISTRY, logger)

    DISPATCHER = create_dispatcher(
        config.dispatch,
        RUNTIME,
        WORKER_REGISTRY,
        RESULTS_DIR,
        logger=logger,
        metrics_recorder=METRICS_RECORDER,
    )

    _ssh_cfg = config.ssh_forward
    if _ssh_cfg.audit_enabled:
        SSH_AUDIT_SERVICE = SshAuditService(REDIS_CLIENT)

    if _ssh_cfg.enabled:
        SSH_FORWARD_SERVICE = SshForwardService(
            redis_client=REDIS_CLIENT,
            node_registry=NODE_REGISTRY,
            worker_registry=WORKER_REGISTRY,
            ssh_audit=SSH_AUDIT_SERVICE,
            bind_host=_ssh_cfg.bind_host,
            public_host=_ssh_cfg.public_host,
            port_start=_ssh_cfg.port_start,
            port_end=_ssh_cfg.port_end,
            logger=logger,
        )

    WATCHDOG = WorkerWatchdog(
        REDIS_CLIENT.sync,
        WORKER_REGISTRY,
        RUNTIME,
        DISPATCHER,
        logger,
        enabled=config.watchdog.enabled,
        check_interval=config.watchdog.check_interval,
        grace_seconds=config.watchdog.grace_sec,
    )

    EVENT_MONITOR = EventMonitor(
        redis_client=REDIS_CLIENT.sync,
        logger=logger,
        runtime=RUNTIME,
        dispatcher=DISPATCHER,
        worker_registry=WORKER_REGISTRY,
        node_registry=NODE_REGISTRY,
        metrics_recorder=METRICS_RECORDER,
        watchdog=WATCHDOG,
        ssh_proxy_enabled=config.ssh_forward.proxy_enabled,
        ssh_forward=SSH_FORWARD_SERVICE,
        results_dir=RESULTS_DIR,
        log_stream_ttl_sec=config.log_stream.ttl_sec,
    )

    LOG_ARCHIVER = TaskLogArchiver(
        redis=REDIS_CLIENT.sync,
        runtime=RUNTIME,
        results_dir=RESULTS_DIR,
        logger=logger,
        flush_interval_sec=config.log_stream.archive_flush_interval_sec,
        flush_max_entries=config.log_stream.archive_flush_max_entries,
    )

# --------------------------------------------------------------------------- #
# Metrics export & cleanup hooks
# --------------------------------------------------------------------------- #


def _export_metrics_on_exit() -> None:
    try:
        METRICS_RECORDER.finalize_density_series()
        result = METRICS_RECORDER.export_final_report()
        report = result.get("report") or {}
        summary = METRICS_RECORDER.format_report(report)
        if summary:
            logger.info(summary)
        path = result.get("path")
        if path:
            logger.info("Metrics report saved to %s", path)
    except Exception as exc:
        logger.warning("Failed to export metrics summary: %s", exc)


if IS_ROOT_NODE:
    atexit.register(clear_redis_state, REDIS_CLIENT, logger)
atexit.register(_export_metrics_on_exit)

# --------------------------------------------------------------------------- #
# Background threads (root node only)
# --------------------------------------------------------------------------- #

STOP_EVENT = threading.Event()
BACKGROUND_THREADS: list[threading.Thread] = []


def _start_root_threads() -> None:
    """Start orchestrator background threads. Only called on root nodes."""
    assert DISPATCHER is not None
    assert LOG_ARCHIVER is not None
    assert WATCHDOG is not None

    dispatcher = DISPATCHER
    dispatch_thread = threading.Thread(
        target=lambda: dispatcher.dispatch_loop(STOP_EVENT, poll_interval=1.0),
        name="dispatch-loop",
        daemon=True,
    )
    dispatch_thread.start()
    BACKGROUND_THREADS.append(dispatch_thread)

    log_archiver_thread = threading.Thread(
        target=LOG_ARCHIVER.run, args=(STOP_EVENT,), name="log-archiver", daemon=True
    )
    log_archiver_thread.start()
    BACKGROUND_THREADS.append(log_archiver_thread)

    watchdog_thread = WATCHDOG.start(STOP_EVENT)
    if watchdog_thread and watchdog_thread not in BACKGROUND_THREADS:
        BACKGROUND_THREADS.append(watchdog_thread)

    node_registry_thread = NODE_REGISTRY.start()
    BACKGROUND_THREADS.append(node_registry_thread)


def _stop_background() -> None:
    STOP_EVENT.set()
    NODE_REGISTRY.shutdown()
    if RUNTIME is not None:
        RUNTIME.shutdown()
    for thread in BACKGROUND_THREADS:
        thread.join(timeout=2.0)
    BACKGROUND_THREADS.clear()
    if IS_ROOT_NODE:
        clear_redis_state(REDIS_CLIENT, logger)


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #

openapi_tags = [
    {"name": "Health", "description": "Service health and readiness endpoints."},
    {"name": "Documentations", "description": "API documentation endpoints."},
    {"name": "Workflows", "description": "Workflow submission and lifecycle."},
    {"name": "Tasks", "description": "Task inspection and status."},
    {"name": "Results", "description": "Task results and artifact handling."},
    {"name": "Workers", "description": "Worker pool operations and metadata."},
    {"name": "Nodes", "description": "Node registry and worker control."},
    {"name": "SSH", "description": "SSH proxy endpoint for task connectivity."},
    {"name": "System", "description": "System metrics and admin operations."},
    {"name": "Stack", "description": "Local worker lifecycle management."},
]

app = FastAPI(
    title="FlowMesh Server",
    version=FLOWMESH_VERSION,
    openapi_tags=openapi_tags,
    docs_url=None,
    redoc_url=None,
)


async def _load_plugins(stack: AsyncExitStack) -> None:
    """Load FLOWMESH_PLUGINS modules and drain their `HookBindings` into the
    server's runtime registries.

    A plugin's `install()` is either:
      - a sync function returning a `HookBindings`, or
      - an `@asynccontextmanager async def` yielding a `HookBindings` (the
        ctx manager registers on enter, cleans up on exit; e.g. closes a
        SQLAlchemy engine).
    """
    for plugin_name in config.plugins:
        mod = importlib.import_module(plugin_name)
        rv = mod.install()
        if hasattr(rv, "__aenter__"):
            bindings = await stack.enter_async_context(rv)
        elif inspect.iscoroutine(rv):
            bindings = await rv
        else:
            bindings = rv
        if not isinstance(bindings, HookBindings):
            raise TypeError(
                f"{plugin_name}.install() must return HookBindings, got "
                f"{type(bindings).__name__}"
            )
        register(bindings)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    async with AsyncExitStack() as plugin_stack:
        await _load_plugins(plugin_stack)

        # --- System principal resolution ---
        system_principal = await resolve_system_principal(
            config.identity.api_key, logger
        )
        app.state.system_principal = system_principal

        # --- Root-only startup ---
        if IS_ROOT_NODE:
            if SSH_FORWARD_SERVICE is not None:
                await SSH_FORWARD_SERVICE.start()
            _start_root_threads()
            if EVENT_MONITOR is not None:
                EVENT_MONITOR.start()

        # --- Supervisor (all nodes with worker management) ---
        if SUPERVISOR is not None:
            await SUPERVISOR.start(system_principal)
            app.state.node_id = SUPERVISOR.node_id
            # Tell EventMonitor which node this server belongs to so that it can wait
            # for the supervisor's SV_UNREGISTER event on shutdown.
            if EVENT_MONITOR is not None:
                EVENT_MONITOR.set_own_node(SUPERVISOR.node_id)

        try:
            yield
        finally:
            # --- Supervisor shutdown ---
            if SUPERVISOR is not None:
                await SUPERVISOR.stop()
                app.state.node_id = None

            # --- Event monitor shutdown ---
            if EVENT_MONITOR is not None:
                await EVENT_MONITOR.stop()

            # --- Root-only shutdown ---
            _stop_background()
            if SSH_FORWARD_SERVICE is not None:
                await SSH_FORWARD_SERVICE.stop()


app.router.lifespan_context = _lifespan

# --------------------------------------------------------------------------- #
# App state & routers
# --------------------------------------------------------------------------- #

# Shared state (all nodes)
app.state.logger = logger
app.state.node_role = NODE_ROLE
app.state.node_registry = NODE_REGISTRY
app.state.metrics_recorder = METRICS_RECORDER
app.state.redis_client = REDIS_CLIENT
app.state.results_dir = RESULTS_DIR
app.state.supervisor = SUPERVISOR
# resolved during lifespan startup
app.state.node_id = None
app.state.system_principal = None

# Root-only state (None on worker nodes)
app.state.runtime = RUNTIME
app.state.dispatcher = DISPATCHER
app.state.workflow_registry = WORKFLOW_REGISTRY
app.state.worker_registry = WORKER_REGISTRY
app.state.watchdog = WATCHDOG
app.state.event_monitor = EVENT_MONITOR
app.state.ssh_forward = SSH_FORWARD_SERVICE
app.state.ssh_audit = SSH_AUDIT_SERVICE
app.state.ssh_proxy_enabled = config.ssh_forward.proxy_enabled and IS_ROOT_NODE

# Routers — shared
app.include_router(health.router)
app.include_router(docs.router)

v1_prefix = "/api/v1"

# Routers — root only
if IS_ROOT_NODE:
    app.include_router(v1.workflows.router, prefix=v1_prefix)
    app.include_router(v1.workers.router, prefix=v1_prefix)
    app.include_router(v1.nodes.router, prefix=v1_prefix)
    app.include_router(v1.tasks.router, prefix=v1_prefix)
    app.include_router(v1.results.router, prefix=v1_prefix)
    app.include_router(v1.ssh.router, prefix=v1_prefix)
    app.include_router(v1.system.router, prefix=v1_prefix)
    app.include_router(v1.traces.router, prefix=v1_prefix)

# Routers — supervisor (any node with worker management)
if config.worker_management.enabled:
    app.include_router(v1.stack.router, prefix=v1_prefix)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the FlowMesh server.")
    parser.add_argument(
        "--host",
        help="Bind address (defaults to SERVER_APP_HOST env or 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port (defaults to SERVER_APP_PORT, then PORT env, else 8000).",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable auto-reload.",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload.",
    )
    parser.set_defaults(reload=None)
    parser.add_argument(
        "--log-level",
        help="Uvicorn log level (defaults to LOG_LEVEL).",
    )

    args = parser.parse_args(argv)

    host_value = args.host or config.http.host
    port_value = args.port if args.port is not None else config.http.port

    reload_enabled = config.http.reload if args.reload is None else args.reload
    log_level = args.log_level or config.http.log_level

    uvicorn_app = app if not reload_enabled else "server.main:app"

    uvicorn.run(
        uvicorn_app,
        host=host_value,
        port=port_value,
        reload=reload_enabled,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
