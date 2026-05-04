import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from shared.utils.parsing import parse_bool_env, parse_float_env, parse_int_env


class NodeRole(StrEnum):
    ROOT = "root"
    WORKER = "worker"


@dataclass
class LoggingConfig:
    file: str = "server.log"
    level: str = "INFO"
    max_bytes: int = 5_242_880
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        return cls(
            file=os.getenv("LOG_FILE", "server.log"),
            level=(
                os.getenv("SERVER_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
            ).upper(),
            max_bytes=parse_int_env("LOG_MAX_BYTES", 5_242_880),
            backup_count=parse_int_env("LOG_BACKUP_COUNT", 5),
        )


@dataclass
class RedisConfig:
    control_url: str = "redis://localhost:6379/0"
    telemetry_url: str = "redis://localhost:6379/0"
    acl_enabled: bool = False
    username: str = "admin"
    password: str = ""
    tls_ca_file: str | None = None

    @classmethod
    def from_env(cls) -> "RedisConfig":
        redis_url = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        tls_raw = os.getenv("REDIS_TLS_CA_FILE", "").strip()
        return cls(
            control_url=os.getenv("REDIS_CONTROL_URL") or redis_url,
            telemetry_url=os.getenv("REDIS_TELEMETRY_URL") or redis_url,
            acl_enabled=parse_bool_env("REDIS_ACL_ENABLED", False),
            username=os.getenv("REDIS_USERNAME", "admin"),
            password=os.getenv("REDIS_PASSWORD", ""),
            tls_ca_file=tls_raw or None,
        )


@dataclass
class HttpConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "HttpConfig":
        port_default = parse_int_env("PORT", 8000)
        return cls(
            host=os.getenv("SERVER_APP_HOST", "0.0.0.0"),
            port=parse_int_env(
                "SERVER_APP_PORT", parse_int_env("SERVER_HTTP_PORT", port_default)
            ),
            reload=parse_bool_env("SERVER_APP_RELOAD", False),
            log_level=(
                os.getenv("SERVER_APP_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "info"
            ).lower(),
        )


@dataclass
class GrpcConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    tls_ca_file: str = ""
    tls_cert_file: str = ""
    tls_key_file: str = ""

    @classmethod
    def from_env(cls) -> "GrpcConfig":
        return cls(
            host="0.0.0.0",
            port=int(os.getenv("SERVER_GRPC_PORT") or "50051"),
            tls_ca_file=(os.getenv("SERVER_GRPC_TLS_CA_FILE") or "").strip(),
            tls_cert_file=(os.getenv("SERVER_GRPC_TLS_CERT_FILE") or "").strip(),
            tls_key_file=(os.getenv("SERVER_GRPC_TLS_KEY_FILE") or "").strip(),
        )


@dataclass
class SshForwardConfig:
    enabled: bool = True
    proxy_enabled: bool = True
    audit_enabled: bool = True
    bind_host: str = "0.0.0.0"
    public_host: str = "localhost"
    port_start: int = 32000
    port_end: int = 32100

    @classmethod
    def from_env(cls) -> "SshForwardConfig":
        return cls(
            enabled=parse_bool_env("ENABLE_SERVER_SSH_FORWARD", True),
            proxy_enabled=parse_bool_env("ENABLE_SERVER_SSH_PROXY", True),
            audit_enabled=parse_bool_env("ENABLE_SERVER_SSH_CONNECTION_AUDIT", True),
            bind_host=os.getenv("SERVER_SSH_FORWARD_BIND_HOST", "0.0.0.0").strip(),
            public_host=os.getenv(
                "SERVER_SSH_FORWARD_PUBLIC_HOST", "localhost"
            ).strip(),
            port_start=parse_int_env("SERVER_SSH_FORWARD_PORT_START", 32000),
            port_end=parse_int_env("SERVER_SSH_FORWARD_PORT_END", 32100),
        )


@dataclass
class IdentityConfig:
    role: NodeRole = NodeRole.ROOT
    namespace: str = "flowmesh"
    cluster: str = "cluster"
    alias: str = "node"
    tags: list[str] = field(default_factory=list)
    base_url: str = "http://localhost:8000"

    @classmethod
    def from_env(cls) -> "IdentityConfig":
        raw_tags = os.getenv("NODE_TAGS") or ""
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        role_raw = (os.getenv("NODE_ROLE") or NodeRole.ROOT.value).strip().lower()
        try:
            role = NodeRole(role_raw)
        except ValueError as exc:
            raise SystemExit(
                f"NODE_ROLE must be one of: root, worker (got {role_raw!r})"
            ) from exc
        return cls(
            role=role,
            namespace=(os.getenv("NODE_NAMESPACE") or "flowmesh"),
            cluster=(os.getenv("NODE_CLUSTER") or "cluster"),
            alias=(os.getenv("NODE_ALIAS") or "node"),
            tags=tags,
            base_url=(os.getenv("FLOWMESH_BASE_URL") or "http://localhost:8000"),
        )


@dataclass
class DispatchConfig:
    mode: str = "adaptive"
    worker_selection: str = "best_fit"
    selection_jitter: float = 1e-3
    lambda_inference: float = 0.4
    lambda_training: float = 0.8
    lambda_other: float = 0.5
    enable_task_merge: bool = True
    task_merge_max_batch_size: int = 4
    enable_context_reuse: bool = True
    worker_cache_ttl_sec: int = 3600
    enable_stage_weight_stickiness: bool = False

    @classmethod
    def from_env(cls) -> "DispatchConfig":
        return cls(
            mode=os.getenv("ORCHESTRATOR_DISPATCH_MODE", "adaptive"),
            worker_selection=os.getenv("ORCHESTRATOR_WORKER_SELECTION", "best_fit"),
            selection_jitter=parse_float_env("SCHEDULER_SELECTION_JITTER", 1e-3),
            lambda_inference=parse_float_env("SCHEDULER_LAMBDA_INFERENCE", 0.4),
            lambda_training=parse_float_env("SCHEDULER_LAMBDA_TRAINING", 0.8),
            lambda_other=parse_float_env("SCHEDULER_LAMBDA_OTHER", 0.5),
            enable_task_merge=parse_bool_env("ENABLE_TASK_MERGE", True),
            task_merge_max_batch_size=max(
                1, parse_int_env("TASK_MERGE_MAX_BATCH_SIZE", 4)
            ),
            enable_context_reuse=parse_bool_env("ENABLE_CONTEXT_REUSE", True),
            worker_cache_ttl_sec=max(0, parse_int_env("WORKER_CACHE_TTL_SEC", 3600)),
            enable_stage_weight_stickiness=parse_bool_env(
                "ENABLE_STAGE_WEIGHT_STICKINESS", False
            ),
        )


@dataclass
class WatchdogConfig:
    enabled: bool = True
    check_interval: int = 30
    grace_sec: int = 60

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        return cls(
            enabled=parse_bool_env("ENABLE_WORKER_WATCHDOG", True),
            check_interval=max(5, parse_int_env("WORKER_DEATH_CHECK_INTERVAL", 30)),
            grace_sec=max(0, parse_int_env("WORKER_DEATH_GRACE_SEC", 60)),
        )


@dataclass
class MetricsConfig:
    dir: Path | None = None
    enable_density_plot: bool = False
    density_bucket_sec: int = 60

    @classmethod
    def from_env(cls, results_dir: Path) -> "MetricsConfig":
        metrics_env = os.getenv("SERVER_METRICS_DIR")
        metrics_dir: Path | None
        if metrics_env:
            metrics_dir = Path(metrics_env).expanduser().resolve()
        else:
            metrics_dir = results_dir.parent / "metrics"
        return cls(
            dir=metrics_dir,
            enable_density_plot=parse_bool_env(
                "SERVER_METRICS_ENABLE_DENSITY_PLOT", False
            ),
            density_bucket_sec=max(
                1, parse_int_env("SERVER_METRICS_DENSITY_BUCKET_SEC", 60)
            ),
        )


@dataclass
class WorkerManagementConfig:
    enabled: bool = True
    config_path: str = "configs/worker_config.yaml"
    heartbeat_interval: int = 30

    @classmethod
    def from_env(cls) -> "WorkerManagementConfig":
        return cls(
            enabled=parse_bool_env("ENABLE_SUPERVISOR", True),
            config_path=os.getenv("WORKER_CONFIG_PATH", "configs/worker_config.yaml"),
            heartbeat_interval=int(os.getenv("SERVER_HEARTBEAT_INTERVAL") or "30"),
        )


@dataclass
class LogStreamConfig:
    ttl_sec: int = 3600
    archive_flush_interval_sec: float = 5.0
    archive_flush_max_entries: int = 100

    @classmethod
    def from_env(cls) -> "LogStreamConfig":
        return cls(
            ttl_sec=max(0, parse_int_env("LOG_STREAM_TTL_SEC", 3600)),
            archive_flush_interval_sec=max(
                0.1, parse_float_env("TASK_LOG_ARCHIVE_FLUSH_INTERVAL_SEC", 5.0)
            ),
            archive_flush_max_entries=max(
                1, parse_int_env("TASK_LOG_ARCHIVE_FLUSH_MAX_ENTRIES", 100)
            ),
        )


@dataclass
class ServerConfig:
    logging: LoggingConfig
    redis: RedisConfig
    http: HttpConfig
    grpc: GrpcConfig
    ssh_forward: SshForwardConfig
    identity: IdentityConfig
    dispatch: DispatchConfig
    watchdog: WatchdogConfig
    metrics: MetricsConfig
    worker_management: WorkerManagementConfig
    log_stream: LogStreamConfig
    results_dir: Path = Path("./results")

    @classmethod
    def from_env(cls) -> "ServerConfig":
        results_dir = (
            Path(os.getenv("RESULTS_DIR", "").strip() or "./results")
            .expanduser()
            .resolve()
        )
        return cls(
            logging=LoggingConfig.from_env(),
            redis=RedisConfig.from_env(),
            http=HttpConfig.from_env(),
            grpc=GrpcConfig.from_env(),
            ssh_forward=SshForwardConfig.from_env(),
            identity=IdentityConfig.from_env(),
            dispatch=DispatchConfig.from_env(),
            watchdog=WatchdogConfig.from_env(),
            metrics=MetricsConfig.from_env(results_dir),
            worker_management=WorkerManagementConfig.from_env(),
            log_stream=LogStreamConfig.from_env(),
            results_dir=results_dir,
        )
