# worker/config.py
"""Configuration loader for the Worker process.

This module encapsulates all environment-derived configuration so the rest of
the worker code can depend on a structured config object.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from shared.utils.parsing import parse_bool_env, parse_float_env, parse_int_env

from .utils.health import get_hb_config


@dataclass(frozen=True)
class WorkerConfig:
    worker_token: str
    supervisor_grpc_target: str
    supervisor_grpc_tls_ca_b64: str | None
    results_dir: Path
    results_mount_source: str | None
    hb_interval_sec: int
    hb_ttl_sec: int
    hb_file: Path
    namespace: str
    cluster: str
    alias: str
    tags: list[str]
    log_level: str
    cost_per_hour: float
    network_bandwidth_bytes_per_sec: float | None
    executor_idle_cleanup_sec: float | None
    enable_mp_executors: bool
    redis_control_url: str | None
    worker_cache_dir: Path
    data_ttl_sec: int
    grpc_keepalive_time_ms: int | None = None
    grpc_keepalive_timeout_ms: int | None = None
    network_mode: str | None = None
    container_name: str | None = None
    ssh_network_name: str | None = None

    @staticmethod
    def from_env() -> "WorkerConfig":
        worker_token = os.getenv("WORKER_TOKEN", "").strip()
        if not worker_token:
            raise SystemExit("WORKER_TOKEN is required")

        supervisor_grpc_target = (os.getenv("SUPERVISOR_GRPC_TARGET") or "").strip()
        if not supervisor_grpc_target:
            raise SystemExit("SUPERVISOR_GRPC_TARGET is required")

        supervisor_grpc_tls_ca_b64: str | None = (
            os.getenv("SUPERVISOR_GRPC_TLS_CA_B64") or ""
        ).strip() or None

        results_dir = Path(os.getenv("RESULTS_DIR", "./results_workers")).absolute()
        results_dir.mkdir(parents=True, exist_ok=True)

        hb_interval, hb_ttl, hb_file = get_hb_config()

        namespace = os.getenv("WORKER_NAMESPACE", "flowmesh").strip()
        cluster = os.getenv("WORKER_CLUSTER", "cluster").strip()
        container_name = os.getenv("WORKER_CONTAINER_NAME", "").strip() or None
        ssh_network_name = os.getenv("SSH_NETWORK_NAME", "").strip() or None
        alias = os.getenv("WORKER_ALIAS", "").strip() or os.urandom(8).hex()
        tags = [t.strip() for t in os.getenv("WORKER_TAGS", "").split(",") if t.strip()]

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        cost_per_hour = parse_float_env("WORKER_COST_PER_HOUR", 1.0)
        if cost_per_hour < 0:
            raise SystemExit("WORKER_COST_PER_HOUR must be non-negative")

        network_bandwidth_bytes_per_sec = parse_float_env(
            "WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC", None
        )
        if (
            network_bandwidth_bytes_per_sec is not None
            and network_bandwidth_bytes_per_sec <= 0
        ):
            raise SystemExit("WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC must be positive")

        enable_mp_executors = parse_bool_env("WORKER_ENABLE_MP_EXECUTORS", True)
        grpc_keepalive_time_ms = parse_int_env(
            "SUPERVISOR_GRPC_KEEPALIVE_TIME_MS", 300_000
        )
        grpc_keepalive_timeout_ms = parse_int_env(
            "SUPERVISOR_GRPC_KEEPALIVE_TIMEOUT_MS", 10_000
        )
        results_mount_source = os.getenv("RESULTS_MOUNT_SOURCE", "").strip() or None
        network_mode = os.getenv("WORKER_NETWORK_MODE", "").strip() or None
        executor_idle_cleanup_sec = parse_float_env(
            "WORKER_EXECUTOR_IDLE_CLEANUP_SEC", 60
        )

        redis_control_url = os.getenv("REDIS_CONTROL_URL", "").strip() or None
        worker_cache_dir = Path(
            os.getenv("WORKER_CACHE_DIR") or "/tmp/flowmesh_worker_cache"  # noqa: S108
        ).absolute()
        worker_cache_dir.mkdir(parents=True, exist_ok=True)
        data_ttl_sec = parse_int_env("WORKER_DATA_TTL_SEC", 24 * 60 * 60)

        return WorkerConfig(
            worker_token=worker_token,
            supervisor_grpc_target=supervisor_grpc_target,
            supervisor_grpc_tls_ca_b64=supervisor_grpc_tls_ca_b64,
            results_dir=results_dir,
            results_mount_source=results_mount_source,
            hb_interval_sec=hb_interval,
            hb_ttl_sec=hb_ttl,
            hb_file=hb_file,
            namespace=namespace,
            cluster=cluster,
            alias=alias,
            tags=tags,
            log_level=log_level,
            cost_per_hour=cost_per_hour,
            network_bandwidth_bytes_per_sec=network_bandwidth_bytes_per_sec,
            executor_idle_cleanup_sec=executor_idle_cleanup_sec,
            enable_mp_executors=enable_mp_executors,
            redis_control_url=redis_control_url,
            worker_cache_dir=worker_cache_dir,
            data_ttl_sec=data_ttl_sec,
            grpc_keepalive_time_ms=grpc_keepalive_time_ms,
            grpc_keepalive_timeout_ms=grpc_keepalive_timeout_ms,
            network_mode=network_mode,
            container_name=container_name,
            ssh_network_name=ssh_network_name,
        )
