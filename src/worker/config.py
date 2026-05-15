# worker/config.py
"""Configuration loader for the Worker process.

This module encapsulates all environment-derived configuration so the rest of
the worker code can depend on a structured config object.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.schemas.worker import SSHLimits
from shared.utils.parsing import (
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_mem_to_bytes,
)

from .utils.health import get_hb_config


@dataclass(frozen=True)
class WorkerConfig:
    worker_token: str
    owner_principal: dict[str, Any] | None
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
    docker_gpu_runtime: str | None
    ssh_limits: SSHLimits | None
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

        owner_principal: dict[str, Any] | None = None
        if owner_principal_json := os.getenv("WORKER_OWNER_PRINCIPAL_JSON", "").strip():
            try:
                loaded = json.loads(owner_principal_json)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(loaded, dict):
                    owner_principal = loaded

        supervisor_grpc_target = (os.getenv("SUPERVISOR_GRPC_TARGET") or "").strip()
        if not supervisor_grpc_target:
            raise SystemExit("SUPERVISOR_GRPC_TARGET is required")

        supervisor_grpc_tls_ca_b64: str | None = (
            os.getenv("SUPERVISOR_GRPC_TLS_CA_B64") or ""
        ).strip() or None

        results_dir = Path(
            os.getenv("RESULTS_DIR", "").strip() or "./results"
        ).absolute()
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
        docker_gpu_runtime = os.getenv("DOCKER_GPU_RUNTIME", "").strip() or None
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

        ssh_max_cpu = parse_float_env("SSH_MAX_CPU")
        if ssh_max_cpu is not None and ssh_max_cpu <= 0:
            raise SystemExit("SSH_MAX_CPU must be positive")
        ssh_max_memory_raw = os.getenv("SSH_MAX_MEMORY", "").strip() or None
        ssh_max_memory_bytes: int | None = None
        if ssh_max_memory_raw is not None:
            ssh_max_memory_bytes = parse_mem_to_bytes(ssh_max_memory_raw)
            if ssh_max_memory_bytes is None or ssh_max_memory_bytes <= 0:
                raise SystemExit(
                    f"SSH_MAX_MEMORY value {ssh_max_memory_raw!r} is not a valid "
                    "memory string (e.g. '8Gi', '512Mi', or a positive byte count)"
                )
        ssh_max_pids = parse_int_env("SSH_MAX_PIDS")
        if ssh_max_pids is not None and ssh_max_pids <= 0:
            raise SystemExit("SSH_MAX_PIDS must be positive")
        ssh_limits = (
            None
            if (
                ssh_max_cpu is None
                and ssh_max_memory_bytes is None
                and ssh_max_pids is None
            )
            else SSHLimits(
                max_cpu_cores=ssh_max_cpu,
                max_memory_bytes=ssh_max_memory_bytes,
                max_pids=ssh_max_pids,
            )
        )

        return WorkerConfig(
            worker_token=worker_token,
            owner_principal=owner_principal,
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
            docker_gpu_runtime=docker_gpu_runtime,
            ssh_limits=ssh_limits,
            grpc_keepalive_time_ms=grpc_keepalive_time_ms,
            grpc_keepalive_timeout_ms=grpc_keepalive_timeout_ms,
            network_mode=network_mode,
            container_name=container_name,
            ssh_network_name=ssh_network_name,
        )
