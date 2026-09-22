"""Resolved configuration for an SSH session.

Transport-agnostic: every value here is derived from the task spec, the worker
config and the environment, and means the same thing whichever session backend
ends up running the session.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shared.tasks.components.resources import GPURequirements
from shared.tasks.specs.ssh import SSHInputSpec, SSHOutputSpec, SSHSpecStrict
from shared.tasks.worker_message import WorkerHardware
from shared.utils import parse_float_env, parse_mem_to_bytes
from shared.utils.hardware import (
    parse_gpu_memory_bytes,
    select_matching_gpu_indices,
    unified_gpu_memory_satisfies,
)
from worker.config import WorkerConfig

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

# Label applied to every session resource so teardown can find them
LABEL_WORKER = "flowmesh.ssh.worker_id"
LABEL_TASK = "flowmesh.ssh.task_id"
LABEL_SESSION = "flowmesh.ssh.session_id"
LABEL_MANAGED = "flowmesh.ssh.managed"

DEFAULT_IMAGE_CPU: str = (
    f"{os.getenv('FLOWMESH_REGISTRY', 'ghcr.io/mlsys-io')}"
    f"/flowmesh_ssh:{os.getenv('FLOWMESH_VERSION', 'latest')}-cpu"
)
DEFAULT_IMAGE_GPU: str = (
    f"{os.getenv('FLOWMESH_REGISTRY', 'ghcr.io/mlsys-io')}"
    f"/flowmesh_ssh:{os.getenv('FLOWMESH_VERSION', 'latest')}-gpu"
)
DEFAULT_USER = "flowmesh"
DEFAULT_TTL_SEC = 3600
DEFAULT_IDLE_SEC = 900
MAX_TTL_SEC = 28800  # 8 hours
POLL_INTERVAL_SEC = 5
STOP_TIMEOUT_SEC = 30
DEFAULT_INPUTS_ROOT = "/mnt/flowmesh/inputs"
DEFAULT_OUTPUT_PATH = "/mnt/flowmesh/output"
SAFE_MOUNT_ROOT = PurePosixPath("/mnt/flowmesh")
FINISH_SENTINEL_PATH = PurePosixPath("/", "tmp", ".flowmesh_finish").as_posix()


@dataclass(slots=True)
class ResolvedSSHInput:
    stage: str
    task_id: str
    source_path: Path
    mount_path: str


@dataclass(slots=True)
class SSHOutputConfig:
    mount_path: str
    max_bytes: int | None

    @classmethod
    def from_spec(cls, spec: SSHOutputSpec) -> "SSHOutputConfig":
        return cls(
            mount_path=spec.mountPath or DEFAULT_OUTPUT_PATH,
            max_bytes=spec.maxBytes,
        )


@dataclass(slots=True)
class SSHConfig:
    image: str
    interactive: bool
    user: str
    authorized_keys: list[str]
    command: list[str] | None
    entrypoint: list[str] | None
    ttl_sec: float
    idle_sec: float
    access_mode: str
    extra_env: dict[str, object]
    inputs: list[SSHInputSpec]
    output: SSHOutputConfig | None
    poll_interval_sec: float
    stop_timeout_sec: float
    cpu_limit: float | None
    memory_limit_bytes: int | None
    pids_limit: int | None
    gpu_device_ids: list[str]

    @classmethod
    def from_spec(
        cls,
        spec: SSHSpecStrict,
        worker_cfg: WorkerConfig,
        hardware: WorkerHardware | None = None,
        available_uuids: frozenset[str] | None = None,
    ) -> "SSHConfig":
        """Build a resolved config from a task spec, env vars, and defaults."""
        has_gpu = bool(os.getenv("WORKER_HOST_GPU_ID", "").strip())
        fallback_image = DEFAULT_IMAGE_GPU if has_gpu else DEFAULT_IMAGE_CPU
        default_image = os.getenv("SSH_DEFAULT_IMAGE", fallback_image)
        default_user = os.getenv("SSH_DEFAULT_USER", DEFAULT_USER)
        default_ttl_sec = parse_float_env("SSH_DEFAULT_TTL_SEC", DEFAULT_TTL_SEC)
        default_idle_sec = parse_float_env("SSH_DEFAULT_IDLE_SEC", DEFAULT_IDLE_SEC)
        max_ttl_sec = parse_float_env("SSH_MAX_TTL_SEC", MAX_TTL_SEC)
        poll_interval_sec = parse_float_env("SSH_POLL_INTERVAL_SEC", POLL_INTERVAL_SEC)
        stop_timeout_sec = parse_float_env("SSH_STOP_TIMEOUT_SEC", STOP_TIMEOUT_SEC)
        output_cfg = (
            SSHOutputConfig.from_spec(ssh_output)
            if (ssh_output := spec.sshOutput)
            else None
        )
        cpu_limit, memory_limit_bytes, pids_limit = _resolve_resource_limits(
            spec, worker_cfg
        )
        gpu_device_ids = _resolve_gpu_devices(
            spec, worker_cfg, hardware, available_uuids
        )
        ttl_sec = min(spec.ttlSeconds or default_ttl_sec, max_ttl_sec)
        return cls(
            image=spec.image or default_image,
            interactive=bool(spec.interactive),
            user=spec.user or default_user,
            authorized_keys=spec.authorizedKeys or [],
            command=spec.command,
            entrypoint=spec.entrypoint,
            ttl_sec=ttl_sec,
            idle_sec=min(spec.idleTimeoutSeconds or default_idle_sec, ttl_sec),
            access_mode=spec.accessMode or "direct",
            extra_env=dict(spec.env or {}),
            inputs=list(spec.inputs or []),
            output=output_cfg,
            poll_interval_sec=poll_interval_sec,
            stop_timeout_sec=stop_timeout_sec,
            cpu_limit=cpu_limit,
            memory_limit_bytes=memory_limit_bytes,
            pids_limit=pids_limit,
            gpu_device_ids=gpu_device_ids,
        )


def _resolve_resource_limits(
    spec: SSHSpecStrict, worker_cfg: WorkerConfig
) -> tuple[float | None, int | None, int | None]:
    """Resolve effective CPU/memory limits as min(task spec, worker cap).

    Returns ``(cpu_limit, memory_limit_bytes, pids_limit)``. Each of them may be
    ``None`` to mean unbounded — that is, neither the spec nor the cap constrains it.
    """
    spec_cpu: float | None = None
    spec_mem_bytes: int | None = None
    if (res := spec.resources) and (hw := res.hardware):
        if hw.cpu is not None:
            spec_cpu = float(hw.cpu)
        if hw.memory is not None:
            if isinstance(hw.memory, str):
                spec_mem_bytes = parse_mem_to_bytes(hw.memory)
                if spec_mem_bytes is None:
                    raise ExecutionError(
                        f"resources.hardware.memory value {hw.memory!r} is not "
                        "a valid memory string (e.g. '8Gi', '512Mi')"
                    )
            else:
                spec_mem_bytes = int(hw.memory)

    ssh_limits = worker_cfg.ssh_limits
    if ssh_limits is None:
        return spec_cpu, spec_mem_bytes, None

    cpu_limit = _min_or_none(spec_cpu, ssh_limits.max_cpu_cores)
    if (
        spec_cpu is not None
        and ssh_limits.max_cpu_cores is not None
        and spec_cpu > ssh_limits.max_cpu_cores
    ):
        logger.warning(
            "SSH task requested cpu=%s but worker cap is %s; clamping to cap",
            spec_cpu,
            ssh_limits.max_cpu_cores,
        )

    memory_limit_bytes = _min_or_none(spec_mem_bytes, ssh_limits.max_memory_bytes)
    if (
        spec_mem_bytes is not None
        and ssh_limits.max_memory_bytes is not None
        and spec_mem_bytes > ssh_limits.max_memory_bytes
    ):
        logger.warning(
            "SSH task requested memory=%d bytes but worker cap is %d; "
            "clamping to cap",
            spec_mem_bytes,
            ssh_limits.max_memory_bytes,
        )

    return cpu_limit, memory_limit_bytes, ssh_limits.max_pids


def _min_or_none[T: (int, float)](a: T | None, b: T | None) -> T | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _resolve_gpu_devices(
    spec: SSHSpecStrict,
    config: WorkerConfig,
    hardware: WorkerHardware | None,
    available_uuids: frozenset[str] | None = None,
) -> list[str]:
    """Pick the smallest subset of the worker's GPUs that satisfies the spec.

    Returns the *host* device IDs to expose to the SSH session. When the spec
    sets no GPU constraints at all, returns the worker's full host GPU set;
    when only ``type`` or ``memory`` is set without ``count``, defaults to
    slicing a single matching device.
    """
    host_gpu_ids = [
        d_stripped
        for d in os.getenv("WORKER_HOST_GPU_ID", "").split(",")
        if (d_stripped := d.strip())
    ]
    if not config.enable_ssh_gpu_limit:
        return host_gpu_ids

    gpu_req: GPURequirements | None = None
    if (res := spec.resources) and (hw := res.hardware):
        gpu_req = hw.gpu
    if gpu_req is None or (
        gpu_req.count is None and not gpu_req.type and not gpu_req.memory
    ):
        return host_gpu_ids

    requested = gpu_req.count if gpu_req.count is not None else 1
    if requested <= 0:
        return []

    if not host_gpu_ids:
        raise ExecutionError(
            f"SSH task requested {requested} GPU(s) but this worker has none"
        )

    # The supervisor passes WORKER_HOST_GPU_ID in the same order as
    # worker.hardware.gpu.devices, so positions line up 1:1. When metadata is
    # missing or misaligned, fall back to count-only slicing.
    devices = hardware.gpu.devices if hardware is not None else []
    if devices and len(devices) == len(host_gpu_ids) and available_uuids is not None:
        # Drop held devices from both lists together: selection returns positions,
        # so the two must stay aligned. Filtering the selected positions instead
        # would report "no satisfying device" whenever the first match is held.
        paired = [
            (device, host_id)
            for device, host_id in zip(devices, host_gpu_ids, strict=True)
            if device.uuid in available_uuids
        ]
        devices = [device for device, _ in paired]
        host_gpu_ids = [host_id for _, host_id in paired]
        if len(host_gpu_ids) < requested:
            raise ExecutionError(
                f"SSH task requested {requested} GPU(s) but only "
                f"{len(host_gpu_ids)} of this worker's devices are free",
                retryable=True,
            )
    if devices and len(devices) != len(host_gpu_ids):
        logger.warning(
            "WORKER_HOST_GPU_ID (%d) and worker hardware.gpu.devices (%d) "
            "disagree; falling back to count-only slicing",
            len(host_gpu_ids),
            len(devices),
        )
        devices = []

    required_mem_bytes: int | None = None
    if gpu_req.memory:
        required_mem_bytes = parse_gpu_memory_bytes(gpu_req.memory)
        if required_mem_bytes is None:
            raise ExecutionError(
                f"resources.hardware.gpu.memory value {gpu_req.memory!r} is "
                "not a valid memory string (e.g. '40Gi', '80GB')"
            )

    if not devices:
        if len(host_gpu_ids) < requested:
            raise ExecutionError(
                f"SSH task requested {requested} GPU(s) but only "
                f"{len(host_gpu_ids)} are available on this worker"
            )
        return host_gpu_ids[:requested]

    matching_indices = select_matching_gpu_indices(devices, gpu_req, limit=requested)
    if len(matching_indices) >= requested:
        return [host_gpu_ids[idx] for idx in matching_indices]

    if required_mem_bytes is not None and hardware is not None:
        type_only_req = GPURequirements(
            count=gpu_req.count, type=gpu_req.type, memory=None
        )
        type_matching = select_matching_gpu_indices(
            devices, type_only_req, limit=requested
        )
        if len(type_matching) >= requested and unified_gpu_memory_satisfies(
            hardware, required_mem_bytes, requested
        ):
            return [host_gpu_ids[idx] for idx in type_matching]

    raise ExecutionError(
        f"SSH task requested {requested} GPU(s) matching the spec but "
        f"only {len(matching_indices)} satisfying device(s) are available "
        "on this worker"
    )


def normalize_mount_path(path: str, field_name: str) -> str:
    normalized = PurePosixPath(path.strip())
    if not normalized.is_absolute():
        raise ExecutionError(f"{field_name} must be an absolute path")
    if normalized == PurePosixPath("/"):
        raise ExecutionError(f"{field_name} cannot be '/'")
    if normalized != SAFE_MOUNT_ROOT and SAFE_MOUNT_ROOT not in normalized.parents:
        raise ExecutionError(f"{field_name} must be under {SAFE_MOUNT_ROOT.as_posix()}")
    return normalized.as_posix()


def reserve_mount_path(used_mount_paths: set[str], mount_path: str) -> None:
    if mount_path in used_mount_paths:
        raise ExecutionError(f"Duplicate SSH mountPath '{mount_path}'")
    used_mount_paths.add(mount_path)
