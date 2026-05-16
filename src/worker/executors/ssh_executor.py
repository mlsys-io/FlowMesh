"""SSH session executor.

Supports two modes:

**Interactive** (default): Creates an ephemeral Docker container running sshd,
emits a TASK_UPDATE event with connection info, and blocks until the session
ends (TTL, idle timeout, or worker shutdown).

**Non-interactive** (``interactive=false``): Runs a user-provided Docker image
with an optional custom entrypoint/command.
"""

import io
import logging
import os
import shlex
import shutil
import socket
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

import requests

from shared.tasks.components.resources import GPURequirements
from shared.tasks.specs.ssh import (
    SSHInputSpec,
    SSHMountSpec,
    SSHOutputSpec,
    SSHSpecStrict,
)
from shared.tasks.worker_message import WorkerHardware
from shared.utils import new_ssh_session_id, parse_float_env, parse_mem_to_bytes
from shared.utils.hardware import (
    parse_gpu_memory_bytes,
    select_matching_gpu_indices,
    unified_gpu_memory_satisfies,
)
from shared.utils.http import auth_headers
from shared.utils.manifest import ARTIFACTS_DIR, prepare_output_dir
from worker.config import WorkerConfig
from worker.executors.utils.checkpoints import maybe_upload_artifacts

from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    TaskCancelledError,
)

try:
    import docker
    from docker import DockerClient
    from docker.models.containers import Container
    from docker.types import DeviceRequest

    _HAS_DOCKER = True
except Exception:
    _HAS_DOCKER = False
    if TYPE_CHECKING:
        import docker
        from docker import DockerClient
        from docker.models.containers import Container
        from docker.types import DeviceRequest
    else:
        docker = None
        DockerClient = Any
        Container = Any
        DeviceRequest = Any


logger = logging.getLogger(__name__)

# Label applied to every session container so teardown can find them
_LABEL_WORKER = "flowmesh.ssh.worker_id"
_LABEL_TASK = "flowmesh.ssh.task_id"
_LABEL_SESSION = "flowmesh.ssh.session_id"
_LABEL_MANAGED = "flowmesh.ssh.managed"

_DEFAULT_IMAGE_CPU: str = (
    f"{os.getenv('FLOWMESH_REGISTRY', 'ghcr.io/mlsys-io')}"
    f"/flowmesh_ssh:{os.getenv('FLOWMESH_VERSION', 'latest')}-cpu"
)
_DEFAULT_IMAGE_GPU: str = (
    f"{os.getenv('FLOWMESH_REGISTRY', 'ghcr.io/mlsys-io')}"
    f"/flowmesh_ssh:{os.getenv('FLOWMESH_VERSION', 'latest')}-gpu"
)
_DEFAULT_USER = "flowmesh"
_DEFAULT_TTL_SEC = 3600
_DEFAULT_IDLE_SEC = 900
_MAX_TTL_SEC = 28800  # 8 hours
_POLL_INTERVAL_SEC = 5
_STOP_TIMEOUT_SEC = 30
_DEFAULT_INPUTS_ROOT = "/mnt/flowmesh/inputs"
_DEFAULT_OUTPUT_PATH = "/mnt/flowmesh/output"
_SAFE_MOUNT_ROOT = PurePosixPath("/mnt/flowmesh")
_CONTAINER_RESULTS_SOURCE_ROOT = "/root/.flowmesh/results-source"
_RESULT_BUNDLE_TIMEOUT_SEC = 300.0
_FINISH_SENTINEL_PATH = PurePosixPath("/", "tmp", ".flowmesh_finish").as_posix()
_SSH_RUN_ENTRYPOINT_PATH = "/flowmesh-ssh-run.sh"
_SSH_RUN_SCRIPT_SOURCE = (
    Path(__file__).resolve().parent.parent / "docker" / "ssh-run.sh"
)

type DemuxLogStream = Iterator[tuple[bytes | None, bytes | None]]


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
            mount_path=spec.mountPath or _DEFAULT_OUTPUT_PATH,
            max_bytes=spec.maxBytes,
        )


@dataclass(slots=True)
class SSHMountPlan:
    volumes: list[str]
    staged_input_specs: list[tuple[str, str]]
    create_dirs: list[str]
    direct_output_path: Path | None
    copy_output_path: str | None
    staged_inputs_dir: Path | None
    staged_inputs_volume: str | None


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
    extra_env: dict[str, Any]
    inputs: list[SSHInputSpec]
    output: SSHOutputConfig | None
    mounts: list[SSHMountSpec]
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
    ) -> "SSHConfig":
        """Build a resolved config from a task spec, env vars, and defaults."""
        has_gpu = bool(os.getenv("WORKER_HOST_GPU_ID", "").strip())
        fallback_image = _DEFAULT_IMAGE_GPU if has_gpu else _DEFAULT_IMAGE_CPU
        default_image = os.getenv("SSH_DEFAULT_IMAGE", fallback_image)
        default_user = os.getenv("SSH_DEFAULT_USER", _DEFAULT_USER)
        default_ttl_sec = parse_float_env("SSH_DEFAULT_TTL_SEC", _DEFAULT_TTL_SEC)
        default_idle_sec = parse_float_env("SSH_DEFAULT_IDLE_SEC", _DEFAULT_IDLE_SEC)
        max_ttl_sec = parse_float_env("SSH_MAX_TTL_SEC", _MAX_TTL_SEC)
        poll_interval_sec = parse_float_env("SSH_POLL_INTERVAL_SEC", _POLL_INTERVAL_SEC)
        stop_timeout_sec = parse_float_env("SSH_STOP_TIMEOUT_SEC", _STOP_TIMEOUT_SEC)
        output_cfg = (
            SSHOutputConfig.from_spec(ssh_output)
            if (ssh_output := spec.sshOutput)
            else None
        )
        cpu_limit, memory_limit_bytes, pids_limit = _resolve_resource_limits(
            spec, worker_cfg
        )
        gpu_device_ids = _resolve_gpu_devices(spec, worker_cfg, hardware)
        return cls(
            image=spec.image or default_image,
            interactive=bool(spec.interactive),
            user=spec.user or default_user,
            authorized_keys=spec.authorizedKeys or [],
            command=spec.command,
            entrypoint=spec.entrypoint,
            ttl_sec=min(spec.ttlSeconds or default_ttl_sec, max_ttl_sec),
            idle_sec=spec.idleTimeoutSeconds or default_idle_sec,
            access_mode=spec.accessMode or "direct",
            extra_env=dict(spec.env or {}),
            inputs=list(spec.inputs or []),
            output=output_cfg,
            mounts=list(spec.mounts or []),
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
    spec: SSHSpecStrict, config: WorkerConfig, hardware: WorkerHardware | None
) -> list[str]:
    """Pick the smallest subset of the worker's GPUs that satisfies the spec.

    Returns the *host* device IDs to expose to the SSH container. When the spec
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
    if not host_gpu_ids:
        return []

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

    # The supervisor passes WORKER_HOST_GPU_ID in the same order as
    # worker.hardware.gpu.devices, so positions line up 1:1. When metadata is
    # missing or misaligned, fall back to count-only slicing.
    devices = hardware.gpu.devices if hardware is not None else []
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
        # No per-device metadata to filter by — fall back to first-N host IDs.
        if len(host_gpu_ids) < requested:
            raise ExecutionError(
                f"SSH task requested {requested} GPU(s) but only "
                f"{len(host_gpu_ids)} are available on this worker"
            )
        return host_gpu_ids[:requested]

    matching_indices = select_matching_gpu_indices(devices, gpu_req, limit=requested)
    if len(matching_indices) >= requested:
        return [host_gpu_ids[idx] for idx in matching_indices]

    # Unified memory fallback
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


class SSHExecutor(Executor):
    """Executor for SSH tasks (interactive sessions and non-interactive jobs)."""

    name = "ssh"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self._config
        lifecycle = self._lifecycle
        self._worker_name = (
            config.container_name
            or config.alias
            or (lifecycle.worker_id if lifecycle else uuid.uuid4().hex[:8])
        )
        self._docker: DockerClient | None = None
        self._docker_gpu_runtime: str | None = config.docker_gpu_runtime
        self._ssh_network: str | None = None
        self._cancel_event = threading.Event()
        self._finish_event = threading.Event()
        self._current_container: Container | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        if not _HAS_DOCKER:
            raise ExecutionError("Docker SDK is not available (`pip install docker`).")
        self._docker = self._get_docker_client()
        self._ssh_network = self._ensure_ssh_network(self._docker)

    def teardown(self) -> None:
        """Stop all session containers owned by this worker."""
        stop_timeout_sec = parse_float_env("SSH_STOP_TIMEOUT_SEC", _STOP_TIMEOUT_SEC)
        client = self._get_docker_client()
        try:
            containers = client.containers.list(
                filters={"label": f"{_LABEL_WORKER}={self._worker_name}"}
            )
        except Exception as exc:
            logger.warning(
                "Failed to list SSH session containers during teardown: %s", exc
            )
            return
        for c in containers:
            try:
                c.stop(timeout=stop_timeout_sec)
                c.remove(force=True)
                logger.info("Removed SSH session container on teardown")
            except Exception as exc:
                logger.warning("Failed to remove container: %s", exc)

    # ------------------------------------------------------------------ #
    # Main execution
    # ------------------------------------------------------------------ #

    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:
        spec = self.require_spec(task, SSHSpecStrict)
        cfg = SSHConfig.from_spec(spec, self._config, self._hardware)
        access_mode = cfg.access_mode
        interactive = cfg.interactive

        if interactive and access_mode not in ("direct", "proxy", "forward"):
            raise ExecutionError(f"accessMode '{access_mode}' is not supported")

        self.prepare()  # Initialize Docker client and SSH network
        client = self._docker
        assert client is not None

        if interactive:
            ports = {"22/tcp": None}  # assign a random host port
            container_cmd = None
        else:
            ports = {}
            # Resolve the command to pass to the wrapper entrypoint.
            container_cmd = self._resolve_noninteractive_command(client, cfg)

        session_id = new_ssh_session_id()
        container_name = f"{self._worker_name}_ssh-{task.task_id[:8]}-{session_id[:8]}"

        prepare_output_dir(out_dir)  # Ensure output dir exists before mounting
        resolved_inputs = self._resolve_inputs(task, cfg)
        mount_plan = self._build_mount_plan(
            client, out_dir, resolved_inputs, cfg, session_id
        )

        labels = {
            _LABEL_WORKER: self._worker_name,
            _LABEL_TASK: task.task_id,
            _LABEL_SESSION: session_id,
            _LABEL_MANAGED: "true",
        }
        environment = self._build_environment(
            cfg.user,
            cfg.authorized_keys,
            cfg.extra_env,
            mount_plan.staged_input_specs,
            mount_plan.create_dirs,
            interactive,
            cfg.gpu_device_ids,
        )
        kwargs = self._build_run_kwargs(
            cfg,
            container_name,
            environment,
            labels,
            ports,
            mount_plan.volumes,
            container_cmd,
            interactive,
        )

        if interactive:
            container_kind = "SSH session"
            logger.info(
                "Starting %s container (task=%s session=%s mode=%s ttl=%ds)",
                container_kind,
                task.task_id,
                session_id,
                access_mode,
                cfg.ttl_sec,
            )
        else:
            container_kind = "non-interactive"
            logger.info(
                "Starting %s container (task=%s session=%s ttl=%ds cmd=%s)",
                container_kind,
                task.task_id,
                session_id,
                cfg.ttl_sec,
                container_cmd,
            )

        container: Container | None = None
        log_stream: DemuxLogStream | None = None
        exit_code = 0
        try:
            container, log_stream = self._start_container(client, kwargs, interactive)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"Failed to start {container_kind} container: {exc}"
            ) from exc

        assert isinstance(container, Container)
        self._current_container = container
        log_thread: threading.Thread | None = None
        if not interactive:
            log_thread = threading.Thread(
                target=self._stream_container_logs,
                args=(log_stream,),
                daemon=True,
                name=f"flowmesh-container-logs-{task.task_id[:8]}",
            )
            log_thread.start()
        try:
            session_info = (
                self._wait_session_ready(container, session_id, task, cfg)
                if interactive
                else {}
            )
            exit_code = self._wait_for_session(
                container,
                cfg.ttl_sec,
                cfg.idle_sec,
                cfg.poll_interval_sec,
                cfg.output,
                mount_plan,
            )
            result: dict[str, Any] = {"session_id": session_id, "exit_code": exit_code}
            if interactive:
                result.update(session_info)
            else:
                # Keep as fallback — captures any output the streaming thread missed.
                self._save_container_logs(container, out_dir)
                if cfg.command is not None:
                    result["command"] = cfg.command
                if cfg.entrypoint is not None:
                    result["entrypoint"] = cfg.entrypoint
            if mount_plan.copy_output_path:
                self._copy_output_directory(
                    container,
                    mount_plan.copy_output_path,
                    out_dir / ARTIFACTS_DIR,
                )
            maybe_upload_artifacts(task, out_dir, logger=logger, skip_errors=True)
        finally:
            if log_thread is not None:
                # Wait for the thread to drain remaining output before tearing down
                # the container.
                log_thread.join(timeout=30.0)
            self._current_container = None
            self._cancel_event.clear()
            self._finish_event.clear()
            if container is not None:
                self._stop_container(container, container_name, cfg.stop_timeout_sec)
            self._cleanup_mount_plan(client, mount_plan)

        if not (interactive or exit_code == 0):
            raise ExecutionError(
                f"{container_kind.capitalize()} container exited with code {exit_code}"
            )

        return result

    def cancel(self, task_id: str) -> None:
        self._cancel_event.set()
        container = self._current_container
        if container is None:
            return
        try:
            container.stop(timeout=1)
        except Exception:
            logger.debug(
                "Failed to stop SSH container during cancellation", exc_info=True
            )

    def stop(self, task_id: str) -> None:
        self._finish_event.set()
        container = self._current_container
        if container is None:
            return
        try:
            container.stop(timeout=1)
        except Exception:
            logger.debug(
                "Failed to stop SSH container during graceful stop", exc_info=True
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_docker_client(self) -> DockerClient:
        if self._docker is None:
            try:
                self._docker = docker.from_env()
            except Exception as exc:
                raise ExecutionError(
                    f"Docker is not available; cannot run SSH executor: {exc}"
                ) from exc
        return self._docker

    def _ensure_ssh_network(self, client: DockerClient) -> str | None:
        """Create (or reuse) an isolated bridge network for SSH containers.

        The network disables inter-container communication (ICC) so that SSH
        containers from different sessions/tenants cannot reach each other,
        while still allowing outbound internet access.
        """
        net_name = self._config.ssh_network_name
        if not net_name:
            return None
        try:
            existing = client.networks.list(names=[net_name])
            for net in existing:
                labels = net.attrs.get("Labels") or {}
                if net.name == net_name and labels.get(_LABEL_MANAGED) == "true":
                    return net_name
        except Exception:
            pass
        try:
            client.networks.create(
                net_name,
                driver="bridge",
                options={"com.docker.network.bridge.enable_icc": "false"},
                labels={_LABEL_MANAGED: "true"},
            )
            logger.info("Created isolated network")
            return net_name
        except Exception:
            logger.warning(
                "Failed to create isolated network; falling back to default network"
            )
            return None

    def _build_environment(
        self,
        user: str,
        authorized_keys: list[str],
        extra_env: dict[str, Any],
        staged_input_specs: list[tuple[str, str]],
        create_dirs: list[str],
        interactive: bool,
        gpu_device_ids: list[str] | None = None,
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        if interactive:
            env["SSH_USER"] = user
            if authorized_keys:
                env["AUTHORIZED_KEYS"] = "\n".join(authorized_keys)
            env["SSH_UID"] = str(os.getuid())
            env["SSH_GID"] = str(os.getgid())
        if gpu_device_ids:
            # Docker exposes only the sliced devices, which appear as 0..N-1
            # inside the container regardless of their host IDs.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(len(gpu_device_ids))
            )
        if staged_input_specs:
            env["FLOWMESH_STAGED_INPUT_SPECS"] = "\n".join(
                f"{mount_path}\t{target_path}"
                for mount_path, target_path in staged_input_specs
            )
        if create_dirs:
            env["FLOWMESH_CREATE_DIRS"] = "\n".join(create_dirs)
        env["FLOWMESH_FINISH_SENTINEL"] = _FINISH_SENTINEL_PATH
        for k, v in extra_env.items():
            env[str(k)] = str(v)
        return env

    def _build_run_kwargs(
        self,
        cfg: SSHConfig,
        container_name: str,
        environment: dict[str, str],
        labels: dict[str, str],
        ports: dict[str, Any],
        volumes: list[str],
        command: list[str] | None,
        interactive: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "image": cfg.image,
            "name": container_name,
            "environment": environment,
            "labels": labels,
            "ports": ports,
            "detach": True,
            "security_opt": ["no-new-privileges:true"],
        }
        if volumes:
            kwargs["volumes"] = volumes
        if not interactive:
            kwargs["entrypoint"] = [_SSH_RUN_ENTRYPOINT_PATH]
            if command:
                kwargs["command"] = command
        if cfg.cpu_limit is not None:
            kwargs["nano_cpus"] = int(cfg.cpu_limit * 1_000_000_000)
        if cfg.memory_limit_bytes is not None:
            kwargs["mem_limit"] = cfg.memory_limit_bytes
        if cfg.pids_limit is not None:
            kwargs["pids_limit"] = cfg.pids_limit
        if cfg.gpu_device_ids:
            try:
                kwargs["device_requests"] = [
                    DeviceRequest(
                        device_ids=list(cfg.gpu_device_ids), capabilities=[["gpu"]]
                    )
                ]
                if runtime := self._docker_gpu_runtime:
                    kwargs["runtime"] = runtime
            except Exception:
                pass
        if self._ssh_network:
            kwargs["network"] = self._ssh_network
        return kwargs

    def _wait_for_port(self, container: Container, timeout_sec: float = 30.0) -> int:
        """Wait until Docker assigns a host port and sshd accepts connections."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                container.reload()
                if container.status not in ("running", "restarting"):
                    exit_info = container.wait()
                    exit_code = int(exit_info.get("StatusCode", -1))
                    tail = ""
                    try:
                        tail = (
                            container.logs(tail=20)
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
                    except Exception:
                        pass
                    msg = (
                        f"Container {container.name} exited (code {exit_code}) "
                        f"before SSH became ready."
                    )
                    if tail:
                        msg += f"\nContainer output:\n{tail}"
                    raise ExecutionError(msg)
                port_bindings = container.ports.get("22/tcp")
                if port_bindings:
                    host_port = int(port_bindings[0]["HostPort"])
                    if self._is_ssh_ready("127.0.0.1", host_port):
                        return host_port
            except ExecutionError:
                raise
            except Exception:
                pass
            time.sleep(1.0)
        raise ExecutionError(
            f"Timed out waiting for SSH readiness on container {container.name}. "
            f"Ensure the image has an SSH server (e.g. openssh-server) installed "
            f"and configured to start on port 22, or use the default FlowMesh SSH "
            f"image by omitting the image field."
        )

    def _is_ssh_ready(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1.0) as sock:
                sock.settimeout(1.0)
                banner = sock.recv(64)
                return banner.startswith(b"SSH-")
        except OSError:
            return False

    def _wait_session_ready(
        self, container: Container, session_id: str, task: ExecutorTask, cfg: SSHConfig
    ) -> dict[str, Any]:
        access_mode = cfg.access_mode
        expires_at = self._iso_offset(cfg.ttl_sec)
        host_port = self._wait_for_port(container)
        host_name = socket.getfqdn()
        ssh_info: dict[str, Any] = {
            "session_id": session_id,
            "mode": access_mode,
            "username": cfg.user,
            "expires_at": expires_at,
            "host": host_name,
            "port": host_port,
        }
        if access_mode in ("proxy", "forward"):
            if access_mode == "forward":
                # Forward-mode sessions need separate direct connection info
                ssh_info["directHost"] = host_name
                ssh_info["directPort"] = host_port
            ssh_info["_relay_target"] = {
                "host": "127.0.0.1",
                "port": host_port,
            }
            logger.info(
                "SSH %s session ready: host=%s port=%s (task=%s)",
                access_mode,
                host_name,
                host_port,
                task.task_id,
            )
        else:
            logger.info(
                "SSH session ready: host=%s port=%s (task=%s)",
                host_name,
                host_port,
                task.task_id,
            )
        self.emit_update(task.task_id, {"ssh": ssh_info})
        return {
            "expires_at": expires_at,
            "host": None if host_port is None else host_name,
            "port": host_port,
        }

    def _wait_for_session(
        self,
        container: Container,
        ttl_sec: float,
        idle_sec: float,
        poll_interval_sec: float,
        output_cfg: SSHOutputConfig | None,
        mount_plan: SSHMountPlan,
    ) -> int:
        """Block until the container exits or TTL/idle timeout fires.

        Returns the container exit code.
        """
        # TODO(kaiitunnz): Implement idle timeout by tracking container stats
        # and last activity timestamp
        deadline = time.time() + ttl_sec
        while time.time() < deadline:
            if self._cancel_event.is_set():
                raise TaskCancelledError("SSH session cancelled")
            if self._finish_event.is_set() or self._finish_requested(container):
                logger.info("SSH session finish requested; stopping container")
                try:
                    container.stop(timeout=1)
                except Exception:
                    logger.debug(
                        "Failed to stop SSH container during graceful finish",
                        exc_info=True,
                    )
                return 0
            try:
                container.reload()
                self._enforce_output_limit(container, output_cfg, mount_plan)
                if container.status not in ("running", "restarting"):
                    return int(container.wait()["StatusCode"])
            except Exception as exc:
                logger.debug("Container reload error (may have exited): %s", exc)
                if self._finish_event.is_set():
                    return 0
                break
            time.sleep(poll_interval_sec)

        logger.info("SSH session TTL reached; stopping container")
        return 0

    def _stop_container(
        self, container: Container, name: str, stop_timeout_sec: float
    ) -> None:
        try:
            container.stop(timeout=stop_timeout_sec)
        except Exception as exc:
            logger.debug("Error stopping container: %s", exc)
        try:
            container.remove(force=True)
            logger.info("Removed SSH session container")
        except Exception as exc:
            logger.debug("Error removing container: %s", exc)

    def _resolve_noninteractive_command(
        self, client: DockerClient, cfg: SSHConfig
    ) -> list[str]:
        if cfg.entrypoint is not None and cfg.command is not None:
            return cfg.entrypoint + cfg.command
        if cfg.entrypoint is not None:
            return cfg.entrypoint
        if cfg.command is not None:
            return cfg.command

        # Neither set — inspect the image metadata for defaults.
        try:
            image_obj = client.images.get(cfg.image)
            image_config = image_obj.attrs.get("Config", {})
        except Exception:
            try:
                image_obj = client.images.pull(cfg.image)
                image_config = image_obj.attrs.get("Config", {})
            except Exception as exc:
                raise ExecutionError(
                    f"Cannot determine default entrypoint/command for image "
                    f"'{cfg.image}': {exc}"
                ) from exc

        og_entrypoint = image_config.get("Entrypoint") or []
        og_cmd = image_config.get("Cmd") or []
        combined = list(og_entrypoint) + list(og_cmd)
        if not combined:
            raise ExecutionError(
                f"Image '{cfg.image}' has no Entrypoint or Cmd; "
                f"set command or entrypoint in the SSH spec"
            )
        return combined

    def _save_container_logs(self, container: Container, out_dir: Path) -> None:
        """Save container stdout/stderr to the output directory."""
        try:
            logs = container.logs(stdout=True, stderr=True)
            if isinstance(logs, bytes) and logs:
                logs_dir = out_dir / "artifacts" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / "container_output.log"
                log_path.write_bytes(logs)
        except Exception as exc:
            logger.debug("Failed to capture container logs: %s", exc)

    @staticmethod
    def _stream_container_logs(log_stream: DemuxLogStream) -> None:
        """Stream container stdout/stderr as log records.

        The method blocks until the container's output streams are closed
        (i.e. the container exits), ensuring no trailing output is lost.
        """

        def _emit(line: str, stream_name: str) -> None:
            level = logging.WARNING if stream_name == "stderr" else logging.INFO
            logger.log(level, line, extra={"flowmesh_stream": stream_name})

        buffers: dict[str, str] = {"stdout": "", "stderr": ""}
        try:
            for stdout_chunk, stderr_chunk in log_stream:
                for raw_chunk, stream_name in (
                    (stdout_chunk, "stdout"),
                    (stderr_chunk, "stderr"),
                ):
                    if raw_chunk is None:
                        continue
                    chunk: bytes = raw_chunk  # type: ignore[assignment]
                    text = buffers[stream_name] + chunk.decode(
                        "utf-8", errors="replace"
                    )
                    # Emit only complete lines; keep the trailing fragment.
                    if "\n" in text:
                        *complete, remainder = text.split("\n")
                        for line in complete:
                            if line:
                                _emit(line, stream_name)
                        buffers[stream_name] = remainder
                    else:
                        buffers[stream_name] = text
        except Exception:
            logger.debug("Container log stream ended", exc_info=True)

        # Flush any unterminated remainder.
        for stream_name, leftover in buffers.items():
            if leftover:
                _emit(leftover, stream_name)

    def _start_container(
        self, client: DockerClient, kwargs: dict[str, Any], interactive: bool
    ) -> tuple[Container, DemuxLogStream | None]:
        image = kwargs.get("image")
        mode = "interactive" if interactive else "non-interactive"
        log_stream: DemuxLogStream | None = None
        try:
            if interactive:
                container = client.containers.run(**kwargs)
            else:
                container, log_stream = self._run_noninteractive_container(
                    client, kwargs
                )
        except Exception as exc:
            if isinstance(image, str) and "No such image" in str(exc):
                try:
                    logger.info("Pulling missing image %s for %s SSH task", image, mode)
                    client.images.pull(image)
                    if interactive:
                        container = client.containers.run(**kwargs)
                    else:
                        container, log_stream = self._run_noninteractive_container(
                            client, kwargs
                        )
                except Exception as pull_exc:
                    raise ExecutionError(
                        f"Failed to start {mode} container after pulling image "
                        f"'{image}': {pull_exc}"
                    ) from pull_exc
            else:
                raise ExecutionError(
                    f"Failed to start {mode} container: {exc}"
                ) from exc
        assert isinstance(container, Container)
        return container, log_stream

    def _run_noninteractive_container(
        self, client: DockerClient, kwargs: dict[str, Any]
    ) -> tuple[Container, DemuxLogStream]:
        try:
            container = client.containers.create(**kwargs)
        except Exception as exc:
            raise ExecutionError(
                f"Failed to create non-interactive container: {exc}"
            ) from exc
        assert isinstance(container, Container)
        try:
            container.put_archive("/", self._build_ssh_run_archive())
            log_stream = cast(
                DemuxLogStream,
                container.attach(
                    stream=True, logs=True, stdout=True, stderr=True, demux=True
                ),
            )
            container.start()
        except Exception as exc:
            try:
                container.remove(force=True)
            except Exception:
                logger.debug(
                    "Failed to remove non-interactive container after startup error",
                    exc_info=True,
                )
            raise ExecutionError(
                f"Failed to initialize non-interactive container: {exc}"
            ) from exc
        return container, log_stream

    @staticmethod
    def _iso_offset(seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()

    def _resolve_inputs(
        self, task: ExecutorTask, cfg: SSHConfig
    ) -> list[ResolvedSSHInput]:
        if not cfg.inputs:
            return []

        results_root = self._config.results_dir
        resolved: list[ResolvedSSHInput] = []
        upstream_task_ids = task.upstream_task_ids or {}
        for entry in cfg.inputs:
            stage = entry.stage.strip()
            if not stage:
                raise ExecutionError("SSH input stage names must be non-empty")
            task_id = upstream_task_ids.get(stage)
            if not task_id:
                raise ExecutionError(
                    f"Missing resolved upstream task ID for SSH input stage '{stage}'"
                )
            source_path = results_root / task_id
            resolved.append(
                ResolvedSSHInput(
                    stage=stage,
                    task_id=task_id,
                    source_path=source_path,
                    mount_path=self._normalize_mount_path(
                        entry.mountPath or f"{_DEFAULT_INPUTS_ROOT}/{stage}",
                        field_name=f"inputs[{stage}].mountPath",
                    ),
                )
            )
        return resolved

    def _build_mount_plan(
        self,
        client: DockerClient,
        out_dir: Path,
        resolved_inputs: list[ResolvedSSHInput],
        cfg: SSHConfig,
        session_id: str,
    ) -> SSHMountPlan:
        volumes: list[str] = []
        staged_input_specs: list[tuple[str, str]] = []
        create_dirs: list[str] = []
        used_mount_paths: set[str] = set()
        results_source = self._config.results_mount_source
        staged_inputs_dir: Path | None = None
        staged_inputs_volume: str | None = None

        # Stage inputs in an isolated volume/directory
        if results_source and resolved_inputs:
            staged_inputs_volume = self._stage_inputs_in_volume(
                client, resolved_inputs, results_source, session_id
            )
            volumes.append(
                f"{staged_inputs_volume}:{_CONTAINER_RESULTS_SOURCE_ROOT}:ro"
            )
        elif resolved_inputs:
            staged_inputs_dir = self._stage_inputs_locally(resolved_inputs, session_id)

        # Mount resolved inputs
        for resolved in resolved_inputs:
            self._reserve_mount_path(used_mount_paths, resolved.mount_path)
            if results_source:
                # Materialize the requested staged input into the final mount path.
                staged_input_specs.append(
                    (
                        resolved.mount_path,
                        f"{_CONTAINER_RESULTS_SOURCE_ROOT}/{resolved.task_id}",
                    )
                )
            else:
                # Can mount directly from the local staged directory
                assert staged_inputs_dir is not None
                staged_input_path = staged_inputs_dir / resolved.task_id
                volumes.append(f"{staged_input_path}:{resolved.mount_path}:ro")

        direct_output_path: Path | None = None
        copy_output_path: str | None = None
        if cfg.output is not None:
            # Mount output directory
            output_mount_path = self._normalize_mount_path(
                cfg.output.mount_path, field_name="sshOutput.mountPath"
            )
            self._reserve_mount_path(used_mount_paths, output_mount_path)
            artifacts_dir = out_dir / ARTIFACTS_DIR
            if results_source:
                # Copy output back from the container after the session ends.
                create_dirs.append(output_mount_path)
                copy_output_path = output_mount_path
            else:
                # Can mount directly to the output directory
                volumes.append(f"{artifacts_dir}:{output_mount_path}:rw")
                direct_output_path = artifacts_dir

        return SSHMountPlan(
            volumes=volumes,
            staged_input_specs=staged_input_specs,
            create_dirs=create_dirs,
            direct_output_path=direct_output_path,
            copy_output_path=copy_output_path,
            staged_inputs_dir=staged_inputs_dir,
            staged_inputs_volume=staged_inputs_volume,
        )

    def _build_ssh_run_archive(self) -> bytes:
        script_bytes = _SSH_RUN_SCRIPT_SOURCE.read_bytes()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=_SSH_RUN_ENTRYPOINT_PATH.lstrip("/"))
            info.size = len(script_bytes)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(script_bytes))
        return stream.getvalue()

    def _stage_inputs_locally(
        self, resolved_inputs: list[ResolvedSSHInput], session_id: str
    ) -> Path:
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f"flowmesh-ssh-inputs-{session_id[:8]}-")
        )
        for resolved in resolved_inputs:
            destination = staging_dir / resolved.task_id
            if resolved.source_path.exists():
                shutil.copytree(resolved.source_path, destination, dirs_exist_ok=True)
                continue
            self._download_result_bundle(resolved.task_id, staging_dir)
            if not destination.exists():
                raise ExecutionError(
                    "Downloaded SSH input bundle did not create expected directory "
                    f"{destination} for upstream task {resolved.task_id}"
                )
        return staging_dir

    def _stage_inputs_in_volume(
        self,
        client: DockerClient,
        resolved_inputs: list[ResolvedSSHInput],
        results_source: str,
        session_id: str,
    ) -> str:
        volume_name = f"flowmesh_ssh_inputs_{session_id}"
        volume = client.volumes.create(
            name=volume_name,
            labels={
                _LABEL_WORKER: self._worker_name,
                _LABEL_SESSION: session_id,
                _LABEL_MANAGED: "true",
            },
        )
        commands = ["set -e"]
        for resolved in resolved_inputs:
            if resolved.source_path.exists():
                src = shlex.quote(f"/src/{resolved.task_id}")
                dst = shlex.quote(f"/dst/{resolved.task_id}")
                commands.append(f"mkdir -p {dst}")
                commands.append(f"cp -a {src}/. {dst}/")
                continue
            commands.append(self._build_remote_stage_command(resolved.task_id))
        command = " && ".join(commands)
        try:
            run_kwargs: dict[str, Any] = {
                "image": "busybox:1.36.1",
                "command": ["sh", "-lc", command],
                "volumes": [
                    f"{results_source}:/src:ro",
                    f"{volume_name}:/dst:rw",
                ],
                "remove": True,
            }
            if self._config.network_mode:
                run_kwargs["network_mode"] = self._config.network_mode
            client.containers.run(**run_kwargs)
        except Exception:
            try:
                volume.remove(force=True)
            except Exception:
                logger.debug(
                    "Failed to remove staging volume %s after populate failure",
                    volume_name,
                    exc_info=True,
                )
            raise
        return volume_name

    def _build_remote_stage_command(self, task_id: str) -> str:
        url = shlex.quote(self._result_bundle_url(task_id))
        header_parts = [
            f"--header {shlex.quote(f'{k}: {v}')}" for k, v in auth_headers().items()
        ]
        header_prefix = f"{' '.join(header_parts)} " if header_parts else ""
        return f"wget -qO- {header_prefix}{url} | tar -xz -C /dst"

    def _result_bundle_url(self, task_id: str) -> str:
        base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
        if not base_url:
            raise ExecutionError(
                "SSH input result hydration requires FLOWMESH_BASE_URL when "
                "upstream results are not available locally"
            )
        return (
            f"{base_url.rstrip('/')}/api/v1/results/{task_id}/bundle"
            "?include=results&include=artifacts"
        )

    def _download_result_bundle(self, task_id: str, destination_dir: Path) -> None:
        tmp_fd, tmp_str = tempfile.mkstemp(prefix="ssh_bundle_", suffix=".tar.gz")
        os.close(tmp_fd)
        tmp_path = Path(tmp_str)
        try:
            with requests.get(
                self._result_bundle_url(task_id),
                headers=auth_headers(),
                stream=True,
                timeout=_RESULT_BUNDLE_TIMEOUT_SEC,
            ) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as sink:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            sink.write(chunk)
            self._extract_result_bundle(tmp_path, destination_dir)
        except requests.RequestException as exc:
            raise ExecutionError(
                f"Failed to download SSH input result bundle for {task_id}: {exc}"
            ) from exc
        except tarfile.TarError as exc:
            raise ExecutionError(
                f"Failed to unpack SSH input result bundle for {task_id}: {exc}"
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _finish_requested(container: Container) -> bool:
        try:
            result = container.exec_run(
                ["sh", "-lc", f"test -f {shlex.quote(_FINISH_SENTINEL_PATH)}"]
            )
        except Exception:
            return False
        return result.exit_code == 0

    @staticmethod
    def _extract_result_bundle(bundle_path: Path, destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        dest_root = destination_dir.resolve()
        with tarfile.open(bundle_path, mode="r:*") as archive:
            for member in archive:
                member_path = (dest_root / member.name).resolve()
                try:
                    member_path.relative_to(dest_root)
                except ValueError as exc:
                    raise ExecutionError(
                        f"Unsafe path in SSH input result bundle: {member.name}"
                    ) from exc
                archive.extract(member, dest_root, filter="data")

    def _cleanup_mount_plan(
        self, client: DockerClient, mount_plan: SSHMountPlan
    ) -> None:
        if mount_plan.staged_inputs_dir is not None:
            shutil.rmtree(mount_plan.staged_inputs_dir, ignore_errors=True)
        if mount_plan.staged_inputs_volume is not None:
            try:
                client.volumes.get(mount_plan.staged_inputs_volume).remove(force=True)
            except Exception:
                logger.debug(
                    "Failed to remove staged SSH input volume %s",
                    mount_plan.staged_inputs_volume,
                    exc_info=True,
                )

    def _enforce_output_limit(
        self,
        container: Container,
        output_cfg: SSHOutputConfig | None,
        mount_plan: SSHMountPlan,
    ) -> None:
        if output_cfg is None or output_cfg.max_bytes is None:
            return
        max_bytes = output_cfg.max_bytes
        if max_bytes < 0:
            logger.warning(
                "Invalid maxBytes %d in SSH output config; ignoring limit", max_bytes
            )
            return

        if mount_plan.direct_output_path is not None:
            current_size = self._path_size_bytes(mount_plan.direct_output_path)
        elif mount_plan.copy_output_path is not None:
            current_size = self._container_path_size(
                container, mount_plan.copy_output_path
            )
        else:
            current_size = 0

        if current_size <= max_bytes:
            return

        logger.warning(
            "SSH output exceeded maxBytes (%d > %d)", current_size, max_bytes
        )
        try:
            container.stop(timeout=1)
        except Exception as exc:
            logger.debug("Failed to stop SSH container after maxBytes breach: %s", exc)
        raise ExecutionError(
            f"SSH sshOutput exceeded maxBytes ({current_size} > {max_bytes})"
        )

    @staticmethod
    def _path_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    @staticmethod
    def _container_path_size(container: Container, path: str) -> int:
        quoted = shlex.quote(path)
        result = container.exec_run(
            ["sh", "-lc", f"du -sb {quoted} 2>/dev/null | cut -f1 || echo 0"]
        )
        raw = result.output
        if isinstance(raw, bytes):
            output = raw.decode("utf-8", errors="ignore").strip()
        else:
            output = b"".join(raw).decode("utf-8", errors="ignore").strip()
        try:
            return int(output or "0")
        except ValueError:
            return 0

    def _copy_output_directory(
        self, container: Container, source_path: str, destination: Path
    ) -> None:
        self.ensure_dir(destination)
        try:
            stream, _ = container.get_archive(source_path)
        except Exception as exc:
            raise ExecutionError(
                f"Failed to collect SSH output from {source_path}: {exc}"
            ) from exc

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in stream:
                tmp.write(chunk)

        source_name = PurePosixPath(source_path).name
        try:
            with tarfile.open(tmp_path) as archive:
                for member in archive.getmembers():
                    relative = self._relative_archive_path(member.name, source_name)
                    if relative is None:
                        continue
                    target = destination / relative
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    with target.open("wb") as fh:
                        shutil.copyfileobj(extracted, fh)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _relative_archive_path(member_name: str, source_name: str) -> Path | None:
        parts = [
            part for part in PurePosixPath(member_name).parts if part not in ("", ".")
        ]
        if not parts:
            return None
        if source_name in parts:
            parts = parts[parts.index(source_name) + 1 :]
        if not parts or any(part == ".." for part in parts):
            return None
        return Path(*parts)

    @staticmethod
    def _reserve_mount_path(used_mount_paths: set[str], mount_path: str) -> None:
        if mount_path in used_mount_paths:
            raise ExecutionError(f"Duplicate SSH mountPath '{mount_path}'")
        used_mount_paths.add(mount_path)

    @staticmethod
    def _normalize_mount_path(path: str, field_name: str) -> str:
        normalized = PurePosixPath(path.strip())
        if not normalized.is_absolute():
            raise ExecutionError(f"{field_name} must be an absolute path")
        if normalized == PurePosixPath("/"):
            raise ExecutionError(f"{field_name} cannot be '/'")
        if (
            normalized != _SAFE_MOUNT_ROOT
            and _SAFE_MOUNT_ROOT not in normalized.parents
        ):
            raise ExecutionError(
                f"{field_name} must be under {_SAFE_MOUNT_ROOT.as_posix()}"
            )
        return normalized.as_posix()
