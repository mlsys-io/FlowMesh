import asyncio
import json
import os
import re
import threading
from collections import Counter
from enum import StrEnum
from typing import Any

from docker import DockerClient
from docker.errors import NotFound
from docker.models.containers import Container
from docker.types import DeviceRequest
from pydantic import BaseModel, Field

from shared.utils.docker import sanitize_container_name

from ... import env
from ...hooks import PrincipalContext
from ...utils.helpers import get_docker_client, get_logger
from ..resource_manager import GpuArch, ResourceManager
from ..schemas import WorkerHardware, WorkerInfo, WorkerStatus
from .base import (
    ProviderSpec,
    WorkerAdapter,
    WorkerConfig,
    WorkerFactory,
    WorkerTokenType,
)
from .utils import get_worker_image_name

_STOP_TIMEOUT = 30  # seconds
_PROVIDER_NAME = "docker"
_SSH_OWNER_LABEL = "flowmesh.ssh.worker_id"
_SSH_MANAGED_LABEL = "flowmesh.ssh.managed"
_ssh_network_suffix = sanitize_container_name(env.NODE_ALIAS, maxlen=32)
_SSH_NETWORK_NAME = f"flowmesh_ssh_{_ssh_network_suffix or 'default'}"

logger = get_logger()


class _VolumeInitializer:
    VOLUME_INIT_IMAGE = "busybox:1.36.1"
    WORKER_UID = 10001
    WORKER_GID = 10001

    _locks: dict[str, threading.Lock] = {}
    _locks_lock = threading.Lock()
    _initialized: set[str] = set()

    @classmethod
    def ensure(cls, client: DockerClient, volume_name: str, mount_path: str) -> None:
        if not volume_name:
            return
        with cls._locks_lock:
            lock = cls._locks.get(volume_name)
            if lock is None:
                lock = threading.Lock()
                cls._locks[volume_name] = lock
        with lock:
            if volume_name in cls._initialized:
                return
            cls._prepare_volume(client, volume_name, mount_path)
            cls._initialized.add(volume_name)

    @classmethod
    def _prepare_volume(
        cls, client: DockerClient, volume_name: str, mount_path: str
    ) -> None:
        try:
            client.containers.run(
                image=cls.VOLUME_INIT_IMAGE,
                command=[
                    "sh",
                    "-c",
                    (
                        f"mkdir -p {mount_path} && "
                        f"chown -R {cls.WORKER_UID}:{cls.WORKER_GID} {mount_path}"
                    ),
                ],
                volumes={volume_name: {"bind": mount_path, "mode": "rw"}},
                remove=True,
            )
        except Exception as exc:
            logger.warning(
                (
                    "Failed to prepare Docker volume %s: %s. "
                    "Workers may lack write access."
                ),
                volume_name,
                repr(exc),
            )


class WorkerType(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class SSHConfig(BaseModel):
    default_image: str | None = env.SSH_DEFAULT_IMAGE
    """Default container image for SSH sessions"""
    default_user: str | None = env.SSH_DEFAULT_USER
    """Default SSH username"""
    default_ttl_sec: float | None = env.SSH_DEFAULT_TTL_SEC
    """Default session TTL in seconds"""
    default_idle_sec: float | None = env.SSH_DEFAULT_IDLE_SEC
    """Default idle timeout in seconds"""
    max_ttl_sec: float | None = env.SSH_MAX_TTL_SEC
    """Maximum allowed TTL in seconds"""
    poll_interval_sec: float | None = env.SSH_POLL_INTERVAL_SEC
    """Container status poll interval in seconds"""
    stop_timeout_sec: float | None = env.SSH_STOP_TIMEOUT_SEC
    """Seconds to wait when stopping a session container"""

    def to_env(self) -> dict[str, str]:
        """Return env vars to inject into the worker container."""
        mapping = {
            "SSH_DEFAULT_IMAGE": self.default_image,
            "SSH_DEFAULT_USER": self.default_user,
            "SSH_DEFAULT_TTL_SEC": self.default_ttl_sec,
            "SSH_DEFAULT_IDLE_SEC": self.default_idle_sec,
            "SSH_MAX_TTL_SEC": self.max_ttl_sec,
            "SSH_POLL_INTERVAL_SEC": self.poll_interval_sec,
            "SSH_STOP_TIMEOUT_SEC": self.stop_timeout_sec,
        }
        return {k: str(v) for k, v in mapping.items() if v is not None}


class DockerWorkerConfig(WorkerConfig):
    container_name: str | None = None
    """Optional Docker container name"""
    worker_type: WorkerType = WorkerType.CPU
    """Type of worker (cpu or gpu)"""
    cuda_devices: list[int] | None = None
    """List of CUDA devices to use (if any)"""
    gpu_count: int = 1
    """Number of GPUs to auto-pick when ``cuda_devices`` is unset
    (only consulted for GPU workers)."""
    docker_registry: str = env.FLOWMESH_REGISTRY
    """Docker registry to pull worker images from"""
    version: str = env.FLOWMESH_VERSION
    """Worker Docker image version tag"""
    enable_ssh: bool = env.ENABLE_SSH_BY_DEFAULT
    """Whether to enable support for SSH jobs"""
    ssh: SSHConfig = Field(default_factory=SSHConfig)
    """Default SSH session configuration"""

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        if self.worker_type == WorkerType.GPU and (
            isinstance(self.cuda_devices, list) and len(self.cuda_devices) == 0
        ):
            raise ValueError("Expected at least one CUDA device for GPU worker.")


class DockerWorkerInfo(WorkerInfo):
    pass


class DockerWorkerAdapter(WorkerAdapter):
    CONTAINER_RESULTS_DIR: str = "/var/lib/flowmesh-results"
    CONTAINER_HF_CACHE_DIR: str = "/home/appuser/.cache/huggingface"
    HF_CACHE_VOLUME: str | None = "flowmesh_server_hf_cache"
    DOCKER_SOCKET_PATH: str = "/var/run/docker.sock"

    def __init__(
        self,
        token: WorkerTokenType,
        name: str,
        container_name: str,
        cuda_devices: list[int] | None,
        gpu_arch: GpuArch | None,
        config: DockerWorkerConfig,
        docker_client: DockerClient,
        owner: PrincipalContext,
    ) -> None:
        if config.worker_type == WorkerType.GPU and (
            cuda_devices is None or len(cuda_devices) == 0
        ):
            raise ValueError("Expected at least one CUDA device for GPU worker.")

        super().__init__(token, name, config, owner)

        self.config: DockerWorkerConfig
        self.container_name = container_name
        self.cuda_devices = cuda_devices
        self.gpu_arch = gpu_arch

        self._docker = docker_client
        self._status: WorkerStatus = WorkerStatus.STOPPED
        self._hardware: dict[str, Any] | WorkerHardware | None = None

    @property
    def status(self) -> WorkerStatus:
        return self._status

    def set_status(self, status: WorkerStatus) -> None:
        self._status = status

    def get_info(self) -> DockerWorkerInfo:
        hardware = self._hardware
        if isinstance(hardware, dict):
            hardware = WorkerHardware.model_validate(hardware)
            self._hardware = hardware
        return DockerWorkerInfo(
            id=self.worker_id,
            name=self.name,
            provider=_PROVIDER_NAME,
            status=self.status,
            hardware=hardware,
        )

    async def start(self) -> bool:
        self.set_status(WorkerStatus.STARTING)
        try:
            ok = await asyncio.to_thread(self._start)
            if not ok:
                self.set_status(WorkerStatus.STOPPED)
            return ok
        except Exception:
            self.set_status(WorkerStatus.STOPPED)
            raise

    async def prepare(self) -> None:
        self._hardware = await asyncio.to_thread(self._probe_hardware)

    async def stop(self) -> bool:
        prev_status = self.status
        if prev_status in (WorkerStatus.STOPPING, WorkerStatus.STOPPED):
            return True
        self.set_status(WorkerStatus.STOPPING)
        try:
            ok = await asyncio.to_thread(self._stop)
            if not ok:
                self.set_status(prev_status)
            return ok
        except Exception:
            self.set_status(prev_status)
            raise

    def get_image_name(self) -> str:
        return get_worker_image_name(
            self.config.docker_registry, self.config.version, self.gpu_arch
        )

    def _start(self) -> bool:
        existing: Container | None = None
        try:
            existing = self._docker.containers.get(self.container_name)
        except NotFound:
            pass
        except Exception as exc:
            logger.warning(
                "Failed to inspect Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return False

        if existing is not None:
            try:
                existing.reload()
            except Exception as exc:
                logger.warning(
                    "Failed to reload Docker container %s: %s",
                    self.container_name,
                    repr(exc),
                )
                return False

            if existing.status == "running":
                self._is_started = True
                logger.warning("Container %s is already running.", self.container_name)
                return True

            try:
                existing.remove(force=True)
                logger.debug("Removed stale container %s", self.container_name)
            except Exception as exc:
                logger.error(
                    "Failed to remove stale container %s: %s",
                    self.container_name,
                    repr(exc),
                )
                return False

        environment: dict[str, str] = self._base_environment()
        labels: dict[str, str] = self._base_labels()
        volumes: list[str] = []
        self._mount_results(volumes)
        self._mount_hf_cache(volumes)
        self._mount_docker_socket(volumes)
        device_requests, runtime = self._apply_worker_type_settings(environment, labels)
        docker_gid = self._get_docker_socket_gid()

        try:
            run_kwargs: dict[str, Any] = {
                "image": self.get_image_name(),
                "name": self.container_name,
                "environment": environment,
                "labels": labels,
                "volumes": volumes,
                "network_mode": "host",
                "device_requests": device_requests,
                "restart_policy": {"Name": "unless-stopped"},
                "detach": True,
            }
            if runtime is not None:
                run_kwargs["runtime"] = runtime
            if docker_gid:
                run_kwargs["group_add"] = [docker_gid]
            self._docker.containers.run(**run_kwargs)
            self._is_started = True
        except Exception as exc:
            logger.error(
                "Failed to start Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return False

        if self._hardware is None:
            self._hardware = self._probe_hardware()

        return True

    def _probe_hardware(self) -> dict[str, Any] | None:
        logger.debug("Collecting hardware info for worker %s", self.container_name)
        container = self._get_running_container()
        output_prefix = "HW_PROBE_OUTPUT: "
        cmd = self._hardware_probe_cmd(output_prefix)
        output: bytes | None
        if container is None:
            # Probe hardware in a temporary container
            environment: dict[str, str] = self._base_environment()
            device_requests, runtime = self._apply_worker_type_settings(
                environment, None
            )
            try:
                run_kwargs: dict[str, Any] = {
                    "image": self.get_image_name(),
                    "command": cmd,
                    "environment": environment,
                    "network_mode": "host",
                    "device_requests": device_requests,
                    "remove": True,
                    "stdout": True,
                    "stderr": True,
                }
                if runtime is not None:
                    run_kwargs["runtime"] = runtime
                output = self._docker.containers.run(**run_kwargs)
            except Exception as exc:
                logger.warning(
                    "Failed to run hardware probe for worker %s: %s",
                    self.container_name,
                    repr(exc),
                )
                return None
        else:
            # Probe hardware in the existing container
            try:
                result = container.exec_run(cmd, stdout=True, stderr=True)
            except Exception as exc:
                logger.warning(
                    "Failed to exec hardware probe for worker %s: %s",
                    self.container_name,
                    repr(exc),
                )
                return None
            exit_code = getattr(result, "exit_code", None)
            output = getattr(result, "output", None)
            if exit_code not in (0, None):
                logger.warning(
                    "Hardware probe exec failed for %s with exit code %s",
                    self.container_name,
                    exit_code,
                )
            if output is None:
                return None

        return self._parse_hardware_output(output, output_prefix)

    def _stop(self) -> bool:
        is_started = self._is_started
        try:
            container = self._docker.containers.get(self.container_name)
        except NotFound:
            self._is_started = False
            if is_started:
                logger.warning("Container %s not found.", self.container_name)
            return True
        except Exception as exc:
            log_fn = logger.error if is_started else logger.warning
            log_fn(
                "Failed to fetch Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return False

        self._stop_owned_ssh_containers()
        self._remove_owned_ssh_volumes()

        try:
            container.stop(timeout=_STOP_TIMEOUT)
            container.remove()
            self._is_started = False
            return True
        except Exception as exc:
            log_fn = logger.error if is_started else logger.warning
            log_fn(
                "Failed to stop Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return False

    def _stop_owned_ssh_containers(self) -> None:
        try:
            containers = self._docker.containers.list(
                all=True, filters={"label": f"{_SSH_OWNER_LABEL}={self.container_name}"}
            )
        except Exception as exc:
            logger.warning(
                "Failed to list SSH session containers for worker %s: %s",
                self.container_name,
                repr(exc),
            )
            return

        for ssh_container in containers:
            try:
                ssh_container.reload()
                if ssh_container.status == "running":
                    ssh_container.stop(timeout=_STOP_TIMEOUT)
            except Exception as exc:
                logger.warning(
                    "Failed to stop SSH session container %s: %s",
                    ssh_container.name,
                    repr(exc),
                )
            try:
                ssh_container.remove(force=True)
            except Exception as exc:
                logger.warning(
                    "Failed to remove SSH session container %s: %s",
                    ssh_container.name,
                    repr(exc),
                )

    def _remove_owned_ssh_volumes(self) -> None:
        try:
            volumes = self._docker.volumes.list(
                filters={
                    "label": [
                        f"{_SSH_OWNER_LABEL}={self.container_name}",
                        f"{_SSH_MANAGED_LABEL}=true",
                    ]
                }
            )
        except Exception as exc:
            logger.warning(
                "Failed to list SSH staging volumes for worker %s: %s",
                self.container_name,
                repr(exc),
            )
            return

        for volume in volumes:
            try:
                volume.remove(force=True)
            except Exception as exc:
                logger.warning(
                    "Failed to remove SSH staging volume %s: %s",
                    volume.name,
                    repr(exc),
                )

    def _base_environment(self) -> dict[str, str]:
        environment = super()._base_environment()
        environment["RESULTS_DIR"] = self.CONTAINER_RESULTS_DIR
        environment["RESULTS_MOUNT_SOURCE"] = self.config.results_dir
        environment["FLOWMESH_REGISTRY"] = self.config.docker_registry
        environment["FLOWMESH_VERSION"] = self.config.version
        environment["WORKER_NETWORK_MODE"] = f"container:{self.container_name}"
        environment["WORKER_CONTAINER_NAME"] = self.container_name
        environment["SSH_NETWORK_NAME"] = _SSH_NETWORK_NAME
        environment.update(self.config.ssh.to_env())
        return environment

    def _apply_worker_type_settings(
        self,
        environment: dict[str, str],
        labels: dict[str, str] | None,
    ) -> tuple[list[DeviceRequest] | None, str | None]:
        """Apply worker type specific settings to environment and labels.

        Returns:
            device_requests: list[DeviceRequest] | None
                Device requests for Docker container.
            runtime: str | None
                Runtime to use for Docker container.
        """
        device_requests: list[DeviceRequest] | None
        runtime: str | None
        match self.config.worker_type:
            case WorkerType.CPU:
                if labels is not None:
                    labels["flowmesh.worker.type"] = "cpu"
                device_requests = None
                runtime = None
            case WorkerType.GPU:
                assert self.cuda_devices is not None
                assert self.gpu_arch is not None
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in range(len(self.cuda_devices))
                )
                cuda_devices_str = [str(i) for i in self.cuda_devices]
                gpu_ids = ",".join(cuda_devices_str)
                environment["WORKER_HOST_GPU_ID"] = gpu_ids
                gpu_arch = self.gpu_arch.value
                environment["WORKER_HOST_GPU_ARCH"] = gpu_arch
                if labels is not None:
                    labels["flowmesh.worker.type"] = "gpu"
                    labels["flowmesh.worker.gpu_id"] = gpu_ids
                    labels["flowmesh.worker.gpu_arch"] = gpu_arch
                device_requests = [
                    DeviceRequest(device_ids=cuda_devices_str, capabilities=[["gpu"]])
                ]
                runtime = env.DOCKER_GPU_RUNTIME
            case _:
                raise ValueError(f"Unsupported worker type: {self.config.worker_type}")
        return device_requests, runtime

    def _base_labels(self) -> dict[str, str]:
        return {
            "flowmesh.role": "worker",
            "flowmesh.group": "server-workers",
        }

    def _ensure_volume_access(self, source: str, container_path: str) -> None:
        if not source or os.path.isabs(source):
            return
        _VolumeInitializer.ensure(self._docker, source, container_path)

    def _mount_results(self, volumes: list[str]) -> None:
        source = self.config.results_dir
        container_results_dir = self.CONTAINER_RESULTS_DIR
        self._ensure_volume_access(source, container_results_dir)
        result_mnt = f"{source}:{container_results_dir}"
        volumes.append(result_mnt)

    def _mount_hf_cache(self, volumes: list[str]) -> None:
        hf_cache_dir = self.config.hf_cache_dir
        container_cache_dir = self.CONTAINER_HF_CACHE_DIR
        if hf_cache_dir is not None:
            self._ensure_volume_access(hf_cache_dir, container_cache_dir)
            volumes.append(f"{hf_cache_dir}:{container_cache_dir}")
            return

        hf_cache_volume = self.HF_CACHE_VOLUME
        if hf_cache_volume is not None:
            # Create the volume if it doesn't exist
            found = self._docker.volumes.list(filters={"name": hf_cache_volume})
            if not found:
                self._docker.volumes.create(name=hf_cache_volume)
            self._ensure_volume_access(hf_cache_volume, container_cache_dir)
            volumes.append(f"{hf_cache_volume}:{container_cache_dir}")
            return

    def _mount_docker_socket(self, volumes: list[str]) -> None:
        if self.config.enable_ssh:
            volumes.append(f"{self.DOCKER_SOCKET_PATH}:{self.DOCKER_SOCKET_PATH}")

    def _get_docker_socket_gid(self) -> int | None:
        if not self.config.enable_ssh:
            return None
        try:
            gid = os.stat(self.DOCKER_SOCKET_PATH).st_gid
        except OSError as exc:
            logger.warning(
                "Failed to inspect Docker socket %s for worker %s: %s",
                self.DOCKER_SOCKET_PATH,
                self.container_name,
                repr(exc),
            )
            return None
        return gid

    def _hardware_probe_cmd(self, prefix: str | None) -> list[str]:
        cmd = ["python", "-m", "worker.main", "--collect-hw"]
        bandwidth = self.config.network_bandwidth
        if bandwidth is not None:
            cmd.extend(["--bandwidth-bytes-per-sec", str(bandwidth)])
        if prefix:
            cmd.extend(["--collect-hw-prefix", prefix])
        return cmd

    def _get_running_container(self) -> Container | None:
        try:
            container = self._docker.containers.get(self.container_name)
        except NotFound:
            return None
        except Exception as exc:
            logger.warning(
                "Failed to inspect Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return None
        try:
            container.reload()
        except Exception as exc:
            logger.warning(
                "Failed to reload Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return None
        return container if container.status == "running" else None

    def _parse_hardware_output(
        self, output: bytes | str | None, prefix: str | None
    ) -> dict[str, Any] | None:
        if output is None:
            return None
        if isinstance(output, (bytes, bytearray)):
            output_text = output.decode("utf-8", errors="replace").strip()
        else:
            output_text = str(output).strip()
        if not output_text:
            logger.warning(
                "Hardware probe returned no output for %s", self.container_name
            )
            return None
        if prefix:
            # Find the output line and strip the prefix
            for line in output_text.splitlines():
                if line.startswith(prefix):
                    output_text = line.removeprefix(prefix)
                    break
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid hardware probe output for %s: %s",
                self.container_name,
                repr(exc),
            )
            logger.debug("Hardware probe output was: %s", output_text)
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "Hardware probe output for %s is not a JSON object", self.container_name
            )
            return None
        return payload


class DockerWorkerFactory(WorkerFactory):
    _CONTAINER_NAME_MAX_LEN = 128
    _CONTAINER_NAME_ALLOWED_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def __init__(self, system_principal: PrincipalContext) -> None:
        super().__init__(system_principal)
        self._rm = ResourceManager.get_instance()
        self._docker = get_docker_client()
        self._worker_id_registry: Counter[str] = Counter()

    def create_worker(
        self, token: WorkerTokenType, config: DockerWorkerConfig
    ) -> DockerWorkerAdapter:
        cuda_devices: list[int] | None
        gpu_arch: GpuArch | None
        match config.worker_type:
            case WorkerType.CPU:
                cuda_devices = None
                gpu_arch = None
            case WorkerType.GPU:
                if config.cuda_devices is None:
                    cuda_devices, gpu_arch = self._rm.reserve_gpus(n=config.gpu_count)
                else:
                    cuda_devices, gpu_arch = self._rm.reserve_gpus(
                        devices=config.cuda_devices
                    )

        name = self._resolve_worker_name(config)
        container_name = (
            config.container_name
            if config.container_name
            else self._sanitize_container_name(name, config)
        )
        worker = DockerWorkerAdapter(
            token=token,
            name=name,
            container_name=container_name,
            cuda_devices=cuda_devices,
            gpu_arch=gpu_arch,
            config=config,
            docker_client=self._docker,
            owner=self.system_principal,
        )
        return worker

    def destroy_worker(self, worker: WorkerAdapter) -> None:
        if not isinstance(worker, DockerWorkerAdapter):
            raise ValueError("Invalid worker type")

        if worker.cuda_devices:
            self._rm.deallocate_gpus(worker.cuda_devices)

    def cleanup(self) -> None:
        self._remove_ssh_network()

    def _remove_ssh_network(self) -> None:
        network_name = _SSH_NETWORK_NAME
        try:
            existing = self._docker.networks.list(names=[network_name])
            for net in existing:
                labels = net.attrs.get("Labels") or {}
                if (
                    net.name == network_name
                    and labels.get(_SSH_MANAGED_LABEL) == "true"
                ):
                    net.remove()
                    logger.info("Removed SSH network %s", network_name)
        except Exception:
            pass

    def _get_next_worker_id(self, prefix: str) -> int:
        containers: list[Container] = self._docker.containers.list(
            all=True, filters={"name": prefix}
        )
        max_id = -1
        for container in containers:
            name = container.name
            assert isinstance(name, str)
            try:
                cur_id = int(name.rsplit("_", maxsplit=1)[-1])
            except ValueError:
                continue
            if cur_id > max_id:
                max_id = cur_id

        registered_id = self._worker_id_registry[prefix]
        if registered_id > max_id:
            container_id = registered_id
        else:
            container_id = max_id + 1
        self._worker_id_registry[prefix] = container_id + 1
        return container_id

    def _resolve_worker_name(self, config: DockerWorkerConfig) -> str:
        return config.worker_alias or self._get_next_worker_name(config.worker_type)

    def _sanitize_container_name(self, value: str, config: DockerWorkerConfig) -> str:
        raw = str(value or "").strip()
        if not raw:
            return self._get_next_worker_name(config.worker_type)
        sanitized = sanitize_container_name(raw, self._CONTAINER_NAME_MAX_LEN)
        if not sanitized or not self._CONTAINER_NAME_ALLOWED_RE.match(sanitized):
            return self._get_next_worker_name(config.worker_type)
        return sanitized

    def _get_next_worker_name(self, worker_type: WorkerType) -> str:
        match worker_type:
            case WorkerType.CPU:
                prefix = "flowmesh_server_worker_cpu_"
            case WorkerType.GPU:
                prefix = "flowmesh_server_worker_gpu_"
            case _:
                raise ValueError(f"Unsupported worker type: {worker_type}")
        next_id = self._get_next_worker_id(prefix)
        return f"{prefix}{next_id}"


def get_provider_spec(system_principal: PrincipalContext) -> ProviderSpec:
    return ProviderSpec(
        name=_PROVIDER_NAME,
        config_cls=DockerWorkerConfig,
        adapter_cls=DockerWorkerAdapter,
        factory=DockerWorkerFactory(system_principal),
    )
