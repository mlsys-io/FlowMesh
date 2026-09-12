"""Docker session backend: one sibling container per SSH session.

Requires a reachable Docker daemon. The session container is isolated from the
worker (its own image, network, cgroup limits and GPU slice) and the supervisor
that dials its relay uplink shares a host with it, so its published port is
reachable on loopback.
"""

import io
import logging
import shlex
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from shared.tasks.worker_message import WorkerHardware
from shared.utils import parse_float_env
from shared.utils.http import auth_headers
from shared.utils.manifest import ARTIFACTS_DIR
from worker.config import WorkerConfig
from worker.executors.utils.docker import (
    DockerUnavailableError,
    docker_available,
    docker_client,
)

from ..base_executor import ExecutionError
from .base import (
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
    count_established_connections,
    is_ssh_ready,
    path_size_bytes,
)
from .config import (
    FINISH_SENTINEL_PATH,
    LABEL_MANAGED,
    LABEL_SESSION,
    LABEL_TASK,
    LABEL_WORKER,
    STOP_TIMEOUT_SEC,
    ResolvedSSHInput,
    SSHConfig,
    normalize_mount_path,
    reserve_mount_path,
)
from .inputs import (
    RESULT_BUNDLE_TIMEOUT_SEC,
    result_bundle_url,
    stage_inputs_locally,
)

try:
    from docker import DockerClient
    from docker.models.containers import Container
    from docker.types import DeviceRequest

    _HAS_DOCKER = True
except Exception:
    _HAS_DOCKER = False
    if TYPE_CHECKING:
        from docker import DockerClient
        from docker.models.containers import Container
        from docker.types import DeviceRequest
    else:
        DockerClient = Any
        Container = Any
        DeviceRequest = Any


logger = logging.getLogger(__name__)

_SESSION_SSH_PORT = 22
_CONTAINER_RESULTS_SOURCE_ROOT = "/root/.flowmesh/results-source"
_SSH_RUN_ENTRYPOINT_PATH = "/flowmesh-ssh-run.sh"
_SSH_RUN_SCRIPT_SOURCE = (
    Path(__file__).resolve().parent.parent.parent / "docker" / "ssh-run.sh"
)

type DemuxLogStream = Iterator[tuple[bytes | None, bytes | None]]


@dataclass(slots=True)
class SSHMountPlan:
    volumes: list[str]
    staged_input_specs: list[tuple[str, str]]
    create_dirs: list[str]
    direct_output_path: Path | None
    copy_output_path: str | None
    staged_inputs_dir: Path | None
    staged_inputs_volume: str | None


class DockerSessionBackend(SSHSessionBackend):
    name = "docker"
    supports_noninteractive = True

    def __init__(
        self, config: WorkerConfig, hardware: WorkerHardware | None = None
    ) -> None:
        super().__init__(config, hardware)
        self._docker: DockerClient | None = None
        self._docker_gpu_runtime: str | None = config.docker_gpu_runtime
        self._ssh_network: str | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return docker_available()

    def prepare(self) -> None:
        if not _HAS_DOCKER:
            raise ExecutionError("Docker SDK is not available (`pip install docker`).")
        self._docker = self._get_docker_client()
        self._ssh_network = self._ensure_ssh_network(self._docker)

    def teardown(self, worker_name: str) -> None:
        stop_timeout_sec = parse_float_env("SSH_STOP_TIMEOUT_SEC", STOP_TIMEOUT_SEC)
        client = self._get_docker_client()
        try:
            containers = client.containers.list(
                filters={"label": f"{LABEL_WORKER}={worker_name}"}
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

    def start_session(self, request: SessionRequest) -> "DockerSession":
        cfg = request.cfg
        client = self._get_docker_client()
        interactive = cfg.interactive

        if interactive:
            ports: dict[str, Any] = {f"{_SESSION_SSH_PORT}/tcp": None}
            container_cmd = None
        else:
            ports = {}
            container_cmd = self._resolve_noninteractive_command(client, cfg)

        container_name = (
            f"{request.worker_name}_ssh-"
            f"{request.task_id[:8]}-{request.session_id[:8]}"
        )
        mount_plan = self._build_mount_plan(
            client,
            request.out_dir,
            request.resolved_inputs,
            cfg,
            request.session_id,
            request.worker_name,
        )
        labels = {
            LABEL_WORKER: request.worker_name,
            LABEL_TASK: request.task_id,
            LABEL_SESSION: request.session_id,
            LABEL_MANAGED: "true",
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
        try:
            container, log_stream = self._start_container(client, kwargs, interactive)
        except Exception:
            self._cleanup_mount_plan(client, mount_plan)
            raise
        return DockerSession(
            client=client,
            container=container,
            container_name=container_name,
            cfg=cfg,
            mount_plan=mount_plan,
            log_stream=log_stream,
        )

    # ------------------------------------------------------------------ #
    # Docker plumbing
    # ------------------------------------------------------------------ #

    def _cuda_visible_devices(self, gpu_device_ids: list[str]) -> str | None:
        # Docker exposes only the sliced devices, which appear as 0..N-1
        # inside the container regardless of their host IDs.
        return ",".join(str(i) for i in range(len(gpu_device_ids)))

    def _get_docker_client(self) -> DockerClient:
        if self._docker is None:
            try:
                self._docker = docker_client()
            except DockerUnavailableError as exc:
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
                if net.name == net_name and labels.get(LABEL_MANAGED) == "true":
                    return net_name
        except Exception:
            pass
        try:
            client.networks.create(
                net_name,
                driver="bridge",
                options={"com.docker.network.bridge.enable_icc": "false"},
                labels={LABEL_MANAGED: "true"},
            )
            logger.info("Created isolated network")
            return net_name
        except Exception:
            logger.warning(
                "Failed to create isolated network; falling back to default network"
            )
            return None

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

    def _resolve_noninteractive_command(
        self, client: DockerClient, cfg: SSHConfig
    ) -> list[str]:
        if cfg.entrypoint is not None and cfg.command is not None:
            return cfg.entrypoint + cfg.command
        if cfg.entrypoint is not None:
            return cfg.entrypoint
        if cfg.command is not None:
            return cfg.command

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
    def _build_ssh_run_archive() -> bytes:
        script_bytes = _SSH_RUN_SCRIPT_SOURCE.read_bytes()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=_SSH_RUN_ENTRYPOINT_PATH.lstrip("/"))
            info.size = len(script_bytes)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(script_bytes))
        return stream.getvalue()

    # ------------------------------------------------------------------ #
    # Mount planning
    # ------------------------------------------------------------------ #

    def _build_mount_plan(
        self,
        client: DockerClient,
        out_dir: Path,
        resolved_inputs: list[ResolvedSSHInput],
        cfg: SSHConfig,
        session_id: str,
        worker_name: str,
    ) -> SSHMountPlan:
        volumes: list[str] = []
        staged_input_specs: list[tuple[str, str]] = []
        create_dirs: list[str] = []
        used_mount_paths: set[str] = set()
        results_source = self._config.results_mount_source
        staged_inputs_dir: Path | None = None
        staged_inputs_volume: str | None = None

        if results_source and resolved_inputs:
            staged_inputs_volume = self._stage_inputs_in_volume(
                client, resolved_inputs, results_source, session_id, worker_name
            )
            volumes.append(
                f"{staged_inputs_volume}:{_CONTAINER_RESULTS_SOURCE_ROOT}:ro"
            )
        elif resolved_inputs:
            staged_inputs_dir = stage_inputs_locally(resolved_inputs, session_id)

        for resolved in resolved_inputs:
            reserve_mount_path(used_mount_paths, resolved.mount_path)
            if results_source:
                staged_input_specs.append(
                    (
                        resolved.mount_path,
                        f"{_CONTAINER_RESULTS_SOURCE_ROOT}/{resolved.task_id}",
                    )
                )
            else:
                assert staged_inputs_dir is not None
                staged_input_path = staged_inputs_dir / resolved.task_id
                volumes.append(f"{staged_input_path}:{resolved.mount_path}:ro")

        direct_output_path: Path | None = None
        copy_output_path: str | None = None
        if cfg.output is not None:
            output_mount_path = normalize_mount_path(
                cfg.output.mount_path, field_name="sshOutput.mountPath"
            )
            reserve_mount_path(used_mount_paths, output_mount_path)
            artifacts_dir = out_dir / ARTIFACTS_DIR
            if results_source:
                create_dirs.append(output_mount_path)
                copy_output_path = output_mount_path
            else:
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

    def _stage_inputs_in_volume(
        self,
        client: DockerClient,
        resolved_inputs: list[ResolvedSSHInput],
        results_source: str,
        session_id: str,
        worker_name: str,
    ) -> str:
        volume_name = f"flowmesh_ssh_inputs_{session_id}"
        volume = client.volumes.create(
            name=volume_name,
            labels={
                LABEL_WORKER: worker_name,
                LABEL_SESSION: session_id,
                LABEL_MANAGED: "true",
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

    @staticmethod
    def _build_remote_stage_command(task_id: str) -> str:
        url = shlex.quote(result_bundle_url(task_id))
        header_parts = [
            f"--header {shlex.quote(f'{k}: {v}')}" for k, v in auth_headers().items()
        ]
        header_prefix = f"{' '.join(header_parts)} " if header_parts else ""
        timeout = int(RESULT_BUNDLE_TIMEOUT_SEC)
        return f"wget -qO- -T {timeout} -t 1 {header_prefix}{url} | tar -xz -C /dst"

    @staticmethod
    def _cleanup_mount_plan(client: DockerClient, mount_plan: SSHMountPlan) -> None:
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


class DockerSession(SSHSession):
    """An SSH session running as a sibling Docker container."""

    def __init__(
        self,
        client: DockerClient,
        container: Container,
        container_name: str,
        cfg: SSHConfig,
        mount_plan: SSHMountPlan,
        log_stream: DemuxLogStream | None,
    ) -> None:
        self._client = client
        self._container = container
        self._container_name = container_name
        self._cfg = cfg
        self._mount_plan = mount_plan
        self._log_stream = log_stream

    def wait_ready(self, timeout_sec: float = 30.0) -> int:
        """Wait until Docker assigns a host port and sshd accepts connections."""
        container = self._container
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
                port_bindings = container.ports.get(f"{_SESSION_SSH_PORT}/tcp")
                if port_bindings:
                    host_port = int(port_bindings[0]["HostPort"])
                    if is_ssh_ready("127.0.0.1", host_port):
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

    def poll(self) -> int | None:
        container = self._container
        container.reload()
        if container.status in ("running", "restarting"):
            return None
        return int(container.wait()["StatusCode"])

    def finish_requested(self) -> bool:
        try:
            result = self._container.exec_run(
                ["sh", "-lc", f"test -f {shlex.quote(FINISH_SENTINEL_PATH)}"]
            )
        except Exception:
            return False
        return result.exit_code == 0

    def established_connections(self) -> int | None:
        try:
            result = self._container.exec_run(
                ["sh", "-lc", "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null"]
            )
        except Exception:
            return None
        if result.exit_code != 0:
            return None
        output = _decode_exec_output(result.output)
        if not output:
            return None
        return count_established_connections(output, _SESSION_SSH_PORT)

    def output_size_bytes(self) -> int | None:
        plan = self._mount_plan
        if plan.direct_output_path is not None:
            return path_size_bytes(plan.direct_output_path)
        if plan.copy_output_path is not None:
            return self._container_path_size(plan.copy_output_path)
        return None

    def collect_output(self, destination: Path) -> None:
        source_path = self._mount_plan.copy_output_path
        if source_path is None:
            return
        destination.mkdir(parents=True, exist_ok=True)
        try:
            stream, _ = self._container.get_archive(source_path)
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
                    relative = _relative_archive_path(member.name, source_name)
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

    def stop(self, timeout_sec: float) -> None:
        try:
            self._container.stop(timeout=timeout_sec)
        except Exception as exc:
            logger.debug("Error stopping container: %s", exc)

    def cleanup(self) -> None:
        try:
            self._container.remove(force=True)
            logger.info("Removed SSH session container")
        except Exception as exc:
            logger.debug("Error removing container: %s", exc)
        DockerSessionBackend._cleanup_mount_plan(self._client, self._mount_plan)

    def drain_logs(self) -> None:
        if self._log_stream is not None:
            self._stream_container_logs(self._log_stream)

    def save_logs(self, out_dir: Path) -> None:
        """Save container stdout/stderr to the output directory."""
        try:
            logs = self._container.logs(stdout=True, stderr=True)
            if isinstance(logs, bytes) and logs:
                logs_dir = out_dir / ARTIFACTS_DIR / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / "container_output.log"
                log_path.write_bytes(logs)
        except Exception as exc:
            logger.debug("Failed to capture container logs: %s", exc)

    def _container_path_size(self, path: str) -> int | None:
        quoted = shlex.quote(path)
        try:
            result = self._container.exec_run(
                ["sh", "-lc", f"du -sb {quoted} 2>/dev/null | cut -f1 || echo 0"]
            )
        except Exception:
            return None
        try:
            return int(_decode_exec_output(result.output) or "0")
        except ValueError:
            return 0

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

        for stream_name, leftover in buffers.items():
            if leftover:
                _emit(leftover, stream_name)


def _decode_exec_output(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore").strip()
    if raw is None:
        return ""
    return b"".join(raw).decode("utf-8", errors="ignore").strip()


def _relative_archive_path(member_name: str, source_name: str) -> Path | None:
    parts = [part for part in PurePosixPath(member_name).parts if part not in ("", ".")]
    if not parts:
        return None
    if source_name in parts:
        parts = parts[parts.index(source_name) + 1 :]
    if not parts or any(part == ".." for part in parts):
        return None
    return Path(*parts)
