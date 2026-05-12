"""Local Docker and compose helpers for FlowMesh tooling."""

import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


class DockerError(RuntimeError):
    """Raised when a docker command fails."""


def ensure_docker_available() -> None:
    """Verify that the docker CLI is available."""
    if shutil.which("docker") is None:
        raise DockerError("docker is required but was not found in PATH")


def compose(
    compose_file: Path,
    env_file: Path | None,
    args: Iterable[str],
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run docker compose with the provided arguments."""
    ensure_docker_available()
    cmd = ["docker", "compose"]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    cmd += ["-f", str(compose_file)]
    cmd += list(args)
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        check=False,
        capture_output=capture_output,
        text=True,
        env=merged_env,
    )


def compose_logs(
    compose_file: Path,
    env_file: Path | None,
    env: Mapping[str, str] | None = None,
    service: str | None = None,
    capture_output: bool = False,
    profile: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Stream compose logs, optionally for a specific service."""
    args: list[str] = []
    if profile:
        args += ["--profile", profile]
    args += ["logs", "-f"]
    if service:
        args.append(service)
    return compose(
        compose_file=compose_file,
        env_file=env_file,
        args=args,
        env=env,
        capture_output=capture_output,
    )


def container_logs(
    container: str,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Stream logs for a single container via docker logs."""
    ensure_docker_available()
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["docker", "logs", "-f", container],
        check=False,
        capture_output=capture_output,
        text=True,
        env=merged_env,
    )


def inspect_image(
    image: str, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    """Inspect a docker image by tag."""
    ensure_docker_available()
    return subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        capture_output=capture_output,
        text=True,
    )


def remove_image(
    image: str, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    """Remove a docker image by tag."""
    ensure_docker_available()
    return subprocess.run(
        ["docker", "rmi", image], check=False, capture_output=capture_output, text=True
    )


def image_env_overrides(image_tag: str | None) -> dict[str, str]:
    """Build ``FLOWMESH_VERSION`` overrides for compose commands."""
    env: dict[str, str] = {}
    if image_tag:
        env["FLOWMESH_VERSION"] = image_tag
    return env


@dataclass
class ServerWorkerStatusRow:
    name: str
    status: str
    worker_type: str
    gpu_id: str


def server_worker_status_rows(
    *, include_exited: bool = False
) -> list[ServerWorkerStatusRow]:
    """Return status rows for local server worker containers."""

    def _inspect(name: str, fmt: str) -> str:
        result = subprocess.run(
            ["docker", "inspect", "--format", fmt, name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    ensure_docker_available()
    cmd = [
        "docker",
        "ps",
        "-a" if include_exited else "",
        "--filter",
        "label=flowmesh.group=server-workers",
        "--format",
        "{{.Names}}\t{{.Status}}",
    ]
    cmd = [part for part in cmd if part]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    rows = result.stdout.strip()
    if not rows:
        return []

    items: list[ServerWorkerStatusRow] = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        name, status = (line.split("\t", 1) + [""])[:2]
        worker_type = _inspect(name, '{{index .Config.Labels "flowmesh.worker.type"}}')
        gpu_id = _inspect(name, '{{index .Config.Labels "flowmesh.worker.gpu_id"}}')
        if not worker_type:
            env_dump = _inspect(name, "{{range .Config.Env}}{{println .}}{{end}}")
            detected_gpu = ""
            for entry in env_dump.splitlines():
                if entry.startswith("WORKER_HOST_GPU_ID="):
                    detected_gpu = entry.split("=", 1)[1]
                    break
            if detected_gpu:
                worker_type = "gpu"
                gpu_id = detected_gpu
            else:
                worker_type = "cpu"
        items.append(
            ServerWorkerStatusRow(
                name=name,
                status=status,
                worker_type=worker_type or "unknown",
                gpu_id=gpu_id or "-",
            )
        )
    return items


@dataclass
class DockerComposeStack:
    """Local compose wrapper."""

    compose_file: Path
    """Path to the compose file used for all stack operations."""
    env_file_var: str
    """
    Environment variable name passed to compose to point at the selected env file.
    """
    load_env: Callable[[Path], None]
    """
    Callback that loads and resolves env-file values before stack operations run.
    """
    ensure_deploy_paths: Callable[[Path], None] | None = None
    """
    Optional callback that prepares required local files and directories before 
    deployment-oriented compose commands.
    """

    def run(
        self,
        args: list[str],
        env_file: Path,
        env: dict[str, str] | None = None,
        to_deploy: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``docker compose`` for this stack."""
        self.load_env(env_file)
        if to_deploy and self.ensure_deploy_paths is not None:
            self.ensure_deploy_paths(Path.cwd())
        compose_env = {self.env_file_var: str(env_file.resolve())}
        if env:
            compose_env.update(env)
        return compose(
            compose_file=self.compose_file,
            env_file=env_file,
            args=args,
            env=compose_env,
        )

    def stream_logs(
        self,
        env_file: Path,
        service: str | None = None,
        profile: str | None = None,
    ) -> int:
        """Stream stack logs and fall back to container logs when needed."""
        self.load_env(env_file)
        compose_env = {self.env_file_var: str(env_file.resolve())}
        if service:
            result = compose_logs(
                compose_file=self.compose_file,
                env_file=env_file,
                env=compose_env,
                service=service,
                capture_output=False,
                profile=profile,
            )
            if result.returncode == 0:
                return 0
            fallback = container_logs(container=service, env=None, capture_output=False)
            return fallback.returncode

        result = compose_logs(
            compose_file=self.compose_file,
            env_file=env_file,
            env=compose_env,
            capture_output=False,
            profile=profile,
        )
        return result.returncode
