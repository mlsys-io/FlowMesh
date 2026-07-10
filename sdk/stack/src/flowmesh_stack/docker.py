"""Local Docker and compose helpers for FlowMesh tooling."""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .images import managed_repos, parse_image_ref

FLOWMESH_IMAGE_SOURCE = "https://github.com/mlsys-io/FlowMesh"
"""``org.opencontainers.image.source`` label carried by every FlowMesh image."""


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


def image_env_overrides(image_tag: str | None) -> dict[str, str]:
    """Build ``FLOWMESH_VERSION`` overrides for compose commands."""
    env: dict[str, str] = {}
    if image_tag:
        env["FLOWMESH_VERSION"] = image_tag
    return env


@dataclass
class ManagedImage:
    """A FlowMesh Docker image present on the local daemon."""

    repo: str
    tag: str | None
    target: str | None
    version: str | None
    image_id: str
    size_bytes: int
    created: datetime
    dangling: bool
    in_use: bool

    @property
    def removal_ref(self) -> str:
        """Reference to pass to ``docker rmi`` (tag when tagged, else image id)."""
        return self.tag if self.tag else self.image_id


@dataclass
class RemovalResult:
    """Outcome of removing a single image reference."""

    ref: str
    ok: bool
    error: str | None = None


_EPOCH = datetime.fromtimestamp(0, tz=UTC)
_TIMESTAMP_FRACTION = re.compile(r"(\.\d{6})\d+")


def _parse_docker_timestamp(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    if not text:
        return _EPOCH
    text = _TIMESTAMP_FRACTION.sub(r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return _EPOCH
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _docker_json_lines(
    result: subprocess.CompletedProcess[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _inspect_image_metadata(image_ids: list[str]) -> dict[str, tuple[int, datetime]]:
    if not image_ids:
        return {}
    result = subprocess.run(
        ["docker", "image", "inspect", *image_ids, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    metadata: dict[str, tuple[int, datetime]] = {}
    for obj in _docker_json_lines(result):
        image_id = obj.get("Id", "")
        if not isinstance(image_id, str) or not image_id:
            continue
        size = obj.get("Size", 0)
        size_bytes = int(size) if isinstance(size, (int, float)) else 0
        metadata[image_id] = (
            size_bytes,
            _parse_docker_timestamp(str(obj.get("Created", ""))),
        )
    return metadata


def container_image_refs() -> set[str]:
    """Return the image ids referenced by every container, running or stopped."""
    ensure_docker_available()
    listing = subprocess.run(
        ["docker", "ps", "-aq"], capture_output=True, text=True, check=False
    )
    container_ids = [
        line.strip() for line in listing.stdout.splitlines() if line.strip()
    ]
    if not container_ids:
        return set()
    inspected = subprocess.run(
        ["docker", "container", "inspect", *container_ids, "--format", "{{.Image}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.strip() for line in inspected.stdout.splitlines() if line.strip()}


def list_managed_images(
    registry: str,
    *,
    include_dangling: bool = False,
    in_use_ids: set[str] | None = None,
) -> list[ManagedImage]:
    """List FlowMesh images on the local daemon.

    Tagged images under a managed repository are attributed to their build target
    and version. When ``include_dangling`` is set, untagged FlowMesh layers
    (identified by the ``org.opencontainers.image.source`` label) are included
    with ``target``/``version`` unset. ``in_use_ids`` marks images referenced by
    a container.
    """
    ensure_docker_available()
    in_use = in_use_ids or set()
    repos = managed_repos(registry)

    images: list[ManagedImage] = []
    listing = subprocess.run(
        ["docker", "image", "ls", "--no-trunc", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    for row in _docker_json_lines(listing):
        repo = str(row.get("Repository", ""))
        tag = str(row.get("Tag", ""))
        image_id = str(row.get("ID", ""))
        if repo not in repos or not image_id or tag == "<none>":
            continue
        ref = f"{repo}:{tag}"
        parsed = parse_image_ref(registry, ref)
        target, version = parsed if parsed else (None, None)
        images.append(
            ManagedImage(
                repo=repo,
                tag=ref,
                target=target,
                version=version,
                image_id=image_id,
                size_bytes=0,
                created=_EPOCH,
                dangling=False,
                in_use=image_id in in_use,
            )
        )

    if include_dangling:
        dangling = subprocess.run(
            [
                "docker",
                "image",
                "ls",
                "--no-trunc",
                "--filter",
                "dangling=true",
                "--filter",
                f"label=org.opencontainers.image.source={FLOWMESH_IMAGE_SOURCE}",
                "--format",
                "{{json .}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for row in _docker_json_lines(dangling):
            image_id = str(row.get("ID", ""))
            if not image_id:
                continue
            images.append(
                ManagedImage(
                    repo=str(row.get("Repository", "<none>")),
                    tag=None,
                    target=None,
                    version=None,
                    image_id=image_id,
                    size_bytes=0,
                    created=_EPOCH,
                    dangling=True,
                    in_use=image_id in in_use,
                )
            )

    metadata = _inspect_image_metadata([image.image_id for image in images])
    for image in images:
        if image.image_id in metadata:
            image.size_bytes, image.created = metadata[image.image_id]
    return images


def remove_images(refs: list[str], *, force: bool = False) -> list[RemovalResult]:
    """Remove image references via ``docker rmi``, reporting each outcome.

    Never aborts mid-batch: a failed removal is recorded and the rest proceed.
    """
    ensure_docker_available()
    results: list[RemovalResult] = []
    for ref in refs:
        args = ["docker", "rmi"]
        if force:
            args.append("-f")
        args.append(ref)
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            results.append(RemovalResult(ref=ref, ok=True))
        else:
            error = (result.stderr or result.stdout).strip()
            results.append(RemovalResult(ref=ref, ok=False, error=error or None))
    return results


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
