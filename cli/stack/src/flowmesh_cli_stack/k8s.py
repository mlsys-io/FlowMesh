"""Kubernetes backend for the FlowMesh stack lifecycle commands."""

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

import typer
from flowmesh.models.nodes import NodeRole
from flowmesh_cli.core import logging
from flowmesh_cli.core.assets import asset_path
from flowmesh_stack.env import ensure_env_file, load_env, parse_env_file
from flowmesh_stack.kubernetes import KubectlError, KubernetesStack
from flowmesh_stack.manifests import ManifestError
from flowmesh_stack.paths import resolve_path

from .utils import DEFAULT_ENV_FILE, STACK_PATH_KEYS, drain_workers, stack_env_example

MANIFEST_ASSETS = (
    "00-namespace.yaml",
    "10-rbac.yaml",
    "20-redis.yaml",
    "30-server.yaml",
)
"""Manifest assets applied in order; the namespace must come first."""

SERVER_WORKLOAD = "deployment/flowmesh-server"
REDIS_WORKLOADS = (
    "statefulset/flowmesh-redis-control",
    "statefulset/flowmesh-redis-telemetry",
)
STACK_WORKLOADS = {
    "server": SERVER_WORKLOAD,
    "redis_control": REDIS_WORKLOADS[0],
    "redis_telemetry": REDIS_WORKLOADS[1],
}
ROLLOUT_TIMEOUT = "300s"

DEFAULT_NAMESPACE = "flowmesh"
DEFAULT_WORKER_CONFIG = "./configs/worker_config.yaml"


class StackBackend(StrEnum):
    COMPOSE = "compose"
    K8S = "k8s"


def resolve_backend(value: str | None, env_file: Path) -> StackBackend:
    """Resolve the stack backend from the option, falling back to the env file."""
    raw = (value or "").strip()
    if not raw:
        raw = parse_env_file(env_file).get("STACK_BACKEND", "").strip()
    if not raw:
        return StackBackend.COMPOSE
    try:
        return StackBackend(raw.lower())
    except ValueError:
        logging.error(
            f"Unknown stack backend {raw!r}; "
            f"expected one of {', '.join(StackBackend)}."
        )
        raise typer.Exit(code=1) from None


def manifest_paths() -> list[Path]:
    return [
        asset_path("flowmesh_cli_stack.assets", "k8s", name) for name in MANIFEST_ASSETS
    ]


def apply_k8s_env(base_dir: Path) -> None:
    """Fill in the derived Kubernetes values the manifests reference.

    ``K8S_WORKER_NAMESPACE`` defaults to the stack namespace and
    ``SERVER_WORKER_CONFIG`` to the path the compose backend bind-mounts,
    because manifest substitution has no nested defaults. The distinctness of
    the worker namespace is precomputed for the same reason: the manifest
    conditionals compare against a value, not against another variable.
    """
    namespace = os.environ.get("K8S_NAMESPACE", "").strip() or DEFAULT_NAMESPACE
    os.environ["K8S_NAMESPACE"] = namespace
    worker_namespace = os.environ.get("K8S_WORKER_NAMESPACE", "").strip() or namespace
    os.environ["K8S_WORKER_NAMESPACE"] = worker_namespace
    os.environ["K8S_WORKER_NAMESPACE_DISTINCT"] = (
        "true" if worker_namespace != namespace else "false"
    )
    os.environ["SERVER_WORKER_CONFIG"] = resolve_path(
        os.environ.get("SERVER_WORKER_CONFIG", ""),
        default=DEFAULT_WORKER_CONFIG,
        base_dir=base_dir,
    ).as_posix()


def node_role() -> NodeRole:
    raw = os.environ.get("NODE_ROLE", "").strip()
    return NodeRole(raw.lower()) if raw else NodeRole.ROOT


def stack(env_file: Path, image_tag: str | None = None) -> KubernetesStack:
    def _load(path: Path) -> None:
        ensure_env_file(path, stack_env_example())
        load_env(path, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
        apply_k8s_env(Path.cwd())
        if image_tag:
            os.environ["FLOWMESH_VERSION"] = image_tag

    _load(env_file)
    return KubernetesStack(
        manifests=manifest_paths(),
        namespace=os.environ["K8S_NAMESPACE"],
        load_env=_load,
        context=os.environ.get("K8S_CONTEXT", "").strip() or None,
        kubeconfig=os.environ.get("K8S_KUBECONFIG", "").strip() or None,
    )


def workloads() -> list[str]:
    """Return the workloads this node runs; Redis only on a root node."""
    if node_role() is NodeRole.ROOT:
        return [*REDIS_WORKLOADS, SERVER_WORKLOAD]
    return [SERVER_WORKLOAD]


def _check(result: subprocess.CompletedProcess) -> None:
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@contextmanager
def _reporting(action: str) -> Iterator[None]:
    try:
        yield
    except (KubectlError, ManifestError) as exc:
        logging.error(f"Failed to {action}: {exc}")
        raise typer.Exit(code=1) from exc


def _resolve_workloads(services: list[str] | None) -> list[str]:
    if not services:
        return workloads()

    requested = list(dict.fromkeys(services))
    unknown = [name for name in requested if name not in STACK_WORKLOADS]
    if unknown:
        logging.error(
            f"Unknown service(s): {', '.join(unknown)}. "
            f"Known services: {', '.join(STACK_WORKLOADS)}."
        )
        raise typer.Exit(code=1)
    return [STACK_WORKLOADS[name] for name in requested]


def up(env_file: Path = DEFAULT_ENV_FILE, image_tag: str | None = None) -> None:
    """Apply the stack manifests and wait for the workloads to become ready."""
    with _reporting("apply the stack"):
        target = stack(env_file, image_tag)
        _check(target.apply(env_file))
        for workload in workloads():
            _check(target.rollout_status(workload, ROLLOUT_TIMEOUT))
    logging.success("FlowMesh stack is up.")


def down(env_file: Path = DEFAULT_ENV_FILE, image_tag: str | None = None) -> None:
    """Drain workers and delete the stack resources."""
    logging.info("Draining workers...")
    drain_workers(env_file)
    logging.info("Shutting down the FlowMesh stack...")
    with _reporting("delete the stack"):
        target = stack(env_file, image_tag)
        _check(target.delete_workers())
        _check(target.delete(env_file))
    logging.success("FlowMesh stack stopped.")


def restart(
    services: list[str] | None = None,
    env_file: Path = DEFAULT_ENV_FILE,
    image_tag: str | None = None,
) -> None:
    """Restart the stack, or the named workloads, in place.

    An image tag change is applied rather than rolled, because a rollout
    restart does not change the pod template's image.
    """
    with _reporting("restart the stack"):
        target = stack(env_file, image_tag)
        requested = _resolve_workloads(services)

        if SERVER_WORKLOAD in requested:
            logging.info("Draining workers...")
            drain_workers(env_file)

        if image_tag:
            _check(target.apply(env_file))
        else:
            for workload in requested:
                _check(target.rollout_restart(workload))

        for workload in requested:
            _check(target.rollout_status(workload, ROLLOUT_TIMEOUT))
    logging.success("FlowMesh stack restarted.")


def logs(service: str | None = None, env_file: Path = DEFAULT_ENV_FILE) -> None:
    """Stream logs from a stack workload."""
    with _reporting("stream logs"):
        target = stack(env_file)
        workload = _resolve_workloads([service])[0] if service else SERVER_WORKLOAD
        _check(target.logs(workload))


def ps(env_file: Path = DEFAULT_ENV_FILE) -> None:
    """Show stack pods and supervisor-managed worker pods."""
    with _reporting("read stack status"):
        target = stack(env_file)
        _check(target.status())
        logging.log("\nWorkers:")
        _check(target.worker_status())


def clean(env_file: Path = DEFAULT_ENV_FILE, image_tag: str | None = None) -> None:
    """Drain workers, delete the stack, and remove its persistent volumes."""
    down(env_file, image_tag)
    logging.info("Removing stack volumes...")
    with _reporting("remove stack volumes"):
        _check(stack(env_file, image_tag).delete_volumes())
    logging.success("FlowMesh stack cleaned.")
