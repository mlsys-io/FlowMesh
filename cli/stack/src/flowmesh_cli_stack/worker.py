"""Worker management commands."""

import json
import os
from pathlib import Path

import typer
from flowmesh.exceptions import FlowMeshError
from flowmesh_cli.core import logging
from flowmesh_cli.core.typer import get_typer
from flowmesh_stack.docker import DockerError, image_env_overrides
from flowmesh_stack.env import load_env
from flowmesh_stack.images import get_image_ref
from flowmesh_stack.workers import (
    create_workers,
    operate_workers,
    pull_images,
    select_worker_images,
)

from .utils import DEFAULT_ENV_FILE, STACK_PATH_KEYS, stack_node_client

app = get_typer(help="Create and manage workers on the local node.")


@app.command("list")
def worker_list(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to load defaults"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Node API base URL"),
    token: str = typer.Option(
        "", "--token", help="Bearer token", envvar=["FLOWMESH_API_KEY"]
    ),
) -> None:
    """List all workers managed by the local node."""
    client = stack_node_client(env_file, base_url, token or None)
    try:
        workers = client.list_workers()
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(workers, indent=2))


@app.command("up")
def worker_up(
    kind: str = typer.Argument(
        "cpu", help="cpu|gpu when not using --config; ignored if --config is provided"
    ),
    count: int = typer.Argument(
        1, help="CPU worker count (used when kind=cpu and no --config provided)"
    ),
    targets: str = typer.Option(
        "all",
        "--targets",
        "-t",
        help="GPU ids comma-separated or 'all' (used when kind=gpu and no --config)",
    ),
    config: list[Path] | None = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Path(s) to worker init config (JSON or YAML). "
            "Repeat --config to provide multiple files."
        ),
    ),
    config_raw: list[str] | None = typer.Option(
        None,
        "--config-raw",
        help=(
            "Inline worker init config (JSON or YAML). "
            "Repeat --config-raw to provide multiple configs."
        ),
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to load defaults"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Node API base URL"),
    token: str = typer.Option(
        "", "--token", help="Bearer token", envvar=["FLOWMESH_API_KEY"]
    ),
) -> None:
    """Create and start one or more workers from presets or a custom config file."""
    client = stack_node_client(env_file, base_url, token or None)
    logging.info("Creating workers...")
    try:
        created = create_workers(
            client,
            kind=kind,
            count=count,
            targets=targets,
            config_paths=config,
            config_raw=config_raw,
        )
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    for label, worker_info in created:
        worker_name = worker_info.get("name", "<unknown>")
        logging.success(f"Created {label} '{worker_name}':")
        logging.log(json.dumps(worker_info, indent=2))


@app.command("start")
def worker_start(
    names: list[str] = typer.Argument(..., help="Worker name(s) or 'all'"),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to load defaults"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Node API base URL"),
    token: str = typer.Option(
        "", "--token", help="Bearer token", envvar=["FLOWMESH_API_KEY"]
    ),
) -> None:
    """Start a stopped worker container."""
    client = stack_node_client(env_file, base_url, token or None)
    try:
        started = operate_workers(client, names, operation="start")
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    if not started:
        logging.warning("No workers found.")
        return
    for name in started:
        logging.success(f"Started worker {name}")


@app.command("stop")
def worker_stop(
    names: list[str] = typer.Argument(..., help="Worker name(s) or 'all'"),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to load defaults"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Node API base URL"),
    token: str = typer.Option(
        "", "--token", help="Bearer token", envvar=["FLOWMESH_API_KEY"]
    ),
) -> None:
    """Stop a running worker container without removing it."""
    client = stack_node_client(env_file, base_url, token or None)
    try:
        stopped = operate_workers(client, names, operation="stop")
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    if not stopped:
        logging.warning("No workers found.")
        return
    for name in stopped:
        logging.success(f"Stopped worker {name}")


@app.command("down")
def worker_down(
    names: list[str] = typer.Argument(..., help="Worker name(s) or 'all'"),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to load defaults"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Node API base URL"),
    token: str = typer.Option(
        "", "--token", help="Bearer token", envvar=["FLOWMESH_API_KEY"]
    ),
) -> None:
    """Destroy a worker or all workers, removing containers and associated resources."""
    client = stack_node_client(env_file, base_url, token or None)
    if "all" in names:
        if len(names) != 1:
            logging.error("Use either 'all' or worker names, not both.")
            raise typer.Exit(code=1)
        try:
            logging.info("Destroying all workers...")
            client.destroy_all_workers()
        except FlowMeshError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1)
        logging.success("Destroyed all workers")
        return

    try:
        destroyed = operate_workers(client, names, operation="destroy")
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    if not destroyed:
        logging.warning("No workers found.")
        return
    for name in destroyed:
        logging.success(f"Destroyed worker {name}")


@app.command("pull")
def worker_pull(
    kinds: list[str] = typer.Argument(..., help="cpu|gpu|ssh-cpu|ssh-gpu|all"),
    builder: bool = typer.Option(False, "--builder", "-b", help="Pull builder images"),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to load defaults"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION for worker images"
    ),
) -> None:
    """Pull worker or builder Docker images from the registry."""
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
    registry = os.getenv("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io")
    version = image_env_overrides(image_tag).get(
        "FLOWMESH_VERSION", os.getenv("FLOWMESH_VERSION", "dev")
    )

    kind_images: dict[str, str] = {
        "cpu": get_image_ref(registry, version, "flowmesh_worker_cpu"),
        "gpu": get_image_ref(registry, version, "flowmesh_worker_gpu"),
        "ssh-cpu": get_image_ref(registry, version, "flowmesh_ssh_cpu"),
        "ssh-gpu": get_image_ref(registry, version, "flowmesh_ssh_gpu"),
    }
    builder_images: dict[str, str] = {
        "gpu": get_image_ref(registry, version, "flowmesh_worker_gpu_builder"),
    }

    try:
        images = select_worker_images(
            kinds,
            images=kind_images,
            builder_images=builder_images,
            builder=builder,
        )
        for image in images:
            logging.info(f"Pulling {'builder' if builder else 'worker'} image: {image}")
        pull_images(images)
    except (DockerError, FlowMeshError) as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.success(f"{'Builder' if builder else 'Worker'} images pulled.")
