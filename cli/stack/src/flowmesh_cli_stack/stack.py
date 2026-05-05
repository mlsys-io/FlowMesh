"""Stack management commands."""

import os
import subprocess
from pathlib import Path

import typer
from flowmesh_cli.core import logging
from flowmesh_cli.core.assets import AssetNotFoundError, asset_path
from flowmesh_cli.core.typer import get_typer
from flowmesh_stack.docker import (
    DockerComposeStack,
    DockerError,
    ensure_docker_available,
    image_env_overrides,
    inspect_image,
    remove_image,
)
from flowmesh_stack.doctor import DoctorFinding, run_doctor_checks
from flowmesh_stack.env import ensure_env_file, load_env
from flowmesh_stack.env_schema import render_env_example
from flowmesh_stack.images import BUILD_GROUPS, BUILD_TARGETS, get_image_ref

from .env_schema import STACK_ENV_SCHEMA
from .utils import (
    DEFAULT_ENV_FILE,
    STACK_PATH_KEYS,
    apply_stack_resource_env,
    ensure_deploy_paths,
    stack_bake_file,
    stack_compose_file,
    stack_env_example,
    stack_node_client,
)
from .worker import worker_pull

app = get_typer(help="Build, manage, and run the FlowMesh stack.")


def _stack() -> DockerComposeStack:
    def _load(env_file: Path) -> None:
        ensure_env_file(env_file, stack_env_example())
        load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
        try:
            apply_stack_resource_env()
        except ValueError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1)

    return DockerComposeStack(
        compose_file=stack_compose_file(),
        env_file_var="STACK_ENV_FILE",
        load_env=_load,
        ensure_deploy_paths=ensure_deploy_paths,
    )


def _compose(
    args: list[str], env_file: Path, env: dict[str, str] | None, to_deploy: bool = False
) -> None:
    ensure_env_file(env_file, stack_env_example())
    result = _stack().run(args, env_file=env_file, env=env, to_deploy=to_deploy)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _run_bake(mode: str, targets: list[str] | None, env_file: Path) -> None:
    ensure_env_file(env_file, stack_env_example())
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)

    try:
        ensure_docker_available()
    except DockerError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)

    buildx_check = subprocess.run(
        ["docker", "buildx", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if buildx_check.returncode != 0:
        if buildx_check.stdout:
            logging.log(buildx_check.stdout)
        if buildx_check.stderr:
            logging.log(buildx_check.stderr, err=True)
        logging.error("docker buildx is required for dev build/push")
        raise typer.Exit(code=buildx_check.returncode)

    bake_file = stack_bake_file()
    if not bake_file.exists():
        logging.error(f"Bake file not found: {bake_file}")
        raise typer.Exit(code=1)

    build_created = (
        subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip()
    )
    registry = os.getenv("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io")
    version = os.getenv("FLOWMESH_VERSION", "dev")
    cache_version = os.getenv("FLOWMESH_CACHE_VERSION", "").strip() or version
    env = {
        "REGISTRY": registry,
        "VERSION": version,
        "BUILD_REF": os.getenv("FLOWMESH_BUILD_REF", "local"),
        "BUILD_CREATED": build_created,
    }

    batches: list[list[str]] = []
    if targets is None:
        batches = [["builders"], ["server", "workers"]]
    else:
        batches = [targets]

    for batch_targets in batches:
        args = ["docker", "buildx", "bake", "-f", str(bake_file)]
        if mode == "push":
            args.append("--push")
        else:
            args.append("--load")
        args.extend(batch_targets)
        args.extend(["--set", "*.args.BUILDKIT_INLINE_CACHE=1"])

        # Resolve cache-from for each target
        selected_targets: list[str]
        if batch_targets:
            selected_targets = []
            for target in batch_targets:
                if target in BUILD_GROUPS:
                    selected_targets.extend(BUILD_GROUPS[target])
                elif target in BUILD_TARGETS:
                    selected_targets.append(target)
                else:
                    continue
        else:
            selected_targets = list(BUILD_TARGETS)
        for target in selected_targets:
            image_ref = get_image_ref(registry, cache_version, target)
            args += ["--set", f"{target}.cache-from=type=registry,ref={image_ref}"]

        result = subprocess.run(args, env={**os.environ, **env}, check=False, text=True)
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)


def _log_finding(fnd: DoctorFinding) -> None:
    match fnd.level:
        case "note":
            logging.log(fnd.message)
        case "warning":
            logging.warning(fnd.message)
        case "error":
            logging.error(fnd.message)


@app.command()
def build(
    targets: list[str] | None = typer.Argument(
        None, help="Optional bake targets", metavar="[TARGETS]..."
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for stack compose/bake"
    ),
) -> None:
    """Build FlowMesh Docker images locally using buildx."""
    _run_bake("load", targets, env_file)
    logging.success("Images built locally.")


@app.command()
def push(
    targets: list[str] | None = typer.Argument(
        None, help="Optional bake targets", metavar="[TARGETS]..."
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for stack compose/bake"
    ),
) -> None:
    """Build FlowMesh Docker images and push them to the container registry."""
    _run_bake("push", targets, env_file)
    logging.success("Images pushed.")


@app.command()
def pull(
    services: list[str] | None = typer.Argument(
        None, help="Optional services to pull", metavar="[SERVICES]..."
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
) -> None:
    """Pull Docker images for stack services from the registry."""
    args = ["pull"] + (services or [])
    _compose(args, env_file=env_file, env=image_env_overrides(image_tag))


@app.command()
def pullall(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
) -> None:
    """Pull all images required for the stack."""
    pull(services=None, env_file=env_file, image_tag=image_tag)
    worker_pull(kinds=["all"], builder=True, env_file=env_file, image_tag=image_tag)
    worker_pull(kinds=["all"], builder=False, env_file=env_file, image_tag=image_tag)


@app.command()
def up(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
) -> None:
    """Start the stack including server and Redis."""
    _compose(
        ["up", "-d", "--wait"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        to_deploy=True,
    )
    logging.success("FlowMesh stack is up.")


def _drain_workers(env_file: Path) -> None:
    """Destroy all dynamically spawned workers before stopping the server."""
    try:
        client = stack_node_client(env_file, base_url=None, token=None)
        client.destroy_all_workers()
    except Exception as exc:
        logging.warning(f"Unable to drain workers; continuing shutdown. {exc}")


@app.command()
def down(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
) -> None:
    """Drain workers and stop the stack."""
    logging.info("Draining workers...")
    _drain_workers(env_file)
    logging.info("Shutting down the FlowMesh stack...")
    _compose(["down"], env_file=env_file, env=image_env_overrides(image_tag))
    logging.success("FlowMesh stack stopped.")


@app.command()
def restart(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
) -> None:
    """Drain workers and restart the stack."""
    logging.info("Draining workers...")
    _drain_workers(env_file)
    _compose(["down"], env_file=env_file, env=image_env_overrides(image_tag))
    _compose(
        ["up", "-d", "--wait"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        to_deploy=True,
    )
    logging.success("FlowMesh stack is up.")


@app.command()
def logs(
    service: str | None = typer.Argument(None, help="Optional service name"),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
) -> None:
    """Stream logs from stack services or a specific service container."""
    code = _stack().stream_logs(env_file=env_file, service=service)
    if code != 0:
        raise typer.Exit(code=code)


@app.command()
def ps(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
) -> None:
    """Display running status of stack containers and worker containers."""
    _compose(["ps"], env_file=env_file, env=None)
    logging.log("\nWorkers:")
    subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=flowmesh.group=server-workers",
            "--format",
            "  {{.Names}}\t{{.Status}}",
        ],
        check=False,
    )


@app.command("status")
def status_cmd(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
) -> None:
    """Display running status of stack containers (alias for ps)."""
    ps(env_file=env_file)


@app.command()
def clean(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
) -> None:
    """Drain workers, stop the stack, and remove all containers and volumes."""
    logging.info("Draining workers...")
    _drain_workers(env_file)
    logging.info("Removing stack containers and volumes...")
    _compose(["down", "-v"], env_file=env_file, env=image_env_overrides(image_tag))
    logging.success("FlowMesh stack cleaned.")


def _write_env_example(package: str, filename: str, schema, errors: list[str]) -> None:
    try:
        path = asset_path(package, filename)
    except AssetNotFoundError as exc:
        message = f"Unable to resolve asset {package}/{filename}: {exc}"
        logging.error(message)
        errors.append(message)
        return
    path.write_text(render_env_example(schema))
    logging.success(f"Wrote {path}")


@app.command("env-examples")
def env_examples() -> None:
    """Generate env example files from the shared schema."""
    errors: list[str] = []
    _write_env_example(
        "flowmesh_cli_stack.assets",
        ".env.example",
        STACK_ENV_SCHEMA,
        errors,
    )
    if errors:
        raise typer.Exit(code=1)


@app.command()
def doctor(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to validate"
    )
) -> None:
    """Verify Docker access and validate stack env configuration."""
    report = run_doctor_checks(env_file, STACK_ENV_SCHEMA, callback=_log_finding)
    if report.errors:
        logging.error(f"Doctor found {len(report.errors)} issue(s).")
        raise typer.Exit(code=1)
    logging.success("Doctor checks passed.")


@app.command("init")
def init(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file to write"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force initialization; overwrite existing files"
    ),
) -> None:
    """Create or overwrite the stack env file from the example template."""
    example = stack_env_example()
    if not example.exists():
        logging.error(f"Env example not found: {example}")
        raise typer.Exit(code=1)
    if env_file.exists() and not force:
        if not typer.confirm(f"{env_file} exists. Overwrite?", default=False):
            logging.info("Keeping existing env file.")
            return
    env_file.write_text(example.read_text())
    logging.success(f"Wrote {env_file} from {example.name}.")


@app.command("purge")
def purge(
    version: str = typer.Argument(
        ..., help="FlowMesh version to purge from local Docker"
    ),
    targets: list[str] | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Optional specific image targets to purge "
        "(e.g., flowmesh_server, flowmesh_worker_gpu, etc.)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List images to be purged without deleting them"
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for docker"
    ),
) -> None:
    """Purge FlowMesh Docker images for a specific version from local Docker."""
    try:
        ensure_docker_available()
    except DockerError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)

    logging.info(f"Purging FlowMesh Docker images with version '{version}'...")
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)

    if targets is None:
        targets = list(BUILD_TARGETS)
    else:
        invalid = [t for t in targets if t not in BUILD_TARGETS]
        if invalid:
            logging.error(f"Invalid targets specified: {', '.join(invalid)}")
            raise typer.Exit(code=1)

    images_to_purge: list[str] = []
    registry = os.getenv("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io")
    for target in targets:
        image_ref = get_image_ref(registry=registry, version=version, target=target)
        result = inspect_image(image_ref, capture_output=True)  # Check if image exists
        if result.returncode == 0:
            images_to_purge.append(image_ref)
        else:
            logging.warning(f"Image not found: {image_ref}")

    if not images_to_purge:
        logging.info("No images to purge.")
        return

    if dry_run:
        logging.info("Images to be purged:")
        for image in images_to_purge:
            logging.log(f"  {image}")
        return

    error = False
    for image in images_to_purge:
        result = remove_image(image, capture_output=True)
        if result.returncode == 0:
            logging.success(f"Removed image: {image}")
        else:
            logging.error(f"Failed to remove image: {image}")
            if result.stdout:
                logging.log(result.stdout)
            if result.stderr:
                logging.log(result.stderr, err=True)
            error = True
    if error:
        raise typer.Exit(code=1)
