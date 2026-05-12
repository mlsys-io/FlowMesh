"""Stack management commands."""

import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import typer
from flowmesh.models.nodes import NodeRole
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
from flowmesh_stack.env import ensure_env_file, load_env, parse_env_file
from flowmesh_stack.env_schema import render_env_example
from flowmesh_stack.images import (
    BUILD_GROUPS,
    BUILD_TARGETS,
    expand_build_targets,
    get_cache_ref,
    get_image_ref,
    get_push_platforms,
)

from .env_schema import STACK_ENV_SCHEMA, deploy_overrides, role_overrides
from .utils import (
    DEFAULT_ENV_FILE,
    STACK_PATH_KEYS,
    apply_stack_resource_env,
    ensure_deploy_paths,
    parse_node_role,
    resolve_package_version,
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
    args: list[str],
    env_file: Path,
    env: dict[str, str] | None,
    to_deploy: bool = False,
    profile: str | None = None,
) -> None:
    ensure_env_file(env_file, stack_env_example())
    full_args = (["--profile", profile] if profile else []) + args
    result = _stack().run(full_args, env_file=env_file, env=env, to_deploy=to_deploy)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _node_role(env_file: Path) -> NodeRole:
    """Return the configured NODE_ROLE (root | worker), defaulting to root if unset."""
    raw = parse_env_file(env_file).get("NODE_ROLE", "").strip()
    try:
        return NodeRole(raw.lower()) if raw else NodeRole.ROOT
    except ValueError:
        logging.error(
            f"NODE_ROLE={raw!r} is not a recognized role; expected 'root' or 'worker'."
        )
        raise typer.Exit(code=1)


def _resolve_build_targets(batch_targets: list[str]) -> list[str]:
    resolved: list[str] = []
    for target in batch_targets:
        if target in BUILD_GROUPS:
            resolved.extend(BUILD_GROUPS[target])
        elif target in BUILD_TARGETS:
            resolved.append(target)
    return expand_build_targets(resolved)


def _platform_overrides(mode: str, targets: list[str]) -> list[tuple[str, str]]:
    if mode == "load":
        return [(target, "local") for target in targets]
    return [(target, get_push_platforms(target)) for target in targets]


def _resolve_bake_batches(
    targets: list[str] | None, no_builder: bool = False
) -> list[list[str]]:
    if targets is None:
        default_targets = ["server", "workers"]
        if no_builder:
            return [default_targets]
        return [["builders"], default_targets]

    builder_targets = {"builders", "flowmesh_worker_gpu_builder"}
    if no_builder and any(target in builder_targets for target in targets):
        logging.error("--no-builder cannot be used with explicit builder targets.")
        raise typer.Exit(code=1)

    return [targets]


def _require_bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        logging.error(f"{name} is required but was not found in PATH.")
        raise typer.Exit(code=1)
    return path


def _parse_buildx_field(output: str, field: str) -> str | None:
    needle = f"{field}:"
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(needle):
            return line.split(":", 1)[1].strip() or None
    return None


def _inspect_buildx_builder(
    docker_bin: str, name: str | None
) -> subprocess.CompletedProcess[str]:
    args = [docker_bin, "buildx", "inspect"]
    if name is not None:
        args.append(name)
    return subprocess.run(  # nosec B603: argv list, absolute binary path.
        args, capture_output=True, text=True, check=False
    )


def _get_active_buildx_builder(docker_bin: str) -> str | None:
    result = _inspect_buildx_builder(docker_bin, None)
    if result.returncode != 0:
        return None
    return _parse_buildx_field(result.stdout, "Name")


def _ensure_buildx_builder_ready(
    docker_bin: str,
    builder: str,
    expected_driver: str,
    missing_hint: str | None = None,
) -> None:
    """Verify ``builder`` exists and uses ``expected_driver`` before bake runs."""
    result = _inspect_buildx_builder(docker_bin, builder)
    if result.returncode != 0:
        logging.error(f"Buildx builder '{builder}' is not available.")
        if result.stderr:
            logging.log(result.stderr.strip(), err=True)
        if missing_hint:
            logging.log(missing_hint)
        raise typer.Exit(code=1)
    driver = _parse_buildx_field(result.stdout, "Driver")
    if driver != expected_driver:
        logging.error(
            f"Buildx builder '{builder}' uses driver '{driver or 'unknown'}'; "
            f"'{expected_driver}' is required."
        )
        raise typer.Exit(code=1)


def _switch_active_buildx_builder(docker_bin: str, target: str, force: bool) -> None:
    """If the active buildx builder differs from ``target``, switch to it.

    Prompts for confirmation unless ``force`` is true; aborts the command on
    decline so the user never silently builds against an unintended builder.
    """
    active = _get_active_buildx_builder(docker_bin)
    if active == target:
        return
    if not force:
        prompt = (
            f"Active buildx builder is '{active or 'unknown'}'; "
            f"switch to '{target}'?"
        )
        if not typer.confirm(prompt, default=False):
            logging.error(f"Aborted; '{target}' is not the active buildx builder.")
            raise typer.Exit(code=1)
    result = subprocess.run(  # nosec B603: argv list, absolute binary path.
        [docker_bin, "buildx", "use", target],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            logging.log(result.stdout)
        if result.stderr:
            logging.log(result.stderr, err=True)
        logging.error(f"Failed to switch active buildx builder to '{target}'.")
        raise typer.Exit(code=result.returncode)
    logging.info(f"Switched active buildx builder to '{target}'.")


# Driver split: load uses the native docker driver (local cache, image goes
# straight into the daemon's image store); push uses docker-container (registry
# cache in/out, multi-platform).
_BUILD_DEFAULT_BUILDER = "default"
_PUSH_DEFAULT_BUILDER = "flowmesh-multiarch"
_PUSH_BUILDER_MISSING_HINT = (
    "Create the builder, then retry:\n"
    f"docker buildx create --name {_PUSH_DEFAULT_BUILDER} "
    "--driver docker-container --bootstrap"
)


def _run_bake(
    mode: str,
    targets: list[str] | None,
    env_file: Path,
    builder: str,
    force: bool,
    no_builder: bool = False,
    image_tag: str | None = None,
    build_ref: str | None = None,
) -> None:
    ensure_env_file(env_file, stack_env_example())
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)

    try:
        ensure_docker_available()
    except DockerError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    docker_bin = _require_bin("docker")

    buildx_check = subprocess.run(
        [
            docker_bin,
            "buildx",
            "version",
        ],  # nosec B603: argv list, absolute binary path.
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

    if mode == "push":
        _ensure_buildx_builder_ready(
            docker_bin,
            builder,
            "docker-container",
            missing_hint=_PUSH_BUILDER_MISSING_HINT,
        )
    else:
        _ensure_buildx_builder_ready(docker_bin, builder, "docker")
    _switch_active_buildx_builder(docker_bin, builder, force)

    build_created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = os.getenv("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io")
    version = image_tag if image_tag else os.getenv("FLOWMESH_VERSION", "dev")
    cache_version = os.getenv("FLOWMESH_CACHE_VERSION", "").strip() or "cache"
    env: dict[str, str] = {
        "REGISTRY": registry,
        "VERSION": version,
        "BUILD_REF": (
            build_ref if build_ref else os.getenv("FLOWMESH_BUILD_REF", "local")
        ),
        "BUILD_CREATED": build_created,
    }

    for batch_targets in _resolve_bake_batches(targets, no_builder=no_builder):
        args = [
            docker_bin,
            "buildx",
            "bake",
            "-f",
            str(bake_file),
            "--builder",
            builder,
        ]
        if mode == "push":
            args.append("--push")
        else:
            args.append("--load")
        args.extend(batch_targets)

        selected_targets = _resolve_build_targets(batch_targets)
        if mode == "push":
            for target in selected_targets:
                cache_ref = get_cache_ref(registry, cache_version, target)
                args += [
                    "--set",
                    f"{target}.cache-from=type=registry,ref={cache_ref}",
                    "--set",
                    f"{target}.cache-to=type=registry,ref={cache_ref},mode=max",
                ]
        for target, platform in _platform_overrides(mode, selected_targets):
            args += ["--set", f"{target}.platform={platform}"]

        result = subprocess.run(  # nosec B603: argv list, absolute binary path.
            args,
            env={**os.environ, **env},
            check=False,
            text=True,
        )
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
    no_builder: bool = typer.Option(
        False,
        "--no-builder",
        help="Skip exporting the standalone GPU builder image.",
    ),
    builder: str = typer.Option(
        _BUILD_DEFAULT_BUILDER,
        "--builder",
        help="Buildx builder to use; must use the native 'docker' driver.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Skip the confirmation prompt when switching the active buildx builder.",
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
    build_ref: str | None = typer.Option(
        None, "--build-ref", help="Override FLOWMESH_BUILD_REF"
    ),
) -> None:
    """Build FlowMesh Docker images locally using buildx."""
    _run_bake(
        "load",
        targets,
        env_file,
        builder=builder,
        force=force,
        no_builder=no_builder,
        image_tag=image_tag,
        build_ref=build_ref,
    )
    logging.success("Images built locally.")


@app.command()
def push(
    targets: list[str] | None = typer.Argument(
        None, help="Optional bake targets", metavar="[TARGETS]..."
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for stack compose/bake"
    ),
    no_builder: bool = typer.Option(
        False,
        "--no-builder",
        help="Skip publishing the standalone GPU builder image.",
    ),
    builder: str = typer.Option(
        _PUSH_DEFAULT_BUILDER,
        "--builder",
        help="Buildx builder to use; must use the 'docker-container' driver.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Skip the confirmation prompt when switching the active buildx builder.",
    ),
    image_tag: str | None = typer.Option(
        None, "--image-tag", help="Override FLOWMESH_VERSION"
    ),
    build_ref: str | None = typer.Option(
        None, "--build-ref", help="Override FLOWMESH_BUILD_REF"
    ),
) -> None:
    """Build FlowMesh Docker images and push them to the container registry."""
    _run_bake(
        "push",
        targets,
        env_file,
        builder=builder,
        force=force,
        no_builder=no_builder,
        image_tag=image_tag,
        build_ref=build_ref,
    )
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
    profile = "root" if _node_role(env_file) == NodeRole.ROOT else None
    _compose(
        args, env_file=env_file, env=image_env_overrides(image_tag), profile=profile
    )


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
    """Start the stack.

    On root nodes (NODE_ROLE=root, the default), the local Redis services are
    started alongside the server. On worker nodes (NODE_ROLE=worker), Redis
    services are skipped — the worker is expected to connect to the root
    node's Redis via REDIS_CONTROL_URL / REDIS_TELEMETRY_URL.
    """
    profile = "root" if _node_role(env_file) == NodeRole.ROOT else None
    _compose(
        ["up", "-d", "--wait"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        to_deploy=True,
        profile=profile,
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
    _compose(
        ["down"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        profile="root",
    )
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
    _compose(
        ["down"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        profile="root",
    )
    profile = "root" if _node_role(env_file) == NodeRole.ROOT else None
    _compose(
        ["up", "-d", "--wait"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        to_deploy=True,
        profile=profile,
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
    code = _stack().stream_logs(env_file=env_file, service=service, profile="root")
    if code != 0:
        raise typer.Exit(code=code)


@app.command()
def ps(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for compose"
    ),
) -> None:
    """Display running status of stack containers and worker containers."""
    _compose(["ps"], env_file=env_file, env=None, profile="root")
    logging.log("\nWorkers:")
    docker_bin = _require_bin("docker")
    subprocess.run(
        [
            docker_bin,
            "ps",
            "-a",
            "--filter",
            "label=flowmesh.group=server-workers",
            "--format",
            "  {{.Names}}\t{{.Status}}",
        ],
        check=False,
    )  # nosec B603: argv list, absolute binary path.


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
    _compose(
        ["down", "-v"],
        env_file=env_file,
        env=image_env_overrides(image_tag),
        profile="root",
    )
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
    role: str = typer.Option(
        NodeRole.ROOT.value,
        "--role",
        help="Target NODE_ROLE for the generated env file (root|worker).",
    ),
    deploy: bool = typer.Option(
        False,
        "--deploy",
        help=(
            "Pin FLOWMESH_VERSION to the installed flowmesh-cli-stack package version"
            "(falls back to 'latest' if package metadata is missing)."
        ),
    ),
) -> None:
    """Create or overwrite the stack env file rendered from the schema."""
    node_role = parse_node_role(role)
    if env_file.exists() and not force:
        if not typer.confirm(f"{env_file} exists. Overwrite?", default=False):
            logging.info("Keeping existing env file.")
            return
    deploy_version: str | None = None
    if deploy:
        deploy_version = resolve_package_version()
        if deploy_version is None:
            logging.warning(
                "Unable to resolve flowmesh-cli-stack version; "
                "falling back to FLOWMESH_VERSION=latest. "
                "Edit .env if you need a specific version."
            )
            deploy_version = "latest"
    overrides = {
        **role_overrides(node_role),
        **deploy_overrides(deploy, deploy_version),
    }
    env_file.write_text(render_env_example(STACK_ENV_SCHEMA, overrides=overrides))
    logging.success(f"Wrote {env_file} (NODE_ROLE={node_role.value}).")


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
