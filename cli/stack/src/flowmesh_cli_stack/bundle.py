"""Bundle commands."""

import shutil
import subprocess
import sys
import tarfile
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer
from flowmesh.models.nodes import NodeRole
from flowmesh_cli.core import logging
from flowmesh_cli.core.typer import get_typer
from packaging.version import InvalidVersion, Version

from . import stack as stack_module
from .utils import DEFAULT_ENV_FILE

app = get_typer(
    help="Package FlowMesh deployments into portable bundles for distribution."
)

_TLS_SERVER_SUBDIR = "secrets/tls/server"
_TLS_REDIS_SUBDIR = "secrets/tls/redis"
_WORKER_CONFIG_FILE = "configs/worker_config.yaml"

_SERVER_TLS_SOURCES = (
    Path(_TLS_SERVER_SUBDIR) / "server-ca.pem",
    Path(_TLS_SERVER_SUBDIR) / "server.key",
    Path(_TLS_SERVER_SUBDIR) / "server.pem",
)
_REDIS_TLS_CA_SOURCE = Path(_TLS_REDIS_SUBDIR) / "redis-ca.pem"
_REDIS_TLS_CERT_SOURCES = (
    Path(_TLS_REDIS_SUBDIR) / "redis-server.pem",
    Path(_TLS_REDIS_SUBDIR) / "redis-server.key",
)
_WORKER_CONFIG_SOURCE = Path(_WORKER_CONFIG_FILE)


def _copy_redis_tls_assets(dest: Path, include_tls: bool, *, ca_only: bool) -> None:
    if not include_tls:
        return
    tls_dir = dest / _TLS_REDIS_SUBDIR
    sources: tuple[Path, ...] = (_REDIS_TLS_CA_SOURCE,)
    if not ca_only:
        sources = sources + _REDIS_TLS_CERT_SOURCES
    copied = False
    missing: list[str] = []
    for src in sources:
        if src.exists():
            tls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tls_dir / src.name)
            copied = True
        else:
            missing.append(src.name)
    if not copied:
        logging.warning(
            "Warning: Redis TLS assets not found; bundle created without TLS."
        )
    elif missing:
        missing_str = ", ".join(missing)
        logging.warning(f"Warning: Redis TLS assets missing: {missing_str}")


def _copy_server_assets(
    dest: Path, include_tls: bool, role: NodeRole = NodeRole.ROOT
) -> None:
    if include_tls:
        tls_dir = dest / _TLS_SERVER_SUBDIR
        copied = False
        missing: list[str] = []
        for src in _SERVER_TLS_SOURCES:
            if src.exists():
                tls_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tls_dir / src.name)
                copied = True
            else:
                missing.append(src.name)
        if not copied:
            logging.warning(
                "Warning: TLS assets not found; bundle created without TLS."
            )
        elif missing:
            missing_str = ", ".join(missing)
            logging.warning(f"Warning: TLS assets missing: {missing_str}")
    # Worker nodes don't host Redis (compose root profile gates it), so they
    # only need the CA to verify the root's TLS.
    _copy_redis_tls_assets(dest, include_tls, ca_only=role == NodeRole.WORKER)

    if _WORKER_CONFIG_SOURCE.exists():
        worker_config_dest = dest / _WORKER_CONFIG_FILE
        worker_config_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_WORKER_CONFIG_SOURCE, worker_config_dest)
    else:
        logging.warning(f"Warning: worker config not found: {_WORKER_CONFIG_SOURCE}")


def _scaffold_server_assets(dest: Path, include_tls: bool) -> None:
    """Scaffold the bundle directory layout in-place at ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)

    if include_tls:
        for subdir in (_TLS_SERVER_SUBDIR, _TLS_REDIS_SUBDIR):
            target = dest / subdir
            existed = target.is_dir()
            target.mkdir(parents=True, exist_ok=True)
            logging.log(f"{'kept' if existed else 'created'} {subdir}/")

    worker_config = dest / _WORKER_CONFIG_FILE
    if worker_config.exists():
        logging.log(f"kept {_WORKER_CONFIG_FILE}")
    else:
        worker_config.parent.mkdir(parents=True, exist_ok=True)
        worker_config.touch()
        logging.log(f"created {_WORKER_CONFIG_FILE}")


def _build_cli_wheels(wheel_dir: Path) -> None:
    """Build wheels for the CLI and SDK into wheel_dir."""
    wheel_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd().resolve()
    packages = [
        repo_root / "sdk",
        repo_root / "sdk" / "stack",
        repo_root / "cli",
        repo_root / "cli" / "stack",
    ]
    for pkg in packages:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                str(pkg),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logging.log(result.stdout)
            logging.log(result.stderr, err=True)
            raise typer.Exit(code=result.returncode)


def _published_cli_spec() -> str:
    """Return the published FlowMesh CLI package spec for this release."""
    try:
        package_version = version("flowmesh-cli-stack")
    except PackageNotFoundError:
        logging.error("Unable to resolve installed flowmesh-cli-stack version.")
        raise typer.Exit(code=1) from None
    try:
        parsed = Version(package_version)
    except InvalidVersion:
        logging.error(
            f"Installed flowmesh-cli-stack version {package_version!r} is not a "
            "valid PEP 440 version."
        )
        raise typer.Exit(code=1) from None
    if parsed.is_prerelease or parsed.is_devrelease or parsed.local is not None:
        logging.error(
            f"Installed flowmesh-cli-stack version {package_version!r} is not a "
            "published release; the bundle's install.sh would fail on PyPI. "
            "Install a release of flowmesh-cli-stack first, or pass "
            "--include-wheels to bundle local wheels instead."
        )
        raise typer.Exit(code=1)
    # Workspace versions are kept in lock-step by scripts/dev/bump_version.py,
    # so flowmesh-cli-stack's version is also the matching flowmesh-metapackage
    # version on PyPI.
    return f"flowmesh[cli]=={package_version}"


def _write_install_script(
    dest: Path, *, package_spec: str | None = None, include_wheels: bool = False
) -> None:
    """Write an install.sh script to set up a venv and install FlowMesh CLI."""
    script_path = dest / "install.sh"
    if include_wheels:
        install_block = '"$UV_BIN" pip install ./wheels/*.whl'
    else:
        assert package_spec is not None
        install_block = f"""\
FLOWMESH_PACKAGE_SPEC="${{FLOWMESH_PACKAGE_SPEC:-{package_spec}}}"
FLOWMESH_INDEX_URL="${{FLOWMESH_INDEX_URL:-}}"
FLOWMESH_EXTRA_INDEX_URL="${{FLOWMESH_EXTRA_INDEX_URL:-}}"

INSTALL_ARGS=("$FLOWMESH_PACKAGE_SPEC")
if [ -n "$FLOWMESH_INDEX_URL" ]; then
  INSTALL_ARGS=(--index-url "$FLOWMESH_INDEX_URL" "${{INSTALL_ARGS[@]}}")
fi
if [ -n "$FLOWMESH_EXTRA_INDEX_URL" ]; then
  INSTALL_ARGS=(--extra-index-url "$FLOWMESH_EXTRA_INDEX_URL" "${{INSTALL_ARGS[@]}}")
fi
"$UV_BIN" pip install "${{INSTALL_ARGS[@]}}"
"""

    script = f"""#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${{VENV_DIR:-.venv}}"
UV_BIN="${{UV_BIN:-uv}}"
PYTHON_REQ="${{FLOWMESH_PYTHON:-3.12}}"
ENV_FILE=".env"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  UV_BIN="${{UV_BIN:-uv}}"
fi
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv install failed or not found in PATH." >&2
  exit 1
fi

"$UV_BIN" python install "$PYTHON_REQ"

if [ ! -d "$VENV_DIR" ]; then
  "$UV_BIN" venv "$VENV_DIR" --python "$PYTHON_REQ"
fi

source "$VENV_DIR/bin/activate"
"$UV_BIN" pip install --upgrade pip
{install_block}
echo "Installed flowmesh CLI into $VENV_DIR."
echo "Activate it with 'source $VENV_DIR/bin/activate'."
if [ ! -f "$ENV_FILE" ]; then
  flowmesh stack init --env-file "$ENV_FILE"
fi
echo "Configure $ENV_FILE before executing FlowMesh."
echo "Then run:"
echo "  flowmesh stack pull"
echo "  flowmesh stack up"
echo "To stop services:"
echo "  flowmesh stack down"
"""
    script_path.write_text(script)
    script_path.chmod(script_path.stat().st_mode | 0o111)


def _create_bundle_tarball(
    tar_path: Path,
    include_tls: bool,
    *,
    include_wheels: bool,
    role: NodeRole = NodeRole.ROOT,
) -> None:
    """Create a deployable bundle as a tar.gz with a top-level prefix."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "flowmesh_server_bundle"
    with tempfile.TemporaryDirectory(prefix="flowmesh-bundle-") as tmp:
        staging_root = Path(tmp) / prefix
        staging_root.mkdir(parents=True, exist_ok=True)
        _copy_server_assets(staging_root, include_tls=include_tls, role=role)
        if include_wheels:
            wheel_dir = staging_root / "wheels"
            _build_cli_wheels(wheel_dir)
            _write_install_script(staging_root, include_wheels=True)
        else:
            _write_install_script(
                staging_root,
                package_spec=_published_cli_spec(),
                include_wheels=False,
            )
        with tarfile.open(tar_path, mode="w:gz") as tf:
            # Ensure we archive the top-level prefix directory.
            tf.add(staging_root, arcname=prefix)


@app.command("export")
def bundle_export(
    role: str = typer.Argument(
        NodeRole.ROOT.value,
        help="Target NODE_ROLE for the bundle (root|worker).",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output tar.gz path (default: ./dist/flowmesh_server_bundle.tar.gz)",
    ),
    no_tls: bool = typer.Option(False, "--no-tls", help="Exclude TLS assets"),
    include_wheels: bool = typer.Option(
        False,
        "--include-wheels",
        help=(
            "Bundle local CLI/SDK wheels instead of installing "
            "published flowmesh[cli]."
        ),
    ),
) -> None:
    """Create a deployment bundle for the server."""
    normalized_role = role.strip().lower()
    try:
        node_role = NodeRole(normalized_role)
    except ValueError:
        logging.error(f"Invalid role {role!r}; expected one of {', '.join(NodeRole)}.")
        raise typer.Exit(code=1)
    logging.info(f"Creating server bundle for role={normalized_role}...")
    tar_path = output
    if tar_path is None:
        tar_path = Path("./dist/flowmesh_server_bundle.tar.gz")
        tar_path.parent.mkdir(parents=True, exist_ok=True)
    _create_bundle_tarball(
        tar_path,
        include_tls=not no_tls,
        include_wheels=include_wheels,
        role=node_role,
    )
    logging.success(f"Bundle created at {tar_path}")


@app.command("init")
def bundle_init(
    dest: Path = typer.Option(
        Path("."),
        "--dest",
        help="Directory to scaffold the bundle layout in (default: current dir).",
    ),
    no_tls: bool = typer.Option(
        False, "--no-tls", help="Skip TLS placeholder directories."
    ),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE,
        "--env-file",
        help="Env file to write under --dest (or absolute path).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing env file without prompting.",
    ),
) -> None:
    """Scaffold an empty bundle layout in --dest."""
    logging.info(f"Initializing server bundle layout in '{dest}'...")
    _scaffold_server_assets(dest, include_tls=not no_tls)
    resolved_env = env_file if env_file.is_absolute() else dest / env_file
    resolved_env.parent.mkdir(parents=True, exist_ok=True)
    stack_module.init(env_file=resolved_env, force=force)
    # Paths in the next-steps block are intentionally relative to dest so
    # they remain accurate after the user runs the cd line.
    env_hint = env_file if not env_file.is_absolute() else resolved_env
    cd_hint = "" if dest == Path(".") else f"  cd {dest}\n"
    env_arg = "" if env_file == DEFAULT_ENV_FILE else f" --env-file {env_hint}"
    tls_hint = (
        f"  drop TLS certs into {_TLS_SERVER_SUBDIR}/ and {_TLS_REDIS_SUBDIR}/\n"
        if not no_tls
        else ""
    )
    logging.success(f"Bundle layout ready at '{dest}'.")
    logging.log(
        f"Next steps:\n{cd_hint}"
        f"  edit {env_hint} and {_WORKER_CONFIG_FILE}\n"
        f"{tls_hint}"
        f"  flowmesh stack pull{env_arg}\n"
        f"  flowmesh stack up{env_arg}"
    )
