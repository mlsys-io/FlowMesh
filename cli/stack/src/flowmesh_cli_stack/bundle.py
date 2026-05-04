"""Bundle commands."""

import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import typer
from flowmesh_cli.core import logging
from flowmesh_cli.core.typer import get_typer

app = get_typer(
    help="Package FlowMesh deployments into portable bundles for distribution."
)


def _copy_redis_tls_assets(dest: Path, include_tls: bool, *, ca_only: bool) -> None:
    if not include_tls:
        return
    tls_dir = dest / "tls" / "redis"
    ca_src = Path("secrets/tls/redis/redis-ca.pem")
    cert_src = Path("secrets/tls/redis/redis-server.pem")
    key_src = Path("secrets/tls/redis/redis-server.key")
    copied = False
    missing: list[str] = []
    if ca_src.exists():
        tls_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ca_src, tls_dir / ca_src.name)
        copied = True
    else:
        missing.append(ca_src.name)
    if not ca_only:
        for src in (cert_src, key_src):
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


def _copy_server_assets(dest: Path, include_tls: bool) -> None:
    if include_tls:
        tls_dir = dest / "tls" / "server"
        ca_src = Path("secrets/tls/server/server-ca.pem")
        key_src = Path("secrets/tls/server/server.key")
        pem_src = Path("secrets/tls/server/server.pem")
        copied = False
        missing: list[str] = []
        for src in (ca_src, key_src, pem_src):
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
    _copy_redis_tls_assets(dest, include_tls, ca_only=True)

    default_worker_config = Path("configs/worker_config.yaml")
    if default_worker_config.exists():
        shutil.copy2(default_worker_config, dest / "worker_config.yaml")
    else:
        logging.warning(f"Warning: worker config not found: {default_worker_config}")


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


def _write_install_script(dest: Path) -> None:
    """Write an install.sh script to set up a venv and install bundled wheels."""
    script_path = dest / "install.sh"
    script = """#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
UV_BIN="${UV_BIN:-uv}"
PYTHON_REQ="${FLOWMESH_PYTHON:-3.12}"
ENV_FILE=".env"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  UV_BIN="${UV_BIN:-uv}"
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
"$UV_BIN" pip install ./wheels/*.whl
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


def _create_bundle_tarball(tar_path: Path, include_tls: bool) -> None:
    """Create a deployable bundle as a tar.gz with a top-level prefix."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "flowmesh_server_bundle"
    with tempfile.TemporaryDirectory(prefix="flowmesh-bundle-") as tmp:
        staging_root = Path(tmp) / prefix
        staging_root.mkdir(parents=True, exist_ok=True)
        _copy_server_assets(staging_root, include_tls=include_tls)
        wheel_dir = staging_root / "wheels"
        _build_cli_wheels(wheel_dir)
        _write_install_script(staging_root)
        with tarfile.open(tar_path, mode="w:gz") as tf:
            # Ensure we archive the top-level prefix directory.
            tf.add(staging_root, arcname=prefix)


@app.command("export")
def bundle_export(
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output tar.gz path (default: ./dist/flowmesh_server_bundle.tar.gz)",
    ),
    no_tls: bool = typer.Option(False, "--no-tls", help="Exclude TLS assets"),
) -> None:
    """Create a self-contained deployment bundle for the server."""
    logging.info("Creating server bundle...")
    tar_path = output
    if tar_path is None:
        tar_path = Path("./dist/flowmesh_server_bundle.tar.gz")
        tar_path.parent.mkdir(parents=True, exist_ok=True)
    _create_bundle_tarball(tar_path, include_tls=not no_tls)
    logging.success(f"Bundle created at {tar_path}")
