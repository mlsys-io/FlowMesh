"""Validate built FlowMesh distributions and umbrella extras."""

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

RUNTIME_TOP_LEVELS = {"server", "shared", "worker"}


def _python_bin(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _script_bin(env_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / f"{name}.exe"
    return env_dir / "bin" / name


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)  # nosec B603: fixed argv list, no shell.


def _root_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("flowmesh-*-py3-none-any.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected one root flowmesh wheel, found {len(wheels)}")
    return wheels[0]


def _check_root_wheel(root_wheel: Path) -> None:
    """Ensure the metapackage wheel does not ship runtime source modules."""
    with ZipFile(root_wheel) as zf:
        bad = [
            name
            for name in zf.namelist()
            if name.split("/", 1)[0] in RUNTIME_TOP_LEVELS
        ]
    if bad:
        raise SystemExit(
            "root flowmesh wheel contains runtime source: " + ", ".join(bad[:5])
        )


def _create_venv(env_dir: Path) -> Path:
    venv.EnvBuilder(with_pip=True).create(env_dir)
    return _python_bin(env_dir)


def _smoke_extra(
    dist_dir: Path, root_wheel: Path, extra: str, import_name: str
) -> None:
    """Install one umbrella extra in a fresh venv and import its public module."""
    with tempfile.TemporaryDirectory(prefix=f"flowmesh-{extra}-smoke-") as tmp:
        env_dir = Path(tmp) / ".venv"
        python = _create_venv(env_dir)
        _run(
            [
                python.as_posix(),
                "-m",
                "pip",
                "install",
                "--find-links",
                dist_dir.as_posix(),
                f"{root_wheel.as_posix()}[{extra}]",
            ]
        )
        _run([python.as_posix(), "-c", f"import {import_name}"])


def _smoke_cli(dist_dir: Path, root_wheel: Path) -> None:
    """Install the CLI umbrella extra in a fresh venv and run `flowmesh --help`."""
    with tempfile.TemporaryDirectory(prefix="flowmesh-cli-smoke-") as tmp:
        env_dir = Path(tmp) / ".venv"
        python = _create_venv(env_dir)
        _run(
            [
                python.as_posix(),
                "-m",
                "pip",
                "install",
                "--find-links",
                dist_dir.as_posix(),
                f"{root_wheel.as_posix()}[cli]",
            ]
        )
        _run([_script_bin(env_dir, "flowmesh").as_posix(), "--help"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        default="dist",
        type=Path,
        help="Directory containing wheels built by `uv build`.",
    )
    args = parser.parse_args()

    dist_dir = args.dist.resolve()
    if not dist_dir.is_dir():
        raise SystemExit(f"Distribution directory does not exist: {dist_dir}")

    root_wheel = _root_wheel(dist_dir)
    _check_root_wheel(root_wheel)
    _smoke_extra(dist_dir, root_wheel, "sdk", "flowmesh")
    _smoke_extra(dist_dir, root_wheel, "hook", "flowmesh_hook")
    _smoke_cli(dist_dir, root_wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
