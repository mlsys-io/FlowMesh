"""Update synchronized FlowMesh package versions and internal pins."""

import argparse
import re
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PYPROJECTS: tuple[Path, ...] = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "cli" / "pyproject.toml",
    REPO_ROOT / "cli" / "stack" / "pyproject.toml",
    REPO_ROOT / "hook" / "pyproject.toml",
    REPO_ROOT / "sdk" / "pyproject.toml",
    REPO_ROOT / "sdk" / "stack" / "pyproject.toml",
)
SDK_VERSION_MODULE = REPO_ROOT / "sdk" / "src" / "flowmesh" / "_version.py"
CLI_VERSION_MODULE = REPO_ROOT / "cli" / "src" / "flowmesh_cli" / "_version.py"
SHARED_VERSION_MODULE = REPO_ROOT / "src" / "shared" / "_version.py"
FIRST_PARTY_DISTRIBUTIONS: tuple[str, ...] = (
    "flowmesh-cli-stack",
    "flowmesh-sdk-stack",
    "flowmesh-cli",
    "flowmesh-hook",
    "flowmesh-sdk",
    "flowmesh",
)

_VERSION_RE = re.compile(r'(?m)^version = "[^"]+"$')
_STATIC_VERSION_RE = re.compile(r'(?m)^_STATIC_VERSION = "[^"]+"$')
_SHARED_RUNTIME_VERSION_RE = re.compile(r'(?m)^FLOWMESH_RELEASE_VERSION = "[^"]+"$')
_PIN_RE = re.compile(
    r"(?P<name>\b(?:"
    + "|".join(re.escape(name) for name in FIRST_PARTY_DISTRIBUTIONS)
    + r")\b)(?P<extras>\[[^\]]+\])?==(?P<version>[^\"'\s,\]]+)"
)


def _normalize_version(raw: str) -> str:
    try:
        return str(Version(raw.removeprefix("v")))
    except InvalidVersion as exc:
        raise SystemExit(f"Version is not PEP 440: {raw!r} ({exc}).")


def _render(text: str, version: str, path: Path) -> str:
    versioned, count = _VERSION_RE.subn(f'version = "{version}"', text, count=1)
    if count != 1:
        rel = path.relative_to(REPO_ROOT)
        raise SystemExit(f"Expected one project version line in {rel}.")
    return _PIN_RE.sub(
        lambda m: f"{m.group('name')}{m.group('extras') or ''}=={version}",
        versioned,
    )


def _render_literal_assignment(
    text: str,
    version: str,
    path: Path,
    pattern: re.Pattern[str],
    name: str,
) -> str:
    if len(pattern.findall(text)) != 1:
        rel = path.relative_to(REPO_ROOT)
        raise SystemExit(f"Expected one {name} line in {rel}.")
    rendered = pattern.sub(f'{name} = "{version}"', text, count=1)
    return rendered


def _render_static_version_module(text: str, version: str, path: Path) -> str:
    return _render_literal_assignment(
        text,
        version,
        path,
        _STATIC_VERSION_RE,
        "_STATIC_VERSION",
    )


def _render_shared_version_module(text: str, version: str) -> str:
    return _render_literal_assignment(
        text,
        version,
        SHARED_VERSION_MODULE,
        _SHARED_RUNTIME_VERSION_RE,
        "FLOWMESH_RELEASE_VERSION",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Synchronized release version, e.g. 0.1.1.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the package files are not already set to the version.",
    )
    args = parser.parse_args()

    version = _normalize_version(args.version)
    rendered: list[tuple[Path, str, str]] = []
    for path in PACKAGE_PYPROJECTS:
        current = path.read_text()
        rendered.append((path, current, _render(current, version, path)))
    for static_module in (SDK_VERSION_MODULE, CLI_VERSION_MODULE):
        current = static_module.read_text()
        rendered.append(
            (
                static_module,
                current,
                _render_static_version_module(current, version, static_module),
            )
        )
    shared_version_current = SHARED_VERSION_MODULE.read_text()
    rendered.append(
        (
            SHARED_VERSION_MODULE,
            shared_version_current,
            _render_shared_version_module(shared_version_current, version),
        )
    )

    changed = [path for path, current, updated in rendered if current != updated]
    if args.check:
        if changed:
            print("Package versions need updates:")
            for path in changed:
                print(f"- {path.relative_to(REPO_ROOT)}")
            return 1
        print(f"Package versions are already set to {version}.")
        return 0

    for path, current, updated in rendered:
        if current != updated:
            path.write_text(updated)

    if changed:
        print(f"Updated package versions and internal pins to {version}:")
        for path in changed:
            print(f"- {path.relative_to(REPO_ROOT)}")
    else:
        print(f"Package versions are already set to {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
