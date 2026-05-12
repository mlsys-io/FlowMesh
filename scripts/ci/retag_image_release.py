"""Retag FlowMesh container images as ``:latest`` for a release tag.

Iterates ``flowmesh-sdk-stack``'s canonical ``BUILD_TARGETS`` and, for
each target, writes a ``:latest`` tag (preserving the ``-cpu`` /
``-gpu`` variant suffix encoded in the image reference template) that
points at the same manifest digest as ``--tag``. The retag is a pure
registry rewrite via ``docker buildx imagetools create``; no rebuild
and no daemon image-store interaction.

Downgrade protection: if a ``:latest`` reference already exists and its
``org.opencontainers.image.version`` label parses to a version greater
than or equal to ``--tag``, the script aborts unless ``--force`` is
given. A ``:latest`` that exists but lacks a readable version label is
also rejected unless ``--force``, on the principle that an unattributed
tag should not be silently overwritten.

All preconditions are checked across every target before any retag
executes, so the script either retags the full set or none of it.
"""

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any

from flowmesh_stack.images import BUILD_TARGETS, get_image_ref
from packaging.version import InvalidVersion, Version


class MissingVersionLabel(Exception):
    """Existing ``:latest`` exists but its version label is unreadable."""


class TransientInspectError(Exception):
    """``imagetools inspect`` failed in a way that is not a clear 'not found'.

    Examples include registry 5xx, auth failures, and network timeouts —
    states where treating ``:latest`` as absent would silently disable the
    downgrade guard.
    """


_MISSING_REF_STDERR_PATTERNS = (
    "not found",
    "manifest unknown",
)


def _docker_bin() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("docker binary not found on PATH")
    return docker


def _imagetools_inspect(
    docker: str, ref: str, output_format: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603: argv list, no shell, docker path from shutil.which()
        [docker, "buildx", "imagetools", "inspect", ref, "--format", output_format],
        capture_output=True,
        text=True,
        check=False,
    )


def _existing_latest_version(docker: str, latest_ref: str) -> Version | None:
    """Resolve the version of an existing ``:latest``.

    Returns ``None`` when ``:latest`` definitively does not exist on the
    registry (inspect failed with a stderr pattern matching a missing
    reference). Raises :class:`TransientInspectError` for any other
    inspect failure so the caller can decide whether to abort or proceed
    under ``--force``. Raises :class:`MissingVersionLabel` when
    ``:latest`` exists but the expected OCI version label cannot be
    parsed from any platform.
    """

    result = _imagetools_inspect(docker, latest_ref, "{{json .Image}}")
    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        if any(pattern in stderr_lower for pattern in _MISSING_REF_STDERR_PATTERNS):
            return None
        raise TransientInspectError(
            f"{latest_ref}: imagetools inspect failed with rc={result.returncode}: "
            f"{result.stderr.strip()}"
        )
    try:
        images: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MissingVersionLabel(
            f"{latest_ref}: imagetools .Image output is not valid JSON ({exc})"
        )
    for blob in images.values():
        labels = blob.get("config", {}).get("Labels") or {}
        version_label = labels.get("org.opencontainers.image.version")
        if not version_label:
            continue
        try:
            return Version(version_label.removeprefix("v"))
        except InvalidVersion as exc:
            raise MissingVersionLabel(
                f"{latest_ref}: image.version={version_label!r} is not PEP 440 ({exc})"
            )
    raise MissingVersionLabel(
        f"{latest_ref}: no readable image.version label on any platform"
    )


def _plan_retags(
    docker: str, tag: str, registry: str, force: bool
) -> list[tuple[str, str]]:
    new_version = Version(tag.removeprefix("v"))
    plan: list[tuple[str, str]] = []
    for target in BUILD_TARGETS:
        source_ref = get_image_ref(registry, tag, target)
        latest_ref = get_image_ref(registry, "latest", target)
        try:
            existing = _existing_latest_version(docker, latest_ref)
        except MissingVersionLabel as exc:
            if not force:
                print(
                    f"::error::{exc}; pass --force to retag anyway",
                    file=sys.stderr,
                )
                sys.exit(1)
            existing = None
        except TransientInspectError as exc:
            if not force:
                print(
                    f"::error::{exc}. The downgrade guard cannot run while "
                    f"inspect fails for unknown reasons; resolve the registry "
                    f"error or pass --force to retag anyway.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                f"::warning::{exc}; --force set, skipping downgrade check",
                file=sys.stderr,
            )
            existing = None
        if existing is not None and existing >= new_version and not force:
            print(
                f"::error::{latest_ref} already points at version {existing} "
                f"(>= {new_version}); skipping retag. Pass --force to override.",
                file=sys.stderr,
            )
            sys.exit(1)
        plan.append((source_ref, latest_ref))
    return plan


def _execute_retag(docker: str, source_ref: str, latest_ref: str) -> None:
    print(f"Retagging {latest_ref} -> {source_ref}")
    result = subprocess.run(  # nosec B603: argv list, no shell, docker path from shutil.which()
        [docker, "buildx", "imagetools", "create", "-t", latest_ref, source_ref],
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"::error::failed to retag {latest_ref}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    docker = _docker_bin()
    plan = _plan_retags(docker, args.tag, args.registry, args.force)
    for source_ref, latest_ref in plan:
        _execute_retag(docker, source_ref, latest_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
