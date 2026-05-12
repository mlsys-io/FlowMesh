"""Verify FlowMesh container images for a release.

Resolves the expected image references for ``--tag`` against the
``flowmesh-sdk-stack`` canonical map, queries each one through
``docker buildx imagetools inspect``, and asserts:

* The published manifest is a multi-arch OCI image index.
* The platform set (excluding buildx attestation manifests with
  ``platform.architecture=unknown``) matches the target's declared
  ``PUSH_PLATFORMS`` entry.
* Every per-platform image config carries
  ``org.opencontainers.image.version == tag`` and
  ``org.opencontainers.image.revision == commit``.

Writes a Markdown digest table to ``--markdown-file`` and, when given,
appends ``is_release_tag=<bool>`` to ``--github-output`` for downstream
workflow gating. A tag is treated as a release (eligible for ``:latest``
retag) when it parses as PEP 440 and is neither a pre-release, dev
release, nor carries a local version segment. Post-releases
(``vX.Y.Z.postN``) are eligible.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from flowmesh_stack.images import BUILD_TARGETS, PUSH_PLATFORMS, get_image_ref
from packaging.version import InvalidVersion, Version

# Buildx emits one of these mediatypes for multi-arch publications depending
# on driver / BuildKit version. The legacy Docker manifest list is just as
# valid a multi-arch index as the OCI form.
ACCEPTED_INDEX_MEDIATYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)


def _docker_bin() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("docker binary not found on PATH")
    return docker


def _imagetools_inspect(docker: str, ref: str, output_format: str) -> dict[str, Any]:
    result = subprocess.run(  # nosec B603: argv list, no shell, docker path from shutil.which()
        [docker, "buildx", "imagetools", "inspect", ref, "--format", output_format],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"imagetools inspect {ref} (format={output_format!r}) failed: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _is_release(tag: str) -> bool:
    try:
        parsed = Version(tag.removeprefix("v"))
    except InvalidVersion:
        return False
    if parsed.is_prerelease or parsed.is_devrelease:
        return False
    if parsed.local is not None:
        return False
    return True


def _check_target(
    docker: str, target: str, registry: str, tag: str, commit: str
) -> tuple[str, str | None, list[str]]:
    ref = get_image_ref(registry, tag, target)
    errors: list[str] = []

    manifest = _imagetools_inspect(docker, ref, "{{json .Manifest}}")
    actual_media = manifest.get("mediaType")
    if actual_media not in ACCEPTED_INDEX_MEDIATYPES:
        errors.append(
            f"{ref}: mediaType {actual_media!r} is not a multi-arch index "
            f"(expected one of {sorted(ACCEPTED_INDEX_MEDIATYPES)})"
        )
        return ref, manifest.get("digest"), errors

    expected_platforms = set(PUSH_PLATFORMS[target].split(","))
    actual_platforms: set[str] = set()
    for entry in manifest.get("manifests", []):
        platform = entry.get("platform", {})
        arch = platform.get("architecture", "")
        if arch == "unknown":
            continue
        actual_platforms.add(f"{platform.get('os', '')}/{arch}")
    if actual_platforms != expected_platforms:
        errors.append(
            f"{ref}: platforms {sorted(actual_platforms)}, "
            f"expected {sorted(expected_platforms)}"
        )

    images = _imagetools_inspect(docker, ref, "{{json .Image}}")
    for platform in sorted(expected_platforms):
        blob = images.get(platform, {})
        labels = blob.get("config", {}).get("Labels") or {}
        version_label = labels.get("org.opencontainers.image.version")
        if version_label != tag:
            errors.append(
                f"{ref} [{platform}]: image.version={version_label!r}, expected {tag!r}"
            )
        revision_label = labels.get("org.opencontainers.image.revision")
        if revision_label != commit:
            errors.append(
                f"{ref} [{platform}]: image.revision={revision_label!r}, "
                f"expected {commit!r}"
            )

    return ref, manifest.get("digest"), errors


def _write_markdown(
    tag: str, rows: list[tuple[str, str | None, str]], path: Path
) -> None:
    lines = [
        f"### FlowMesh container images for `{tag}`",
        "",
        "| Image | Platforms | Digest |",
        "|---|---|---|",
    ]
    for ref, digest, platforms in rows:
        digest_str = f"`{digest}`" if digest else "_unavailable_"
        lines.append(f"| `{ref}` | `{platforms}` | {digest_str} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--markdown-file", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    docker = _docker_bin()
    rows: list[tuple[str, str | None, str]] = []
    all_errors: list[str] = []

    for target in BUILD_TARGETS:
        ref, digest, errors = _check_target(
            docker, target, args.registry, args.tag, args.commit
        )
        if errors:
            for err in errors:
                print(f"::error::{err}", file=sys.stderr)
            all_errors.extend(errors)
        rows.append((ref, digest, PUSH_PLATFORMS[target]))

    _write_markdown(args.tag, rows, args.markdown_file)

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as f:
            f.write(f"is_release_tag={'true' if _is_release(args.tag) else 'false'}\n")

    if all_errors:
        print(
            f"::error::{len(all_errors)} image verification failure(s)", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
