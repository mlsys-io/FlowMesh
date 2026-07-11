"""FlowMesh Docker image reference management.

Provides the canonical mapping from build targets to image references,
used by both the CLI dev commands and programmatic build/deploy scripts.
"""

_TARGET_COMPONENTS: dict[str, tuple[str, str]] = {
    "flowmesh_server": ("flowmesh_server", ""),
    "flowmesh_worker_cpu": ("flowmesh_worker", "-cpu"),
    "flowmesh_worker_gpu_builder": ("flowmesh_worker_builder", "-gpu"),
    "flowmesh_worker_gpu": ("flowmesh_worker", "-gpu"),
    "flowmesh_ssh_cpu": ("flowmesh_ssh", "-cpu"),
    "flowmesh_ssh_gpu": ("flowmesh_ssh", "-gpu"),
}
BUILD_TARGETS: dict[str, str] = {
    target: f"{{registry}}/{repo}:{{version}}{suffix}"
    for target, (repo, suffix) in _TARGET_COMPONENTS.items()
}
"""Mapping from build target name to image reference format string."""

CACHE_TARGETS: dict[str, str] = {
    target: f"{{registry}}/{repo}:{{scope}}{suffix}"
    for target, (repo, suffix) in _TARGET_COMPONENTS.items()
}
"""Mapping from build target name to registry cache reference format string."""

BUILD_GROUPS: dict[str, list[str]] = {
    "server": ["flowmesh_server"],
    "workers": [
        "flowmesh_worker_cpu",
        "flowmesh_worker_gpu",
        "flowmesh_ssh_cpu",
        "flowmesh_ssh_gpu",
    ],
    "builders": ["flowmesh_worker_gpu_builder"],
}
"""Mapping from group name to list of build targets."""
BUILD_GROUPS["default"] = [
    target for group in ("server", "workers") for target in BUILD_GROUPS[group]
]

BUILD_DEPENDENCIES: dict[str, list[str]] = {
    "flowmesh_worker_gpu": ["flowmesh_worker_gpu_builder"],
}
"""Auxiliary build targets that must be configured alongside a selected target."""

PUSH_PLATFORMS: dict[str, str] = {
    target: "linux/amd64,linux/arm64" for target in BUILD_TARGETS
}
"""Default platform matrix to publish for each build target."""


def get_image_ref(registry: str, version: str, target: str) -> str:
    """Resolve a Docker image reference for a build target.

    Args:
        registry: Container registry (e.g. ``ghcr.io/mlsys-io``).
        version: Image version tag (e.g. ``dev``, ``0.1.0``).
        target: Build target name (must be a key in :data:`BUILD_TARGETS`).

    Raises:
        ValueError: If the target is unknown.
    """
    if target not in BUILD_TARGETS:
        raise ValueError(f"Unknown build target: {target}")
    return BUILD_TARGETS[target].format(registry=registry, version=version)


def get_cache_ref(registry: str, scope: str, target: str) -> str:
    """Resolve a Docker registry cache reference for a build target."""

    if target not in CACHE_TARGETS:
        raise ValueError(f"Unknown build target: {target}")
    normalized_scope = (scope or "").strip()
    if not normalized_scope or normalized_scope == "cache":
        cache_scope = "cache"
    elif normalized_scope.startswith("cache-"):
        cache_scope = normalized_scope
    else:
        cache_scope = f"cache-{normalized_scope}"
    return CACHE_TARGETS[target].format(registry=registry, scope=cache_scope)


def expand_build_targets(targets: list[str]) -> list[str]:
    """Expand explicit targets with any dependent helper targets.

    The returned list preserves first-seen order and inserts helper targets ahead
    of any target that requires them.
    """

    expanded: list[str] = []
    seen: set[str] = set()

    def _visit(target: str) -> None:
        if target in seen:
            return
        for dep in BUILD_DEPENDENCIES.get(target, []):
            _visit(dep)
        seen.add(target)
        expanded.append(target)

    for target in targets:
        _visit(target)
    return expanded


def get_push_platforms(target: str) -> str:
    """Resolve the default push platform matrix for a build target."""

    if target not in PUSH_PLATFORMS:
        raise ValueError(f"Unknown build target: {target}")
    return PUSH_PLATFORMS[target]


_RESERVED_CACHE_VERSION = "cache"
_RESERVED_CACHE_PREFIX = "cache-"


def _split_repo_tag(ref: str) -> tuple[str, str] | None:
    """Split an image ref into ``(repo, tag)``, or ``None`` when it has no tag.

    Splits on the final colon and applies the Docker rule that a colon whose right side
    contains ``/`` is a registry port, not a tag separator (e.g.
    ``localhost:5000/org/img``).
    """
    repo, sep, tag = ref.rpartition(":")
    return (repo, tag) if sep and "/" not in tag else None


def managed_repos(registry: str) -> set[str]:
    """Return the distinct image repositories FlowMesh publishes under a registry."""
    return {f"{registry}/{_TARGET_COMPONENTS[target][0]}" for target in BUILD_TARGETS}


def parse_image_ref(registry: str, ref: str) -> tuple[str, str] | None:
    """Map a concrete image reference back to its ``(target, version)``.

    Inverse of :func:`get_image_ref`. Returns ``None`` for references outside the
    managed repositories, references without a tag, and cache-scope tags reserved by
    :data:`CACHE_TARGETS` (``cache`` / ``cache-*``), which are registry-only and must
    never be treated as prunable versions.
    """
    split = _split_repo_tag(ref)
    if split is None:
        return None
    repo, tag = split
    for target, (target_repo, suffix) in _TARGET_COMPONENTS.items():
        if repo != f"{registry}/{target_repo}":
            continue
        if suffix and not tag.endswith(suffix):
            continue
        version = tag.removesuffix(suffix)
        if not version:
            continue
        if version == _RESERVED_CACHE_VERSION or version.startswith(
            _RESERVED_CACHE_PREFIX
        ):
            return None
        return target, version
    return None
