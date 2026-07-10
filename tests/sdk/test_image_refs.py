"""Tests for image-reference parsing in ``flowmesh_stack.images``."""

import pytest
from flowmesh_stack.images import (
    BUILD_TARGETS,
    get_image_ref,
    managed_repos,
    parse_image_ref,
)

REGISTRY = "ghcr.io/mlsys-io"


@pytest.mark.parametrize("target", list(BUILD_TARGETS))
def test_parse_image_ref_round_trips_every_target(target: str) -> None:
    ref = get_image_ref(REGISTRY, "1.2.3", target)
    assert parse_image_ref(REGISTRY, ref) == (target, "1.2.3")


def test_parse_image_ref_distinguishes_cpu_and_gpu_on_shared_repo() -> None:
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_worker:dev-cpu") == (
        "flowmesh_worker_cpu",
        "dev",
    )
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_worker:dev-gpu") == (
        "flowmesh_worker_gpu",
        "dev",
    )


def test_parse_image_ref_builder_repo_is_not_worker_repo() -> None:
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_worker_builder:dev-gpu") == (
        "flowmesh_worker_gpu_builder",
        "dev",
    )


def test_parse_image_ref_preserves_versions_with_dashes() -> None:
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_server:my-feature-1") == (
        "flowmesh_server",
        "my-feature-1",
    )


def test_parse_image_ref_rejects_unmanaged_and_malformed() -> None:
    assert parse_image_ref(REGISTRY, "ubuntu:22.04") is None
    assert parse_image_ref(REGISTRY, "other.io/mlsys-io/flowmesh_server:dev") is None
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_server") is None
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_worker:dev-tpu") is None


def test_parse_image_ref_rejects_reserved_cache_tags() -> None:
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_server:cache") is None
    assert parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_worker:cache-gpu") is None
    assert (
        parse_image_ref(REGISTRY, f"{REGISTRY}/flowmesh_worker_builder:cache-v2-gpu")
        is None
    )


def test_managed_repos_returns_distinct_repos() -> None:
    assert managed_repos(REGISTRY) == {
        f"{REGISTRY}/flowmesh_server",
        f"{REGISTRY}/flowmesh_worker",
        f"{REGISTRY}/flowmesh_worker_builder",
        f"{REGISTRY}/flowmesh_ssh",
    }
