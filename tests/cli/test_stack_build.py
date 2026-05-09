import pytest
import typer
from flowmesh_cli_stack.stack import (
    _parse_buildx_driver,
    _platform_overrides,
    _resolve_bake_batches,
    _resolve_build_targets,
)
from flowmesh_stack.images import get_cache_ref


def test_resolve_build_targets_expands_gpu_builder_dependency() -> None:
    targets = _resolve_build_targets(["workers"])
    assert targets == [
        "flowmesh_worker_cpu",
        "flowmesh_worker_gpu_builder",
        "flowmesh_worker_gpu",
        "flowmesh_ssh_cpu",
        "flowmesh_ssh_gpu",
    ]


def test_resolve_bake_batches_include_builder_by_default() -> None:
    assert _resolve_bake_batches(None) == [["builders"], ["server", "workers"]]


def test_resolve_bake_batches_skip_standalone_builder_when_requested() -> None:
    assert _resolve_bake_batches(None, no_builder=True) == [["server", "workers"]]


def test_resolve_bake_batches_reject_explicit_builder_target_with_no_builder() -> None:
    with pytest.raises(typer.Exit):
        _resolve_bake_batches(["flowmesh_worker_gpu_builder"], no_builder=True)


def test_platform_overrides_use_local_for_build_mode() -> None:
    overrides = _platform_overrides(
        "load",
        ["flowmesh_server", "flowmesh_worker_gpu_builder", "flowmesh_worker_gpu"],
    )
    assert overrides == [
        ("flowmesh_server", "local"),
        ("flowmesh_worker_gpu_builder", "local"),
        ("flowmesh_worker_gpu", "local"),
    ]


def test_platform_overrides_use_multiarch_for_push_mode() -> None:
    overrides = _platform_overrides(
        "push",
        ["flowmesh_server", "flowmesh_worker_gpu_builder", "flowmesh_worker_gpu"],
    )
    assert overrides == [
        ("flowmesh_server", "linux/amd64,linux/arm64"),
        ("flowmesh_worker_gpu_builder", "linux/amd64,linux/arm64"),
        ("flowmesh_worker_gpu", "linux/amd64,linux/arm64"),
    ]


def test_parse_buildx_driver_returns_named_driver() -> None:
    output = """Name:   default
Driver: docker-container
Nodes:
Name:      default0
"""
    assert _parse_buildx_driver(output) == "docker-container"


def test_get_cache_ref_uses_stable_target_specific_tags() -> None:
    assert (
        get_cache_ref("ghcr.io/mlsys-io", "cache", "flowmesh_server")
        == "ghcr.io/mlsys-io/flowmesh_server:cache"
    )
    assert (
        get_cache_ref("ghcr.io/mlsys-io", "cache", "flowmesh_worker_gpu")
        == "ghcr.io/mlsys-io/flowmesh_worker:cache-gpu"
    )
    assert (
        get_cache_ref("ghcr.io/mlsys-io", "v2", "flowmesh_worker_gpu_builder")
        == "ghcr.io/mlsys-io/flowmesh_worker_builder:cache-v2-gpu"
    )
    assert (
        get_cache_ref("ghcr.io/mlsys-io", "cache-v2", "flowmesh_worker_gpu_builder")
        == "ghcr.io/mlsys-io/flowmesh_worker_builder:cache-v2-gpu"
    )
