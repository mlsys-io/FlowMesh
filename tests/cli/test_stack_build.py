from flowmesh_cli_stack.stack import (
    _parse_buildx_driver,
    _platform_overrides,
    _resolve_build_targets,
)


def test_resolve_build_targets_expands_gpu_builder_dependency() -> None:
    targets = _resolve_build_targets(["workers"])
    assert targets == [
        "flowmesh_worker_cpu",
        "flowmesh_worker_gpu_builder",
        "flowmesh_worker_gpu",
        "flowmesh_ssh_cpu",
        "flowmesh_ssh_gpu",
    ]


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
