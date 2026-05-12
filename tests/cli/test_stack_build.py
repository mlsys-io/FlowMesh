from subprocess import CompletedProcess
from typing import Any
from unittest.mock import patch

import pytest
import typer
from flowmesh_cli_stack import stack as stack_module
from flowmesh_cli_stack.stack import (
    _ensure_buildx_builder_ready,
    _parse_buildx_field,
    _platform_overrides,
    _resolve_bake_batches,
    _resolve_build_targets,
    _switch_active_buildx_builder,
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


def test_parse_buildx_field_returns_named_value() -> None:
    output = """Name:   default
Driver: docker-container
Nodes:
Name:      default0
"""
    assert _parse_buildx_field(output, "Driver") == "docker-container"
    assert _parse_buildx_field(output, "Name") == "default"
    assert _parse_buildx_field(output, "Missing") is None


def _ok(stdout: str) -> CompletedProcess[str]:
    return CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _err(returncode: int = 1, stderr: str = "not found") -> CompletedProcess[str]:
    return CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_ensure_buildx_builder_ready_accepts_matching_driver() -> None:
    inspect = _ok("Name: default\nDriver: docker\n")
    with patch.object(stack_module, "_inspect_buildx_builder", return_value=inspect):
        _ensure_buildx_builder_ready("/usr/bin/docker", "default", "docker")


def test_ensure_buildx_builder_ready_rejects_driver_mismatch() -> None:
    inspect = _ok("Name: default\nDriver: docker-container\n")
    with patch.object(stack_module, "_inspect_buildx_builder", return_value=inspect):
        with pytest.raises(typer.Exit):
            _ensure_buildx_builder_ready("/usr/bin/docker", "default", "docker")


def test_ensure_buildx_builder_ready_errors_when_builder_missing() -> None:
    with patch.object(stack_module, "_inspect_buildx_builder", return_value=_err()):
        with pytest.raises(typer.Exit):
            _ensure_buildx_builder_ready(
                "/usr/bin/docker",
                "flowmesh-multiarch",
                "docker-container",
                missing_hint="hint",
            )


def test_switch_active_buildx_builder_noop_when_already_active() -> None:
    inspect = _ok("Name: default\nDriver: docker\n")
    with (
        patch.object(stack_module, "_inspect_buildx_builder", return_value=inspect),
        patch.object(stack_module.subprocess, "run") as run_mock,
    ):
        _switch_active_buildx_builder("/usr/bin/docker", "default", force=False)
        run_mock.assert_not_called()


def test_switch_active_buildx_builder_runs_use_when_force() -> None:
    inspect = _ok("Name: other\nDriver: docker-container\n")
    use_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with (
        patch.object(stack_module, "_inspect_buildx_builder", return_value=inspect),
        patch.object(
            stack_module.subprocess, "run", return_value=use_result
        ) as run_mock,
        patch.object(stack_module.typer, "confirm") as confirm_mock,
    ):
        _switch_active_buildx_builder("/usr/bin/docker", "default", force=True)
        confirm_mock.assert_not_called()
        assert run_mock.call_args.args[0][:4] == [
            "/usr/bin/docker",
            "buildx",
            "use",
            "default",
        ]


def test_switch_active_buildx_builder_prompts_when_not_forced() -> None:
    inspect = _ok("Name: other\nDriver: docker-container\n")
    use_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with (
        patch.object(stack_module, "_inspect_buildx_builder", return_value=inspect),
        patch.object(stack_module.subprocess, "run", return_value=use_result),
        patch.object(stack_module.typer, "confirm", return_value=True) as confirm_mock,
    ):
        _switch_active_buildx_builder("/usr/bin/docker", "default", force=False)
        confirm_mock.assert_called_once()


def test_switch_active_buildx_builder_aborts_on_decline() -> None:
    inspect = _ok("Name: other\nDriver: docker-container\n")
    with (
        patch.object(stack_module, "_inspect_buildx_builder", return_value=inspect),
        patch.object(stack_module.subprocess, "run") as run_mock,
        patch.object(stack_module.typer, "confirm", return_value=False),
    ):
        with pytest.raises(typer.Exit):
            _switch_active_buildx_builder("/usr/bin/docker", "default", force=False)
        run_mock.assert_not_called()


def _capture_bake_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> list[list[str]]:
    """Stub out the bake subprocess and capture the argv each call would launch."""
    captured: list[list[str]] = []

    def fake_run(args: list[str], **_: Any) -> CompletedProcess[str]:
        captured.append(args)
        return CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stack_module.subprocess, "run", fake_run)
    monkeypatch.setattr(stack_module, "ensure_docker_available", lambda: None)
    monkeypatch.setattr(stack_module, "_require_bin", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        stack_module,
        "_inspect_buildx_builder",
        lambda _b, name: _ok(f"Name: {name or 'default'}\nDriver: docker\n"),
    )
    monkeypatch.setattr(
        stack_module, "_get_active_buildx_builder", lambda _b: "default"
    )
    monkeypatch.setattr(
        stack_module, "_resolve_bake_batches", lambda _t, no_builder=False: [["server"]]
    )
    monkeypatch.setattr(
        stack_module, "_resolve_build_targets", lambda batch: ["flowmesh_server"]
    )
    monkeypatch.setattr(stack_module, "_platform_overrides", lambda _m, _t: [])
    monkeypatch.setattr(stack_module, "ensure_env_file", lambda *_a, **_k: None)
    monkeypatch.setattr(stack_module, "load_env", lambda *_a, **_k: None)
    bake_file = tmp_path / "bake.hcl"
    bake_file.write_text("")
    monkeypatch.setattr(stack_module, "stack_bake_file", lambda: bake_file)
    monkeypatch.setattr(
        stack_module, "stack_env_example", lambda: tmp_path / ".env.example"
    )
    # buildx version check returns success.
    return captured


def _flatten(args: list[str]) -> str:
    return " ".join(args)


def test_run_bake_load_uses_local_cache_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured = _capture_bake_args(monkeypatch, tmp_path)
    stack_module._run_bake(
        "load", None, tmp_path / ".env", builder="default", force=True
    )
    bake_calls = [a for a in captured if "bake" in a]
    assert bake_calls, "expected a bake invocation"
    cmd = _flatten(bake_calls[0])
    assert "--builder default" in cmd
    assert "--load" in cmd
    assert "cache-from=type=registry" not in cmd
    assert "cache-to=type=registry" not in cmd


def test_run_bake_push_uses_registry_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured = _capture_bake_args(monkeypatch, tmp_path)
    monkeypatch.setattr(
        stack_module,
        "_inspect_buildx_builder",
        lambda _b, name: _ok(
            f"Name: {name or 'flowmesh-multiarch'}\nDriver: docker-container\n"
        ),
    )
    monkeypatch.setattr(
        stack_module, "_get_active_buildx_builder", lambda _b: "flowmesh-multiarch"
    )
    stack_module._run_bake(
        "push",
        None,
        tmp_path / ".env",
        builder="flowmesh-multiarch",
        force=True,
    )
    bake_calls = [a for a in captured if "bake" in a]
    assert bake_calls, "expected a bake invocation"
    cmd = _flatten(bake_calls[0])
    assert "--builder flowmesh-multiarch" in cmd
    assert "--push" in cmd
    assert "cache-from=type=registry" in cmd
    assert "cache-to=type=registry" in cmd


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
