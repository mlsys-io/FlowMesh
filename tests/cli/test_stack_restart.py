"""Service-scoped `flowmesh stack restart [SERVICE]...` behavior."""

from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from flowmesh.models.nodes import NodeRole
from flowmesh_cli_stack import stack as stack_module


def _restart(**kwargs: object) -> None:
    defaults: dict[str, object] = {
        "services": None,
        "env_file": Path(".env"),
        "image_tag": None,
        "pull": True,
    }
    defaults.update(kwargs)
    stack_module.restart(**defaults)  # type: ignore[arg-type]


def test_restart_server_drains_then_recreates_only_server() -> None:
    with (
        patch.object(stack_module, "_drain_workers") as drain,
        patch.object(stack_module, "_compose") as compose,
        patch.object(stack_module, "_node_role", return_value=NodeRole.ROOT),
        patch.object(stack_module, "image_env_overrides", return_value={}),
    ):
        _restart(services=["server"])

    drain.assert_called_once()
    up_args = compose.call_args.args[0]
    assert up_args[:5] == ["up", "-d", "--no-deps", "--force-recreate", "--wait"]
    assert up_args[-1] == "server"
    assert "--pull" in up_args


def test_restart_multiple_services_drains_once_and_recreates_all() -> None:
    with (
        patch.object(stack_module, "_drain_workers") as drain,
        patch.object(stack_module, "_compose") as compose,
        patch.object(stack_module, "_node_role", return_value=NodeRole.ROOT),
        patch.object(stack_module, "image_env_overrides", return_value={}),
    ):
        _restart(services=["server", "redis_control"])

    # A single drain (server manages workers) and a single recreate carrying
    # both services in the order given.
    drain.assert_called_once()
    compose.assert_called_once()
    up_args = compose.call_args.args[0]
    assert up_args[:5] == ["up", "-d", "--no-deps", "--force-recreate", "--wait"]
    assert up_args[-2:] == ["server", "redis_control"]


def test_restart_dedupes_repeated_services() -> None:
    with (
        patch.object(stack_module, "_drain_workers"),
        patch.object(stack_module, "_compose") as compose,
        patch.object(stack_module, "_node_role", return_value=NodeRole.ROOT),
        patch.object(stack_module, "image_env_overrides", return_value={}),
    ):
        _restart(services=["server", "server"])

    up_args = compose.call_args.args[0]
    assert up_args.count("server") == 1


def test_restart_no_pull_omits_pull_flag() -> None:
    with (
        patch.object(stack_module, "_drain_workers"),
        patch.object(stack_module, "_compose") as compose,
        patch.object(stack_module, "_node_role", return_value=NodeRole.ROOT),
        patch.object(stack_module, "image_env_overrides", return_value={}),
    ):
        _restart(services=["server"], pull=False)

    assert "--pull" not in compose.call_args.args[0]


def test_restart_redis_service_does_not_drain_workers() -> None:
    with (
        patch.object(stack_module, "_drain_workers") as drain,
        patch.object(stack_module, "_compose"),
        patch.object(stack_module, "_node_role", return_value=NodeRole.ROOT),
        patch.object(stack_module, "image_env_overrides", return_value={}),
    ):
        _restart(services=["redis_control"])

    drain.assert_not_called()


def test_restart_unknown_service_exits_without_acting() -> None:
    with (
        patch.object(stack_module, "_drain_workers") as drain,
        patch.object(stack_module, "_compose") as compose,
    ):
        with pytest.raises(typer.Exit):
            _restart(services=["bogus"])

    drain.assert_not_called()
    compose.assert_not_called()


def test_restart_unknown_in_a_set_exits_without_acting() -> None:
    with (
        patch.object(stack_module, "_drain_workers") as drain,
        patch.object(stack_module, "_compose") as compose,
    ):
        with pytest.raises(typer.Exit):
            _restart(services=["server", "bogus"])

    drain.assert_not_called()
    compose.assert_not_called()
