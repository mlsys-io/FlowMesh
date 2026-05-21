import os
from pathlib import Path
from unittest.mock import patch

import pytest
from flowmesh_cli_stack.utils import (
    _PLUGIN_DATA_ALIAS,
    STACK_SLUG_ENV,
    STACK_SUFFIX_ENV,
    WORKER_RESULTS_DIR_ENV,
    apply_plugin_data_env,
    apply_stack_resource_env,
    stack_resource_env_overrides,
)


def test_stack_resource_env_overrides_use_defaults_without_suffix() -> None:
    overrides = stack_resource_env_overrides({})
    assert overrides[STACK_SLUG_ENV] == "flowmesh_node"


def test_stack_resource_env_overrides_append_sanitized_suffix() -> None:
    overrides = stack_resource_env_overrides({STACK_SUFFIX_ENV: "alice.dev"})
    assert overrides[STACK_SLUG_ENV] == "flowmesh_node_alice.dev"


def test_stack_resource_env_overrides_reject_invalid_suffix() -> None:
    with pytest.raises(ValueError, match=STACK_SUFFIX_ENV):
        stack_resource_env_overrides({STACK_SUFFIX_ENV: "!!!"})


def test_apply_stack_resource_env_defaults_results_dirs_from_suffix() -> None:
    with patch.dict(
        os.environ,
        {
            STACK_SUFFIX_ENV: "alice.dev",
            WORKER_RESULTS_DIR_ENV: "",
        },
        clear=True,
    ):
        apply_stack_resource_env()
        assert os.environ[WORKER_RESULTS_DIR_ENV] == "flowmesh_node_alice.dev_results"


def test_apply_plugin_data_env_empty_resolves_default(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"FLOWMESH_PLUGIN_DATA_DIR": ""}, clear=True):
        apply_plugin_data_env(tmp_path)
        expected = (tmp_path / "plugin-data").as_posix()
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == expected
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ
        assert not (tmp_path / "plugin-data").exists()  # routing only; no mkdir


def test_apply_plugin_data_env_relative_path_is_cwd_resolved(tmp_path: Path) -> None:
    env = {"FLOWMESH_PLUGIN_DATA_DIR": "./custom-data"}
    with patch.dict(os.environ, env, clear=True):
        apply_plugin_data_env(tmp_path)
        expected = (tmp_path / "custom-data").as_posix()
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == expected
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ


def test_apply_plugin_data_env_absolute_path_passthrough(tmp_path: Path) -> None:
    abs_path = "/var/lib/flowmesh-plugin"
    with patch.dict(os.environ, {"FLOWMESH_PLUGIN_DATA_DIR": abs_path}, clear=True):
        apply_plugin_data_env(tmp_path)
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == abs_path
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ


def test_apply_plugin_data_env_tilde_is_path(tmp_path: Path) -> None:
    env = {"FLOWMESH_PLUGIN_DATA_DIR": "~/flowmesh-data"}
    with patch.dict(os.environ, env, clear=True):
        apply_plugin_data_env(tmp_path)
        # Resolved against base_dir; the leading ~ keeps it in path mode.
        assert "FLOWMESH_PLUGIN_DATA_VOLUME" not in os.environ
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"]  # set to something resolved


def test_apply_plugin_data_env_bare_name_routes_to_volume(tmp_path: Path) -> None:
    env = {"FLOWMESH_PLUGIN_DATA_DIR": "my_external_vol"}
    with patch.dict(os.environ, env, clear=True):
        apply_plugin_data_env(tmp_path)
        assert os.environ["FLOWMESH_PLUGIN_DATA_VOLUME"] == "my_external_vol"
        assert os.environ["FLOWMESH_PLUGIN_DATA_DIR"] == _PLUGIN_DATA_ALIAS
