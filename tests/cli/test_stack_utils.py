import os
from unittest.mock import patch

import pytest
from flowmesh_cli_stack.utils import (
    STACK_SLUG_ENV,
    STACK_SUFFIX_ENV,
    WORKER_RESULTS_DIR_ENV,
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
