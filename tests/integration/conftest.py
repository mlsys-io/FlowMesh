"""
Pytest configuration for FlowMesh end-to-end integration tests.

Registers CLI options so the suite can be driven without pre-setting env vars:

    pytest tests/integration/ --host-url http://myserver:8000 --api-key flm-...

The options are synced into environment variables during pytest_configure so
that module-level constants and the pytestmark skip-condition in test_e2e.py
pick them up at collection time (before any fixtures run).
"""

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e", "FlowMesh end-to-end tests")
    group.addoption(
        "--host-url",
        default=None,
        metavar="URL",
        help="FlowMesh host base URL (overrides FLOWMESH_HOST_URL env var)",
    )
    group.addoption(
        "--api-key",
        default=None,
        metavar="KEY",
        help="FlowMesh API key (overrides FLOWMESH_API_KEY env var)",
    )
    group.addoption(
        "--task-yaml",
        default=None,
        metavar="PATH",
        help="Path to workflow YAML to submit (overrides TASK_YAML env var)",
    )
    group.addoption(
        "--e2e-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="Max seconds to wait for task completion (overrides E2E_TIMEOUT_SEC)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end integration tests that require a live FlowMesh host",
    )

    # Sync CLI options into env vars *before* test modules are collected so
    # that module-level constants and pytestmark conditions in test_e2e.py see
    # the right values.  os.environ.setdefault is used so an explicit env var
    # always takes precedence over a CLI flag.
    _sync_opt(config, "--host-url", "FLOWMESH_HOST_URL")
    _sync_opt(config, "--api-key", "FLOWMESH_API_KEY")
    _sync_opt(config, "--task-yaml", "TASK_YAML")
    if (timeout := config.getoption("--e2e-timeout")) is not None:
        os.environ.setdefault("E2E_TIMEOUT_SEC", str(timeout))


def _sync_opt(config: pytest.Config, opt: str, env_var: str) -> None:
    """If *opt* was passed on the CLI, set *env_var* unless already present."""
    value: str | None = config.getoption(opt)
    if value is not None:
        os.environ.setdefault(env_var, value)
