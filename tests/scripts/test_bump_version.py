"""Regression tests for ``scripts/dev/bump_version.py``."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "dev" / "bump_version.py"


@pytest.fixture(scope="module")
def bump_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bump_version", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_static_version_module_updates_static_version(bump_module):
    source = '_STATIC_VERSION = "0.1.0"\n'
    assert (
        bump_module._render_static_version_module(
            source, "0.1.1", bump_module.SDK_VERSION_MODULE
        )
        == '_STATIC_VERSION = "0.1.1"\n'
    )


def test_render_shared_version_module_updates_runtime_version(bump_module):
    source = 'FLOWMESH_RELEASE_VERSION = "0.1.0"\n'
    assert (
        bump_module._render_shared_version_module(source, "0.1.1")
        == 'FLOWMESH_RELEASE_VERSION = "0.1.1"\n'
    )


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("VERSION = '0.1.0'\n", "Expected one _STATIC_VERSION line"),
        (
            '_STATIC_VERSION = "0.1.0"\n_STATIC_VERSION = "0.1.0"\n',
            "Expected one _STATIC_VERSION line",
        ),
    ],
)
def test_render_static_version_module_requires_single_version_line(
    bump_module, source, match
):
    with pytest.raises(SystemExit, match=match):
        bump_module._render_static_version_module(
            source, "0.1.1", bump_module.SDK_VERSION_MODULE
        )


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("VERSION = '0.1.0'\n", "Expected one FLOWMESH_RELEASE_VERSION line"),
        (
            'FLOWMESH_RELEASE_VERSION = "0.1.0"\nFLOWMESH_RELEASE_VERSION = "0.1.0"\n',
            "Expected one FLOWMESH_RELEASE_VERSION line",
        ),
    ],
)
def test_render_shared_version_module_requires_single_version_line(
    bump_module, source, match
):
    with pytest.raises(SystemExit, match=match):
        bump_module._render_shared_version_module(source, "0.1.1")
