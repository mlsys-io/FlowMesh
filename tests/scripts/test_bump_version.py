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


def test_render_sdk_init_updates_version_constant(bump_module):
    source = '__version__ = "0.1.0"\n'
    assert bump_module._render_sdk_init(source, "0.1.1") == '__version__ = "0.1.1"\n'


def test_render_sdk_init_requires_single_version_line(bump_module):
    with pytest.raises(SystemExit, match="Expected one __version__ line"):
        bump_module._render_sdk_init("VERSION = '0.1.0'\n", "0.1.1")
