"""Regression tests for ``scripts/ci/check_release_version.py``."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "ci" / "check_release_version.py"


@pytest.fixture(scope="module")
def release_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_release_version", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_runtime_versions_accepts_matching_literals(
    release_module, monkeypatch, tmp_path
):
    sdk_version = tmp_path / "sdk_version.py"
    sdk_version.write_text('_STATIC_VERSION = "0.1.1"\n')
    shared_version = tmp_path / "shared_version.py"
    shared_version.write_text('FLOWMESH_VERSION = "0.1.1"\n')

    monkeypatch.setattr(release_module, "SDK_VERSION_MODULE", sdk_version)
    monkeypatch.setattr(release_module, "SHARED_VERSION_MODULE", shared_version)

    release_module._check_runtime_versions(Version("0.1.1"))


def test_check_runtime_versions_rejects_mismatched_literal(
    release_module, monkeypatch, tmp_path
):
    sdk_version = tmp_path / "sdk_version.py"
    sdk_version.write_text('_STATIC_VERSION = "0.1.0"\n')
    shared_version = tmp_path / "shared_version.py"
    shared_version.write_text('FLOWMESH_VERSION = "0.1.1"\n')

    monkeypatch.setattr(release_module, "SDK_VERSION_MODULE", sdk_version)
    monkeypatch.setattr(release_module, "SHARED_VERSION_MODULE", shared_version)

    with pytest.raises(SystemExit, match="expected release 0.1.1"):
        release_module._check_runtime_versions(Version("0.1.1"))
