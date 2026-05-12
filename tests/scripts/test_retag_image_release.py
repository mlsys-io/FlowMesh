"""Regression tests for ``scripts/ci/retag_image_release.py``.

Covers the three return states of ``_existing_latest_version`` — missing
reference, transient inspect failure, and successful version parsing —
plus the label-parsing branches that distinguish "no version label" from
"version label is not PEP 440".
"""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "ci" / "retag_image_release.py"


@pytest.fixture(scope="module")
def retag_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("retag_image_release", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_completed_process(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _image_blob_with_version(version: str) -> dict[str, Any]:
    return {
        "linux/amd64": {
            "config": {"Labels": {"org.opencontainers.image.version": version}}
        }
    }


def test_missing_reference_returns_none(retag_module, monkeypatch):
    fake = MagicMock(
        return_value=_mock_completed_process(
            "", returncode=1, stderr="ghcr.io/foo: manifest unknown"
        )
    )
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    assert (
        retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo") is None
    )


def test_missing_reference_via_not_found_stderr(retag_module, monkeypatch):
    fake = MagicMock(
        return_value=_mock_completed_process(
            "", returncode=1, stderr="image was not found in registry"
        )
    )
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    assert (
        retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo") is None
    )


def test_transient_inspect_failure_raises(retag_module, monkeypatch):
    fake = MagicMock(
        return_value=_mock_completed_process(
            "", returncode=1, stderr="503 Service Unavailable"
        )
    )
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    with pytest.raises(retag_module.TransientInspectError):
        retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo")


def test_version_label_parsed(retag_module, monkeypatch):
    payload = json.dumps(_image_blob_with_version("v0.1.0"))
    fake = MagicMock(return_value=_mock_completed_process(payload))
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    result = retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo")
    assert result == Version("0.1.0")


def test_version_label_without_v_prefix_parsed(retag_module, monkeypatch):
    payload = json.dumps(_image_blob_with_version("0.2.0"))
    fake = MagicMock(return_value=_mock_completed_process(payload))
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    result = retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo")
    assert result == Version("0.2.0")


def test_missing_version_label_raises(retag_module, monkeypatch):
    payload = json.dumps({"linux/amd64": {"config": {"Labels": {}}}})
    fake = MagicMock(return_value=_mock_completed_process(payload))
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    with pytest.raises(retag_module.MissingVersionLabel):
        retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo")


def test_invalid_version_label_raises(retag_module, monkeypatch):
    payload = json.dumps(_image_blob_with_version("not-a-version"))
    fake = MagicMock(return_value=_mock_completed_process(payload))
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    with pytest.raises(retag_module.MissingVersionLabel):
        retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo")


def test_malformed_inspect_json_raises(retag_module, monkeypatch):
    fake = MagicMock(return_value=_mock_completed_process("{not json"))
    monkeypatch.setattr(retag_module, "_imagetools_inspect", fake)
    with pytest.raises(retag_module.MissingVersionLabel):
        retag_module._existing_latest_version("/usr/bin/docker", "ghcr.io/foo")
