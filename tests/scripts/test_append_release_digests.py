"""Regression tests for ``scripts/ci/append_release_digests.py``.

Exercises the body-normalization contract added to ``_read_current_body``
so a body-less GitHub Release does not leak the literal ``null`` token
into the published notes after the digest block is appended.
"""

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "ci" / "append_release_digests.py"


@pytest.fixture(scope="module")
def append_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "append_release_digests", _MODULE_PATH
    )
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


def test_null_body_normalizes_to_empty(append_module, monkeypatch):
    fake_run = MagicMock(return_value=_mock_completed_process("null\n"))
    monkeypatch.setattr(append_module.subprocess, "run", fake_run)
    assert append_module._read_current_body("/usr/bin/gh", "v0.1.0") == ""


def test_empty_body_normalizes_to_empty(append_module, monkeypatch):
    fake_run = MagicMock(return_value=_mock_completed_process(""))
    monkeypatch.setattr(append_module.subprocess, "run", fake_run)
    assert append_module._read_current_body("/usr/bin/gh", "v0.1.0") == ""


def test_whitespace_only_body_normalizes_to_empty(append_module, monkeypatch):
    fake_run = MagicMock(return_value=_mock_completed_process("\n\n  \n"))
    monkeypatch.setattr(append_module.subprocess, "run", fake_run)
    assert append_module._read_current_body("/usr/bin/gh", "v0.1.0") == ""


def test_real_body_preserved(append_module, monkeypatch):
    fake_run = MagicMock(return_value=_mock_completed_process("Existing notes here.\n"))
    monkeypatch.setattr(append_module.subprocess, "run", fake_run)
    assert (
        append_module._read_current_body("/usr/bin/gh", "v0.1.0")
        == "Existing notes here.\n"
    )


def test_strip_existing_block_empty_input(append_module):
    assert append_module._strip_existing_block("") == ""


def test_strip_existing_block_removes_prior_block(append_module):
    body = (
        "Existing prelude.\n"
        "\n"
        f"{append_module.START_MARKER}\n"
        "| old | digest | row |\n"
        f"{append_module.END_MARKER}\n"
        "Trailing line.\n"
    )
    assert append_module._strip_existing_block(body) == (
        "Existing prelude.\n\nTrailing line."
    )


def test_strip_existing_block_preserves_body_without_markers(append_module):
    body = "Standalone notes.\nNo markers here.\n"
    assert (
        append_module._strip_existing_block(body)
        == "Standalone notes.\nNo markers here."
    )
