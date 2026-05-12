"""Regression tests for ``scripts/ci/check_image_release.py``.

Pins the ``_is_release`` predicate that gates the ``:latest`` retag job:
plain releases and post-releases are eligible; pre-releases, dev
releases, local versions, and non-PEP-440 strings are not.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "ci" / "check_image_release.py"


@pytest.fixture(scope="module")
def check_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_image_release", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tag",
    ["v0.1.0", "0.1.0", "v1.2.3", "v0.1.0.post1", "v10.20.30"],
)
def test_release_tags_eligible(check_module, tag):
    assert check_module._is_release(tag) is True


@pytest.mark.parametrize(
    "tag",
    [
        "v0.1.0rc1",
        "v0.1.0-rc1",
        "v0.1.0a1",
        "v0.1.0b2",
        "v0.1.0.dev1",
        "v0.1.0+local",
        "not-a-version",
        "v",
        "",
    ],
)
def test_non_release_tags_rejected(check_module, tag):
    assert check_module._is_release(tag) is False
