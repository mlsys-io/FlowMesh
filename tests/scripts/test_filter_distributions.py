"""Regression tests for ``scripts/ci/filter_distributions.py``.

Pin the policy that drives the release workflow's staggered PyPI
Trusted Publisher onboarding: ``*`` is a no-op, an empty match set
errors out, and only filenames matching one of the comma-separated
patterns survive.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "ci" / "filter_distributions.py"


@pytest.fixture(scope="module")
def filter_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("filter_distributions", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_dist(dist_dir: Path) -> list[str]:
    names = [
        "flowmesh-0.1.0-py3-none-any.whl",
        "flowmesh-0.1.0.tar.gz",
        "flowmesh_sdk-0.1.0-py3-none-any.whl",
        "flowmesh_sdk-0.1.0.tar.gz",
        "flowmesh_cli-0.1.0-py3-none-any.whl",
        "flowmesh_cli-0.1.0.tar.gz",
    ]
    for name in names:
        (dist_dir / name).write_bytes(b"")
    return sorted(names)


def test_star_pattern_is_no_op(filter_module, tmp_path, capsys):
    seeded = _seed_dist(tmp_path)
    rc = filter_module.filter_directory(tmp_path, "*")
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == seeded


def test_single_pattern_keeps_only_matches(filter_module, tmp_path):
    _seed_dist(tmp_path)
    rc = filter_module.filter_directory(tmp_path, "flowmesh-0.1.0*")
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "flowmesh-0.1.0-py3-none-any.whl",
        "flowmesh-0.1.0.tar.gz",
    ]


def test_multi_pattern_union(filter_module, tmp_path):
    _seed_dist(tmp_path)
    rc = filter_module.filter_directory(tmp_path, "flowmesh-0.1.0*,flowmesh_cli*")
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "flowmesh-0.1.0-py3-none-any.whl",
        "flowmesh-0.1.0.tar.gz",
        "flowmesh_cli-0.1.0-py3-none-any.whl",
        "flowmesh_cli-0.1.0.tar.gz",
    ]


def test_no_matches_errors(filter_module, tmp_path, capsys):
    _seed_dist(tmp_path)
    rc = filter_module.filter_directory(tmp_path, "nonexistent-*")
    assert rc == 1
    captured = capsys.readouterr()
    assert "matched no built distributions" in captured.err
    # All files should have been deleted before the error fired.
    assert list(tmp_path.iterdir()) == []


def test_whitespace_only_pattern_errors(filter_module, tmp_path, capsys):
    _seed_dist(tmp_path)
    rc = filter_module.filter_directory(tmp_path, " , , ")
    assert rc == 1
    assert "no usable patterns parsed" in capsys.readouterr().err


def test_patterns_stripped_of_whitespace(filter_module, tmp_path):
    _seed_dist(tmp_path)
    rc = filter_module.filter_directory(tmp_path, " flowmesh_sdk* , flowmesh_cli* ")
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "flowmesh_cli-0.1.0-py3-none-any.whl",
        "flowmesh_cli-0.1.0.tar.gz",
        "flowmesh_sdk-0.1.0-py3-none-any.whl",
        "flowmesh_sdk-0.1.0.tar.gz",
    ]
