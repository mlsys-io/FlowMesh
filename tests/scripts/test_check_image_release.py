"""Regression tests for ``scripts/ci/check_image_release.py``.

Pins the ``_is_release`` predicate that gates the ``:latest`` retag job
and the per-target verification logic in ``_check_target``: mediatype
rejection short-circuits, attestation manifests are ignored when
building the platform set, and per-platform ``image.version`` /
``image.revision`` labels are required to match the release tag and
commit.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

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


_OCI_INDEX = "application/vnd.oci.image.index.v1+json"
_DOCKER_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
_TARGET = "flowmesh_server"
_REGISTRY = "ghcr.io/mlsys-io"
_TAG = "v0.1.0"
_COMMIT = "abc123def456"
_REF = "ghcr.io/mlsys-io/flowmesh_server:v0.1.0"


def _valid_manifest(media: str = _OCI_INDEX) -> dict[str, Any]:
    return {
        "mediaType": media,
        "digest": "sha256:deadbeef",
        "manifests": [
            {"platform": {"os": "linux", "architecture": "amd64"}},
            {"platform": {"os": "linux", "architecture": "arm64"}},
            {"platform": {"os": "unknown", "architecture": "unknown"}},
        ],
    }


def _valid_images(version: str = _TAG, revision: str = _COMMIT) -> dict[str, Any]:
    labels = {
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.revision": revision,
    }
    return {
        "linux/amd64": {"config": {"Labels": dict(labels)}},
        "linux/arm64": {"config": {"Labels": dict(labels)}},
    }


def _wire_inspect(
    monkeypatch, check_module, manifest: dict[str, Any], images: dict[str, Any]
) -> None:
    def fake(docker: str, ref: str, output_format: str) -> dict[str, Any]:
        if "Manifest" in output_format:
            return manifest
        return images

    monkeypatch.setattr(check_module, "_imagetools_inspect", fake)


def test_check_target_happy_path(check_module, monkeypatch):
    _wire_inspect(monkeypatch, check_module, _valid_manifest(), _valid_images())
    ref, digest, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert ref == _REF
    assert digest == "sha256:deadbeef"
    assert errors == []


def test_check_target_accepts_docker_manifest_list(check_module, monkeypatch):
    _wire_inspect(
        monkeypatch, check_module, _valid_manifest(_DOCKER_LIST), _valid_images()
    )
    _, _, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert errors == []


def test_check_target_rejects_non_index_mediatype(check_module, monkeypatch):
    manifest = _valid_manifest("application/vnd.oci.image.manifest.v1+json")
    _wire_inspect(monkeypatch, check_module, manifest, _valid_images())
    _, digest, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert digest == "sha256:deadbeef"
    assert len(errors) == 1
    assert "is not a multi-arch index" in errors[0]


def test_check_target_platform_mismatch(check_module, monkeypatch):
    manifest = _valid_manifest()
    manifest["manifests"] = [
        {"platform": {"os": "linux", "architecture": "amd64"}},
    ]
    _wire_inspect(monkeypatch, check_module, manifest, _valid_images())
    _, _, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert any("platforms" in err for err in errors)


def test_check_target_attestation_manifests_ignored(check_module, monkeypatch):
    manifest = _valid_manifest()
    # Extra attestation manifest should not break the platform-set check.
    manifest["manifests"].append(
        {"platform": {"os": "unknown", "architecture": "unknown"}}
    )
    _wire_inspect(monkeypatch, check_module, manifest, _valid_images())
    _, _, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert errors == []


def test_check_target_missing_labels(check_module, monkeypatch):
    images = {
        "linux/amd64": {"config": {"Labels": {}}},
        "linux/arm64": {"config": {"Labels": {}}},
    }
    _wire_inspect(monkeypatch, check_module, _valid_manifest(), images)
    _, _, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert len(errors) == 4
    assert any("image.version=None" in err for err in errors)
    assert any("image.revision=None" in err for err in errors)


def test_check_target_version_label_drift(check_module, monkeypatch):
    images = _valid_images(version="v0.0.9")
    _wire_inspect(monkeypatch, check_module, _valid_manifest(), images)
    _, _, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert all("image.version='v0.0.9'" in err for err in errors)
    assert len(errors) == 2  # one per platform


def test_check_target_revision_label_drift(check_module, monkeypatch):
    images = _valid_images(revision="0000000")
    _wire_inspect(monkeypatch, check_module, _valid_manifest(), images)
    _, _, errors = check_module._check_target(
        "/usr/bin/docker", _TARGET, _REGISTRY, _TAG, _COMMIT
    )
    assert all("image.revision='0000000'" in err for err in errors)
    assert len(errors) == 2
