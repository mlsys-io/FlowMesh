"""KubernetesStack: what a teardown deletes, and what it must leave behind."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from flowmesh_stack.kubernetes import (
    DATA_KINDS,
    KubectlError,
    KubernetesStack,
    ensure_kubectl_available,
)

MANIFEST = """
apiVersion: v1
kind: Namespace
metadata:
  name: flowmesh
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flowmesh-server
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flowmesh-results
"""


def _stack(tmp_path: Path) -> KubernetesStack:
    manifest = tmp_path / "stack.yaml"
    manifest.write_text(MANIFEST)
    return KubernetesStack(
        manifests=[manifest],
        namespace="flowmesh",
        load_env=lambda _: None,
    )


def _stdin_kinds(run: MagicMock) -> list[str]:
    stream = run.call_args.kwargs["stdin"]
    return [doc["kind"] for doc in yaml.safe_load_all(stream) if doc]


class TestTeardown:
    def test_delete_keeps_the_namespace_and_claims(self, tmp_path: Path) -> None:
        """Deleting the namespace would cascade to every claim inside it."""
        stack = _stack(tmp_path)
        with patch.object(stack, "_run") as run:
            stack.delete(tmp_path / ".env")

        kinds = _stdin_kinds(run)
        assert kinds == ["Deployment"]
        assert not DATA_KINDS & set(kinds)

    def test_delete_can_be_asked_for_everything(self, tmp_path: Path) -> None:
        stack = _stack(tmp_path)
        with patch.object(stack, "_run") as run:
            stack.delete(tmp_path / ".env", keep_data=False)

        assert _stdin_kinds(run) == ["Namespace", "Deployment", "PersistentVolumeClaim"]

    def test_delete_tolerates_missing_resources(self, tmp_path: Path) -> None:
        stack = _stack(tmp_path)
        with patch.object(stack, "_run") as run:
            stack.delete(tmp_path / ".env")

        assert "--ignore-not-found" in run.call_args.args[0]

    def test_apply_sends_every_document(self, tmp_path: Path) -> None:
        stack = _stack(tmp_path)
        with patch.object(stack, "_run") as run:
            stack.apply(tmp_path / ".env")

        assert _stdin_kinds(run) == ["Namespace", "Deployment", "PersistentVolumeClaim"]


class TestKubectl:
    def test_missing_kubectl_is_reported(self) -> None:
        with patch("flowmesh_stack.kubernetes.shutil.which", return_value=None):
            with pytest.raises(KubectlError, match="kubectl is required"):
                ensure_kubectl_available()
