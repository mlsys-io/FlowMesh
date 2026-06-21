"""Tests for the executor registry and safe import mechanism."""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

import worker.executors as executors_pkg
from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_config
from worker.executors import EXECUTOR_MODULES, EXECUTOR_REGISTRY, IMPORT_ERRORS
from worker.executors.base_executor import Executor, ExecutorTask
from worker.main import build_capabilities


class TestExecutorRegistry:
    def test_registry_has_expected_keys(self) -> None:
        expected = {
            "vllm",
            "vllm_lora",
            "ppo",
            "dpo",
            "sft",
            "lora_sft",
            "image_classification_training",
            "default",
            "rag",
            "agent",
            "echo",
            "data_profiling",
            "data_retrieval",
            "diffusers",
            "api",
            "ssh",
            "omni_text2image",
            "omni_text2speech",
            "omni_text2audio",
            "omni_text2general",
        }
        assert set(EXECUTOR_REGISTRY.keys()) == expected

    def test_unavailable_executors_tracked(self) -> None:
        """Executors that failed to import have their errors recorded."""
        # GPU executors (vllm, etc.) are expected to fail without CUDA
        for key, cls in EXECUTOR_REGISTRY.items():
            if cls is None:
                assert key in IMPORT_ERRORS or EXECUTOR_MODULES[key][0] in IMPORT_ERRORS

    def test_training_executors_are_wrapped_for_isolation(self) -> None:
        """Training executors run in a subprocess for GPU cleanup; ensure the
        image classification executor is in the wrap set (and thus instantiated
        by the worker), guarding against registry/worker drift."""
        from worker.main import _EXECUTORS_TO_WRAP

        assert "image_classification_training" in _EXECUTORS_TO_WRAP
        assert "image_classification_training" in EXECUTOR_REGISTRY

    def test_import_executor_does_not_crash(self) -> None:
        """The registry should load without raising, even when deps are missing."""
        # If we got here, the import at module level already succeeded
        assert isinstance(EXECUTOR_REGISTRY, Mapping)
        assert isinstance(IMPORT_ERRORS, dict)

    def test_import_executor_rejects_non_executor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered name that resolves to a non-Executor is dropped and recorded."""
        fake_module = SimpleNamespace(NotAnExecutor=object)
        monkeypatch.setattr(
            executors_pkg.importlib, "import_module", lambda *a, **k: fake_module
        )
        executors_pkg._IMPORT_ERRORS.pop("NotAnExecutor", None)

        result = executors_pkg._import_executor("NotAnExecutor", ".does_not_matter")

        assert result is None
        assert (
            "not a subclass of Executor"
            in executors_pkg._IMPORT_ERRORS["NotAnExecutor"]
        )


class _StubExecutor(Executor):
    def run(self, task: ExecutorTask, out_dir: Path) -> BaseExecutorResult:
        raise NotImplementedError


class _StubEcho(_StubExecutor):
    supported_task_types = frozenset({TaskType.ECHO})


class _StubInference(_StubExecutor):
    supported_task_types = frozenset({TaskType.INFERENCE, TaskType.EMBEDDING})


class TestSupportedTaskTypes:
    def test_every_available_executor_declares_task_types(self) -> None:
        for key, cls in EXECUTOR_REGISTRY.items():
            if cls is not None:
                assert cls.supported_task_types, f"{key} declares no task types"

    def test_default_executor_serves_inference_and_embedding(self) -> None:
        cls = EXECUTOR_REGISTRY["default"]
        assert cls is not None
        assert {TaskType.INFERENCE, TaskType.EMBEDDING} <= cls.supported_task_types


class _EmptyCaps(Executor):
    """Stand-in for a live MPExecutor wrapper: reports no task types of its own."""

    def run(self, task: ExecutorTask, out_dir: Path) -> BaseExecutorResult:
        raise NotImplementedError


class TestBuildCapabilities:
    @staticmethod
    def _executors(*keys: str) -> dict[str, Executor]:
        # Values report empty supported_task_types, mirroring MPExecutor wrappers
        # which do not forward it; build_capabilities must read the registry class
        # by key instead, or wrapped executors would advertise nothing (the
        # inference/training regression).
        inst = _EmptyCaps(make_worker_config())
        return {key: inst for key in keys}

    def test_unions_from_registry_class_not_instance(self) -> None:
        registry: dict[str, type[Executor] | None] = {
            "echo": _StubEcho,
            "inference": _StubInference,
        }
        caps = build_capabilities(self._executors("echo", "inference"), registry)
        assert caps.supported_task_types == {
            TaskType.ECHO,
            TaskType.INFERENCE,
            TaskType.EMBEDDING,
        }

    def test_skips_unknown_and_unavailable_keys(self) -> None:
        registry: dict[str, type[Executor] | None] = {
            "echo": _StubEcho,
            "missing": None,
        }
        caps = build_capabilities(self._executors("echo", "missing", "ghost"), registry)
        assert caps.supported_task_types == frozenset({TaskType.ECHO})

    def test_empty_is_safe(self) -> None:
        assert build_capabilities({}, {}).supported_task_types == frozenset()
