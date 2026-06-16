"""Tests for the executor registry and safe import mechanism."""

from types import SimpleNamespace

import pytest

import worker.executors as executors_pkg
from worker.executors import EXECUTOR_CLASS_NAMES, EXECUTOR_REGISTRY, IMPORT_ERRORS


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

    def test_class_names_match_registry(self) -> None:
        """Every key in EXECUTOR_REGISTRY has a corresponding class name."""
        assert set(EXECUTOR_CLASS_NAMES.keys()) == set(EXECUTOR_REGISTRY.keys())

    def test_unavailable_executors_tracked(self) -> None:
        """Executors that failed to import have their errors recorded."""
        # GPU executors (vllm, etc.) are expected to fail without CUDA
        for key, cls in EXECUTOR_REGISTRY.items():
            if cls is None:
                assert (
                    key in IMPORT_ERRORS or EXECUTOR_CLASS_NAMES[key] in IMPORT_ERRORS
                )

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
        assert isinstance(EXECUTOR_REGISTRY, dict)
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
