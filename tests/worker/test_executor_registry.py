"""Tests for the executor registry and safe import mechanism."""

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

    def test_safe_import_does_not_crash(self) -> None:
        """The registry should load without raising, even when deps are missing."""
        # If we got here, the import at module level already succeeded
        assert isinstance(EXECUTOR_REGISTRY, dict)
        assert isinstance(IMPORT_ERRORS, dict)
