"""Tests for inference backend classification shared by the parser and runner."""

from shared.tasks.specs import InferenceBackend, InferenceSpecStrict


def _spec(**fields: object) -> InferenceSpecStrict:
    return InferenceSpecStrict.model_validate({"taskType": "inference", **fields})


class TestInferenceBackend:
    def test_enforce_cpu_is_transformers(self) -> None:
        assert _spec(enforce_cpu=True).backend() is InferenceBackend.TRANSFORMERS

    def test_transformers_only_is_transformers(self) -> None:
        spec = _spec(model={"transformers": {"torch_dtype": "float32"}})
        assert spec.backend() is InferenceBackend.TRANSFORMERS

    def test_explicit_vllm_is_vllm(self) -> None:
        spec = _spec(model={"vllm": {"gpu_memory_utilization": 0.9}})
        assert spec.backend() is InferenceBackend.VLLM

    def test_transformers_and_vllm_is_vllm(self) -> None:
        spec = _spec(
            model={
                "transformers": {"torch_dtype": "float32"},
                "vllm": {"gpu_memory_utilization": 0.9},
            }
        )
        assert spec.backend() is InferenceBackend.VLLM

    def test_empty_vllm_config_is_not_vllm(self) -> None:
        # An empty vLLM dict is treated as unconfigured (matching the runner): with
        # a transformers model it stays on transformers, otherwise it is AUTO.
        with_transformers = _spec(model={"transformers": {"x": 1}, "vllm": {}})
        assert with_transformers.backend() is InferenceBackend.TRANSFORMERS
        assert _spec(model={"vllm": {}}).backend() is InferenceBackend.AUTO

    def test_lora_adapter_is_vllm(self) -> None:
        spec = _spec(model={"adapters": [{"type": "lora"}]})
        assert spec.backend() is InferenceBackend.VLLM

    def test_non_lora_adapter_is_vllm(self) -> None:
        spec = _spec(model={"adapters": [{"type": "ia3"}]})
        assert spec.backend() is InferenceBackend.VLLM

    def test_no_model_is_auto(self) -> None:
        assert _spec().backend() is InferenceBackend.AUTO

    def test_unhinted_model_is_auto(self) -> None:
        spec = _spec(model={"source": {"identifier": "gpt2"}})
        assert spec.backend() is InferenceBackend.AUTO
