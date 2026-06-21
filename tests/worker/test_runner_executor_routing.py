"""Guard tests: serve tasks must route to the vllm_serve executor, not default.

The serve→vllm_serve mapping is an inline elif in Runner.start(); coverage
is via the executor registry (that the right class is registered under the
right key and declares the right task type) and via
Runner._select_inference_executor_key for inference sub-routing.
"""

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors import EXECUTOR_REGISTRY
from worker.executors.vllm_serve_executor import VLLMServeExecutor
from worker.runner import Runner


class TestServeRoutingViaRegistry:
    def test_vllm_serve_key_is_registered(self) -> None:
        assert "vllm_serve" in EXECUTOR_REGISTRY

    def test_vllm_serve_executor_handles_serve_task_type(self) -> None:
        cls = EXECUTOR_REGISTRY.get("vllm_serve")
        assert cls is not None
        assert TaskType.SERVE in cls.supported_task_types

    def test_vllm_serve_executor_only_handles_serve(self) -> None:
        assert VLLMServeExecutor.supported_task_types == frozenset({TaskType.SERVE})


class TestInferenceExecutorKeySelection:
    """Tests for Runner._select_inference_executor_key."""

    def _key(self, spec: InferenceSpecStrict) -> str:
        # _select_inference_executor_key does not use self — pass None as placeholder
        return Runner._select_inference_executor_key(None, spec)  # type: ignore[arg-type]

    def test_vllm_backend_maps_to_vllm(self) -> None:
        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            model=ModelConfig(source=ModelSource(identifier="Qwen/Qwen3-1.7B")),
        )
        assert self._key(spec) == "vllm"
