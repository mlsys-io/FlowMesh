import pytest

from shared.tasks.components.model import (
    ModelConfig,
    ModelConfigTemplate,
    ModelSource,
    ModelSourceTemplate,
)
from shared.tasks.components.resources import (
    GPURequirements,
    HardwareRequirements,
    ResourcesSpec,
)
from shared.tasks.gpu_usage import _ALWAYS_GPU, _NEVER_GPU, task_uses_gpu
from shared.tasks.specs import (
    EchoSpecStrict,
    EmbeddingSpecStrict,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    SSHSpecStrict,
)
from shared.tasks.task_type import TaskType

_DATA = {"type": "list", "items": ["hi"]}


def _model(**kwargs) -> ModelConfig:
    return ModelConfig(source=ModelSource(identifier="org/m"), **kwargs)


def _inference(**kwargs) -> InferenceSpecStrict:
    kwargs.setdefault("model", _model())
    return InferenceSpecStrict(taskType=TaskType.INFERENCE, data=_DATA, **kwargs)


def _embedding(model: ModelConfig) -> EmbeddingSpecStrict:
    return EmbeddingSpecStrict(taskType=TaskType.EMBEDDING, data=_DATA, model=model)


def _ssh(gpu: GPURequirements | None) -> SSHSpecStrict:
    resources = (
        ResourcesSpec(hardware=HardwareRequirements(gpu=gpu))
        if gpu is not None
        else None
    )
    return SSHSpecStrict(taskType=TaskType.SSH, resources=resources)


class TestEveryTaskTypeIsClassified:
    def test_no_task_type_falls_through(self) -> None:
        # A new TaskType must be classified deliberately rather than inheriting
        # the conservative default by accident.
        decided_by_spec = {TaskType.INFERENCE, TaskType.EMBEDDING, TaskType.SSH}
        assert _ALWAYS_GPU | _NEVER_GPU | decided_by_spec == set(TaskType)

    def test_always_and_never_are_disjoint(self) -> None:
        assert not (_ALWAYS_GPU & _NEVER_GPU)

    def test_cpu_type_is_not_gpu_using(self) -> None:
        assert task_uses_gpu(EchoSpecStrict(taskType=TaskType.ECHO)) is False


class TestInference:
    def test_vllm_backend_uses_gpu(self) -> None:
        assert task_uses_gpu(_inference(model=_model(vllm={"dtype": "auto"}))) is True

    def test_auto_backend_uses_gpu(self) -> None:
        # No vllm/adapters/transformers hints: the runner prefers vLLM.
        assert task_uses_gpu(_inference()) is True

    def test_transformers_defaults_to_gpu(self) -> None:
        spec = _inference(model=_model(transformers={"mode": "text-generation"}))
        assert task_uses_gpu(spec) is True

    @pytest.mark.parametrize(
        "device_map", ["auto", "balanced", "balanced_low_0", "cuda"]
    )
    def test_non_cpu_device_maps_use_gpu(self, device_map: str) -> None:
        spec = _inference(model=_model(transformers={"device_map": device_map}))
        assert task_uses_gpu(spec) is True

    def test_explicit_cpu_device_map_does_not(self) -> None:
        spec = _inference(model=_model(transformers={"device_map": "cpu"}))
        assert task_uses_gpu(spec) is False

    def test_enforce_cpu_wins(self) -> None:
        assert task_uses_gpu(_inference(enforce_cpu=True)) is False

    def test_enforce_cpu_false_is_ignored(self) -> None:
        assert task_uses_gpu(_inference(enforce_cpu=False)) is True


class TestUnresolvedTemplates:
    def test_placeholder_enforce_cpu_is_not_read_as_cpu(self) -> None:
        # A truthiness test would answer False here and wave the task onto a
        # held card. Only a literal True pins to CPU.
        spec = InferenceSpecTemplate(
            taskType=TaskType.INFERENCE,
            data=_DATA,
            model=ModelConfigTemplate(source=ModelSourceTemplate(identifier="org/m")),
            enforce_cpu="${params.cpu}",
        )
        assert task_uses_gpu(spec) is True


class TestEmbedding:
    def test_vllm_embedding_uses_gpu(self) -> None:
        assert task_uses_gpu(_embedding(_model(vllm={}))) is True

    def test_transformers_cpu_embedding_does_not(self) -> None:
        spec = _embedding(_model(transformers={"device_map": "cpu"}))
        assert task_uses_gpu(spec) is False


class TestSSH:
    def test_declared_count_is_gpu_using(self) -> None:
        assert task_uses_gpu(_ssh(GPURequirements(count=1))) is True

    def test_memory_without_count_is_still_gpu_using(self) -> None:
        # _resolve_gpu_devices resolves this to one device.
        assert task_uses_gpu(_ssh(GPURequirements(memory="40Gi"))) is True

    def test_no_gpu_block_is_not(self) -> None:
        assert task_uses_gpu(_ssh(None)) is False
