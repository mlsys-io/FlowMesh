from typing import Any, get_args

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
from shared.tasks.envelope import TaskSpecStrict, TaskSpecTemplate
from shared.tasks.specs import (
    EchoSpecStrict,
    EmbeddingSpecStrict,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    SSHSpecStrict,
)
from shared.tasks.specs.common import TaskSpecStrictBase, TaskSpecTemplateBase
from shared.tasks.task_type import TaskType

_DATA = {"type": "list", "items": ["hi"]}

# Every spec that does not answer for itself and so inherits ``False``. Listed by
# name so that adding a task type forces a deliberate choice rather than silently
# being treated as CPU-only and placed on a worker whose card is held.
_INHERITS_DEFAULT = {
    "AgentSpec",
    "ApiSpec",
    "DataProfilingSpec",
    "DataRetrievalSpec",
    "EchoSpec",
    "RagSpec",
}


_SpecBase = type[TaskSpecStrictBase] | type[TaskSpecTemplateBase]


def _union_members(alias: Any) -> tuple[_SpecBase, ...]:
    return get_args(get_args(alias.__value__)[0])


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
    @pytest.mark.parametrize(
        ("alias", "base", "suffix"),
        [
            (TaskSpecStrict, TaskSpecStrictBase, "Strict"),
            (TaskSpecTemplate, TaskSpecTemplateBase, "Template"),
        ],
    )
    def test_only_expected_specs_inherit_the_default(
        self, alias: Any, base: _SpecBase, suffix: str
    ) -> None:
        inherited = {
            spec.__name__.removesuffix(suffix)
            for spec in _union_members(alias)
            if spec.uses_gpu is base.uses_gpu
        }
        assert inherited == _INHERITS_DEFAULT

    def test_strict_and_template_agree_on_which_specs_decide(self) -> None:
        def deciding(alias: Any, base: _SpecBase, suffix: str) -> set[str]:
            return {
                spec.__name__.removesuffix(suffix)
                for spec in _union_members(alias)
                if spec.uses_gpu is not base.uses_gpu
            }

        assert deciding(TaskSpecStrict, TaskSpecStrictBase, "Strict") == deciding(
            TaskSpecTemplate, TaskSpecTemplateBase, "Template"
        )

    def test_cpu_type_is_not_gpu_using(self) -> None:
        assert EchoSpecStrict(taskType=TaskType.ECHO).uses_gpu() is False


class TestInference:
    def test_vllm_backend_uses_gpu(self) -> None:
        assert _inference(model=_model(vllm={"dtype": "auto"})).uses_gpu() is True

    def test_auto_backend_uses_gpu(self) -> None:
        # No vllm/adapters/transformers hints: the runner prefers vLLM.
        assert _inference().uses_gpu() is True

    def test_transformers_defaults_to_gpu(self) -> None:
        spec = _inference(model=_model(transformers={"mode": "text-generation"}))
        assert spec.uses_gpu() is True

    @pytest.mark.parametrize(
        "device_map", ["auto", "balanced", "balanced_low_0", "cuda"]
    )
    def test_non_cpu_device_maps_use_gpu(self, device_map: str) -> None:
        spec = _inference(model=_model(transformers={"device_map": device_map}))
        assert spec.uses_gpu() is True

    def test_explicit_cpu_device_map_does_not(self) -> None:
        spec = _inference(model=_model(transformers={"device_map": "cpu"}))
        assert spec.uses_gpu() is False

    def test_enforce_cpu_wins(self) -> None:
        assert _inference(enforce_cpu=True).uses_gpu() is False

    def test_enforce_cpu_false_is_ignored(self) -> None:
        assert _inference(enforce_cpu=False).uses_gpu() is True

    def test_enforce_cpu_outranks_a_non_cpu_device_map(self) -> None:
        spec = _inference(
            model=_model(transformers={"device_map": "auto"}), enforce_cpu=True
        )
        assert spec.uses_gpu() is False

    def test_enforce_cpu_outranks_a_vllm_model(self) -> None:
        # validate_dispatchable rejects this pairing, but the answer must not
        # depend on validation having run first.
        spec = _inference(model=_model(vllm={"dtype": "auto"}), enforce_cpu=True)
        assert spec.uses_gpu() is False


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
        assert spec.uses_gpu() is True


class TestEmbedding:
    def test_vllm_embedding_uses_gpu(self) -> None:
        assert _embedding(_model(vllm={})).uses_gpu() is True

    def test_transformers_cpu_embedding_does_not(self) -> None:
        assert (
            _embedding(_model(transformers={"device_map": "cpu"})).uses_gpu() is False
        )


class TestSSH:
    def test_declared_count_is_gpu_using(self) -> None:
        assert _ssh(GPURequirements(count=1)).uses_gpu() is True

    def test_memory_without_count_is_still_gpu_using(self) -> None:
        # _resolve_gpu_devices resolves this to one device.
        assert _ssh(GPURequirements(memory="40Gi")).uses_gpu() is True

    def test_no_gpu_block_is_not(self) -> None:
        assert _ssh(None).uses_gpu() is False
