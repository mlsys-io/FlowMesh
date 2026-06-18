from enum import StrEnum
from typing import Literal

from ..placeholders import TemplateBool, TemplateInt
from ..task_type import TaskType
from .common import (
    ModelInferSpecStrict,
    ModelInferSpecTemplate,
    ParallelSpec,
    ParallelSpecTemplate,
)


class InferenceBackend(StrEnum):
    """The executor backend an inference task resolves to."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    AUTO = "auto"


class InferenceSpecStrict(ModelInferSpecStrict):
    taskType: Literal[TaskType.INFERENCE]

    sloSeconds: int | None = None
    parallel: ParallelSpec | None = None
    enforce_cpu: bool | None = None

    def backend(self) -> InferenceBackend:
        return _inference_backend(self)


class InferenceSpecTemplate(ModelInferSpecTemplate):
    taskType: Literal[TaskType.INFERENCE]

    sloSeconds: TemplateInt | None = None
    parallel: ParallelSpecTemplate | None = None
    enforce_cpu: TemplateBool | None = None

    def backend(self) -> InferenceBackend:
        return _inference_backend(self)


def _inference_backend(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
) -> InferenceBackend:
    """Classify the executor backend an inference task resolves to.

    The HF transformers executor runs on a GPU when one is available, so only ``VLLM``
    (explicit vLLM config or adapters) strictly requires a GPU. ``AUTO`` is the unhinted
    case: the runner prefers vLLM but falls back to the transformers executor when vLLM
    is absent.
    """
    if spec.enforce_cpu is True:
        return InferenceBackend.TRANSFORMERS
    model = spec.model
    if model is None:
        return InferenceBackend.AUTO
    if model.vllm or model.adapters:
        return InferenceBackend.VLLM
    if model.transformers:
        return InferenceBackend.TRANSFORMERS
    return InferenceBackend.AUTO
