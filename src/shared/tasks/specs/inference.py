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

    def validate_dispatchable(self) -> None:
        _validate_inference_dispatchable(self)


class InferenceSpecTemplate(ModelInferSpecTemplate):
    taskType: Literal[TaskType.INFERENCE]

    sloSeconds: TemplateInt | None = None
    parallel: ParallelSpecTemplate | None = None
    enforce_cpu: TemplateBool | None = None

    def backend(self) -> InferenceBackend:
        return _inference_backend(self)

    def validate_dispatchable(self) -> None:
        _validate_inference_dispatchable(self)


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


def _validate_inference_dispatchable(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
) -> None:
    model = spec.model
    if model and (adapters := model.adapters):
        for adapter in adapters:
            if not adapter.path and not adapter.url and not adapter.task_id:
                raise ValueError(
                    f"adapter {adapter.name or adapter.type!r} specifies no path, "
                    "url, or task_id and cannot be loaded."
                )

    if isinstance(spec.enforce_cpu, str):  # Unresolved template placeholder
        return
    if spec.enforce_cpu is True:
        if model and model.vllm:
            raise ValueError(
                "enforce_cpu is set but the model configures a vLLM backend."
            )
        return

    if spec.backend() is not InferenceBackend.VLLM:
        return
    hardware = spec.resources.hardware if spec.resources else None
    gpu = hardware.gpu if hardware else None
    if gpu and gpu.count is not None and gpu.count >= 1:
        return
    raise ValueError(
        "inference task resolves to the vLLM backend but requests no GPU. Set "
        "spec.resources.hardware.gpu.count >= 1, or use enforce_cpu / a "
        "transformers-only model for CPU inference."
    )
