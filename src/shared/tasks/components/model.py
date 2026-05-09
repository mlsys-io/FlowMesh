from typing import Any

from .._base import StrictBaseModel, TemplateBaseModel
from ..placeholders import TemplateBool


class ModelSource(StrictBaseModel):
    type: str | None = None
    identifier: str | None = None
    revision: str | None = None
    trust_remote_code: bool | None = None


class ModelSourceTemplate(TemplateBaseModel):
    type: str | None = None
    identifier: str | None = None
    revision: str | None = None
    trust_remote_code: TemplateBool | None = None


class AdapterConfig(StrictBaseModel):
    type: str
    path: str | None = None
    url: str | None = None
    name: str | None = None
    kwargs: dict[str, Any] | None = None


class AdapterConfigTemplate(TemplateBaseModel):
    type: str
    path: str | None = None
    url: str | None = None
    name: str | None = None
    kwargs: dict[str, Any] | None = None


class ModelConfig(StrictBaseModel):
    """
    Common model configuration used across multiple executors.

    This is intentionally shallow/structured:
    - top-level keys are strict (no unknown keys)
    - nested provider kwargs are free-form dict leaves (e.g. `vllm`, `transformers`)
    """

    source: ModelSource | None = None
    config: dict[str, Any] | None = None
    vllm: dict[str, Any] | None = None
    transformers: dict[str, Any] | None = None
    diffusers: dict[str, Any] | None = None
    adapters: list[AdapterConfig] | None = None

    def has_lora_adapter(self) -> bool:
        adapters = self.adapters or []
        for adapter in adapters:
            if (adapter.type or "").strip().lower() == "lora":
                return True
        return False


class ModelConfigTemplate(TemplateBaseModel):
    source: ModelSourceTemplate | None = None
    config: dict[str, Any] | None = None
    vllm: dict[str, Any] | None = None
    transformers: dict[str, Any] | None = None
    diffusers: dict[str, Any] | None = None
    adapters: list[AdapterConfigTemplate] | None = None

    def has_lora_adapter(self) -> bool:
        adapters = self.adapters or []
        for adapter in adapters:
            if (adapter.type or "").strip().lower() == "lora":
                return True
        return False
