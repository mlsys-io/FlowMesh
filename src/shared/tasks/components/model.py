from typing import Any, Literal

from pydantic import field_validator

from .._base import StrictBaseModel, TemplateBaseModel
from ..placeholders import TemplateBool, TemplateInt

type AdapterApplyMode = Literal["runtime", "merge"]


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
    apply: AdapterApplyMode = "runtime"
    id: int | None = None
    archive_format: str = "auto"
    headers: dict[str, str] | None = None
    task_id: str | None = None

    @field_validator("apply", mode="before")
    @classmethod
    def _normalize_apply(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("archive_format", mode="before")
    @classmethod
    def _normalize_archive_format(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class AdapterConfigTemplate(TemplateBaseModel):
    type: str
    path: str | None = None
    url: str | None = None
    name: str | None = None
    apply: str = "runtime"
    id: TemplateInt | None = None
    archive_format: str = "auto"
    headers: dict[str, str] | None = None
    task_id: str | None = None


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


class ModelConfigTemplate(TemplateBaseModel):
    source: ModelSourceTemplate | None = None
    config: dict[str, Any] | None = None
    vllm: dict[str, Any] | None = None
    transformers: dict[str, Any] | None = None
    diffusers: dict[str, Any] | None = None
    adapters: list[AdapterConfigTemplate] | None = None
