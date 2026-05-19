from typing import Any

from pydantic import Field, model_validator

from ...schemas.result import BaseExecutorResult
from .._base import StrictBaseModel, TemplateBaseModel
from ..components import (
    AdapterConfig,
    AdapterConfigTemplate,
    ModelConfig,
    ModelConfigTemplate,
    OutputSpec,
    OutputSpecTemplate,
    PostprocessSpec,
    PostprocessSpecTemplate,
    ResourcesSpec,
    ShardSpec,
    ShardSpecTemplate,
)
from ..placeholders import TemplateBool, TemplateInt


class ParallelSpec(StrictBaseModel):
    enabled: bool | None = None
    max_shards: int | None = None


class ParallelSpecTemplate(TemplateBaseModel):
    enabled: TemplateBool | None = None
    max_shards: TemplateInt | None = None


class ConditionSpec(StrictBaseModel):
    """Condition that must be met for this task to be dispatched.

    When the condition is not met, the server marks the task as completed
    immediately without dispatching it to a worker.
    """

    node: str = Field(description="Upstream task ID whose result to check.")
    field: str = Field(
        description="Dot-separated path into the upstream result "
        "(e.g. ``result.verdict``)."
    )
    equals: str = Field(
        description="Expected value. Task only dispatches if ``actual == equals``."
    )


def _validate_condition_depends_on[T: "TaskSpecStrictBase | TaskSpecTemplateBase"](
    spec: T,
) -> T:
    condition = spec.condition
    if condition is None:
        return spec
    depends_on = spec.dependsOn
    if not depends_on:
        return spec
    dependency_names = {
        dep_stripped for dep in depends_on if (dep_stripped := dep.strip())
    }
    node = condition.node.strip()
    if node not in dependency_names:
        raise ValueError(f"condition.node '{condition.node}' must appear in dependsOn.")
    return spec


class TaskSpecStrictBase(StrictBaseModel):
    resources: ResourcesSpec | None = None
    output: OutputSpec | None = None
    dependsOn: list[str] | None = None
    condition: ConditionSpec | None = None
    shard: ShardSpec | None = None

    # Server-injected stage context (reserve the user-facing key `_upstreamResults`)
    upstreamResults: dict[str, BaseExecutorResult] | None = Field(
        default=None, alias="_upstreamResults"
    )

    @model_validator(mode="after")
    def _check_condition_depends_on(self) -> "TaskSpecStrictBase":
        _validate_condition_depends_on(self)
        return self

    def get_artifacts(self) -> list[str]:
        output = self.output
        if output is None:
            return []
        artifacts = output.artifacts
        if artifacts is None:
            return []
        return artifacts.copy()


class TaskSpecTemplateBase(TemplateBaseModel):
    resources: ResourcesSpec | None = None
    output: OutputSpecTemplate | None = None
    dependsOn: list[str] | None = None
    condition: ConditionSpec | None = None
    shard: ShardSpecTemplate | None = None

    upstreamResults: dict[str, BaseExecutorResult] | None = Field(
        default=None, alias="_upstreamResults"
    )

    @model_validator(mode="after")
    def _check_condition_depends_on(self) -> "TaskSpecTemplateBase":
        _validate_condition_depends_on(self)
        return self

    def get_artifacts(self) -> list[str]:
        output = self.output
        if output is None:
            return []
        artifacts = output.artifacts
        if artifacts is None:
            return []
        return artifacts.copy()


type TaskSpecBase = TaskSpecStrictBase | TaskSpecTemplateBase


class ModelSpecStrict(TaskSpecStrictBase):
    model: ModelConfig | None = None

    @property
    def model_name(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.identifier  # type: ignore

    @property
    def model_revision(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.revision  # type: ignore

    @property
    def model_trust_remote_code(self) -> bool:
        model = self.model
        source = model and model.source
        return bool(source and source.trust_remote_code)

    @property
    def adapters(self) -> list[AdapterConfig] | None:
        model = self.model
        return None if model is None else model.adapters


class ModelSpecTemplate(TaskSpecTemplateBase):
    model: ModelConfigTemplate | None = None

    @property
    def model_name(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.identifier  # type: ignore

    @property
    def model_revision(self) -> str | None:
        return (model := self.model) and (source := model.source) and source.revision  # type: ignore

    @property
    def model_trust_remote_code(self) -> bool:
        model = self.model
        source = model and model.source
        return bool(source and source.trust_remote_code)

    @property
    def adapters(self) -> list[AdapterConfigTemplate] | None:
        model = self.model
        return None if model is None else model.adapters


class ModelInferSpecStrict(ModelSpecStrict):
    data: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    postprocess: PostprocessSpec | None = None


class ModelInferSpecTemplate(ModelSpecTemplate):
    data: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    postprocess: PostprocessSpecTemplate | None = None
