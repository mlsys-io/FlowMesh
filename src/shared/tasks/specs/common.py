from typing import Any, Self

from pydantic import Field, SerializeAsAny, model_validator

from ...schemas.result import BaseExecutorResult
from ...utils.redact import has_redacted_credential_fields, redact_credential_fields
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
from ..components.output import OutputDestinationHTTP, OutputDestinationHTTPTemplate
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
        description="Dot-separated path into the upstream result payload "
        "(e.g. ``items.0.output``)."
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


def _redact_output(
    output: OutputSpec | OutputSpecTemplate | None,
) -> OutputSpec | OutputSpecTemplate | None:
    if output is None:
        return output
    dest = output.destination
    if (
        isinstance(dest, (OutputDestinationHTTP, OutputDestinationHTTPTemplate))
        and dest.headers is not None
    ):
        redacted_dest = dest.model_copy(
            update={"headers": redact_credential_fields(dest.headers)}
        )
        return output.model_copy(update={"destination": redacted_dest})
    return output


def _output_has_redacted_credentials(
    output: OutputSpec | OutputSpecTemplate | None,
) -> bool:
    dest = output.destination if output else None
    if isinstance(dest, (OutputDestinationHTTP, OutputDestinationHTTPTemplate)):
        return has_redacted_credential_fields(dest.headers)
    return False


class TaskSpecStrictBase(StrictBaseModel):
    resources: ResourcesSpec | None = None
    output: OutputSpec | None = None
    dependsOn: list[str] | None = None
    condition: ConditionSpec | None = None
    shard: ShardSpec | None = None

    # Server-injected stage context (reserve the user-facing key `_upstreamResults`)
    upstreamResults: dict[str, SerializeAsAny[BaseExecutorResult]] | None = Field(
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

    def validate_dispatchable(self) -> None:
        """Validate spec-internal invariants for a runnable task.

        Called at submit and again before dispatch. Overrides must raise ``ValueError``
        for misconfigurations.
        """
        return None

    def redact_credentials(self) -> Self:
        """Redact credential-shaped headers in the output destination, if any."""
        redacted_output = _redact_output(self.output)
        if redacted_output is self.output:
            return self
        return self.model_copy(update={"output": redacted_output})

    def has_redacted_credentials(self) -> bool:
        """Whether this spec contains a redacted credential marker."""
        return _output_has_redacted_credentials(self.output)


class TaskSpecTemplateBase(TemplateBaseModel):
    resources: ResourcesSpec | None = None
    output: OutputSpecTemplate | None = None
    dependsOn: list[str] | None = None
    condition: ConditionSpec | None = None
    shard: ShardSpecTemplate | None = None

    upstreamResults: dict[str, SerializeAsAny[BaseExecutorResult]] | None = Field(
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

    def validate_dispatchable(self) -> None:
        """Validate spec-internal invariants for a runnable task.

        Called at submit and again before dispatch. Overrides must defer
        placeholder-dependent checks and raise ``ValueError`` for genuine
        misconfigurations.
        """
        return None

    def redact_credentials(self) -> Self:
        """Redact credential-shaped headers in the output destination, if any."""
        redacted_output = _redact_output(self.output)
        if redacted_output is self.output:
            return self
        return self.model_copy(update={"output": redacted_output})

    def has_redacted_credentials(self) -> bool:
        """Whether this spec contains a redacted credential marker."""
        return _output_has_redacted_credentials(self.output)


type TaskSpecBase = TaskSpecStrictBase | TaskSpecTemplateBase


def _redact_model_config(
    model: ModelConfig | ModelConfigTemplate | None,
) -> ModelConfig | ModelConfigTemplate | None:
    if model is None:
        return None
    adapters = model.adapters
    redacted_adapters = (
        [
            adapter.model_copy(
                update={"headers": redact_credential_fields(adapter.headers)}
            )
            for adapter in adapters
        ]
        if adapters is not None
        else None
    )
    return model.model_copy(
        update={
            "config": redact_credential_fields(model.config),
            "vllm": redact_credential_fields(model.vllm),
            "transformers": redact_credential_fields(model.transformers),
            "diffusers": redact_credential_fields(model.diffusers),
            "adapters": redacted_adapters,
        }
    )


def _model_has_redacted_credentials(
    model: ModelConfig | ModelConfigTemplate | None,
) -> bool:
    return model is not None and has_redacted_credential_fields(
        model.model_dump(mode="python")
    )


class ModelSpecStrict(TaskSpecStrictBase):
    model: ModelConfig | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(update={"model": _redact_model_config(spec.model)})

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or _model_has_redacted_credentials(
            self.model
        )

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

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(update={"model": _redact_model_config(spec.model)})

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or _model_has_redacted_credentials(
            self.model
        )

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

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "data": redact_credential_fields(spec.data),
                "inference": redact_credential_fields(spec.inference),
                "checkpoint": redact_credential_fields(spec.checkpoint),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {
                "data": self.data,
                "inference": self.inference,
                "checkpoint": self.checkpoint,
            }
        )


class ModelInferSpecTemplate(ModelSpecTemplate):
    data: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    postprocess: PostprocessSpecTemplate | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "data": redact_credential_fields(spec.data),
                "inference": redact_credential_fields(spec.inference),
                "checkpoint": redact_credential_fields(spec.checkpoint),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {
                "data": self.data,
                "inference": self.inference,
                "checkpoint": self.checkpoint,
            }
        )
