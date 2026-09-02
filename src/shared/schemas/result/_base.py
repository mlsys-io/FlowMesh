# Base executor-result models. This module avoids importing ``shared.tasks`` so
# it binds before ``catalog`` pulls ``TaskType`` (whose import chain re-enters
# this module). ``children`` is a forward reference to ``AnyExecutorResult``,
# resolved when ``__init__`` rebuilds the model — hence the future import below.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from ..artifact import ArtifactContext

if TYPE_CHECKING:
    from .catalog import AnyExecutorResult


class DropNoneModel(BaseModel):
    """Omits optional declared fields whose value is ``None`` from the output.

    An optional field left unset drops out of the payload instead of emitting
    ``null``. Required fields (no default) are always kept, even when ``None``,
    so the payload still round-trips. Extra passthrough keys are kept as-is, so
    an explicit ``null`` in an open payload (API bodies, connector rows) stays.
    """

    @model_serializer(mode="wrap")
    def _drop_none_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        dumped = handler(self)
        if not isinstance(dumped, dict):
            return dumped
        optional = {
            field.alias or name
            for name, field in type(self).model_fields.items()
            if not field.is_required()
        }
        return {
            key: value
            for key, value in dumped.items()
            if value is not None or key not in optional
        }


class StrictModel(DropNoneModel):
    """Base for exact nested payload models; rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class BaseExecutorResult(DropNoneModel):
    """Common shape for every executor's result payload.

    ``extra="allow"`` keeps this the permissive fallback of the discriminated
    union: legacy ``results.json`` without a ``task_type`` and condition-skip
    payloads round-trip through it without losing fields.
    """

    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    ok: bool = Field(default=True, description="Whether task execution succeeded.")
    children: dict[str, SerializeAsAny[AnyExecutorResult]] = Field(
        default_factory=dict,
        exclude_if=lambda v: not v,
        description="Per-child result payloads for task merging.",
    )
    artifacts_: ArtifactContext | None = Field(
        default=None,
        alias="_artifacts",
        description="Resolution context for relative artifact refs.",
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "artifacts_" in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} may not redefine the internal "
                "BaseExecutorResult.artifacts_ field"
            )


class StrictExecutorResult(BaseExecutorResult):
    """Base for concrete per-task-type results; exact fields, alias output."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)
