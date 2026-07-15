from typing import Annotated, Literal

from pydantic import Field

from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class ServeSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.SERVE]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward", "proxy"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
    apiKey: str | None = Field(default=None, min_length=1)

    def validate_dispatchable(self) -> None:
        _validate_serve_dispatchable(self)


class ServeSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.SERVE]
    ttlSeconds: Annotated[float, Field(gt=0)] | None = None
    readinessTimeoutSeconds: Annotated[float, Field(gt=0)] | None = None
    accessMode: Literal["direct", "forward", "proxy"] | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
    apiKey: str | None = Field(default=None, min_length=1)

    def validate_dispatchable(self) -> None:
        _validate_serve_dispatchable(self)


def _validate_serve_dispatchable(spec: "ServeSpecStrict | ServeSpecTemplate") -> None:
    """A serve task always launches a persistent vLLM GPU server, so it must
    request at least one GPU or GPU scheduling and accounting are bypassed."""
    hardware = spec.resources.hardware if spec.resources else None
    gpu = hardware.gpu if hardware else None
    if gpu and gpu.count is not None and gpu.count >= 1:
        return
    raise ValueError(
        "serve task launches a persistent vLLM server but requests no GPU. "
        "Set spec.resources.hardware.gpu.count >= 1."
    )
