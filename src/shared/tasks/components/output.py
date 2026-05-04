from typing import Annotated, Any, Literal

from pydantic import Field

from .._base import StrictBaseModel, TemplateBaseModel
from ..placeholders import TemplateFloat


class OutputDestinationLocal(StrictBaseModel):
    type: Literal["local"] = "local"


class OutputDestinationHTTP(StrictBaseModel):
    type: Literal["http"] = "http"
    url: str | None = None
    method: str | None = None
    headers: dict[str, Any] | None = None
    timeoutSec: float | None = None


OutputDestination = Annotated[
    OutputDestinationLocal | OutputDestinationHTTP,
    Field(discriminator="type"),
]


class OutputSpec(StrictBaseModel):
    destination: OutputDestination | None = None
    artifacts: list[str] | None = None


class OutputDestinationLocalTemplate(TemplateBaseModel):
    type: Literal["local"] = "local"


class OutputDestinationHTTPTemplate(TemplateBaseModel):
    type: Literal["http"] = "http"
    url: str | None = None
    method: str | None = None
    headers: dict[str, Any] | None = None
    timeoutSec: TemplateFloat | None = None


OutputDestinationTemplate = Annotated[
    OutputDestinationLocalTemplate | OutputDestinationHTTPTemplate,
    Field(discriminator="type"),
]


class OutputSpecTemplate(TemplateBaseModel):
    destination: OutputDestinationTemplate | None = None
    artifacts: list[str] | None = None
