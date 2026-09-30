import re
from collections.abc import Iterator
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel

from ._base import StrictBaseModel, TemplateBaseModel

PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def is_placeholder(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(PLACEHOLDER_PATTERN.fullmatch(value.strip()))


def placeholder_fields(model: BaseModel) -> Iterator[tuple[str, Any]]:
    """The ``(name, value)`` fields of ``model`` that may hold stage references."""
    exempt = (
        model.placeholder_exempt_fields
        if isinstance(model, (StrictBaseModel, TemplateBaseModel))
        else frozenset()
    )
    for name, value in model:
        if name not in exempt:
            yield name, value


def _validate_placeholder_string(value: Any) -> str:
    if not isinstance(value, str) or not is_placeholder(value):
        raise ValueError("Expected a placeholder string like ${...}")
    return value


type PlaceholderString = Annotated[str, AfterValidator(_validate_placeholder_string)]

type TemplateBool = bool | PlaceholderString
type TemplateInt = int | PlaceholderString
type TemplateFloat = float | PlaceholderString
