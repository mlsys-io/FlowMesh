from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", from_attributes=True, populate_by_name=True
    )

    # Fields whose text is taken verbatim: ``${...}`` in them is never read as a
    # stage reference (e.g. source code, where it is ordinary syntax).
    placeholder_exempt_fields: ClassVar[frozenset[str]] = frozenset()


class TemplateBaseModel(BaseModel):
    """
    Template-time base model.

    Still forbids unknown keys, but allows placeholder-friendly scalar types
    (e.g. int | "${...}") in *explicit* fields of concrete models.
    """

    model_config = ConfigDict(
        extra="forbid", from_attributes=True, populate_by_name=True
    )

    placeholder_exempt_fields: ClassVar[frozenset[str]] = frozenset()
