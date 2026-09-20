from typing import Any

from pydantic import BaseModel


def copy_preserving_fields_set[M: BaseModel](model: M, update: dict[str, Any]) -> M:
    """``model_copy(update=...)`` that preserves the original fields-set.

    ``model_copy`` marks every updated key as set. A caller that only rewrites
    values of fields that already exist (e.g. credential redaction) must keep the
    original fields-set, or ``exclude_unset`` dumps would surface fields the caller
    never set.
    """
    copy = model.model_copy(update=update)
    # Bypass pydantic's __setattr__ (which rejects writes to the internal
    # __pydantic_fields_set__) to set it directly, as pydantic does internally.
    object.__setattr__(
        copy, "__pydantic_fields_set__", set(model.__pydantic_fields_set__)
    )
    return copy
