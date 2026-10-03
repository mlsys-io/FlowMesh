from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

_MISSING = object()
_NULLS = frozenset({"", "null", "None"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _query_items(queries: Mapping) -> list[tuple[str, str]]:
    multi_items = getattr(queries, "multi_items", None)
    if callable(multi_items):
        return list(multi_items())  # type: ignore
    return list(queries.items())


@dataclass(frozen=True)
class _Accepted:
    """The values one key accepts, in every form a field's value is compared as."""

    strings: frozenset[str]
    bools: frozenset[bool]
    null: bool

    @classmethod
    def of(cls, values: frozenset[str]) -> "_Accepted":
        spellings = {value.strip().lower() for value in values}
        bools = {True} if spellings & _TRUTHY else set()
        if spellings & _FALSY:
            bools.add(False)
        return cls(values, frozenset(bools), bool(values & _NULLS))

    def matches(self, value: Any, key: str) -> bool:
        if value is None:
            return self.null
        if isinstance(value, bool):
            return value in self.bools
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(str(member) in self.strings for member in value)
        if key == "tags" and isinstance(value, str):
            tags = (tag.strip() for tag in value.split(","))
            return any(tag in self.strings for tag in tags if tag)
        return str(value) in self.strings


def _get_nested_value(data: Any, key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def filter_models_by_queries[T: BaseModel](
    models: list[T], queries: Mapping
) -> list[T]:
    """Filter Pydantic models by HTTP-style query parameters.

    This function is used by list endpoints that accept arbitrary query params
    (e.g., FastAPI/Starlette ``Request.query_params``). It applies **exact**
    matching for fields that exist on the model, and ignores unknown keys.

    Supported query behaviors:

    - **Exact match (default)**: ``?status=IDLE`` matches when
      ``model.status == "IDLE"``.
    - **Repeated keys (OR semantics)**: ``?status=IDLE&status=BUSY`` matches when
      the field equals **any** provided value.
    - **Nested keys via dot-notation**: ``?env.region=us-east-1`` will traverse
      dict-like fields (``{"env": {"region": ...}}``). If traversal fails, the
      key is ignored for that model.
    - **List/set membership**: if the model field is a list/tuple/set, then a
      match occurs when **any** query value equals **any** element (stringified),
      e.g. ``?cached_models=gpt-4o-mini``.
    - **Tag membership**:
      - If the model field is ``list[str]`` (common), membership matching applies.
      - If the model field is a comma-separated string (as some Redis-backed
        models serialize), ``?tags=gpu`` matches if ``"gpu"`` is one of the
        comma-separated tags.
    - **Null-ish matching**: if the model field is ``None``, it matches query
      values ``""``, ``"null"``, or ``"None"``.
    - **Booleans**: query values accept common truthy/falsy spellings
      (``true/false``, ``1/0``, ``yes/no``, ``on/off``).

    Not supported yet:

    - Partial/substring matches, regex, numeric comparisons (``gt/lt``),
      negation (``!=``), case-insensitive matching, or globbing.

    Examples:

    - List workers by owning node and status:
      ``/workers?node_id=nde-1&status=IDLE&status=BUSY``
    - Filter workers by nested hardware fields (from ``WorkerHardware``):
      ``/workers?hardware.cpu.model=Intel(R)%20Xeon(R)&hardware.gpu.cuda_version=12.4``
    - Filter nodes by namespace/cluster/tag (from ``NodeInfo``):
      ``/nodes?namespace=prod&cluster=us-east-1&tags=gpu``
    - List all node-managed workers (``NodeWorkerInfo``) by provider/status:
      ``/nodes/workers?provider=docker&status=IDLE``
    """
    query_map: dict[str, set[str]] = defaultdict(set)
    for key, value in _query_items(queries):
        query_map[str(key)].add(str(value))
    accepted = {
        key: _Accepted.of(frozenset(values)) for key, values in query_map.items()
    }

    if not accepted:
        return models

    filtered = []
    for model in models:
        model_dict = model.model_dump()
        match = True
        for key, values in accepted.items():
            model_value = _get_nested_value(model_dict, key)
            if model_value is _MISSING:
                continue
            if not values.matches(model_value, key):
                match = False
                break
        if match:
            filtered.append(model)
    return filtered
