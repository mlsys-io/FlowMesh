from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

_MISSING = object()


def _query_items(queries: Mapping) -> list[tuple[str, str]]:
    multi_items = getattr(queries, "multi_items", None)
    if callable(multi_items):
        return list(multi_items())  # type: ignore
    return list(queries.items())


def _get_nested_value(data: Any, key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _matches_query(model_value: Any, key: str, query_values: list[str]) -> bool:
    if model_value is None:
        return any(v in ("", "null", "None") for v in query_values)

    if isinstance(model_value, bool):
        normalized_model = "true" if model_value else "false"
        normalized_queries = {v.strip().lower() for v in query_values}
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}
        if normalized_model == "true":
            return bool(normalized_queries & truthy)
        return bool(normalized_queries & falsy)

    if isinstance(model_value, (list, tuple, set)):
        value_set = {str(v) for v in model_value}
        return any(v in value_set for v in query_values)

    if key == "tags" and isinstance(model_value, str):
        tag_set = {t.strip() for t in model_value.split(",") if t.strip()}
        return any(v in tag_set for v in query_values)

    return any(str(model_value) == v for v in query_values)


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
    query_map: dict[str, list[str]] = defaultdict(list)
    for key, value in _query_items(queries):
        query_map[str(key)].append(str(value))

    if not query_map:
        return models

    filtered = []
    for model in models:
        model_dict = model.model_dump()
        match = True
        for key, values in query_map.items():
            model_value = _get_nested_value(model_dict, key)
            if model_value is _MISSING:
                continue
            if not _matches_query(model_value, key, values):
                match = False
                break
        if match:
            filtered.append(model)
    return filtered
