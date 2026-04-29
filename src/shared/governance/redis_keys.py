"""Redis key conventions for cross-task data payloads.

The worker stores each task's output payload at `flowmesh:data:{data_id}` so
downstream tasks can fetch upstream results without a centralized service.
"""

_DATA_PREFIX = "flowmesh:data:"


def data_key(data_id: str) -> str:
    return f"{_DATA_PREFIX}{data_id}"


def workflow_data_pattern() -> str:
    return f"{_DATA_PREFIX}*"
