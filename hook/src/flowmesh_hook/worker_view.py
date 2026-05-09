"""Read-only structural view of a `Worker` consumed by `SupplierResolver`."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkerView(Protocol):
    """Structural view of a `Worker` passed to `SupplierResolver.resolve`."""

    id: str
    node_id: str
    namespace: str
    cluster: str
    tags: list[str]
    env: dict[str, Any]
