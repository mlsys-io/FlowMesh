"""Shared types referenced by the hook protocols.

Kept dependency-free so plugins can import them without pulling in the
server or worker packages.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable


@dataclass(frozen=True)
class PrincipalContext:
    principal_id: str
    org_id: str
    external_id: str
    principal_type: str
    scopes: list[str]


AccessibleIds = Literal["all"] | frozenset[str]


@runtime_checkable
class WorkerView(Protocol):
    """Structural view of a `Worker` exposed to `SupplierResolver`.

    The server's concrete `Worker` Pydantic model satisfies this Protocol
    structurally. Plugins should only read attributes declared here.
    """

    id: str
    node_id: str
    namespace: str
    cluster: str
    tags: list[str]
    env: dict[str, Any]


class UsageRow(TypedDict):
    org_id: str
    principal_id: str
    supplier_id: str | None
    occurred_at: datetime
    cost: Decimal
    task_id: str
    runtime_sec: float
    cost_per_hour: float
    task_status: str
