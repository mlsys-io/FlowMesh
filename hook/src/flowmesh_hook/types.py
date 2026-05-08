"""Shared types referenced by the hook protocols."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol, TypedDict, runtime_checkable


@dataclass(frozen=True)
class PrincipalContext:
    """Identity envelope returned by `IdentityProvider.resolve` and passed to every
    downstream hook."""

    principal_id: str
    """Stable id within the plugin's namespace; stamped on workflows as `owner_id`."""

    org_id: str
    """Tenant the principal belongs to."""

    external_id: str
    """External handle (email, OIDC `sub`, etc.). Plugin-defined."""

    principal_type: str
    """Descriptive label (e.g. `"user"`, `"admin"`). Diagnostic only;
    capability decisions belong in `scopes`."""

    scopes: list[str]
    """Capabilities the principal carries, in a plugin-defined vocabulary."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrincipalContext":
        return cls(
            principal_id=str(payload["principal_id"]),
            org_id=str(payload["org_id"]),
            external_id=str(payload["external_id"]),
            principal_type=str(payload["principal_type"]),
            scopes=list(payload.get("scopes", [])),
        )


class ResourceType(StrEnum):
    """Resource types in the permission contract.

    `RESULT` checks are always paired with a `task_id`; workflow-level
    operations (logs, queries) check `WORKFLOW`.
    """

    WORKFLOW = "workflow"
    TASK = "task"
    RESULT = "result"
    NODE = "node"
    WORKER = "worker"
    SYSTEM = "system"


class ResourceAction(StrEnum):
    """Actions in the permission contract."""

    READ = "read"
    WRITE = "write"
    CANCEL = "cancel"
    ADMIN = "admin"


@runtime_checkable
class WorkerView(Protocol):
    """Read-only structural view of a `Worker` passed to `SupplierResolver`."""

    id: str
    node_id: str
    namespace: str
    cluster: str
    tags: list[str]
    env: dict[str, Any]


class UsageRow(TypedDict):
    """One usage row emitted to `UsageSink` after a task completes."""

    org_id: str
    principal_id: str
    supplier_id: str | None
    occurred_at: datetime
    cost: Decimal
    task_id: str
    runtime_sec: float
    cost_per_hour: float
    task_status: str
