"""FlowMesh's usage row + sink alias.

`lumid_hooks.UsageSink` is generic over the row type; FlowMesh ships a
concrete `UsageRow` shape and exposes `FlowMeshUsageSink` as the parametrized
alias plugins type against.
"""

from datetime import datetime
from decimal import Decimal
from typing import TypedDict

from lumid_hooks import UsageSink


class UsageRow(TypedDict):
    """One usage row emitted to `FlowMeshUsageSink` after a task completes."""

    org_id: str
    principal_id: str
    supplier_id: str | None
    occurred_at: datetime
    cost: Decimal
    task_id: str
    runtime_sec: float
    cost_per_hour: float
    task_status: str


type FlowMeshUsageSink = UsageSink[UsageRow]
