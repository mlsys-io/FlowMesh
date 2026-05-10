"""FlowMesh's resource and action vocabulary.

`lumid-hooks` keeps `kind` / `action` as plain strings; FlowMesh layers a
`StrEnum` on top so call sites get auto-complete and exhaustiveness checks.
The enums are `str`-compatible, so they pass straight into the shared
protocols' `kind: str` / `action: str` parameters with no `.value`.
"""

from enum import StrEnum


class ResourceKind(StrEnum):
    """Resource kinds in FlowMesh's permission contract.

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
    """Actions in FlowMesh's permission contract."""

    READ = "read"
    WRITE = "write"
    CANCEL = "cancel"
    ADMIN = "admin"
