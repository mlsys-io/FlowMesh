"""FlowMesh-specific extension surface.

Carries the bits the FlowMesh server adds on top of `lumid-hooks`:

- `HookBindings` — runtime-checkable Protocol extending the shared one with
  FlowMesh's `supplier_resolvers`. Used by the server's plugin gate.
- `BaseBindings` — frozen dataclass extending `lumid_hooks.BaseBindings` with
  `supplier_resolvers`. Convenience class for FlowMesh-only plugins.
- `ResourceKind` / `ResourceAction` — FlowMesh resource and action enums.
- `SupplierResolver` / `WorkerView` — supplier attribution at dispatch time.
- `UsageRow` / `FlowMeshUsageSink` — FlowMesh's usage row shape and parametrized
  sink alias.

Shared protocols (`IdentityProvider`, `SubmissionGuard`, `PermissionChecker`,
`ResourceRegistrar`, `UsageSink`) and shared types (`PrincipalContext`,
`ResourceRef`) come from `lumid-hooks`; import them from there directly.
"""

from .bindings import BaseBindings, HookBindings
from .resource_kinds import ResourceAction, ResourceKind
from .supplier import SupplierResolver
from .usage import FlowMeshUsageSink, UsageRow
from .worker_view import WorkerView

__all__ = [
    "BaseBindings",
    "FlowMeshUsageSink",
    "HookBindings",
    "ResourceAction",
    "ResourceKind",
    "SupplierResolver",
    "UsageRow",
    "WorkerView",
]
