"""Server-side runtime registries for the plugin hooks.

The protocols and shared types live in `lumid_hooks`; FlowMesh-specific
extensions (`HookBindings`, `BaseBindings`, `ResourceType`, `ResourceAction`,
`SupplierResolver`, `WorkerView`, `UsageRow`, `FlowMeshUsageSink`) live in
`flowmesh_hook`. This module owns the mutable lists the plugin loader drains
plugin bindings into and that the server iterates at call time.

`register()` accepts any object satisfying `lumid_hooks.HookBindings`
(shared-only plugin); FlowMesh-specific `supplier_resolvers` is drained via
`isinstance` narrowing on the FlowMesh `HookBindings` Protocol.
"""

from flowmesh_hook import (
    BaseBindings,
    FlowMeshUsageSink,
    HookBindings,
    ResourceAction,
    ResourceType,
    SupplierResolver,
    UsageRow,
    WorkerView,
)
from lumid_hooks import HookBindings as SharedHookBindings
from lumid_hooks import (
    IdentityProvider,
    PermissionChecker,
    PrincipalContext,
    ResourceRef,
    ResourceRegistrar,
    SubmissionGuard,
    UsageSink,
)

IDENTITY_PROVIDERS: list[IdentityProvider] = []
SUBMISSION_GUARDS: list[SubmissionGuard] = []
USAGE_SINKS: list[FlowMeshUsageSink] = []
PERMISSION_CHECKERS: list[PermissionChecker] = []
SUPPLIER_RESOLVERS: list[SupplierResolver] = []
RESOURCE_REGISTRARS: list[ResourceRegistrar] = []


def register(bindings: SharedHookBindings) -> None:
    """Append every binding to the matching runtime registry."""
    IDENTITY_PROVIDERS.extend(bindings.identity_providers)
    SUBMISSION_GUARDS.extend(bindings.submission_guards)
    USAGE_SINKS.extend(bindings.usage_sinks)
    PERMISSION_CHECKERS.extend(bindings.permission_checkers)
    RESOURCE_REGISTRARS.extend(bindings.resource_registrars)
    if isinstance(bindings, HookBindings):
        SUPPLIER_RESOLVERS.extend(bindings.supplier_resolvers)


__all__ = [
    "BaseBindings",
    "FlowMeshUsageSink",
    "HookBindings",
    "IDENTITY_PROVIDERS",
    "IdentityProvider",
    "PERMISSION_CHECKERS",
    "PermissionChecker",
    "PrincipalContext",
    "RESOURCE_REGISTRARS",
    "ResourceAction",
    "ResourceRef",
    "ResourceRegistrar",
    "ResourceType",
    "SUBMISSION_GUARDS",
    "SUPPLIER_RESOLVERS",
    "SubmissionGuard",
    "SupplierResolver",
    "USAGE_SINKS",
    "UsageRow",
    "UsageSink",
    "WorkerView",
    "register",
]
