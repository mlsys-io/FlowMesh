"""Server-side runtime registries for the plugin hooks.

The protocols and types themselves live in `flowmesh_hook`; this module
owns the mutable lists the plugin loader drains `HookBindings` into and
that the server iterates at call time.
"""

from flowmesh_hook import (
    HookBindings,
    IdentityProvider,
    PermissionChecker,
    PrincipalContext,
    ResourceAction,
    ResourceRegistrar,
    ResourceType,
    SubmissionGuard,
    SupplierResolver,
    UsageRow,
    UsageSink,
    WorkerView,
)

IDENTITY_PROVIDERS: list[IdentityProvider] = []
SUBMISSION_GUARDS: list[SubmissionGuard] = []
USAGE_SINKS: list[UsageSink] = []
PERMISSION_CHECKERS: list[PermissionChecker] = []
SUPPLIER_RESOLVERS: list[SupplierResolver] = []
RESOURCE_REGISTRARS: list[ResourceRegistrar] = []


def register(bindings: HookBindings) -> None:
    """Append every binding to the matching runtime registry."""
    IDENTITY_PROVIDERS.extend(bindings.identity_providers)
    SUBMISSION_GUARDS.extend(bindings.submission_guards)
    USAGE_SINKS.extend(bindings.usage_sinks)
    PERMISSION_CHECKERS.extend(bindings.permission_checkers)
    SUPPLIER_RESOLVERS.extend(bindings.supplier_resolvers)
    RESOURCE_REGISTRARS.extend(bindings.resource_registrars)


__all__ = [
    "HookBindings",
    "IDENTITY_PROVIDERS",
    "IdentityProvider",
    "PERMISSION_CHECKERS",
    "PermissionChecker",
    "PrincipalContext",
    "RESOURCE_REGISTRARS",
    "ResourceAction",
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
