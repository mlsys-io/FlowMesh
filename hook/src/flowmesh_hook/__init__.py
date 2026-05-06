"""FlowMesh plugin contract surface.

Plugins return a `HookBindings` from `install()`; the server drains it
into its runtime registries on startup.
"""

from .bindings import HookBindings
from .identity import IdentityProvider
from .permissions import PermissionChecker
from .submission import SubmissionGuard
from .supplier import SupplierResolver
from .types import AccessibleIds, PrincipalContext, UsageRow, WorkerView
from .usage import UsageSink

__all__ = [
    "AccessibleIds",
    "HookBindings",
    "IdentityProvider",
    "PermissionChecker",
    "PrincipalContext",
    "SubmissionGuard",
    "SupplierResolver",
    "UsageRow",
    "UsageSink",
    "WorkerView",
]
