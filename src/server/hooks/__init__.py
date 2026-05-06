"""Plugin hooks for extending the server.

- `IdentityProvider` — resolve a bearer token to a `PrincipalContext`.
- `SubmissionGuard` — gate workflow submission.
- `UsageSink` — fan-out per-task usage rows.
- `PermissionChecker` — filter list endpoints and gate point reads/mutations.
- `SupplierResolver` — stamp `TaskRecord.supplier_id` at dispatch time.

Plugins append to the module-level lists; core iterates them at call time.
"""

from .identity import IdentityProvider
from .permissions import AccessibleIds, PermissionChecker
from .submission import SubmissionGuard
from .supplier import SupplierResolver
from .usage import UsageRow, UsageSink

IDENTITY_PROVIDERS: list[IdentityProvider] = []
SUBMISSION_GUARDS: list[SubmissionGuard] = []
USAGE_SINKS: list[UsageSink] = []
PERMISSION_CHECKERS: list[PermissionChecker] = []
SUPPLIER_RESOLVERS: list[SupplierResolver] = []

__all__ = [
    "AccessibleIds",
    "IDENTITY_PROVIDERS",
    "IdentityProvider",
    "PERMISSION_CHECKERS",
    "PermissionChecker",
    "SUBMISSION_GUARDS",
    "SUPPLIER_RESOLVERS",
    "SubmissionGuard",
    "SupplierResolver",
    "USAGE_SINKS",
    "UsageRow",
    "UsageSink",
]
