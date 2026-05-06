"""Plugin hooks for extending the server.

- `IdentityProvider` — resolve a bearer token to a `PrincipalContext`.
- `SubmissionGuard` — gate workflow submission.
- `UsageSink` — fan-out per-task usage rows.
- `PermissionChecker` — filter list endpoints and gate point reads/mutations.

Plugins append to the module-level lists; core iterates them at call time.
"""

from .identity import IdentityProvider
from .permissions import AccessibleIds, PermissionChecker
from .submission import SubmissionGuard
from .usage import UsageRow, UsageSink

IDENTITY_PROVIDERS: list[IdentityProvider] = []
SUBMISSION_GUARDS: list[SubmissionGuard] = []
USAGE_SINKS: list[UsageSink] = []
PERMISSION_CHECKERS: list[PermissionChecker] = []

__all__ = [
    "AccessibleIds",
    "IDENTITY_PROVIDERS",
    "IdentityProvider",
    "PERMISSION_CHECKERS",
    "PermissionChecker",
    "SUBMISSION_GUARDS",
    "SubmissionGuard",
    "USAGE_SINKS",
    "UsageRow",
    "UsageSink",
]
