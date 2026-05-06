"""`HookBindings` — the value a plugin returns from `install()`.

Frozen aggregate of the protocol implementations a plugin contributes. The
server drains each field into its runtime registries on startup; plugins
never touch the registries directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from .identity import IdentityProvider
from .permissions import PermissionChecker
from .submission import SubmissionGuard
from .supplier import SupplierResolver
from .usage import UsageSink


@dataclass(frozen=True)
class HookBindings:
    identity_providers: Sequence[IdentityProvider] = field(default_factory=tuple)
    submission_guards: Sequence[SubmissionGuard] = field(default_factory=tuple)
    usage_sinks: Sequence[UsageSink] = field(default_factory=tuple)
    permission_checkers: Sequence[PermissionChecker] = field(default_factory=tuple)
    supplier_resolvers: Sequence[SupplierResolver] = field(default_factory=tuple)
