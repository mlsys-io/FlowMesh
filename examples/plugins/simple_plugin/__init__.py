"""`simple_plugin` — runnable example exercising every FlowMesh hook.

In-memory state lives in `state` and every hook reads/writes it.

NOT FOR PRODUCTION. See README.md.
"""

from flowmesh_hook import BaseBindings

from .identity import SimpleIdentityProvider
from .permissions import SimplePermissionChecker
from .registrar import SimpleResourceRegistrar
from .submission import SimpleSubmissionGuard
from .supplier import SimpleSupplierResolver
from .usage import SimpleUsageSink


def install() -> BaseBindings:
    return BaseBindings(
        identity_providers=[SimpleIdentityProvider()],
        submission_guards=[SimpleSubmissionGuard()],
        usage_sinks=[SimpleUsageSink()],
        permission_checkers=[SimplePermissionChecker()],
        supplier_resolvers=[SimpleSupplierResolver()],
        resource_registrars=[SimpleResourceRegistrar()],
    )


__all__ = ["install"]
