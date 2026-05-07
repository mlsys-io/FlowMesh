"""Resource lifecycle hook.

Plugins implementing this protocol learn about resource creation and
destruction so they can seed their own ACL / ownership tables. The server
fires `register` after a resource is persisted and `deregister` after a
resource is hard-deleted or self-terminated.

Fires for `WORKFLOW`, `TASK`, `NODE`, and `WORKER`. `RESULT` ownership
is inferred from the owning task by the `PermissionChecker` and does not
fire individually.

`register` always runs from a request handler with an authenticated
actor. `deregister` may also fire from system-initiated reaps (heartbeat
cleanup, supervisor self-shutdown) where no actor is available — in
those cases `principal` is `None`.

Multiple registrars compose: every registrar's `register` runs in
registration order; failures propagate and abort the originating request.
With no registrars registered both methods are no-ops.
"""

import logging
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .types import PrincipalContext, ResourceType


@runtime_checkable
class ResourceRegistrar(Protocol):
    name: str

    async def register(
        self,
        principal: PrincipalContext,
        resource_type: ResourceType,
        resource_id: str,
        metadata: Mapping[str, Any],
        logger: logging.Logger,
    ) -> None:
        """Record that `principal` created `resource_id` of `resource_type`.

        `metadata` carries resource-specific context (e.g. workflow name,
        worker hardware shape) that the plugin may persist alongside the
        ownership row. Plugins should ignore unknown keys.
        """
        ...

    async def deregister(
        self,
        principal: PrincipalContext | None,
        resource_type: ResourceType,
        resource_id: str,
        logger: logging.Logger,
    ) -> None:
        """Record that `resource_id` of `resource_type` no longer exists.

        `principal` is the actor performing the destruction when one exists,
        or `None` for system-initiated reaps (heartbeat-driven cleanup,
        process self-shutdown) where no authenticated request triggered the
        removal. Plugins typically drop the ownership row regardless.
        """
        ...
