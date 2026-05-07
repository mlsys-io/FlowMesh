"""`ResourceRegistrar` example: maintain an in-memory ownership table."""

import logging
from collections.abc import Mapping
from typing import Any

from flowmesh_hook import PrincipalContext, ResourceType

from . import state


class SimpleResourceRegistrar:
    name = "simple_plugin.registrar"

    async def register(
        self,
        principal: PrincipalContext,
        resource_type: ResourceType,
        resource_id: str,
        metadata: Mapping[str, Any],
        logger: logging.Logger,
    ) -> None:
        state.OWNERSHIP[(resource_type, resource_id)] = principal.principal_id
        logger.info(
            "%s: register %s/%s -> principal_id=%s (metadata_keys=%s)",
            self.name,
            resource_type.value,
            resource_id,
            principal.principal_id,
            sorted(metadata.keys()),
        )

    async def deregister(
        self,
        principal: PrincipalContext | None,
        resource_type: ResourceType,
        resource_id: str,
        logger: logging.Logger,
    ) -> None:
        owner = state.OWNERSHIP.pop((resource_type, resource_id), None)
        actor = principal.principal_id if principal is not None else "<system>"
        logger.info(
            "%s: deregister %s/%s (was_owner=%s, actor=%s)",
            self.name,
            resource_type.value,
            resource_id,
            owner,
            actor,
        )
