"""`ResourceRegistrar` example: maintain an in-memory ownership table."""

import logging

from lumid_hooks import PrincipalContext, ResourceRef

from . import state


class SimpleResourceRegistrar:
    name = "simple_plugin.registrar"

    async def register(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        if resource.id is None:
            logger.warning(
                "%s: skipping register for kind-level ref kind=%s",
                self.name,
                resource.kind,
            )
            return
        state.OWNERSHIP[(resource.kind, resource.id)] = principal.principal_id
        logger.info(
            "%s: register %s/%s -> principal_id=%s (metadata_keys=%s)",
            self.name,
            resource.kind,
            resource.id,
            principal.principal_id,
            sorted(resource.metadata.keys()),
        )

    async def deregister(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        if resource.id is None:
            return
        owner = state.OWNERSHIP.pop((resource.kind, resource.id), None)
        logger.info(
            "%s: deregister %s/%s (was_owner=%s, actor=%s)",
            self.name,
            resource.kind,
            resource.id,
            owner,
            principal.principal_id,
        )
