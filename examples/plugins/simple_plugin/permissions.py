"""`PermissionChecker` example: admin bypass + ownership-based gating.

Admin principals (any with the `"admin"` scope) bypass every check. Other
principals can only see and act on resources they registered themselves —
ownership is read from `state.OWNERSHIP`, populated by
`SimpleResourceRegistrar`.
"""

import logging

from fastapi import HTTPException
from flowmesh_hook import PrincipalContext, ResourceAction, ResourceType

from . import state

_ADMIN_SCOPE = "admin"


def _is_admin(principal: PrincipalContext) -> bool:
    return _ADMIN_SCOPE in principal.scopes


class SimplePermissionChecker:
    name = "simple_plugin.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        resource_type: ResourceType,
        action: ResourceAction,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        if _is_admin(principal):
            logger.info(
                "%s: admin %s -> no filter on %s/%s",
                self.name,
                principal.principal_id,
                resource_type.value,
                action.value,
            )
            return None
        owned = frozenset(
            rid
            for (rt, rid), owner in state.OWNERSHIP.items()
            if rt == resource_type and owner == principal.principal_id
        )
        logger.info(
            "%s: principal_id=%s sees %d %s(s)",
            self.name,
            principal.principal_id,
            len(owned),
            resource_type.value,
        )
        return owned

    async def require(
        self,
        principal: PrincipalContext,
        resource_type: ResourceType,
        resource_id: str | None,
        action: ResourceAction,
        logger: logging.Logger,
    ) -> None:
        if _is_admin(principal):
            logger.info(
                "%s: admin %s -> allow %s on %s/%s",
                self.name,
                principal.principal_id,
                action.value,
                resource_type.value,
                resource_id if resource_id is not None else "<type-level>",
            )
            return
        if resource_id is None:
            # Type-level / fleet-level check. Allow any principal carrying at
            # least one scope; tighten this in real plugins.
            if not principal.scopes:
                logger.warning(
                    "%s: deny scope-less type-level %s on %s",
                    self.name,
                    action.value,
                    resource_type.value,
                )
                raise HTTPException(
                    status_code=403,
                    detail="principal has no scopes for type-level action",
                )
            logger.info(
                "%s: principal_id=%s allowed type-level %s on %s",
                self.name,
                principal.principal_id,
                action.value,
                resource_type.value,
            )
            return
        owner = state.OWNERSHIP.get((resource_type, resource_id))
        if owner == principal.principal_id:
            logger.info(
                "%s: owner %s -> allow %s on %s/%s",
                self.name,
                principal.principal_id,
                action.value,
                resource_type.value,
                resource_id,
            )
            return
        logger.warning(
            "%s: deny %s on %s/%s for principal_id=%s (owner=%s)",
            self.name,
            action.value,
            resource_type.value,
            resource_id,
            principal.principal_id,
            owner,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"principal {principal.principal_id} may not {action.value} "
                f"{resource_type.value}/{resource_id}"
            ),
        )
