"""`PermissionChecker` example: admin bypass + ownership-based gating.

Admin principals (any with the `"admin"` scope) bypass every check. Other
principals can only see and act on resources they registered themselves —
ownership is read from `state.OWNERSHIP`, populated by
`SimpleResourceRegistrar`.
"""

import logging

from fastapi import HTTPException
from lumid_hooks import PrincipalContext, ResourceRef

from . import state

_ADMIN_SCOPE = "admin"


def _is_admin(principal: PrincipalContext) -> bool:
    return _ADMIN_SCOPE in principal.scopes


class SimplePermissionChecker:
    name = "simple_plugin.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        if _is_admin(principal):
            logger.info(
                "%s: admin %s -> no filter on %s/%s",
                self.name,
                principal.principal_id,
                kind,
                action,
            )
            return None
        owned = frozenset(
            rid
            for (k, rid), owner in state.OWNERSHIP.items()
            if k == kind and owner == principal.principal_id
        )
        logger.info(
            "%s: principal_id=%s sees %d %s(s)",
            self.name,
            principal.principal_id,
            len(owned),
            kind,
        )
        return owned

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        if _is_admin(principal):
            logger.info(
                "%s: admin %s -> allow %s on %s/%s",
                self.name,
                principal.principal_id,
                action,
                resource.kind,
                resource.id if resource.id is not None else "<kind-level>",
            )
            return
        if resource.id is None:
            # Kind-level / fleet-level check. Allow any principal carrying at
            # least one scope; tighten this in real plugins.
            if not principal.scopes:
                logger.warning(
                    "%s: deny scope-less kind-level %s on %s",
                    self.name,
                    action,
                    resource.kind,
                )
                raise HTTPException(
                    status_code=403,
                    detail="principal has no scopes for kind-level action",
                )
            logger.info(
                "%s: principal_id=%s allowed kind-level %s on %s",
                self.name,
                principal.principal_id,
                action,
                resource.kind,
            )
            return
        owner = state.OWNERSHIP.get((resource.kind, resource.id))
        if owner == principal.principal_id:
            logger.info(
                "%s: owner %s -> allow %s on %s/%s",
                self.name,
                principal.principal_id,
                action,
                resource.kind,
                resource.id,
            )
            return
        logger.warning(
            "%s: deny %s on %s/%s for principal_id=%s (owner=%s)",
            self.name,
            action,
            resource.kind,
            resource.id,
            principal.principal_id,
            owner,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"principal {principal.principal_id} may not {action} "
                f"{resource.kind}/{resource.id}"
            ),
        )
