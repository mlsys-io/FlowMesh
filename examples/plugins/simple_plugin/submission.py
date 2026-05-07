"""`SubmissionGuard` example: deny principals listed in `state.BLOCKED_PRINCIPALS`."""

import logging

from fastapi import HTTPException
from flowmesh_hook import PrincipalContext

from . import state


class SimpleSubmissionGuard:
    name = "simple_plugin.guard"

    async def check(self, principal: PrincipalContext, logger: logging.Logger) -> None:
        if principal.principal_id in state.BLOCKED_PRINCIPALS:
            logger.warning(
                "%s: blocking submission for principal_id=%s",
                self.name,
                principal.principal_id,
            )
            raise HTTPException(
                status_code=403,
                detail="principal blocked by simple_plugin",
            )
        logger.info(
            "%s: allowing submission for principal_id=%s",
            self.name,
            principal.principal_id,
        )
