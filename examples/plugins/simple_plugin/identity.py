"""`IdentityProvider` example: bearer-token lookup against an in-memory dict."""

import logging

from flowmesh_hook import PrincipalContext

from . import state


class SimpleIdentityProvider:
    name = "simple_plugin.identity"

    async def resolve(
        self, raw_token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        principal = state.TOKENS.get(raw_token)
        if principal is None:
            logger.info("%s: unknown token, deferring", self.name)
            return None
        logger.info(
            "%s: resolved token to principal_id=%s scopes=%s",
            self.name,
            principal.principal_id,
            principal.scopes,
        )
        return principal
