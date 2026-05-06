"""Identity-provider hook.

Resolves bearer tokens to a `PrincipalContext`. Each provider may claim a
token (return PrincipalContext), pass it on (return None), or reject it as
invalid by raising an exception (FastAPI's HTTPException is the documented
choice for terminal failures, but the protocol does not import fastapi).
"""

import logging
from typing import Protocol, runtime_checkable

from .types import PrincipalContext


@runtime_checkable
class IdentityProvider(Protocol):
    name: str

    async def resolve(
        self, raw_token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        """Authenticate `raw_token`.

        Return a `PrincipalContext` to claim the token, `None` to defer to the
        next provider, or raise an exception for terminal failures.
        """
        ...
