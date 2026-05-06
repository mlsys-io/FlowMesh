"""Minimal auth surface for OSS.

OSS ships no native API-key auth. The semantic is:

- With no `IdentityProvider` plugins registered, `authenticate_api_key`
  returns a default admin principal — auth is effectively a no-op and every
  caller is admin.
- Once at least one provider is registered, every bearer token is routed
  through the chain in registration order. The first provider returning a
  non-`None` `PrincipalContext` wins; if none claim the token, 401 is raised.

Routers consume the chain via `authenticate_request`, a FastAPI dependency
that pulls the bearer token from the request header before invoking
`authenticate_api_key`. Permission helpers (`resolve_accessible_ids`,
`require_permission`) compose the registered `PermissionChecker` chain;
with no checkers registered both helpers short-circuit to "no filter, no
gate", preserving OSS-only behaviour.
"""

import logging

from fastapi import HTTPException, Request, status
from flowmesh_hook import PrincipalContext, ResourceAction, ResourceType

from ..hooks import IDENTITY_PROVIDERS, PERMISSION_CHECKERS


def default_principal() -> PrincipalContext:
    """Synthetic admin principal returned when no auth is configured."""
    return PrincipalContext(
        principal_id="admin",
        org_id="local",
        external_id="local",
        principal_type="admin",
        scopes=["*"],
    )


async def authenticate_api_key(
    raw_key: str, logger: logging.Logger
) -> PrincipalContext:
    """Resolve a bearer token to a `PrincipalContext` via registered providers.

    With no providers registered, returns `default_principal()` — auth is off,
    every caller is admin.
    """
    if not IDENTITY_PROVIDERS:
        return default_principal()

    for provider in IDENTITY_PROVIDERS:
        resolved = await provider.resolve(raw_key, logger)
        if resolved is not None:
            return resolved

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No identity provider accepted the token",
    )


async def authenticate_request(request: Request) -> PrincipalContext:
    """FastAPI dependency: extract the bearer token and run the auth chain."""
    auth_header = request.headers.get("Authorization", "")
    bearer_prefix = "Bearer "
    raw_token = (
        auth_header[len(bearer_prefix) :]
        if auth_header.startswith(bearer_prefix)
        else ""
    )
    return await authenticate_api_key(raw_token, request.app.state.logger)


async def resolve_accessible_ids(
    principal: PrincipalContext,
    resource_type: ResourceType,
    action: ResourceAction,
    logger: logging.Logger,
) -> frozenset[str] | None:
    """Compose `PermissionChecker.accessible_ids` across registered checkers.

    Returns `None` to indicate "no filter" — either no checkers are
    registered, or some checker explicitly returned `None`. Otherwise
    returns the union of all checker-permitted id sets.
    """
    if not PERMISSION_CHECKERS:
        return None

    accumulated: set[str] = set()
    for checker in PERMISSION_CHECKERS:
        result = await checker.accessible_ids(principal, resource_type, action, logger)
        if result is None:
            return None
        accumulated.update(result)
    return frozenset(accumulated)


async def require_permission(
    principal: PrincipalContext,
    resource_type: ResourceType,
    resource_id: str,
    action: ResourceAction,
    logger: logging.Logger,
) -> None:
    """Run every registered `PermissionChecker.require`. Raises on first deny."""
    for checker in PERMISSION_CHECKERS:
        await checker.require(principal, resource_type, resource_id, action, logger)
