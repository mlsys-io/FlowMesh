"""Minimal auth surface.

FlowMesh ships no native API-key auth. The semantic is:

- With no `IdentityProvider` plugins registered, `authenticate_api_key`
  returns a default admin principal — auth is effectively a no-op and every
  caller is admin.
- Once at least one provider is registered, every bearer token is routed
  through the chain in registration order. The first provider returning a
  non-`None` `PrincipalContext` wins; if none claim the token, 401 is raised.

Routers consume the chain via `authenticate_connection`, which works for
both REST and WebSocket endpoints (both extend Starlette's `HTTPConnection`).
Permission helpers (`resolve_accessible_ids`, `require_permission`) compose
the registered `PermissionChecker` chain; with no checkers registered both
helpers short-circuit to "no filter, no gate".
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketException, status
from flowmesh_hook import ResourceAction, ResourceKind
from lumid_hooks import PrincipalContext, ResourceRef
from starlette.requests import HTTPConnection

from ..hooks import IDENTITY_PROVIDERS, PERMISSION_CHECKERS, RESOURCE_REGISTRARS


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


async def resolve_system_principal(
    api_key: str, logger: logging.Logger
) -> PrincipalContext:
    """Resolve the principal that represents this server for system-driven actions
    (boot-time worker spawn, supervisor self-registration, heartbeat reaper, etc.)."""
    if not IDENTITY_PROVIDERS:
        return default_principal()
    try:
        for provider in IDENTITY_PROVIDERS:
            resolved = await provider.resolve(api_key, logger)
            if resolved is not None:
                return resolved
    except Exception as exc:
        logger.warning(
            "IdentityProvider raised resolving system principal: %s; "
            "falling back to default admin principal.",
            exc,
        )
        return default_principal()
    logger.warning(
        "No IdentityProvider claimed the system FLOWMESH_API_KEY; "
        "falling back to default admin principal."
    )
    return default_principal()


async def authenticate_connection(conn: HTTPConnection) -> PrincipalContext:
    """FastAPI dependency: extract the bearer token and run the auth chain.

    Works for both REST routes and WebSocket endpoints — both `Request` and
    `WebSocket` are `HTTPConnection` subclasses.
    """
    auth_header = conn.headers.get("Authorization", "")
    bearer_prefix = "Bearer "
    raw_token = (
        auth_header[len(bearer_prefix) :]
        if auth_header.startswith(bearer_prefix)
        else ""
    )
    return await authenticate_api_key(raw_token, conn.app.state.logger)


async def authenticate_websocket(
    websocket: WebSocket,
    token: str | None = None,
) -> PrincipalContext:
    """FastAPI dependency for WebSocket auth.

    Tries the `Authorization: Bearer ...` header first, falling back to a
    `?token=...` query param for browser clients that can't set headers.
    Raises `WebSocketException(4401)` on failure so FastAPI closes the
    socket before the route body runs.
    """
    try:
        return await authenticate_connection(websocket)
    except HTTPException:
        pass
    if token:
        try:
            return await authenticate_api_key(token, websocket.app.state.logger)
        except HTTPException:
            pass
    raise WebSocketException(code=4401, reason="unauthorized")


async def resolve_accessible_ids(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    action: ResourceAction,
    logger: logging.Logger,
) -> frozenset[str] | None:
    """Compose `PermissionChecker.accessible_ids` across registered checkers.

    Conjunctive: returns the intersection of the id sets returned by every
    checker. A checker returning `None` imposes no filter and is skipped.
    Returns `None` ("no filter") only when no checkers are registered or
    every checker returns `None`; otherwise returns a (possibly empty)
    `frozenset[str]`.
    """
    if not PERMISSION_CHECKERS:
        return None

    accumulated: frozenset[str] | None = None
    for checker in PERMISSION_CHECKERS:
        result = await checker.accessible_ids(
            principal, resource_kind.value, action.value, logger
        )
        if result is None:
            continue
        accumulated = result if accumulated is None else accumulated & result
    return accumulated


async def require_permission(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    resource_id: str | None,
    action: ResourceAction,
    logger: logging.Logger,
) -> None:
    """Run every registered `PermissionChecker.require`. Raises on first deny.

    `resource_id=None` is a type-level / fleet-level check (e.g. "may the
    principal create workflows" before a `wfl-` id has been minted, or any
    `SYSTEM`-scoped check).
    """
    resource = ResourceRef(kind=resource_kind.value, id=resource_id)
    for checker in PERMISSION_CHECKERS:
        await checker.require(principal, resource, action.value, logger)


async def register_resource(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    resource_id: str,
    metadata: Mapping[str, Any],
    logger: logging.Logger,
) -> None:
    """Notify every registered `ResourceRegistrar` that a resource was created.

    Fires for `WORKFLOW`, `TASK`, `NODE`, and `WORKER`. `RESULT` ownership
    is inferred from the owning task and does not fire. With no registrars
    registered this is a no-op.
    """
    resource = ResourceRef(kind=resource_kind.value, id=resource_id, metadata=metadata)
    for registrar in RESOURCE_REGISTRARS:
        await registrar.register(principal, resource, logger)


async def deregister_resource(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    resource_id: str,
    logger: logging.Logger,
) -> None:
    """Notify every registered `ResourceRegistrar` that a resource was destroyed.

    `principal` is always a real `PrincipalContext` — the calling admin for
    request-driven destructions, the resolved system principal (see
    `resolve_system_principal`) for heartbeat reaps and self-shutdown.
    """
    resource = ResourceRef(kind=resource_kind.value, id=resource_id)
    for registrar in RESOURCE_REGISTRARS:
        await registrar.deregister(principal, resource, logger)


async def refresh_resources(
    resources: Iterable[ResourceRef],
    logger: logging.Logger,
) -> frozenset[str]:
    """Notify every registered `ResourceRegistrar` of the current live set.

    Called once during startup reconcile with every live workflow / task /
    node / worker. A registrar whose `refresh` raises is logged and its
    name is returned in the failed set; the sweep does not abort — pass
    the result to `purge_stale_resources(..., skip=...)` so failed
    registrars don't wipe rows they never marked live.
    """
    refs = list(resources)
    failed: set[str] = set()
    for registrar in RESOURCE_REGISTRARS:
        try:
            await registrar.refresh(refs, logger)
        except Exception:
            logger.exception(
                "ResourceRegistrar %s.refresh failed; skipping its purge_stale.",
                registrar.name,
            )
            failed.add(registrar.name)
    return frozenset(failed)


async def purge_stale_resources(
    logger: logging.Logger,
    *,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Tell each `ResourceRegistrar` to drop records the reconcile sweep
    didn't touch.

    Called once after `refresh_resources`. Registrars whose name is in
    `skip` are bypassed — typically the set of registrars whose `refresh`
    raised in the same sweep, so they don't wipe their rows on a partial
    refresh.
    """
    for registrar in RESOURCE_REGISTRARS:
        if registrar.name in skip:
            continue
        await registrar.purge_stale(logger)
