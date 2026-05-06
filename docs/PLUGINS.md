# Plugin extension points

External integrations (auth, submission policy, usage tracking,
authorisation) plug into the server through four protocol hooks defined
in `src/server/hooks/`:

- `IdentityProvider` — resolve a bearer token to a `PrincipalContext`
  (iterated from `auth/security.py`). Routers consume the chain via the
  `authenticate_request` FastAPI dependency, which extracts the bearer
  token from the `Authorization` header. With no providers registered,
  auth is a no-op and the dependency returns a default admin principal.
- `SubmissionGuard` — pre-submit precondition (iterated from
  `routers/v1/workflows.py`).
- `UsageSink` — fan-out per-task usage rows after a task completes
  (iterated from `services/monitoring.py`). Typical consumers: billing,
  audit, observability.
- `PermissionChecker` — filter list endpoints (`accessible_ids`) and
  gate point reads / mutations (`require`), iterated from
  `auth/security.py` via `resolve_accessible_ids` / `require_permission`.
  Multiple checkers compose: `accessible_ids` is `"all"` if any checker
  returns `"all"`, otherwise the union of returned id sets; `require`
  requires every checker to pass. With no checkers registered the
  helpers are no-ops, matching OSS's open-by-default behaviour.

## How plugins are loaded

A plugin is any Python module that exposes a top-level `install()`. The
server loads `FLOWMESH_PLUGINS` (comma-separated module names) inside
its FastAPI lifespan and treats `install()` as either:

- a sync function returning `None` — the plugin appends its adapters to
  the registries in `server.hooks` and returns; or
- an `@asynccontextmanager async def install()` — the plugin owns
  resources with a lifecycle (a SQLAlchemy engine, an HTTP client, a
  background task) that need teardown on server shutdown. The loader
  holds an `AsyncExitStack`, enters each ctx-manager `install()` on
  startup, and unwinds them in reverse order on shutdown.

Plugins live anywhere on `sys.path` — in-tree under
`src/server/<name>/`, sibling-mounted under `/app/src/<name>/`, or a
pip-installable wheel. Core never references plugin module names; each
plugin self-filters internally.

## Plugins with their own DB

OSS ships no DB itself. Plugins that need persistence bring their own
engine and manage it via the ctx-manager `install()` form:

```python
# myorg_auth_plugin/__init__.py
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.hooks import IDENTITY_PROVIDERS


class _MyOrgAuth:
    name = "myorg.auth"
    def __init__(self, sessionmaker): self._sm = sessionmaker
    async def resolve(self, raw_token, logger):
        async with self._sm() as session:
            ...


@asynccontextmanager
async def install():
    engine = create_async_engine(os.environ["MYORG_DATABASE_URL"])
    IDENTITY_PROVIDERS.append(_MyOrgAuth(async_sessionmaker(engine)))
    try:
        yield
    finally:
        await engine.dispose()
```
