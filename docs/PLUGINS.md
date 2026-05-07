# Plugin extension points

External integrations (auth, submission policy, usage tracking,
authorisation, supplier attribution, resource lifecycle) plug into the
server through six protocol hooks defined in the standalone
**`flowmesh-hook`** package:

- `IdentityProvider` — resolve a bearer token to a `PrincipalContext`.
  Routers and WebSocket endpoints pull the bearer from the
  `Authorization` header and run it through the provider chain; the
  first non-`None` result wins. With no providers registered, auth is
  a no-op and a default admin principal is used. Workers
  self-authenticate the same way, sending their `FLOWMESH_API_KEY` as
  a bearer on every server call.
- `SubmissionGuard` — pre-submit precondition on the principal; e.g.,
  reject when the principal has insufficient balance.
- `UsageSink` — fan-out per-task usage rows after a task completes.
  Typical consumers: billing, audit, observability.
- `PermissionChecker` — filter list endpoints (`accessible_ids`) and
  gate point reads / mutations (`require`) via `resolve_accessible_ids`
  / `require_permission`. Multiple checkers compose. With no checkers
  registered the helpers are no-ops. `require` accepts `resource_id=None`
  for type-level / fleet-level checks (e.g. "may the principal create
  workflows", "may the principal read system metrics").
- `SupplierResolver` — map an assigned worker (`WorkerView`) to a
  supplier id at dispatch time. The first non-`None` result wins and
  is stamped on `TaskRecord.supplier_id`; `UsageSink`s receive that
  value. With no resolvers registered, `supplier_id` stays at `""`.
- `ResourceRegistrar` — observe resource lifecycle. The server fires
  `register` after a resource is persisted (`WORKFLOW`, `TASK`, `NODE`,
  `WORKER`) and `deregister` after one is hard-deleted. Plugins use
  these to seed their own ACL / ownership tables so subsequent
  `PermissionChecker` calls have data to decide on. `RESULT` ownership
  is inferred from the owning task; `RESULT` permission checks are
  always paired with a `task_id`, and workflow-level operations check
  `WORKFLOW`.

The `ResourceType` enum covers `WORKFLOW`, `TASK`, `RESULT`, `NODE`,
`WORKER`, and `SYSTEM`; `ResourceAction` covers `READ`, `WRITE`,
`CANCEL`, and `ADMIN`. Plugins use `principal.scopes` to discriminate
user-vs-supplier-vs-admin capabilities.

The `flowmesh-hook` package has no runtime dependencies and does not
import the server or worker packages, so plugin wheels stay tiny and
can be installed without the heavy core stack.

## How plugins are loaded

A plugin is any Python module that exposes a top-level `install()`
returning a `HookBindings` — the frozen aggregate of the protocol
implementations the plugin contributes. The server loads
`FLOWMESH_PLUGINS` (comma-separated module names) inside its FastAPI
lifespan and treats `install()` as either:

- a sync function returning a `HookBindings` directly; or
- an `@asynccontextmanager async def install()` that yields a
  `HookBindings`. Use this form for plugins owning resources with a
  lifecycle (a SQLAlchemy engine, an HTTP client, a background task)
  that need teardown on server shutdown. The loader holds an
  `AsyncExitStack`, enters each ctx-manager `install()` on startup,
  and unwinds them in reverse order on shutdown.

The loader drains every plugin's `HookBindings` into the server's
runtime registries. Plugins never touch those registries directly.

Plugins live anywhere on `sys.path` — in-tree under
`src/server/<name>/`, sibling-mounted under `/app/src/<name>/`, or a
pip-installable wheel. Core never references plugin module names; each
plugin self-filters internally.

## Minimal sync plugin

```python
# myorg_supplier_plugin/__init__.py
from flowmesh_hook import HookBindings, WorkerView


class _MyOrgSupplier:
    name = "myorg.supplier"

    def resolve(self, worker: WorkerView) -> str | None:
        return worker.env.get("supplier_id") or None


def install() -> HookBindings:
    return HookBindings(supplier_resolvers=[_MyOrgSupplier()])
```

## Plugins with their own DB

FlowMesh ships no DB itself. Plugins that need persistence bring their own
engine and manage it via the ctx-manager `install()` form:

```python
# myorg_auth_plugin/__init__.py
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from flowmesh_hook import HookBindings


class _MyOrgAuth:
    name = "myorg.auth"

    def __init__(self, sessionmaker):
        self._sm = sessionmaker

    async def resolve(self, raw_token, logger):
        async with self._sm() as session:
            ...


@asynccontextmanager
async def install():
    engine = create_async_engine(os.environ["MYORG_DATABASE_URL"])
    try:
        yield HookBindings(
            identity_providers=[_MyOrgAuth(async_sessionmaker(engine))],
        )
    finally:
        await engine.dispose()
```
