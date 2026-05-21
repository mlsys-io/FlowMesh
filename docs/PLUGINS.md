# Plugin extension points

External integrations (auth, submission policy, usage tracking,
authorisation, supplier attribution, resource lifecycle) plug into the
server through six protocol hooks split across two packages:

- **[`lumid-hooks`](https://github.com/mlsys-io/lumid.hooks)** — the
  shared contract surface used by Lumid projects. Carries the five hooks
  generic enough to share (`IdentityProvider`, `SubmissionGuard`,
  `PermissionChecker`, `ResourceRegistrar`, `UsageSink`) along with
  `PrincipalContext` and `ResourceRef`.
- **`flowmesh-hook`** — FlowMesh-specific extensions: the `HookBindings`
  Protocol (extending the shared one with `supplier_resolvers`), a
  `BaseBindings` frozen dataclass plugin authors can return directly,
  the `ResourceKind` / `ResourceAction` enums, `SupplierResolver` and its
  `WorkerView`, and the FlowMesh `UsageRow` / `FlowMeshUsageSink` typed alias.

The hooks:

- `IdentityProvider` (shared) — resolve a bearer token to a
  `PrincipalContext`. Routers and WebSocket endpoints pull the bearer
  from the `Authorization` header and run it through the provider
  chain; the first non-`None` result wins. With no providers
  registered, auth is a no-op and a default admin principal is used.
  Workers self-authenticate the same way, sending their
  `FLOWMESH_API_KEY` as a bearer on every server call.
- `SubmissionGuard` (shared) — pre-submit precondition on the
  principal; e.g., reject when the principal has insufficient balance.
- `UsageSink[Row]` (shared, generic) — fan-out usage rows after a unit
  of work completes. FlowMesh parametrizes it as `FlowMeshUsageSink =
  UsageSink[UsageRow]`; typical consumers: billing, audit,
  observability.
- `PermissionChecker` (shared) — filter list endpoints
  (`accessible_ids(principal, kind, action, logger)`) and gate point
  reads / mutations (`require(principal, resource: ResourceRef,
  action, logger)`) via `resolve_accessible_ids` / `require_permission`
  on the server side. Multiple checkers compose conjunctively: `require`
  denies if any checker denies, and `accessible_ids` returns the
  intersection of returned id sets (checkers returning `None` impose no
  filter and are skipped). With no checkers registered, or every checker
  returning `None`, both helpers are no-ops. `ResourceRef.id == None`
  denotes a kind-level / fleet-level check (e.g. "may the principal create
  workflows", "may the principal read system metrics").
- `SupplierResolver` (FlowMesh-specific) — map an assigned worker
  (`WorkerView`) to a supplier id at dispatch time. The first non-`None`
  result wins and is stamped on `TaskRecord.supplier_id`; `FlowMeshUsageSink`s
  receive that value. With no resolvers registered, `supplier_id` stays
  at `""`.
- `ResourceRegistrar` (shared) — observe resource lifecycle. The server
  fires `register(principal, resource: ResourceRef, logger)` after a
  resource is persisted (`WORKFLOW`, `TASK`, `NODE`, `WORKER`) and
  `deregister` after one is hard-deleted or self-terminated. `principal`
  on both calls is always a real `PrincipalContext`: the calling admin
  for request-driven mutations, or a server-resolved *system principal*
  for boot-time / heartbeat-driven paths. The system principal is
  `FLOWMESH_API_KEY` run through the `IdentityProvider` chain at
  startup, falling back to the synthetic admin when no providers are
  registered. Plugins use these to seed their own ACL / ownership
  tables so subsequent `PermissionChecker` calls have data to decide
  on. `RESULT` ownership is inferred from the owning task; `RESULT`
  permission checks are always paired with a `task_id`, and
  workflow-level operations check `WORKFLOW`. At startup the server
  runs a reconcile sweep: it enumerates every live workflow, task,
  worker, and node and calls `refresh(resources, logger)` once per
  registrar with the full batch, then `purge_stale(logger)` once.
  Persistent registrars use this pair to drop records for resources
  the server no longer knows about — stateless registrars implement
  both as no-ops.

The shared protocols treat `kind` and `action` as plain strings —
`lumid-hooks` does not enumerate kinds. FlowMesh layers the
`ResourceKind` and `ResourceAction` `StrEnum`s on top so call sites
inside FlowMesh get auto-complete and exhaustiveness checks; values
like `ResourceKind.WORKFLOW` pass straight into the protocol's
`kind: str` parameter (no `.value` needed).

`flowmesh-hook` depends only on `lumid-hooks` (transitively `pydantic`)
and does not import the server or worker packages, so plugin wheels
stay tiny and can be installed without the heavy core stack.

## Where plugin authors import from

| Symbol | Package | Notes |
|--------|---------|-------|
| `IdentityProvider`, `SubmissionGuard`, `PermissionChecker`, `ResourceRegistrar`, `UsageSink` | `lumid_hooks` | Shared protocols. |
| `PrincipalContext`, `ResourceRef` | `lumid_hooks` | Shared types. |
| `HookBindings`, `BaseBindings` | `flowmesh_hook` | Protocol (six fields, gate type) and frozen dataclass (convenience base, returned by `install()`). |
| `ResourceKind`, `ResourceAction` | `flowmesh_hook` | FlowMesh resource and action enums. |
| `SupplierResolver`, `WorkerView` | `flowmesh_hook` | FlowMesh-specific dispatch hook. |
| `UsageRow`, `FlowMeshUsageSink` | `flowmesh_hook` | FlowMesh's usage row + parametrized sink alias. |

A plugin that only implements shared hooks may depend on `lumid-hooks`
alone; FlowMesh-specific plugins additionally depend on `flowmesh-hook`.

## How plugins are loaded

A plugin is any Python module that exposes a top-level `install()`
returning an object that satisfies `lumid_hooks.HookBindings` (the
shared Protocol describing the five-field aggregate). FlowMesh's
`BaseBindings` is a frozen dataclass that satisfies the Protocol with
empty default factories; FlowMesh-only plugins typically return that.
A cross-host plugin can return any object whose attributes match the
expected names — structural typing handles the rest. The server
loads `FLOWMESH_PLUGINS` (comma-separated module names) inside its
FastAPI lifespan and treats `install()` as either:

- a sync function returning the bindings directly; or
- an `@asynccontextmanager async def install()` that yields the
  bindings. Use this form for plugins owning resources with a
  lifecycle (a SQLAlchemy engine, an HTTP client, a background task)
  that need teardown on server shutdown. The loader holds an
  `AsyncExitStack`, enters each ctx-manager `install()` on startup,
  and unwinds them in reverse order on shutdown.

The loader drains every plugin's bindings into the server's runtime
registries; FlowMesh-specific `supplier_resolvers` is drained when the
returned object also satisfies `flowmesh_hook.HookBindings`. Plugins
never touch those registries directly.

Plugins live anywhere on `sys.path` — in-tree under
`src/server/<name>/`, host-mounted under `/app/plugins/<name>/` (the
canonical deployment path; see below), or a pip-installable wheel.
Core never references plugin module names; each plugin self-filters
internally.

## Deploying with `flowmesh stack`

The prebuilt server image puts `/app/plugins` on `PYTHONPATH`, and
`flowmesh stack` bind-mounts `${FLOWMESH_PLUGIN_DIR:-./plugins}` from
the host into that location. So a typical deployment is:

```
mkdir -p plugins/myorg_auth
# ... lay out plugins/myorg_auth/__init__.py exposing install()
echo "FLOWMESH_PLUGINS=myorg_auth" >> .env
flowmesh stack up
```

Each subdirectory of `FLOWMESH_PLUGIN_DIR` is importable as a
top-level module. The mount is read-only, so the plugin code is
treated as static deployment artifact.

For writable persistence, `FLOWMESH_PLUGIN_DATA_DIR` (default
`./plugin-data`) is mounted read-write at `/app/plugin-data`. A path
value is a host bind-mount (auto-created on `stack up`); a bare name
is an external Docker volume of that name.

This handles plugin **code** without rebuilding the server image.
When that isn't enough, build a thin overlay on top of the prebuilt
image. Two patterns, pick by need:

Plugin needs extra Python deps but ships its code via the host mount:

```dockerfile
FROM ghcr.io/mlsys-io/flowmesh_server:<pinned-tag>
RUN pip install <your-deps>
```

Plugin is fully baked into the image (no host mount needed):

```dockerfile
FROM ghcr.io/mlsys-io/flowmesh_server:<pinned-tag>
COPY dist/myplugin-1.0-py3-none-any.whl /tmp/
RUN pip install /tmp/myplugin-1.0-py3-none-any.whl \
 && rm /tmp/myplugin-1.0-py3-none-any.whl
```

Push the result to your registry and point the stack at the new tag
via `FLOWMESH_REGISTRY` / `FLOWMESH_VERSION` (or `flowmesh stack up
--image-tag <tag>`).

## Minimal sync plugin

```python
# myorg_supplier_plugin/__init__.py
from flowmesh_hook import BaseBindings, WorkerView


class _MyOrgSupplier:
    name = "myorg.supplier"

    def resolve(self, worker: WorkerView) -> str | None:
        return worker.namespace or None


def install() -> BaseBindings:
    return BaseBindings(supplier_resolvers=[_MyOrgSupplier()])
```

## Plugins with their own DB

FlowMesh ships no DB itself. Plugins that need persistence bring their own
engine and manage it via the ctx-manager `install()` form:

```python
# myorg_auth_plugin/__init__.py
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from flowmesh_hook import BaseBindings
from lumid_hooks import PrincipalContext


class _MyOrgAuth:
    name = "myorg.auth"

    def __init__(self, sessionmaker):
        self._sm = sessionmaker

    async def resolve(self, raw_token, logger) -> PrincipalContext | None:
        async with self._sm() as session:
            ...


@asynccontextmanager
async def install():
    engine = create_async_engine(os.environ["MYORG_DATABASE_URL"])
    try:
        yield BaseBindings(
            identity_providers=[_MyOrgAuth(async_sessionmaker(engine))],
        )
    finally:
        await engine.dispose()
```

## Worked example: every hook over an in-memory store

For a runnable end-to-end example exercising all six hook protocols against
an in-memory store, see [`examples/plugins/simple_plugin/`](../examples/plugins/simple_plugin/).

For an example exercising **only the shared subset** (no
`SupplierResolver`, no FlowMesh resource enums) against an in-memory
store, see [`examples/simple_plugin/`](https://github.com/mlsys-io/lumid.hooks/tree/main/examples/simple_plugin)
in the `lumid-hooks` repo.
