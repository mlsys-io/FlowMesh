# `simple_plugin`

A self-contained example FlowMesh plugin that exercises every hook protocol
against an in-memory store.

> **Not for production.** Tokens are plaintext in source, all state is
> dropped on restart, and every hook is permissive by design. Use this only
> for poking at the contract.

## What each hook does here

| File | Hook | Behavior |
|------|------|----------|
| `identity.py` | `IdentityProvider` | Looks the bearer token up in `state.TOKENS`. Returns the matching `PrincipalContext`, or `None` (defer to next provider). |
| `submission.py` | `SubmissionGuard` | Rejects with HTTP 403 if `principal_id` is in `state.BLOCKED_PRINCIPALS`. |
| `usage.py` | `UsageSink` | Appends each `UsageRow` to `state.USAGE_LEDGER`; logs row count and total cost. |
| `permissions.py` | `PermissionChecker` | Admin scope bypasses every check. Otherwise `accessible_ids` returns the resources the principal owns; `require` allows type-level (`resource_id is None`) actions for any non-empty scope and concrete-id actions only when the principal is the registered owner. |
| `supplier.py` | `SupplierResolver` | Returns `worker.namespace`. |
| `registrar.py` | `ResourceRegistrar` | Records `(resource_type, resource_id) -> principal_id` in `state.OWNERSHIP` on `register`; drops the row on `deregister`. |

`state.py` holds every shared dict / set / list. `__init__.py` wires the six
hook classes into a `HookBindings` and exposes `install()`.

## Demo principals

| Token | `principal_id` | `org_id` | `scopes` |
|-------|----------------|----------|----------|
| `demo-admin-token` | `alice` | `demo` | `["admin"]` |
| `demo-user-token` | `bob` | `demo` | `["user"]` |

`alice` bypasses `PermissionChecker`; `bob` only sees the workflows / tasks
he submitted himself.

Edit `tokens.json` (next to `state.py`) to add, remove, or rename tokens —
`state.py` reads it at import time and builds `PrincipalContext`s from each
entry's fields.

## Enabling it on a deployed stack

Two paths — pick one.

**Point `FLOWMESH_PLUGIN_DIR` at this directory.** No copy needed:

```bash
# .env
FLOWMESH_PLUGIN_DIR=./examples/plugins
FLOWMESH_PLUGINS=simple_plugin
```

**Or copy / symlink into the default plugin dir** and keep
`FLOWMESH_PLUGIN_DIR` at its default:

```bash
ln -s "$(pwd)/examples/plugins/simple_plugin" ./plugins/simple_plugin
echo 'FLOWMESH_PLUGINS=simple_plugin' >> .env
```

Then `flowmesh stack up`. Authenticate requests with one of the demo tokens:

```bash
TOKEN=demo-admin-token
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/workflows
```

## Inspecting what fired

Every hook logs through the injected logger, so the easiest way to watch
the plugin in action is to tail the server log:

```bash
docker logs -f flowmesh_node_server | grep simple_plugin
```

You should see one line per `resolve` / `check` / `emit` / `accessible_ids`
/ `require` / `register` / `deregister` call.

## Caveats

- Every store is in-process Python state. A server restart wipes it.
- Tokens are committed plaintext. Rotate them by editing `state.TOKENS` (or
  better: write your own plugin and don't ship secrets in the repo).
- `PermissionChecker` here is intentionally permissive (any non-empty scope
  passes type-level checks). Real plugins should map specific scopes to
  specific (resource_type, action) pairs.
- All six hooks are implemented purely for demonstration. Real plugins
  ship the subset they need; absent hooks fall through to the runtime's
  documented default.
