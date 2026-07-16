# API reference (common endpoints)

Server runs at `http://localhost:8000` by default. The router source of
truth is `src/server/routers/v1/*.py`.

Workflow and task statuses (used in payloads and filters):
`PENDING`, `DISPATCHED`, `FAILED`, `CANCELLED`, `DONE`. Worker statuses:
`STARTING`, `IDLE`, `BUSY`, `STOPPING`, `STOPPED`.

## Authentication

Every endpoint under `/api/v1/*` (REST and WebSocket) authenticates via
the `Authorization: Bearer <token>` header. The token is routed through
the registered `IdentityProvider` chain (see `docs/PLUGINS.md`); with
no providers registered, every caller resolves to a default admin
principal. After authentication, every resource-scoped endpoint runs
the registered `PermissionChecker` chain; with no checkers registered,
all calls succeed (open by default). The classification of resource type
and action per endpoint lives in `src/server/routers/v1/`. Workers
self-authenticate the same way, sending `FLOWMESH_API_KEY` as the bearer.

## Workflows

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/workflows` | Submit a workflow. Body is YAML (`text/plain`) or JSON; set `Workflow-Format: n8n` for n8n graphs. |
| POST | `/api/v1/workflows/validate` | Parse without executing. |
| GET | `/api/v1/workflows` | List workflows (`workflow_id`, `owner`, `status`, cursor pagination). |
| GET | `/api/v1/workflows/{id}` | Workflow details + per-task summary. |
| GET | `/api/v1/workflows/{id}/logs` | Query logs (`limit`, `before`/`after` cursors). |
| GET | `/api/v1/workflows/{id}/logs/stream` | SSE log stream. |
| POST | `/api/v1/workflows/{id}/cancel` | Cancel a workflow and all in-flight tasks. |

## Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/tasks` | List tasks. Filters: `workflow_id`, `status`, `task_type`, `assigned_worker`. |
| GET | `/api/v1/tasks/{id}` | Task details. |
| GET | `/api/v1/tasks/{id}/logs` | Query task logs. |
| GET | `/api/v1/tasks/{id}/logs/stream` | SSE task log stream. |

## Results

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/results` | Submit task result (worker → server). |
| GET | `/api/v1/results/{task_id}` | Get task result JSON. |
| GET | `/api/v1/results/{task_id}/bundle` | Download tar.gz bundle (`?include=results,artifacts,logs,all`). |
| POST | `/api/v1/results/{task_id}/files` | Upload artifact (multipart). |
| GET | `/api/v1/results/{task_id}/files/{filename}` | Download artifact. |
| GET | `/api/v1/results/{task_id}/logs` | Download archived `logs.jsonl`. |

## Traces

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/traces/workflows/{workflow_id}/{trace_type}` | Fetch workflow trace JSONL rows. `trace_type` is `spans`, `assets`, or `lineage`. |
| GET | `/api/v1/traces/workflows/analyze/{workflow_id}` | Run the trace analyzer and return a profile summary. |
| POST | `/api/v1/traces/tasks/{task_id}/{trace_type}` | Upload a per-task trace JSONL file. `trace_type` is `spans`, `assets`, or `lineage`. |

## Workers and nodes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/workers` | List workers. Filters: `alias`, `namespace`, `cluster`, `status`, `tags`. |
| GET | `/api/v1/workers/{id}` | Worker details + hardware. |
| GET | `/api/v1/nodes` | List nodes (supervisors). |
| POST | `/api/v1/nodes/register` | Register a node. |
| GET | `/api/v1/nodes/{id}/workers` | List workers under a node. |
| POST | `/api/v1/nodes/{id}/workers/register` | Register worker under node. |
| POST | `/api/v1/nodes/{id}/workers/{name}/{start,stop}` | Start/stop a worker. |

`/api/v1/stack/workers/...` wraps node-registered workers with local-only
container lifecycle and is what `flowmesh stack worker {up,down,...}`
calls.

## SSH

| Method | Path | Description |
|--------|------|-------------|
| WS | `/api/v1/ssh/tasks/{task_id}/proxy` | WebSocket SSH proxy for proxy-mode SSH tasks. |
| GET | `/api/v1/ssh/connections` | List active server-audited SSH proxy/forward connections. |

Server policy toggles: `ENABLE_SERVER_SSH_PROXY`,
`ENABLE_SERVER_PORT_FORWARD`, `ENABLE_SERVER_SSH_CONNECTION_AUDIT`.

## Serve

| Method | Path | Description |
|--------|------|-------------|
| ANY | `/api/v1/serve/tasks/{task_id}/{upstream_path:path}` | HTTP reverse proxy to a `proxy`-mode serve task's vLLM server. |

PAT-exempt: authenticated solely by the task's vLLM api-key, not a Lumid PAT.
Gated by `ENABLE_SERVER_SERVE_PROXY`.

## System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Top-level health check. |
| GET | `/api/v1/system/version` | Server version. |
| GET | `/api/v1/system/metrics` | System metrics snapshot. |

## Cursor pagination

List endpoints (`/api/v1/workflows`, `/api/v1/tasks`, log queries)
accept `limit` and `before` / `after` cursors. The cursor is an opaque
base64 of `(timestamp, id)`; do not parse client-side.
