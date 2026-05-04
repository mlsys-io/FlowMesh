# SDK usage (Python)

The Python SDK lives in `sdk/src/flowmesh/`. The public surface exposes
two clients — `FlowMesh` (sync) and `AsyncFlowMesh` (async) — both
configured the same way and exposing the same resource groups
(`workflows`, `tasks`, `workers`, `nodes`, `results`, `system`,
`ssh`). Both handle pagination, error formatting, and SSE streaming for
you; prefer them over raw HTTP calls.

## Sync client

```python
from flowmesh import FlowMesh

client = FlowMesh(base_url="http://localhost:8000", api_key="...")

# Submit a workflow from a YAML file.
wf = client.workflows.submit_yaml(open("templates/echo_local.yaml").read())

# Stream logs until the workflow finishes.
for ev in client.workflows.stream_logs(wf.workflow_id):
    print(ev.line)

# Inspect tasks and pull a result.
tasks = client.tasks.list(workflow_id=wf.workflow_id)
result = client.results.get(tasks[0].task_id)
```

## Async client

```python
from flowmesh import AsyncFlowMesh

async with AsyncFlowMesh(base_url="...", api_key="...") as client:
    wf = await client.workflows.submit_yaml(yaml_text)
    async for ev in client.workflows.stream_logs(wf.workflow_id):
        ...
```

## Common operations

- **Submit YAML / JSON / n8n** — `client.workflows.submit_yaml(text)`,
  `submit_json(payload)`, `submit_n8n(graph)`.
- **Validate without executing** — `client.workflows.validate_yaml(text)`.
- **List with filters and pagination** — `client.workflows.list(status=...,
  before=..., limit=...)`, same shape on `client.tasks.list(...)`.
- **Stream logs** — `client.workflows.stream_logs(wf_id)` and
  `client.tasks.stream_logs(task_id)` yield server-sent events; the
  iterator stops when the source closes.
- **Pull artifacts** — `client.results.get(task_id)` for the result
  payload, `client.results.download_bundle(task_id, include="all")`
  for the tar.gz.
- **Cancel** — `client.workflows.cancel(wf_id)`.

## Cursor pagination

List endpoints take `limit` and `before` / `after` cursors. Cursors are
opaque base64 strings — pass them through; do not parse them. The SDK
exposes `next_cursor` on the response object so you can paginate by
threading it back in.

## Errors

The SDK raises `flowmesh.FlowMeshError` (and a small set of subclasses
for auth / not-found / rate-limit) instead of returning HTTP error
shapes. Wrap your calls accordingly.
