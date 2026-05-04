# SDK usage (Python)

The Python SDK lives in `sdk/src/flowmesh/`. The public surface exposes
two clients — `FlowMesh` (sync) and `AsyncFlowMesh` (async) — both
configured the same way and exposing the same resource groups
(`workflows`, `tasks`, `workers`, `nodes`, `results`, `system`,
`ssh`, `traces`). Both handle pagination, error formatting, and SSE
streaming for you; prefer them over raw HTTP calls.

## Sync client

```python
from flowmesh import FlowMesh

client = FlowMesh(base_url="http://localhost:8000", api_key="...")

# Submit a workflow from a YAML file.
workflow_text = open("templates/echo_local.yaml").read()
wf = client.workflows.submit(workflow_text)

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
    wf = await client.workflows.submit(yaml_text)
    async for ev in client.workflows.stream_logs(wf.workflow_id):
        ...
```

## Common operations

- **Submit YAML / JSON / n8n** — `client.workflows.submit(text_or_mapping)`;
  pass `workflow_format="n8n"` for n8n graphs.
- **Validate without executing** — `client.workflows.validate(text_or_mapping)`.
- **List with filters and pagination** — `client.workflows.list(status=...)`
  and `client.tasks.list(workflow_id=..., status=...)`; pass raw cursor
  params with `query_params=[("before", cursor), ("limit", "100")]`.
- **Stream logs** — `client.workflows.stream_logs(wf_id)` and
  `client.tasks.stream_logs(task_id)` yield server-sent events; the
  iterator stops when the source closes.
- **Pull artifacts** — `client.results.get(task_id)` for the result
  payload, `client.results.download_bundle(task_id, include="all")`
  for the tar.gz.
- **Fetch and analyze traces** — `client.traces.fetch(wf_id, "spans")`
  yields JSONL rows; `client.traces.analyze(wf_id)` returns a profile
  summary.
- **Cancel** — `client.workflows.cancel(wf_id)`.

## Cursor pagination

Cursor-enabled calls take `limit` and `before` / `after` params.
Cursors are opaque base64 strings — pass them through; do not parse
them.

## Errors

The SDK raises `flowmesh.FlowMeshError` (and a small set of subclasses
for auth / not-found / rate-limit) instead of returning HTTP error
shapes. Wrap your calls accordingly.
