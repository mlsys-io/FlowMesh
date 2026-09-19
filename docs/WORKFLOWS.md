# Workflow YAML format

Workflows are submitted as YAML (or JSON) to `POST /api/v1/workflows`
(see [`docs/API.md`](API.md)). The `examples/templates/` directory contains
runnable examples for each shape; this page documents the spec
hierarchy and the cross-cutting features.

## Single task

```yaml
apiVersion: flowmesh/v1
kind: InferenceTask
metadata:
  name: hello-inference
spec:
  taskType: inference
  resources:
    hardware: { gpu: { type: any, count: 1 } }
  model:
    source: { type: huggingface, identifier: TinyLlama/TinyLlama-1.1B-Chat-v1.0 }
    vllm: { gpu_memory_utilization: 0.5 }
  data:
    type: list
    items:
      - - role: user
          content: What is the capital of France?
  inference: { max_tokens: 64, temperature: 0.0 }
  output:
    destination: { type: http }
```

## Multi-stage DAG

```yaml
apiVersion: flowmesh/v1
kind: Workflow
spec:
  stages:
    - name: extract
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Extract entities from: {{input}}"
    - name: summarize
      dependsOn: [extract]
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Summarize: {{extract.output}}"
```

`spec.stages[].dependsOn` declares the DAG edges; the dispatcher
schedules each stage once all of its dependencies are `DONE`.
Substitutions like `{{extract.output}}` are resolved against the
upstream stage's result.

## Graph DAG

`taskType: graph_template` — topology-aware multi-input prompts with
parent output substitution and validation. See
`src/worker/executors/utils/graph_templates.py` for the templating
contract.

## API task

`taskType: api` issues one HTTP request per row of `spec.data`, in parallel,
and returns the responses row-aligned in `APIResult.items`. A single request
is a one-row `spec.data`. `spec.data` is required, exactly as for the vLLM
executor; it supports the same data types (`list`, `dataset`, `graph_template`,
`dataframe`).

By default it routes to the Nebula endpoint and authenticates with the worker's `NEBULA_API_TOKEN`.

`spec.api.url` overrides the endpoint; when absent, the executor uses `NEBULA_API_BASE_URL` (appending `/v1/chat/completions`). `spec.api.headers` may supply an `Authorization` header directly.

Credential handling: a caller-supplied `Authorization` header is always used as-is and never overwritten. With no header, `NEBULA_API_TOKEN` is injected only when the call is on the Nebula url (no custom `spec.api.url`) — the Nebula token is never sent to a custom endpoint. A Nebula-path call with no token available fails closed.

`spec.api.retries` (default `0`) sets how many times a transient failure is retried before the task fails. A transient failure is a connection error or an HTTP status of 5xx, 408, or 429; other 4xx statuses are never retried. Each retry waits a fixed 1s backoff. A cancelled task stops retrying immediately.

```yaml
spec:
  taskType: api
  api:
    method: POST
    headers:
      Content-Type: application/json
    body:
      model: gpt-4o
      messages:
        - role: user
          content: Hello
    retries: 3
    response:
      parse_json: true
```

### Batching

When `spec.data` is present, the task batches: one request is issued per row,
and the results are returned row-aligned in `APIResult.items`. Each row's
prompt is substituted for the `{{prompt}}` placeholder in the request body.
Server-side stage references are `${...}`; `{{prompt}}` is a worker-side
per-row slot, so it is not touched by server-side resolution. A failure in any
row fails the whole task rather than shifting the remaining rows.
`spec.api.concurrency` bounds the number of in-flight requests and is capped
at 8 (the default); values above 8 are clamped down. Cancelling the task
prevents not-yet-started rows from issuing and marks the task cancelled once
in-flight requests return; a request already inside the HTTP call is not
interrupted.

```yaml
spec:
  taskType: api
  data:
    type: list
    items:
      - Explain vector databases
      - Explain attention
  api:
    method: POST
    body:
      model: gpt-4o
      messages:
        - role: user
          content: "{{prompt}}"
    response:
      parse_json: true
```

## data_retrieval: type lumid

`type: lumid` routes the retrieval through lumid-data-app (HTTP). Three
modes are supported; all require `lumid_data_url` and `lumid_data_token`.

`lumid_data_token` is the bearer forwarded to lumid-data-app (shared lum.id
auth). Set it to your lum.id PAT, or to a key from lumid-data-app's
`LUMID_API_KEYS` for local dev.

```yaml
# SQL mode — single rendered query per param row
data:
  type: lumid
  mode: sql
  lumid_data_url: "http://127.0.0.1:5101"
  lumid_data_token: "${LUMID_PAT}"   # your lum.id PAT, or a local dev key
  template: "SELECT symbol, close FROM demo.fact_ohlc_10m ORDER BY timestamp LIMIT 5"
  output_format: jsonl   # jsonl (default) or csv

# Agent mode — NL description dispatched to the data agent
data:
  type: lumid
  mode: agent
  lumid_data_url: "http://127.0.0.1:5101"
  lumid_data_token: "${LUMID_PAT}"
  description: "Retrieve the latest 10 OHLC rows for NVDA from the demo schema"
  schema_scope: demo
  max_steps: 20
  output_format: jsonl

# S3 Object mode — fetch raw blobs by key
data:
  type: lumid
  mode: s3
  lumid_data_url: "http://127.0.0.1:5101"
  lumid_data_token: "${LUMID_PAT}"
  template: "demo/unstructured/news-html/{slug}"
  params:
    - label: slug
      data:
        type: list
        items:
          - 2024-01-15-nvda-earnings.html
```

## Schedule hints

Workflows can declare scheduling preferences via
`metadata.annotations.schedule_hint`:

- `epoch_groups: [[<task_name>, ...], ...]` — epoch-ordered execution;
  tasks in epoch `n` only dispatch after every task in epoch `n-1`
  succeeds.
- `schedule_in_epoch_order: true` — for dependent DAGs, prefer
  position-in-epoch tie-breaks during dispatch.
