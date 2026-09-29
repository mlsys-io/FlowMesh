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

`taskType: api` sends one HTTP request per `spec.data` row, in parallel, and
returns one `APIResult.items` entry per row, in order. `spec.data` is required
and accepts `list`, `dataset`, `graph_template`, and `dataframe`; row metadata
and `dataframe` table grouping are not applied.

By default it routes to the Nebula endpoint and authenticates with the worker's `NEBULA_API_TOKEN`.

`spec.api.url` overrides the endpoint; when absent, the executor uses `NEBULA_API_BASE_URL` (appending `/v1/chat/completions`). `spec.api.headers` may supply an `Authorization` header directly.

Credential handling: a caller-supplied `Authorization` header is always used as-is and never overwritten. With no header, `NEBULA_API_TOKEN` is injected only when the call is on the Nebula url (no custom `spec.api.url`) — the Nebula token is never sent to a custom endpoint. A Nebula-path call with no token available fails closed.

`spec.api.retries` (default `0`, at most `10`) sets how many times a transient failure is retried before the task fails. A transient failure is a connection error or an HTTP status of 5xx, 408, or 429; other 4xx statuses are never retried. Retries back off exponentially: the first waits 1s and each later one doubles, capped at 60s. When a retryable response carries a `Retry-After` header (seconds or an HTTP date), that wait is used instead, also capped at 60s. Each retry logs a warning with the attempt count and the wait. A cancelled task stops retrying immediately.

```yaml
spec:
  taskType: api
  data:
    type: list
    items:
      - Hello
  api:
    method: POST
    headers:
      Content-Type: application/json
    body:
      model: gpt-4o
      messages:
        - role: user
          content: "{{prompt}}"
    retries: 3
    response:
      parse_json: true
```

### Per-row prompts

Each row's prompt replaces `{{prompt}}` in the request body; a value that is
exactly `{{prompt}}` takes the prompt as-is, so a message-list row fills
`messages`. `spec.api.concurrency` (default and maximum 8) bounds in-flight
requests. Any failed row fails the task. Cancelling the task skips rows that
have not started and marks it cancelled once in-flight requests return.

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

## Python task

`taskType: python` runs one function from `spec.code` in its own container on
the Docker session backend; a worker without Docker does not accept python
tasks. The function is called as `entrypoint(inputs)`, or with no arguments if
it takes none. `inputs` maps each input stage to the directory its result is
mounted at.

| Field | Meaning |
|-------|---------|
| `code` | Python source defining the entrypoint (at most 256 KiB). Passed verbatim: `${...}` in it is ordinary text, not a stage reference. |
| `entrypoint` | Function to call (default `main`). |
| `image` | Container image providing `python3` (default `python:3.12-slim`). |
| `requirements` | pip specifiers installed before the call. Requires `network: bridge`. |
| `network` | `none` (default) or `bridge`. |
| `inputs` | Upstream stages to mount, as for SSH tasks. When omitted, every direct dependency is mounted at `/mnt/flowmesh/inputs/<stage>`. |
| `timeoutSeconds` | Wall-clock limit (default 600, at most 3600). Reaching it fails the task. |
| `env` | Extra environment variables. |
| `emits` | Metric names the code must report; a missing one fails the task. |
| `pythonOutput.maxBytes` | Cap on the output directory. |
| `resources` | CPU, memory and GPU requests, capped by the worker's SSH limits. |

The return value must be JSON-serialisable. It is written to
`artifacts/result.json` and returned as `PythonResult.value`. Metrics come
from a `"metrics"` mapping in the return value and from any `metrics.json` the
code writes into `$FLOWMESH_OUTPUT`. Both are merged into
`artifacts/metrics.json` and `PythonResult.metrics`, and must be finite
numbers. The task succeeds only when the function returns and every promised
metric is present. An exception, a non-serialisable value, a timeout or a
memory kill fails it, and the code's own exception message becomes the task
error.

The code runs as uid 65534 with no effective capabilities, `/tmp` as its
writable scratch space, and no network unless `network: bridge` is set. It
sees GPUs only when `resources.hardware.gpu` asks for them.

```yaml
spec:
  stages:
    - name: prepare
      spec:
        taskType: echo
        data:
          type: list
          items: [the quick brown fox]
    - name: score
      dependsOn: [prepare]
      spec:
        taskType: python
        emits: [mean_words]
        code: |
          import json, os

          def main(inputs):
              path = os.path.join(inputs["prepare"], "results.json")
              items = json.load(open(path))["result"]["items"]
              words = [len(str(i["output"]).split()) for i in items]
              return {"metrics": {"mean_words": sum(words) / len(words)}}
```

See `examples/templates/python_two_stage.yaml` for a runnable workflow.

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
