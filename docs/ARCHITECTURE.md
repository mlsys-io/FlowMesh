# Architecture

FlowMesh is a service fabric for running LLM agentic workflows on
distributed GPU workers. The server parses a workflow (YAML / JSON / n8n),
turns it into a DAG of tasks, dispatches each task to a worker, and
collects results and artifacts.

## Workspace layout

The codebase is a **uv workspace** with these packages:

| Package | Path | Purpose |
|---------|------|---------|
| `flowmesh` (root) | `src/` | Server, Worker, shared schemas |
| `flowmesh-sdk` | `sdk/` | Public Python SDK |
| `flowmesh-sdk-stack` | `sdk/stack/` | Stack/node helpers |
| `flowmesh-cli` | `cli/` | Typer CLI (`flowmesh ...`) |
| `flowmesh-cli-stack` | `cli/stack/` | Stack deployment commands |
| `flowmesh-hook` | `hook/` | Plugin hook protocol interfaces |

Only the SDK, CLI, stack helper, hook, and lightweight `flowmesh`
metapackage distributions are published to PyPI. The runtime source under
`src/` is copied into server and worker images directly and is not included in
the published `flowmesh` wheel.

## Topology

```
Client (CLI / SDK / API) ──▶ Server (FastAPI orchestrator, :8000)
                                │
                                ├── Redis (control + telemetry pub/sub, log streams)
                                │
                                └─▶ Supervisor (per-node) ──gRPC──▶ Worker (executor)
```

The runtime is two top-level processes:

1. **Server** (`src/server/`) — FastAPI orchestrator at `:8000`. Hosts
   workflow / task / dispatch logic and the **Supervisor subsystem**
   (`src/server/supervisor/`), which manages per-node worker lifecycle,
   runs the worker-facing gRPC server (`:50051`), and drives the
   Docker / Vast.ai worker adapters.
2. **Worker** (`src/worker/`) — stateless executor. Connects to a
   supervisor via gRPC, receives tasks, runs the matching executor,
   reports results.

## Communication

- **server ↔ supervisor (same node)** — `multiprocessing.Queue`.
- **server ↔ supervisor (across nodes)** — Redis pub/sub.
- **supervisor ↔ worker** — bidirectional gRPC. Proto stubs at
  `src/shared/grpc/supervisor/v1/`.
- **client ↔ server** — REST.

## Object IDs

3-char prefixes: `wfl-` workflows, `tsk-` tasks, `ssn-` SSH sessions,
`scn-` SSH connection rows, `cmd-` supervisor commands. Always use
`new_*_id()` helpers in `src/shared/utils/ids.py`. Never use `uuid4()` or
`secrets.token_hex` for IDs.

## Task state machine

`PENDING → DISPATCHED → (DONE | FAILED | CANCELLED)`. Retried tasks
cycle back to `PENDING` until exhausted.

## Directory map

```
src/
  server/               FastAPI orchestrator
    auth/                 Helpers for calling plugins' auth and permission check hooks
    clients/              Client wrappers to connect to external services like Redis
    dispatcher/           Dispatch loop, worker selector, stage stickiness, context reuse
    governance/           Governance schemas and trace analysis
    hooks/                Plugin extension ABCs + registries
    main.py               Entrypoint, FLOWMESH_PLUGINS loader, EventMonitor wiring
    registries/           Worker / Node registries (Redis-backed)
    routers/v1/           workflows, tasks, results, workers, nodes, ssh, stack, system
    schemas/              REST API request and response schemas
    services/             monitoring, log streaming, ssh forwarding, runtime
    supervisor/           Per-node agent (gRPC server, adapters, lifecycle)
    task/                 parser, runtime, models, merge / epoch helpers
    utils/                concurrent, helpers, logging, misc, time
  shared/
    grpc/supervisor/v1/   Generated proto stubs (server + worker)
    schemas/              Cross-cutting schemas
    tasks/                Workflow/task spec models
    utils/                JSON, parsing, time, ids
  worker/
    docker/               Worker Dockerfiles (CPU + GPU)
    executors/            Executor implementations
      mixins/               data, governance, inference, training
      utils/                artifacts, checkpoints, data_utils, distributed,
                            graph_templates, huggingface, safe_eval
    runner.py             Task lifecycle (execute, write results, upload artifacts)
cli/                    Typer CLI (`flowmesh`)
hook/                   Plugin hook protocol interfaces
sdk/                    Public Python SDK
proto/                  gRPC service definition
templates/              Example workflow YAMLs
examples/               Sample artifacts
tests/{server,worker,shared,cli,sdk}/
scripts/dev/            compile_protos, sync_requirements, check_env_examples
```

## Key runtime behavior

- **Task merging.** Compatible adjacent tasks in a DAG (same `taskType`,
  model, hardware shape, and merge key) coalesce into a single dispatch.
  Merged children ride on `WorkerTaskMessage.merged_children`; the worker
  writes per-child results into `result.children`; the dispatcher fans
  out synthetic `TASK_SUCCEEDED` / `TASK_FAILED` events. Disable with
  `ENABLE_TASK_MERGE=false`.
- **Stage stickiness** (`ENABLE_STAGE_WEIGHT_STICKINESS=true`) — the
  dispatcher pins stages that reference an upstream stage's checkpoint
  to the worker that produced it, falling back to normal selection when
  unavailable or stale. Mostly relevant for training pipelines reusing
  on-disk checkpoints.
- **Context reuse.** Workers report cached models/datasets in their
  `WorkerHardware`. The dispatcher's `_cached_worker_candidates` filters
  to workers whose cache covers the task's references; entries older
  than `WORKER_CACHE_TTL_SEC` are ignored.
- **Cursor pagination.** List endpoints accept `limit` and `before` /
  `after` cursors. The cursor is an opaque base64 of `(timestamp, id)`;
  do not parse client-side.
- **Redis channels.** The runtime uses three namespaces:
  - `flowmesh:control:*` — control plane (task assignments,
    cancellations, worker lifecycle).
  - `flowmesh:telemetry:*` — telemetry (heartbeats, status updates).
  - `flowmesh:logs:task:{task_id}` and
    `flowmesh:logs:workflow:{wfl_id}` — log streams, bounded by
    `LOG_STREAM_MAXLEN_TASK` / `LOG_STREAM_MAXLEN_WORKFLOW` and
    expired `LOG_STREAM_TTL_SEC` after close.

## Plugin extension points

Server extension points are loaded via the `FLOWMESH_PLUGINS` env var.
Full contract, loader semantics, and a worked example live in
[`docs/PLUGINS.md`](PLUGINS.md).
