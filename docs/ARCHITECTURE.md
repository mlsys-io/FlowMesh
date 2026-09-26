# Architecture

FlowMesh is a service fabric for running LLM agentic workflows on
distributed GPU workers. The server parses a workflow (YAML / JSON / n8n),
turns it into a DAG of tasks, dispatches each task to a worker, and
collects results and artifacts.

## Workspace layout

The codebase is a **uv workspace** with these packages:

| Package | Path | Purpose |
|---------|------|---------|
| `flowmesh` (root) | `pyproject.toml` | Lightweight PyPI metapackage |
| `flowmesh-sdk` | `sdk/` | Public Python SDK |
| `flowmesh-sdk-stack` | `sdk/stack/` | Stack/node helpers |
| `flowmesh-cli` | `cli/` | Typer CLI (`flowmesh ...`) |
| `flowmesh-cli-stack` | `cli/stack/` | Stack deployment commands |
| `flowmesh-hook` | `hook/` | Plugin hook protocol interfaces |
| Runtime source | `src/` | Server, Worker, shared runtime modules |

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
- **supervisor ↔ worker** — gRPC. The worker opens every connection: it streams
  tasks down, pushes events and logs up, and opens a bidirectional `Relay`
  stream per relayed TCP connection. Proto stubs at
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

Retries are routed to a worker that has not already failed the task and
stop once every eligible worker has been tried or `max_attempts` is
reached; the terminal error is the executor's own message. Controlled
executor errors are not retried. A task that no worker can satisfy fails
after `TASK_NO_WORKER_GRACE_SEC`.

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
examples/               Workflow YAMLs, sample configs, plugin examples
tests/{server,worker,shared,cli,sdk}/
scripts/dev/            compile_protos, sync_requirements, check_env_examples
```

## Key runtime behavior

- **Task merging.** Compatible adjacent tasks in a DAG (same `taskType`,
  model, hardware shape, and merge key) coalesce into a single dispatch.
  Merged children ride on `WorkerTaskMessage.merged_children`; the worker
  writes per-child results into `result.children`; the dispatcher fans
  out synthetic `TASK_SUCCEEDED` / `TASK_FAILED` events. Disable with
  `ENABLE_TASK_MERGE=false`. When only some children of a batch fail, they
  fail individually with their dependents and the parent dispatches with its
  remaining valid children. The failed children are persisted before the
  parent's linkage update (children-first); durable state is per-workflow and a
  batch can span workflows, so this cannot always be one atomic write. A crash
  can then only leave a child durably failed while the parent still lists it as
  merged — reconciled on the parent's next dispatch — never a live child
  stranded under a parent that no longer lists it.
- **Stage stickiness** (`ENABLE_STAGE_WEIGHT_STICKINESS=true`) — the
  dispatcher pins stages that reference an upstream stage's checkpoint
  to the worker that produced it, falling back to normal selection when
  unavailable or stale. Mostly relevant for training pipelines reusing
  on-disk checkpoints.
- **Context reuse.** Workers report cached models/datasets in their
  `WorkerHardware`. The dispatcher's `_cached_worker_candidates` filters
  to workers whose cache covers the task's references; entries older
  than `WORKER_CACHE_TTL_SEC` are ignored.
- **Worker capabilities.** Beyond hardware fit, each worker advertises the set
  of task types it can service, and the dispatcher routes a task only to workers
  that advertise its type. A worker advertises a type only when its executor came
  up — e.g. SSH requires a session backend (a reachable Docker daemon, or an
  `sshd` binary on a root worker, which gives each session its own OS account),
  and training or omni types require their (often GPU-only) dependencies — so a
  worker missing that executor isn't a candidate, rather than being handed a
  task it would fail.
- **Foreign GPU occupancy.** A worker samples per-device GPU memory on each
  heartbeat and reports any device a process outside FlowMesh is holding. The
  worker stays `IDLE` and keeps taking CPU work; only the held devices leave the
  pool, filtered in `idle_satisfying_pool` and never in `hw_satisfies`, so the
  task waits for the card instead of failing as unschedulable. The worker
  refuses a GPU task that reaches it anyway, which reroutes rather than dying in
  executor init. A reading is only trusted when nothing of the worker's own is
  loaded — no task running, no GPU-using executor still warm, past
  `WORKER_FOREIGN_GPU_GRACE_SEC` — so a worker never gates itself on its own
  model. A suppressed reading keeps its last value; an unreadable NVML clears to
  "no opinion", since only a fresh clear reading releases a latch. Occupancy that
  arrives while a GPU executor is warm is therefore not detected until that
  executor unloads, which
  `WORKER_EXECUTOR_IDLE_CLEANUP_SEC` bounds. Each refusal also restarts the
  post-task grace window, so a worker being handed GPU work it keeps refusing
  takes longer to re-measure. An SSH task that declares no `gpu` block at all is
  handed the worker's whole device set, held cards included — occupancy filters
  which devices a *declared* request receives, not whether an undeclared session
  sees them. Disable with `WORKER_FOREIGN_GPU_GATE=false`.
- **Session relays are worker-initiated.** A `proxy` or `forward` session, and a
  proxied `serve` endpoint, are reached over a gRPC stream the *worker* opens to
  its supervisor, which bridges it to the Redis `up`/`down` streams the client
  is already reading. The supervisor never connects to a worker, so a worker
  behind NAT serves sessions like any other. An executor publishes its local
  port to the worker's endpoint registry when it starts listening; the
  supervisor names only that endpoint, never a host and port, and the worker
  refuses an id it has not published. Only `direct` mode needs the worker to be
  reachable. An endpoint no relay mode can carry is reported as `direct` at the
  address it listens on, and fails its task only when it published no address.
- **Stale worker reaping.** The watchdog deletes a dead worker's registry record
  (`WORKERS_SET_KEY` membership + `worker_key` hash) after it has been stale past
  `WORKER_REAP_GRACE_SEC`, so a worker that leaves without a clean `UNREGISTER` — a
  crash, or an `external` worker that re-enrolled under a new id — disappears instead of
  lingering as a permanent ghost. Live and briefly-disconnected workers are never reaped.
- **Worker and node identity.** A worker's alias is assigned by its
  supervisor: the supervisor passes it as `WORKER_ALIAS` to the workers it
  launches, an external worker reads it from its token, and registration
  records the alias the supervisor verified from the worker's token. Aliases
  are unique per node. A node takes a Redis lease on its `NODE_ALIAS` at registration
  (`nodes:alias:{alias}`), refreshes it with its heartbeat, and releases it on
  unregister; while another live node holds the alias, registration fails with
  `409`. A lease its holder has not refreshed for half its TTL can be taken
  over, so a crash-restarted node reclaims its alias. Taking the lease removes
  any other node record with the alias, so a node that crashed or was taken
  over does not linger beside its replacement. `(node_alias, alias)` is
  therefore a worker's durable, unique address.
- **External workers' hardware and GPUs.** The supervisor cannot probe a worker
  it did not launch, so an `external` worker's hardware is the report it sends
  at registration. If that report names GPUs of the supervisor's own host
  (matched by UUID, since a worker's device index is local to its container),
  the supervisor holds them out of its pool, so Docker workers aren't given
  them and the node's free GPU count excludes them. The hold lasts until the
  worker unregisters, is destroyed, or the supervisor stops, but not when the
  worker crashes: its orchestrator usually restarts it onto the same cards, so
  a crashed worker's hold is freed by destroying it. A card another worker
  already holds is shared with a warning, never refused, and returns to the
  pool only once every holder releases it. Holds live in the supervisor's
  memory: after a restart, the Docker workers in its config reserve first and
  an external worker claims its cards again when it re-registers. The report
  lists every GPU NVML shows the worker, and `CUDA_VISIBLE_DEVICES` does not
  narrow it, so limit an external worker's GPUs at the device level (the
  container's `--gpus`, a Kubernetes device plugin).
- **Worker cordon.** A cordoned worker keeps running and finishes what it was
  already dispatched, but is left out of both the idle pool and the eligibility
  set, so tasks neither go to it nor wait for it. The cordon is keyed on
  `(node_alias, alias)` and lasts until it is uncordoned, independent of any
  worker's lifecycle, so it also applies to a worker that registers under the
  key later.
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

## Service restarts

Any Compose service can be recreated in place with `flowmesh stack restart
[SERVICE ...]`, without a full teardown. The root server survives its own
restart without losing in-flight work: scheduling state is persisted to Redis
and rebuilt on startup (`TaskRuntime.rehydrate`), and task events replay from a
durable stream. Rolling a new image across the cluster one node at a time is one
application. See [`SERVICE_RESTARTS.md`](SERVICE_RESTARTS.md).

## Plugin extension points

Server extension points are loaded via the `FLOWMESH_PLUGINS` env var.
Full contract, loader semantics, and a worked example live in
[`docs/PLUGINS.md`](PLUGINS.md).
