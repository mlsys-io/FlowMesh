# AGENTS.md — FlowMesh

This file provides guidance to coding agents (Claude Code, Codex, Cursor, and
similar tools) working in this repository. It follows the cross-agent
`AGENTS.md` convention; agent-specific entry files (e.g. `CLAUDE.md`) at this
level simply import from here.

## Project Overview

FlowMesh is a service fabric for running LLM agentic workflows on distributed
GPU workers. It accepts workflow definitions (YAML, JSON, or n8n graph format),
parses them into a DAG of tasks, schedules and dispatches each task to a
suitable worker, and collects results and artifacts. It supports inference
(vLLM, HF transformers, diffusers), training (SFT, LoRA, DPO, PPO),
retrieval-augmented generation, agent execution, SSH-style interactive
sessions, and arbitrary container jobs.

The codebase is organized as a **uv workspace** with SDK and CLI packages:

| Package | Path | Purpose |
|---------|------|---------|
| `flowmesh` (root) | `src/` | Server, Worker, shared schemas |
| `flowmesh-sdk` | `sdk/` | Public Python SDK for the server API |
| `flowmesh-sdk-stack` | `sdk/stack/` | Stack and node helpers for local deployments |
| `flowmesh-cli` | `cli/` | Typer CLI entry point + core commands |
| `flowmesh-cli-stack` | `cli/stack/` | Stack deployment commands |

## Architecture

```
Client (CLI / SDK / API)
  │
  ▼  HTTP (default 8000)
Server  ─── FastAPI orchestrator
  │  - YAML / JSON / n8n workflow parsing
  │  - DAG dependency resolution, task scheduling
  │  - Dispatch: best-fit / first-fit / min-satisfying worker selection
  │  - Task merging, stage stickiness, context reuse, epoch scheduling
  │  - SSE log streaming, metrics recording
  │  - Redis (control + telemetry channels) for pub/sub and task state
  │
  ├──▶ Redis Pub/Sub ──▶ Supervisor(s) ──▶ gRPC ──▶ Worker(s)
  │                           │                       │
  │                           │  Worker lifecycle     │  Stateless GPU executors
  │                           │  Docker / Vast.ai     │  19 executor types
  │                           │  adapters             │
  │                           │                       │
  │                           ▼                       ▼
  │                      Worker Registry         Results / Artifacts
  │
  └──▶ Redis streams: logs, events, task queues
```

### Components

The runtime is two top-level processes:

1. **Server** (`src/server/`) — Central FastAPI orchestrator (default HTTP
   port 8000). Hosts workflow / task / dispatch logic and the **Supervisor
   subsystem** under `src/server/supervisor/`, which manages per-node worker
   lifecycle, runs the worker-facing gRPC server (default port 50051), and
   drives the Docker / Vast.ai worker adapters.

2. **Worker** (`src/worker/`) — Stateless executor process. Connects to a
   supervisor via gRPC, receives tasks, runs executors, reports results.

### Communication Protocols

- **Server ↔ Supervisor (within a node)**: `multiprocessing.Queue` between
  the parent server and its supervisor child process for command/response,
  plus `nodes:events` Redis pub/sub for telemetry.
- **Server ↔ Supervisor (across nodes)**: Redis pub/sub on
  `node:{id}:dispatch`, `node:{id}:cmds`, `nodes:events`, `nodes:responses`.
- **Supervisor ↔ Worker**: bidirectional gRPC (proto stubs at
  `src/shared/grpc/supervisor/v1/`).
- **Client ↔ Server**: REST API over HTTP.

### Prefer CLI and SDK

When interacting with FlowMesh, prefer the **CLI** (`flowmesh`) or **SDK**
(`flowmesh` Python package) over raw HTTP calls or shell scripts. The CLI and
SDK handle pagination, error formatting, and SSE streaming.

### Hook Plugin Extension Points

External integrations (auth, submission policy, usage tracking) plug into the
server through three protocol hooks defined in `src/server/hooks/`:

- `IdentityProvider` — resolve a bearer token to a `PrincipalContext`
  (iterated from `auth/security.py`). With no providers registered, auth is
  a no-op and `authenticate_api_key` returns a default admin principal.
- `SubmissionGuard` — pre-submit precondition (iterated from
  `routers/v1/workflows.py`).
- `UsageSink` — fan-out per-task usage rows after a task completes
  (iterated from `services/monitoring.py`). Typical consumers: billing,
  audit, observability.

A plugin is any Python module that exposes a top-level `install()`. The
server loads `FLOWMESH_PLUGINS` (comma-separated module names) inside its
FastAPI lifespan and treats `install()` as either:

- a sync function returning `None` — the plugin appends its adapters to the
  registries in `server.hooks` and returns; or
- an `@asynccontextmanager async def install()` — the plugin owns resources
  with a lifecycle (a SQLAlchemy engine, an HTTP client, a background task)
  that need teardown on server shutdown. The loader holds an
  `AsyncExitStack`, enters each ctx-manager `install()` on startup, and
  unwinds them in reverse order on shutdown.

Plugins live anywhere on `sys.path` — in-tree under `src/server/<name>/`,
sibling-mounted under `/app/src/<name>/`, or a pip-installable wheel. Core
never references plugin module names; each plugin self-filters internally.

OSS ships no DB itself. Plugins that need persistence bring their own engine
and manage it via the ctx-manager `install()` form. Example:

```python
# myorg_auth_plugin/__init__.py
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.hooks import IDENTITY_PROVIDERS


class _MyOrgAuth:
    name = "myorg.auth"
    def __init__(self, sessionmaker): self._sm = sessionmaker
    async def resolve(self, raw_token, logger):
        async with self._sm() as session:
            ...

@asynccontextmanager
async def install():
    engine = create_async_engine(os.environ["MYORG_DATABASE_URL"])
    IDENTITY_PROVIDERS.append(_MyOrgAuth(async_sessionmaker(engine)))
    try:
        yield
    finally:
        await engine.dispose()
```

## API Reference (Server: `http://localhost:8000`)

### Workflows

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/workflows` | Submit workflow (YAML `text/plain` or JSON; set `Workflow-Format: n8n` header for n8n) |
| POST | `/api/v1/workflows/validate` | Validate without executing |
| GET | `/api/v1/workflows` | List workflows (filterable) |
| GET | `/api/v1/workflows/{id}` | Get workflow details |
| GET | `/api/v1/workflows/{id}/logs` | Query logs (limit, before/after cursors) |
| GET | `/api/v1/workflows/{id}/logs/stream` | SSE log stream |
| POST | `/api/v1/workflows/{id}/cancel` | Cancel workflow |

Workflow statuses: `PENDING`, `DISPATCHED`, `FAILED`, `CANCELLED`, `DONE`

### Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/tasks` | List tasks (filter by workflow_id, status, task_type, assigned_worker) |
| GET | `/api/v1/tasks/{id}` | Task details |
| GET | `/api/v1/tasks/{id}/logs` | Query task logs |
| GET | `/api/v1/tasks/{id}/logs/stream` | SSE task log stream |

Task statuses: `PENDING`, `DISPATCHED`, `FAILED`, `CANCELLED`, `DONE`

### SSH

| Method | Path | Description |
|--------|------|-------------|
| WS | `/api/v1/ssh/tasks/{task_id}/proxy` | WebSocket SSH proxy for proxy-mode SSH tasks |
| GET | `/api/v1/ssh/connections` | List active server-audited SSH proxy/forward connections |
| TCP | `<SERVER_SSH_FORWARD_PUBLIC_HOST>:<forward_port>` | Raw TCP per-task forward-mode port |

Server policy toggles:
- `ENABLE_SERVER_SSH_PROXY` — WebSocket SSH proxy endpoint
- `ENABLE_SERVER_SSH_FORWARD` — TCP forward listener
- `ENABLE_SERVER_SSH_CONNECTION_AUDIT` — best-effort SSH connection audit

### Results

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/results` | Submit task result (worker → server) |
| GET | `/api/v1/results/{task_id}` | Get task result |
| GET | `/api/v1/results/{task_id}/bundle` | Download tar.gz bundle (`?include=results,artifacts,logs,all`) |
| POST | `/api/v1/results/{task_id}/files` | Upload artifact (multipart) |
| GET | `/api/v1/results/{task_id}/files/{filename}` | Download artifact |
| GET | `/api/v1/results/{task_id}/logs` | Download archived logs.jsonl |

### Workers

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/workers` | List workers (filter by alias, namespace, cluster, status, tags) |
| GET | `/api/v1/workers/{id}` | Worker details |

Worker statuses: `STARTING`, `IDLE`, `BUSY`, `STOPPING`, `STOPPED`

### Nodes (Supervisors)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/nodes` | List nodes |
| POST | `/api/v1/nodes/register` | Register a node |
| GET | `/api/v1/nodes/{id}` | Node details |
| GET | `/api/v1/nodes/{id}/workers` | List workers under a node |
| POST | `/api/v1/nodes/{id}/workers/register` | Register worker under node |
| POST | `/api/v1/nodes/{id}/workers/{name}/start` | Start worker |
| POST | `/api/v1/nodes/{id}/workers/{name}/stop` | Stop worker |

### Stack (local single-node lifecycle)

`/api/v1/stack/workers/...` — wraps node-registered workers with local-only
container lifecycle, used by `flowmesh stack worker {up,down,start,stop}`.

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Top-level health check |
| GET | `/api/v1/system/metrics` | System metrics snapshot |

## Object Identifiers

Object IDs use 3-character type prefixes:

- `wfl-` workflows
- `tsk-` tasks
- `ssn-` SSH session containers
- `scn-` SSH connection (server-side audit row)
- `cmd-` supervisor commands

ID factories live in `src/shared/utils/ids.py`. Always use `new_*_id()`
helpers; never use `uuid4()` or `secrets.token_hex` for IDs.

## SDK Usage

### Sync Client

```python
from flowmesh import FlowMesh

client = FlowMesh(base_url="http://localhost:8000", api_key="...")

# Submit a workflow
wf = client.workflows.submit_yaml(open("templates/echo_local.yaml").read())

# Watch progress
for ev in client.workflows.stream_logs(wf.workflow_id):
    print(ev.line)

# Inspect tasks
tasks = client.tasks.list(workflow_id=wf.workflow_id)
result = client.results.get(tasks[0].task_id)
```

### Async Client

```python
from flowmesh import AsyncFlowMesh

async with AsyncFlowMesh(base_url="...", api_key="...") as client:
    wf = await client.workflows.submit_yaml(...)
    async for ev in client.workflows.stream_logs(wf.workflow_id):
        ...
```

## CLI Commands

The CLI entry point is `flowmesh`. Top-level commands and command groups:

```
flowmesh info | health | logout
flowmesh workflow {submit, validate, list, info, watch, cancel, logs}
flowmesh task {list, info, watch, stop, logs}
flowmesh worker {list, info}
flowmesh node {list, info, worker}
flowmesh node worker {list, start, stop}
flowmesh ssh {connect, run, proxy, connections}
flowmesh result {fetch, download}
flowmesh system {metrics}
flowmesh stack {build, push, pull, pullall, up, down, restart, ps, logs}
flowmesh stack worker {up, start, stop, down, list, pull}
```

`flowmesh stack` (from `flowmesh-cli-stack`) builds Docker images, runs the
local Compose stack, and manages local-only worker containers. Core commands
work against any reachable FlowMesh server.

## Workflow YAML Format

### Single Task

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

### Multi-Stage DAG

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

### Graph DAG

`taskType: graph_template` — topology-aware multi-input prompts with parent
output substitution and validation. See
`src/worker/executors/utils/graph_templates.py` for the templating contract.

### Schedule Hints

Workflows can declare scheduling preferences via
`metadata.annotations.schedule_hint`:

- `epoch_groups: [[<task_name>, ...], ...]` — epoch-ordered execution; tasks
  in epoch `n` only dispatch after every task in epoch `n-1` succeeds.
- `schedule_in_epoch_order: true` — for dependent DAGs, prefer
  position-in-epoch tie-breaks during dispatch.

## Task Types & Executor Registry

The worker resolves `spec.taskType` against an executor registry in
`src/worker/runner.py`. Built-in executors:

| `taskType` | Executor | Use case |
|-----------|----------|----------|
| `echo` | `EchoExecutor` | Echo input back as result (smoke tests) |
| `inference` | `VLLMExecutor` / `TransformersExecutor` | LLM inference |
| `diffusion` | `DiffusersExecutor` | Image / video diffusion models |
| `omni_text2{audio,image,speech,general}` | `Omni*Executor` | Multimodal generation |
| `training` | `SFTExecutor` / `LoRASFTExecutor` / `DPOExecutor` / `PPOExecutor` | Fine-tuning |
| `rag` | `RAGExecutor` | Retrieval-augmented generation |
| `agent` | `AgentExecutor` | Tool-using LLM agent (utu / youtu-agent backend) |
| `data_profiling` | `DataProfilingExecutor` | DataFrame profiling |
| `data_retrieval` | `DataRetrievalExecutor` | DataFrame loading from sources |
| `ssh` | `SSHExecutor` | Interactive SSH session or non-interactive container job |

Helper utilities live in `src/worker/executors/utils/` (`artifacts`,
`checkpoints`, `data_utils`, `graph_templates`, `huggingface`, `safe_eval`).
Cross-cutting behavior is in `src/worker/executors/mixins/`
(`data`, `governance`, `inference`, `training`).

### Agent Executor (utu / youtu-agent)

Requires `UTU_LLM_TYPE`, `UTU_LLM_MODEL`, `UTU_LLM_BASE_URL`, `UTU_LLM_API_KEY`
to run. Optional: `SERPER_API_KEY`, `JINA_API_KEY` for search tools.

## Environment Variables

The canonical declared set lives in
`cli/stack/src/flowmesh_cli_stack/env_schema.py` and is mirrored to
`cli/stack/src/flowmesh_cli_stack/assets/.env.example`. Run
`uv run scripts/dev/check_env_examples.py --write` after schema edits.

### Server (selected)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_CONTROL_URL` | `redis://localhost:6379/0` | Redis control channel |
| `REDIS_TELEMETRY_URL` | `redis://localhost:6380/0` | Redis telemetry channel |
| `DATABASE_URL` | – | Postgres connection string |
| `RESULTS_DIR` | `./results` | Server-side results directory |
| `SERVER_HTTP_PORT` | `8000` | Public HTTP port |
| `SERVER_GRPC_PORT` | `50051` | Supervisor gRPC port |
| `ORCHESTRATOR_DISPATCH_MODE` | `adaptive` | Scheduler mode |
| `ORCHESTRATOR_WORKER_SELECTION` | `best_fit` | `best_fit`, `first_fit`, `min_satisfying` |
| `SCHEDULER_LAMBDA_INFERENCE` | `0.4` | Inference task weight |
| `SCHEDULER_LAMBDA_TRAINING` | `0.8` | Training task weight |
| `SCHEDULER_LAMBDA_OTHER` | `0.5` | Other-task weight |
| `SCHEDULER_SELECTION_JITTER` | `1e-3` | Tie-break jitter |
| `ENABLE_TASK_MERGE` | `true` | DAG-level task coalescing |
| `TASK_MERGE_MAX_BATCH_SIZE` | `4` | Max merged tasks per dispatch |
| `ENABLE_CONTEXT_REUSE` | `true` | Bias toward workers with cached models |
| `WORKER_CACHE_TTL_SEC` | `3600` | Cache metadata TTL |
| `ENABLE_STAGE_WEIGHT_STICKINESS` | `false` | Pin stages to checkpoint-producing workers |
| `ENABLE_WORKER_WATCHDOG` | `true` | Worker death detection |
| `WORKER_DEATH_GRACE_SEC` | `60` | Grace period before marking dead |
| `FLOWMESH_PLUGINS` | – | Comma-separated plugin module names |
| `LOG_LEVEL` | `INFO` | Server log level |

### Worker (selected)

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_TOKEN` | – | Auth token for supervisor gRPC |
| `SUPERVISOR_GRPC_TARGET` | – | Supervisor gRPC endpoint |
| `RESULTS_DIR` | `./results_workers` | Task output directory |
| `WORKER_TAGS` | `` | Scheduler hints |
| `WORKER_COST_PER_HOUR` | `1.0` | Cost metadata |
| `WORKER_UPLOAD_RESULTS` | `false` | Upload results when no destination set |
| `HF_CACHE_DIR` | – | Shared HuggingFace cache mount |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Heartbeat cadence |

### Supervisor (selected)

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_NAMESPACE` / `NODE_CLUSTER` / `NODE_ALIAS` | defaults | Identity |
| `NODE_TAGS` | `` | Scheduler hints (CSV) |
| `SUPERVISOR_GRPC_DISABLE_SERVER_TLS` | `false` | Local-only insecure gRPC |
| `SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS` | `true` | gRPC keepalive |
| `SUPERVISOR_GRPC_EXTERNAL_PORT` | – | External port (when port-forwarded) |
| `SERVER_GRPC_TLS_*` | – | TLS certificate files |

## Directory Structure

```
src/
  server/                      # FastAPI orchestrator
    auth/                      # OSS auth shim (no-op when IDENTITY_PROVIDERS is empty; otherwise delegates to the chain)
    clients/                   # Redis client(s)
    config.py                  # Server config + DispatchConfig
    db/                        # SQLAlchemy session factory and migrations (no-op in OSS)
    dispatcher/                # Dispatch loop, worker selector, stage stickiness, context reuse
    env.py                     # Centralized env reads
    hooks/                     # IdentityProvider / SubmissionGuard / UsageSink ABCs + registries
    main.py                    # Entrypoint, FLOWMESH_PLUGINS loader, EventMonitor wiring
    registries/                # Worker / Node registries (Redis-backed)
    routers/v1/                # workflows, tasks, results, workers, nodes, ssh, stack, system
    schemas/                   # Pydantic models for API + DB
    services/                  # monitoring, log streaming, ssh forwarding, runtime
    supervisor/                # Per-node agent (gRPC server, adapters, lifecycle)
      adapters/                # Docker / Vast.ai worker adapters
      services/                # gRPC server, command listener, task listener, relay
    task/                      # parser, runtime, models, merge / epoch helpers
  shared/
    schemas/                   # Cross-cutting schemas (NodeInfo, etc.)
    grpc/supervisor/v1/        # Generated proto stubs (used by server's supervisor subsystem + worker)
    tasks/                     # Workflow/task spec models
    utils/                     # JSON, parsing, time, ids
  worker/
    config.py                  # Worker config (loaded from env)
    docker/                    # Worker Dockerfiles (CPU + GPU)
    executors/                 # Executor implementations
      mixins/                  # data, governance, inference, training
      utils/                   # artifacts, checkpoints, data_utils, graph_templates, huggingface, safe_eval
    runner.py                  # Task lifecycle (execute, write results, upload artifacts)
    supervisor_client.py       # gRPC client to supervisor
cli/
  src/flowmesh_cli/            # Core CLI commands (flowmesh-cli)
  stack/src/flowmesh_cli_stack/   # Stack deployment commands (flowmesh-cli-stack)
sdk/
  src/flowmesh/                # Public SDK (flowmesh-sdk)
  stack/src/flowmesh_stack/    # Stack helpers (flowmesh-sdk-stack)
proto/
  supervisor/v1/supervisor.proto  # gRPC service definition
templates/                     # Example workflow YAMLs
tests/
  server/                      # Server-side unit tests
  worker/                      # Worker-side unit tests
  shared/                      # Shared schema tests
  cli/                         # CLI tests
  sdk/                         # SDK tests
scripts/dev/                   # Developer scripts (compile_protos.sh, sync_requirements.py, check_env_examples.py)
```

## Development Workflow

### Setup

```bash
pip install uv
uv sync --all-extras                  # All Python deps including dev tooling
uv run scripts/dev/compile_protos.sh  # Regenerate proto stubs (only when proto changes)
```

### Format / lint / type-check

```bash
uv run pre-commit run --all-files
```

Hooks: gitleaks, isort, black, ruff, codespell, mypy, sync_requirements,
check_env_examples (via `scripts/dev/check_env_examples.py`).

### Tests

```bash
uv run pytest tests/ --ignore=tests/worker/test_mp_executor_cleanup_gpu.py
```

The cleanup-gpu test requires NVIDIA hardware and isolated processes; skip it
in normal CI.

### Run locally

```bash
uv run flowmesh stack up                          # Server + Postgres + Redis + Supervisor
uv run flowmesh stack worker up cpu 1             # 1 CPU worker
uv run flowmesh stack worker up gpu --targets 0   # 1 GPU worker pinned to GPU 0
uv run flowmesh workflow submit templates/echo_local.yaml
```

### Docker

Build images locally:

```bash
uv run flowmesh stack build server
uv run flowmesh stack build flowmesh_worker_cpu flowmesh_worker_gpu
uv run flowmesh stack build flowmesh_ssh_cpu flowmesh_ssh_gpu
```

Always rebuild the affected worker image after changing executor code; stale
images silently mask code regressions.

## Code Style

- Python 3.12+ (see `[project]` in `pyproject.toml`).
- Top-level imports only; inline imports only to break circular imports.
- Prefer `typing.Any` over `object` in annotations; only use `object` when
  `Any` is semantically wrong.
- Use `# type: ignore[<error-code>]` only after exhausting alternatives. Never
  use a bare `# type: ignore`.
- Prefer `X | Y` over `typing.Union[X, Y]` and `X | None` over `typing.Optional[X]`.
- Don't write `from __future__ import annotations` unless strictly necessary;
  use `typing.Self` or quoted forward references instead.
- Avoid `hasattr` / `getattr` that bypass type checking. Use `isinstance`
  guards. Acceptable `getattr` uses: dynamic dispatch, providing a default,
  accessing untyped third-party APIs.
- Default to no comments. Comment only when the *why* is non-obvious. Names
  self-document.
- Don't add comments referencing the current task, fix, or callers ("used by
  X", "added for Y", "handles issue #123") — those rot as the codebase
  evolves.

### Security Rules (bandit-enforced)

CI runs `bandit` with no severity / confidence threshold. Every finding must
either have a source-level fix or a documented skip in `[tool.bandit]` in
`pyproject.toml`. Per-line `# nosec` is disallowed — silencing a finding
without a written rationale defeats the audit.

When writing new code, follow these rules so bandit stays green:

- **B113 (request timeouts)** — every `requests.get/post/...` call must pass
  `timeout=`. Hung connections are denial-of-service; no implicit defaults.
- **B202 (archive extraction)** — `tarfile.extractall(..., filter="data")` is
  required (Python 3.12+). For zipfile, iterate `infolist()`, validate that
  each member resolves under the destination, and extract per-member; never
  call `zipfile.extractall` on untrusted archives.
- **B310 (urlopen)** — don't use `urllib.request.urlopen`; use the `requests`
  library and validate the URL scheme (`http`/`https` only) before fetching.
- **B324 (insecure hash)** — `hashlib.md5(..., usedforsecurity=False)` is
  required when MD5 is used for cache keys / fingerprints. Never use MD5 for
  anything that crosses a security boundary.
- **B506 (yaml load)** — always use `yaml.safe_load`. `yaml.load(...,
  Loader=yaml.FullLoader)` is forbidden.
- **B607 (subprocess with partial path)** — prefer the vendored SDK
  (`pynvml`, `docker-py`, `GitPython`) over shelling out via `nvidia-smi` /
  `docker` / `git`. If shelling out is unavoidable, the absolute path must
  be provided.
- **B614 (torch.load)** — `torch.load(..., weights_only=True)` is required.
  Pickle deserialization is RCE waiting to happen.
- **B701 (jinja2 autoescape)** — `Environment(autoescape=select_autoescape())`
  is required. The default `False` is unsafe even for non-HTML templates.
- **B108 (hardcoded /tmp)** — use `tempfile.gettempdir()` or
  `tempfile.NamedTemporaryFile` for local-host work. The literal `"/tmp"` in
  Python source is forbidden; if a Linux container path is genuinely
  intended (e.g. an in-container sentinel), construct it from
  `PurePosixPath` segments rather than as a single string constant.

Skipped rules and the rationale for each are listed in `[tool.bandit]`.
When the rationale stops being true (e.g. a sandbox stops being a sandbox),
remove the skip and fix the call sites — don't widen the skip list silently.

## Commit Conventions

- Single-line subject in imperative mood; no body unless a non-obvious "why"
  is needed.
- Conventional prefixes: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`,
  `style:`, `test:`. Scope optional: `feat(server): ...`.
- Sign off (`--signoff`) is required for code coming from coding agents
  (Claude Code, Codex, Cursor, etc.).
- One logical change per commit. Don't batch unrelated changes.

## Key Patterns

### Task State Machine

`PENDING` → `DISPATCHED` → (`DONE` | `FAILED` | `CANCELLED`).
A retried task transitions back to `PENDING` until exhausted.

### Redis Channel Conventions

- `flowmesh:control:*` — control plane (task assignments, cancellations,
  worker lifecycle).
- `flowmesh:telemetry:*` — telemetry (heartbeats, status updates).
- `flowmesh:logs:task:{task_id}` — per-task log stream.
- `flowmesh:logs:workflow:{workflow_id}` — per-workflow log stream.

Stream lengths are bounded via `LOG_STREAM_MAXLEN_TASK` and
`LOG_STREAM_MAXLEN_WORKFLOW`. Streams expire `LOG_STREAM_TTL_SEC` after close.

### Cursor-Based Pagination

List endpoints (`/api/v1/workflows`, `/api/v1/tasks`, log queries) accept
`limit` and `before` / `after` cursors. The cursor is an opaque base64
encoding of `(timestamp, id)`; do not parse client-side.

### Task Merging

Compatible adjacent tasks in a DAG (same `taskType`, model, hardware shape,
and merge key) coalesce into a single dispatch. The merged children are
carried in `WorkerTaskMessage.merged_children`. The runtime hands the worker
a single message; the worker writes per-child results into a `children`
section of the response. The dispatcher then fans out synthetic
TASK_SUCCEEDED / TASK_FAILED events for each child.

Merge can be disabled with `ENABLE_TASK_MERGE=false`.

### Stage Stickiness

When `ENABLE_STAGE_WEIGHT_STICKINESS=true`, the dispatcher pins stages that
reference an upstream stage's checkpoint to the worker that produced it,
falling back to normal selection if that worker is unavailable or stale.
Mostly relevant for training pipelines where reusing the on-disk checkpoint
saves repeated downloads.

### Context Reuse / Cache Affinity

Workers report cached models and datasets via `WorkerHardware`. The
dispatcher's `_cached_worker_candidates` filters the candidate pool to
workers whose cache covers the task's model/dataset references. Stale cache
entries (older than `WORKER_CACHE_TTL_SEC`) are ignored.
