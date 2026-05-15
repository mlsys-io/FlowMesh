# Environment variables (curated)

The canonical declared set lives in
`cli/stack/src/flowmesh_cli_stack/env_schema.py` and is mirrored to
`cli/stack/src/flowmesh_cli_stack/assets/.env.example`. Run
`uv run scripts/dev/check_env_examples.py --write` after schema edits.

The tables below curate the knobs you actually tune. Anything not
listed here is in `.env.example`.

## Server

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ROLE` | `root` | `root` deploys local Redis; `worker` skips it and connects to the root's Redis via the URLs below |
| `REDIS_CONTROL_URL` | `redis://localhost:6379/0` | Redis control channel. On worker nodes, must point at the root node's reachable Redis endpoint |
| `REDIS_TELEMETRY_URL` | `redis://localhost:6380/0` | Redis telemetry channel. On worker nodes, must point at the root node's reachable Redis endpoint |
| `DATABASE_URL` | – | Postgres connection string |
| `RESULTS_DIR` | `./results` | Server-side results directory |
| `SERVER_RESULTS_DIR` | `flowmesh_results` | Host-side directory/docker volume to mount at `RESULTS_DIR` in the server container |
| `WORKER_RESULTS_DIR` | `flowmesh_results` | Server-side directory/docker volume to mount to worker containers |
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
| `SERVER_CUDA_PROBE_IMAGE` | `nvidia/cuda:12.9.1-base-ubuntu24.04` | CUDA image the server runs briefly to query local GPU names/indices |
| `DOCKER_GPU_RUNTIME` | nvidia | Optional Docker runtime name for GPU probe/worker containers; leave empty unless the host requires a named runtime such as `nvidia` |
| `FLOWMESH_API_KEY` | – | Forwarded to spawned workers as their server-callback bearer |
| `LOG_LEVEL` | `INFO` | Server log level |

**Notes:**
- In Docker deployments, `SERVER_RESULTS_DIR` and `WORKER_RESULTS_DIR`
are the host directories or Docker volumes mounted into the server and
worker containers for storing and reading task results. For workflows
with a local output destination (`spec.output.destination.type="local"`)
that have downstream tasks, both variables must point to the same shared
directory or volume so the server can access the worker's task results.
Otherwise, downstream tasks that depend on upstream outputs will stall
in the dispatching loop indefinitely.
- When multiple deployments share one host, you can set `FLOWMESH_STACK_SUFFIX`
in `.env` to differentiate the deployments so that FlowMesh stack CLI does
not interfere with each other.
- `DOCKER_GPU_RUNTIME` defaults to `nvidia`. On hosts where Docker GPU access
works with `--gpus all` but fails with `--runtime=nvidia` (for example, DGX
Spark), set `DOCKER_GPU_RUNTIME=` in the stack env.

## Worker

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_TOKEN` | – | Auth token for supervisor gRPC |
| `SUPERVISOR_GRPC_TARGET` | – | Supervisor gRPC endpoint |
| `RESULTS_DIR` | `./results` | Task output directory |
| `WORKER_TAGS` | `` | Scheduler hints |
| `WORKER_COST_PER_HOUR` | `1.0` | Cost metadata |
| `WORKER_UPLOAD_RESULTS` | `false` | Upload results when no destination set |
| `HF_CACHE_DIR` | – | Shared HuggingFace cache mount |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Heartbeat cadence |

## Supervisor

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_NAMESPACE` / `NODE_CLUSTER` / `NODE_ALIAS` | defaults | Identity |
| `NODE_TAGS` | `` | Scheduler hints (CSV) |
| `SUPERVISOR_GRPC_DISABLE_SERVER_TLS` | `false` | Local-only insecure gRPC |
| `SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS` | `true` | gRPC keepalive |
| `SUPERVISOR_GRPC_EXTERNAL_PORT` | – | External port (when port-forwarded) |
| `SERVER_GRPC_TLS_*` | – | TLS certificate files |

## SSH session resource caps

When `enable_ssh` is true on a Docker worker, these configured
ceilings bound every SSH session container spawned by that worker.
Unset values mean unbounded (host-wide access).

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_MAX_CPU` | – | Max CPU cores per SSH container (float, e.g. `4` or `2.5`). Sets Docker `nano_cpus`. |
| `SSH_MAX_MEMORY` | – | Max memory per SSH container (e.g. `8Gi`, `512Mi`, or a byte count). Sets Docker `mem_limit`. |
| `SSH_MAX_PIDS` | – | Max PIDs per SSH container. Sets Docker `pids_limit`. Admin-only — not user-overridable. |

The effective CPU/memory limit is `min(spec.resources.hardware, worker
cap)`. A task that requests more than the worker cap is dispatched to
another worker if one has a larger cap; otherwise the dispatcher
follows its standard requeue/retry behavior. The worker logs a startup
warning if SSH is enabled with no cap configured.
