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
| `TASK_NO_WORKER_GRACE_SEC` | `60` | Grace before failing a task no worker can satisfy |
| `TASK_STAGE_RESULT_GRACE_SEC` | `120` | Grace after an upstream stage finishes for its result to reach the server before a dependent that reads it fails |
| `TASK_RESULT_DELIVERY` | `true` | Have workers publish what dependent stages need to the server; set `false` only when every worker shares the server's results directory |
| `ENABLE_WORKER_WATCHDOG` | `true` | Worker death detection |
| `WORKER_DEATH_GRACE_SEC` | `60` | Grace period before marking dead |
| `WORKER_REHYDRATION_GRACE_SEC` | `120` | Extra grace for a worker's rehydrated in-flight tasks after the root restarts, before the watchdog may reclaim them |
| `ENABLE_WORKER_REAPER` | `true` | Delete a worker's registry record once it has been dead past the reap grace; requires `ENABLE_WORKER_WATCHDOG` |
| `WORKER_REAP_GRACE_SEC` | `900` | Grace after a worker is marked dead before its registry record is deleted. Lowering it below a few minutes removes the margin that protects an idle worker across a root restart |
| `FLOWMESH_PLUGINS` | – | Comma-separated plugin module names |
| `FLOWMESH_PLUGIN_DATA_DIR` | `./plugin-data` | Writable mount at `/app/plugin-data` for plugin state. A path -> host bind-mount (auto-created); a bare name -> external Docker volume of that name. |
| `SERVER_CUDA_PROBE_IMAGE` | `nvidia/cuda:12.9.1-base-ubuntu24.04` | CUDA image the server runs briefly to query local GPU names/indices |
| `DOCKER_GPU_RUNTIME` | nvidia | Optional Docker runtime name for GPU probe/worker containers; leave empty unless the host requires a named runtime such as `nvidia` |
| `FLOWMESH_API_KEY` | – | Forwarded to spawned workers as their server-callback bearer |
| `ENABLE_PERSISTENT_PORT_FORWARD` | `true` | Keep port-forward listeners bound between task sessions; disable to bind listeners only for active sessions |
| `ENABLE_SERVER_SSH_PROXY` | `true` | Enable the WebSocket proxy for interactive SSH tasks |
| `ENABLE_SERVER_SERVE_PROXY` | `true` | Enable the HTTP reverse proxy for `serve` tasks |
| `LOG_LEVEL` | `INFO` | Server log level |

**Notes:**
- In Docker deployments, `SERVER_RESULTS_DIR` and `WORKER_RESULTS_DIR`
are the host directories or Docker volumes mounted into the server and
worker containers for storing and reading task results. When workers share
the server's results directory, both variables point to the same directory or
volume, and workers upload nothing for dependent stages.
With `TASK_RESULT_DELIVERY=false`, sharing that directory is the only way an
upstream result reaches the server; a dependent fails once
`TASK_STAGE_RESULT_GRACE_SEC` has passed without it.
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
| `WORKER_UPLOAD_RESULTS` | `false` | Publish every result and artifact to FlowMesh, independently of user output destinations |
| `WORKER_RESULT_TRANSFER_TIMEOUT_SEC` | `1800` | Seconds a worker waits for the server while publishing a result or fetching an upstream one, including the time the server spends packing a large bundle |
| `WORKER_UPLOAD_RETRIES` | `5` | Additional upload attempts after retryable failures; `0` disables retries |
| `WORKER_UPLOAD_BACKOFF_SEC` | `2` | First wait between upload retries, doubled per retry up to 30 s; a `Retry-After` header takes precedence |
| `WORKER_EXECUTOR_IDLE_CLEANUP_SEC` | `60` | Seconds a worker waits before unloading an idle executor to release the resources it holds; higher values avoid reload thrash between tasks but keep those resources reserved while idle |
| `WORKER_FOREIGN_GPU_GATE` | `true` | Report a GPU as unavailable while a process outside FlowMesh is using it |
| `WORKER_FOREIGN_GPU_MEM_MIB` | `1024` | Foreign GPU-memory threshold in MiB |
| `WORKER_FOREIGN_GPU_CONSECUTIVE` | `2` | Consecutive readings required before a device changes availability |
| `WORKER_FOREIGN_GPU_GRACE_SEC` | `90` | Seconds to wait after a task ends before trusting a reading |
| `HF_CACHE_DIR` | – | Shared HuggingFace cache mount |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Heartbeat cadence |
| `SERVE_DEFAULT_TTL_SEC` | `3600` | Default vLLM serve session TTL when `spec.ttlSeconds` is unset |
| `SERVE_MAX_TTL_SEC` | `86400` | Upper bound on vLLM serve session TTL, regardless of `spec.ttlSeconds` |

Upload retries cover results, artifact files, trace files, system delivery
bundles, and the system delivery preflight check. FlowMesh uploads retry on
connection errors, timeouts, and 5xx/408/429 responses. External destinations
retry only when a connection could not be established; HTTP responses and
failures after connecting are returned without another attempt. A valid
`Retry-After` delay overrides the exponential backoff and its 30 s cap.

Workers publish what dependent stages need to the server at
`FLOWMESH_BASE_URL` unless the server sets `TASK_RESULT_DELIVERY=false`.
`WORKER_UPLOAD_RESULTS=true` publishes every result and artifact regardless, so clients can retrieve leaf results from a worker that
does not share the server's results directory. Publishing is best-effort: a
failed transfer leaves the result on the worker. `MODEL_CLEANUP_AFTER_UPLOAD`
keeps training artifacts that dependent stages need.

## Supervisor

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_NAMESPACE` / `NODE_CLUSTER` | defaults | Identity |
| `NODE_ALIAS` | `node` | Node alias; unique among live nodes |
| `NODE_TAGS` | `` | Scheduler hints (CSV) |
| `SUPERVISOR_GRPC_DISABLE_SERVER_TLS` | `false` | Local-only insecure gRPC |
| `SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS` | `true` | gRPC keepalive |
| `SUPERVISOR_GRPC_EXTERNAL_PORT` | – | External port (when port-forwarded) |
| `SERVER_GRPC_TLS_*` | – | TLS certificate files |

## SSH session backend

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_SESSION_BACKEND` | `auto` | Sandbox a session runs in: `docker` (sibling container), `process` (sshd inside the worker), or `auto` — `docker`, falling back to `process`. |
| `ENABLE_UNISOLATED_SSH_SESSION` | `false` | Whether a worker that cannot give a session its own OS account may still serve one. |
| `SSH_NETWORK_NAME` | set by the supervisor | Isolated Docker network sessions join for network access. A python task with `network: bridge` runs only on a worker that has one. |
| `SSH_DIRECT_HOST` | – | Address a `direct` session is advertised at. Unset, the worker uses its tailnet address, else its FQDN. Only `direct` uses it: a relayed session is reached over a stream the worker opens. |

A `proxy` or `forward` session is reached over a relay the worker opens to its
supervisor, so its sshd binds loopback and the worker needs no inbound
reachability.

`process` serves one interactive session per worker and ignores `spec.image`.
Only a root worker can give a session its own account, so `process` is
unavailable on any other worker unless `ENABLE_UNISOLATED_SSH_SESSION` is set —
without an account of its own a session runs as the worker and can read its
environment and credentials. `SSH_MAX_CPU` / `SSH_MAX_MEMORY` / `SSH_MAX_PIDS` /
`SSH_MAX_DISK`, the `ENABLE_SSH_GPU_LIMIT` subset, and `ssh -L` forwarding have
no effect in `process` mode.

## SSH session resource caps

When `enable_ssh` is true on a Docker worker, these configured
ceilings bound every session container spawned by that worker: SSH
sessions and python tasks alike.
Unset values mean unbounded (host-wide access).

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_MAX_CPU` | – | Max CPU cores per session container (float, e.g. `4` or `2.5`). Sets Docker `nano_cpus`. |
| `SSH_MAX_MEMORY` | – | Max memory per session container (e.g. `8Gi`, `512Mi`, or a byte count). Sets Docker `mem_limit`. |
| `SSH_MAX_PIDS` | – | Max PIDs per session container. Sets Docker `pids_limit`. Admin-only — not user-overridable. |
| `SSH_MAX_DISK` | – | Max bytes a session may write inside its container outside its tmpfs and bind mounts (e.g. `20Gi`). |
| `ENABLE_SSH_GPU_LIMIT` | `true` | When `true`, expose only the GPU subset matching the spec (`count` / `type` / `memory`); otherwise expose all worker GPUs. |

The effective CPU/memory limit is `min(spec.resources.hardware, worker
cap)`. A task that requests more than the worker cap is dispatched to
another worker if one has a larger cap; otherwise the dispatcher
follows its standard requeue/retry behavior. The worker logs a startup
warning if SSH is enabled with no cap configured.

## SSH session lifetime

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_DEFAULT_TTL_SEC` | `3600` | Session TTL when `spec.ttlSeconds` is unset |
| `SSH_MAX_TTL_SEC` | `28800` | Upper bound on session TTL, including a python task's `timeoutSeconds` |
| `SSH_DEFAULT_IDLE_SEC` | `900` | Idle timeout when `spec.idleTimeoutSeconds` is unset |

An interactive session is stopped once it has had no established SSH
connection for its idle timeout, which is clamped to the TTL. The idle
clock starts when the session does, so a session nobody ever connects
to is reaped too; set `spec.idleTimeoutSeconds: 0` to disable idle
reaping and rely on the TTL alone. Idle reaping does not apply to
non-interactive tasks, which hold no SSH connection by design.
