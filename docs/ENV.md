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

## Worker

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

## Supervisor

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_NAMESPACE` / `NODE_CLUSTER` / `NODE_ALIAS` | defaults | Identity |
| `NODE_TAGS` | `` | Scheduler hints (CSV) |
| `SUPERVISOR_GRPC_DISABLE_SERVER_TLS` | `false` | Local-only insecure gRPC |
| `SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS` | `true` | gRPC keepalive |
| `SUPERVISOR_GRPC_EXTERNAL_PORT` | – | External port (when port-forwarded) |
| `SERVER_GRPC_TLS_*` | – | TLS certificate files |
