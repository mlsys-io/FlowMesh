# FlowMesh Worker

FlowMesh workers communicate with a per-node **supervisor** that relays tasks,
telemetry, and results to the central server. Each worker opens a gRPC
connection to its supervisor, receives assignments, executes them via the
appropriate executor (Transformers, TRL, vLLM, RAG, agents, etc.), and
persists results plus optional artifacts to either shared storage or the
server via HTTP callbacks.

## Quick Start (local)
### 1. Install dependencies with uv
```bash
# Install uv if it is not already available
pip install uv

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Sync the worker runtime (baseline inference stack)
uv sync --group runtime-worker-core --group runtime-inference

# Optional executors:
# uv sync --group runtime-worker-cpu           # enable every CPU worker component
# uv sync --group runtime-worker-cpu --group runtime-worker-gpu  # add GPU deltas
```

### 2. Launch the worker
```bash
export SUPERVISOR_GRPC_TARGET="localhost:50051" # supervisor gRPC host:port
export RESULTS_DIR=./results
export FLOWMESH_BASE_URL="http://localhost:8000"  # required for HTTP artifact uploads
uv run python worker/main.py
```
At startup the worker:
1. Registers with its supervisor, which forwards the worker roster upstream
   to the server.
2. Streams heartbeats (load/power metrics) through the supervisor.
3. Listens for tasks delivered over the supervisor gRPC stream, selects the
   right executor, and writes outputs under `RESULTS_DIR/<task_id>/`.
4. Archives `final_model/` or `final_lora/` and uploads the bundle when the task
   requests HTTP artifact delivery.

## Environment variables
| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_TOKEN` | – | Worker token, used as a unique ID (required). |
| `SUPERVISOR_GRPC_TARGET` | – | Supervisor gRPC host:port (required). |
| `SERVER_GRPC_TLS_CA_B64` | - | Base64-encoded CA bundle for server TLS; when present, the worker uses a secure gRPC channel. |
| `RESULTS_DIR` | `./results` | Root directory for task outputs. |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Interval between heartbeats. |
| `WORKER_NAMESPACE` | flowmesh | Optional namespace associated with the worker. |
| `WORKER_CLUSTER` | cluster | Optional cluster associated with the worker. |
| `WORKER_CONTAINER_NAME` | – | Docker worker container name, used when available as the worker runtime identity. |
| `WORKER_ALIAS` | random hex | Override to pin a stable worker alias advertised to the orchestrator. |
| `WORKER_TAGS` | empty | Comma-separated tags used by the scheduler. |
| `LOG_LEVEL` | `INFO` | Worker log level. |
| `WORKER_COST_PER_HOUR` | `1.0` | Hourly cost in USD; reported with heartbeats. |
| `FLOWMESH_BASE_URL` | `http://localhost:8000` | Server URL used to build artifact download links and to hydrate cross-node SSH input bundles. |
| `MODEL_ARCHIVE_USE_PIGZ` | `1` | Enable multithreaded `pigz` compression (set `0`/`false` to disable). |
| `MODEL_ARCHIVE_COMPRESSION_LEVEL` | `6` | Gzip compression level (`0-9`). |
| `MODEL_ARCHIVE_PIGZ_THREADS` | – | Force a specific thread count for `pigz`; defaults to all CPUs. |
| `MODEL_ARCHIVE_PIGZ_BIN` | `pigz` | Path to the `pigz` binary. |
| `MODEL_ARCHIVE_TAR_BIN` | `tar` | Tar executable used before compression. |
| `WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC` | empty | Throttle HTTP uploads to emulate limited bandwidth. |
| `WORKER_HB_FILE` | – | Full path to the worker heartbeat file. |
| `WORKER_UPLOAD_RESULTS` | `false` | Whether the worker should always upload results to the server if spec.output.destination is unspecified. |

> The heartbeat TTL is computed automatically as `max(HEARTBEAT_INTERVAL_SEC * 4, 120)`.

> When a task sets `spec.output.destination.type: http`, configure
> `FLOWMESH_BASE_URL` so the worker can upload artifacts to
> `POST /api/v1/results/{task_id}/files` and expose the generated download link.

## Output directories
- Every task receives a dedicated subdirectory under `RESULTS_DIR`.
- Executors write their JSON summary to `<task_id>/results.json`.
- Training executors produce checkpoints and, when HTTP uploads are enabled,
  create `final_model.tar.gz` or `final_lora.tar.gz`.

## HTTP artifact workflow
1. Stage 1 declares `spec.output.destination.type: http`.
2. The worker keeps a local copy and uploads the archive to the orchestrator.
3. The orchestrator serves the bundle at
   `GET /api/v1/results/{task_id}/files/<archive>`.
4. Downstream stages reference the URL via `${stage.result.final_model_archive}`
   (or the LoRA equivalent).

Prefer shared storage? Point both the orchestrator and workers at the same
mount and disable HTTP uploads—the executors still emit artifacts locally and
templates can reference absolute paths.

## Debugging tips
- Inspect `worker.log` for executor output and upload diagnostics.
- Tail the server logs to confirm worker registration, gRPC connections, and forwarded telemetry.
- If a pipeline stalls, confirm the Stage 1 task produced the expected
  `final_*_archive` URL and that the orchestrator results directory contains the
  uploaded `.tar.gz`.

## SSH Tasks

Workers support `taskType: ssh` via `SSHExecutor`. When dispatched an SSH task
the worker:

1. Pulls the configured SSH session image (defaults: `flowmesh_ssh:latest-cpu` /
   `flowmesh_ssh:latest-gpu`).
2. Starts a dedicated Docker container running `sshd` with the task-provided
   `authorizedKeys` injected.
3. Mounts any requested upstream stage results read-only inside the container
   (`inputs`), and optionally a writable output path (`sshOutput`).
4. Emits a `TASK_UPDATE` event with SSH connection metadata:
   - **direct**: worker-published `host`/`port` (random host port for `22/tcp`)
   - **proxy/forward**: private `_relay_target` for the server relay path,
     plus `directHost`/`directPort` for optional `--direct` fallback
5. Blocks until the container exits, the TTL expires, or the task is cancelled.
6. Cleans up the SSH container and copies back `sshOutput` artifacts.

The SSH session container is labelled with `flowmesh.ssh.task_id` and
`flowmesh.ssh.worker_id` so the server can clean it up if the worker
container is stopped externally.

Relevant env vars for SSH tasks:

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_DEFAULT_IMAGE_CPU` | `flowmesh_ssh:latest-cpu` | Default SSH session image (CPU) |
| `SSH_DEFAULT_IMAGE_GPU` | `flowmesh_ssh:latest-gpu` | Default SSH session image (GPU) |
| `ENABLE_SSH_BY_DEFAULT` | `false` | Accept SSH tasks even without an explicit image in the spec |

## Task Merge & Multi-GPU
- **Task merge** lets the orchestrator coalesce duplicate inference/RAG
  requests. Workers receive a parent payload with `merged_children`, and
  executors (e.g. vLLM) emit per-child outputs under `result.children`; the
  runner writes each child result to its own directory. Disabled with
  `ENABLE_TASK_MERGE=false` on the server.
- **Multi-GPU execution**: when multiple GPUs are available, vLLM automatically
  sets `tensor_parallel_size`, and PPO/DPO/SFT executors launch distributed jobs
  via `torchrun`. Override `training.allow_multi_gpu=false` or
  `training.nproc_per_node` to constrain world size.
- **Bandwidth throttling**: set `WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC` to
  simulate limited HTTP throughput; the worker reports the value and delays
  callbacks accordingly.
