#!/usr/bin/env bash
# scripts/ci/run_local.sh — Run the full FlowMesh CI pipeline locally
#
# Mirrors the GitHub Actions CI workflow end-to-end so you can test without
# pushing to GitHub.  Requires: docker, docker compose v2, uv.
#
# Fully isolated from any running FlowMesh services:
#   - Host and supervisor ports are dynamically assigned (no fixed 8000/50051)
#   - Worker container name is scoped to the process PID
#   - Each run gets its own Docker network via compose project name
#
# Usage:
#   ./scripts/ci/run_local.sh [OPTIONS]
#
# Options:
#   --gpu               Run the GPU smoke test instead of the CPU integration test
#   --task-yaml PATH    Override the workflow YAML submitted to the host
#   --timeout SEC       Override E2E wait timeout (default: 120, GPU default: 300)
#   --no-clean          Skip the pre-run docker prune step
#   --no-build          Skip rebuilding the worker image (use cached)
#   --keep              Do not tear down services after the run
#   -h, --help          Show this help

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"

# ── Defaults ────────────────────────────────────────────────────────────────────────────────────────
PROJECT="ci-local-$$"
API_KEY="flm-ci-00000000000000000000000000000000"
GPU=false
TASK_YAML=""
TIMEOUT=""
DO_CLEAN=true
DO_BUILD=true
DO_TEARDOWN=true

WORKER_IMAGE_CPU="ci/flowmesh_worker:latest-cpu"
WORKER_IMAGE_GPU="ci/flowmesh_worker:latest-gpu"

# Populated in section 0; referenced in teardown.
WORKER_NAME=""
_WORKER_CFG=""
_COMPOSE_OVERRIDE=""
HOST_URL="http://localhost:8000"   # overwritten after dc up

# ── Argument parsing ───────────────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)         GPU=true;           shift ;;
    --task-yaml)   TASK_YAML="$2";    shift 2 ;;
    --timeout)     TIMEOUT="$2";      shift 2 ;;
    --no-clean)    DO_CLEAN=false;    shift ;;
    --no-build)    DO_BUILD=false;    shift ;;
    --keep)        DO_TEARDOWN=false; shift ;;
    -h|--help)     sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Colors ──────────────────────────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  _B='\033[0;34m' _G='\033[0;32m' _Y='\033[1;33m' _R='\033[0;31m' _N='\033[0m'
else
  _B='' _G='' _Y='' _R='' _N=''
fi
log()  { echo -e "${_B}[ci]${_N}  $*"; }
ok()   { echo -e "${_G}[ok]${_N}  $*"; }
warn() { echo -e "${_Y}[warn]${_N} $*"; }
fail() { echo -e "${_R}[FAIL]${_N} $*" >&2; }

# ── Compose helpers ────────────────────────────────────────────────────────────────────────────────────────
COMPOSE_FILES=(-f "$DOCKER_DIR/ci.compose.yml")
if $GPU; then
  COMPOSE_FILES+=(-f "$DOCKER_DIR/ci.worker.gpu.yml")
fi

dc() { docker compose -p "$PROJECT" "${COMPOSE_FILES[@]}" "$@"; }

# ── Teardown (trap runs on any exit) ──────────────────────────────────────────────────────────────────
_teardown() {
  local code=$?
  if ! $DO_TEARDOWN; then
    warn "Skipping teardown (--keep).  To clean up manually:"
    echo "  docker compose -p $PROJECT ${COMPOSE_FILES[*]} down -v --remove-orphans"
    return
  fi

  log "Tearing down..."

  # Always dump service logs before removal — essential for diagnosing failures.
  echo
  log "Supervisor logs (last 40 lines):"
  dc logs supervisor --tail=40 2>/dev/null || true
  echo

  if [[ -n "$WORKER_NAME" ]]; then
    log "Worker logs ($WORKER_NAME):"
    docker logs "$WORKER_NAME" 2>&1 | tail -60 || true
    echo
  fi

  # Ask supervisor to stop managed workers gracefully.
  dc exec -T supervisor \
    curl -sf -X DELETE http://localhost:8001/api/v1/workers \
    -H "Authorization: Bearer $API_KEY" 2>/dev/null || true
  sleep 3

  docker rm -f "$WORKER_NAME" 2>/dev/null || true
  dc down -v --remove-orphans 2>/dev/null || true

  # Worker image is intentionally kept: the next build overwrites the tag in-place,
  # so there is always exactly one cached image available for --no-build runs.
  docker image prune -f >/dev/null
  docker volume prune -f >/dev/null

  # Clean up isolation temp files.
  rm -f "${_WORKER_CFG:-}" "${_COMPOSE_OVERRIDE:-}" 2>/dev/null || true

  if [[ $code -eq 0 ]]; then
    ok "Local CI run PASSED"
  else
    fail "Local CI run FAILED (exit $code)"
  fi
}
trap _teardown EXIT

# ── 0. Resolve defaults ──────────────────────────────────────────────────────────────────────────────────────────────────────
if $GPU; then
  WORKER_NAME="ci-worker-gpu-$$"
  WORKER_IMAGE="$WORKER_IMAGE_GPU"
  WORKER_DOCKERFILE="src/worker/docker/Dockerfile.cuda"
  [[ -z "$TIMEOUT" ]] && TIMEOUT=300
  # If --task-yaml was given, run only that one; otherwise run the full GPU suite.
  if [[ -n "$TASK_YAML" ]]; then
    GPU_TASK_YAMLS=("$TASK_YAML")
  else
    GPU_TASK_YAMLS=(
      "$REPO_ROOT/templates/inference_vllm_tiny.yaml"
      "$REPO_ROOT/templates/echo_three_node_graph.yaml"
      "$REPO_ROOT/templates/dag_inference_example.yaml"
      "$REPO_ROOT/templates/conditional_echo_test.yaml"
      "$REPO_ROOT/templates/inference_hf_tiny.yaml"
      "$REPO_ROOT/templates/lora_sft_llama.yaml"
      "$REPO_ROOT/templates/ssh_noninteractive.yaml"
      "$REPO_ROOT/templates/n8n/dag_inference.json"
    )
  fi
else
  WORKER_NAME="ci-worker-cpu-$$"
  WORKER_IMAGE="$WORKER_IMAGE_CPU"
  WORKER_DOCKERFILE="src/worker/docker/Dockerfile.cpu"
  [[ -z "$TASK_YAML" ]] && TASK_YAML="$REPO_ROOT/templates/echo_local.yaml"
  [[ -z "$TIMEOUT"   ]] && TIMEOUT=120
fi

cd "$REPO_ROOT"

# ── 0b. Create isolation artifacts ───────────────────────────────────────────────────────────────────────────────────────────────────────────
# Worker config: project-scoped alias prevents container name clashes when a
# second local CI run or a dev worker with the same name is already running.
_WORKER_CFG="$(mktemp /tmp/ci-worker-cfg-XXXXXX.yml)"
if $GPU; then
  sed "s/ci-worker-gpu/$WORKER_NAME/g" \
    "$DOCKER_DIR/ci.gpu_worker_config.yaml" > "$_WORKER_CFG"
else
  cat > "$_WORKER_CFG" <<EOF
default_worker_config:
  hb_interval: 30
workers:
  - provider: docker
    init_on_start: true
    worker_config:
      worker_alias: $WORKER_NAME
      worker_type: cpu
      enable_ssh: true
EOF
fi

# Compose override: host port is dynamic (avoids silently hitting a production
# host on 8000).  Supervisor gRPC stays on fixed 50051 — workers are spawned with
# SUPERVISOR_GRPC_TARGET=localhost:50051 and cannot follow a random port.
# If 50051 is already taken, dc up fails loudly at startup.
_COMPOSE_OVERRIDE="$(mktemp /tmp/ci-compose-override-XXXXXX.yml)"
cat > "$_COMPOSE_OVERRIDE" <<EOF
services:
  host:
    ports:
      - "127.0.0.1::8000"
  supervisor:
    ports:
      - "50051:50051"
    volumes:
      - $_WORKER_CFG:/etc/supervisor/worker_config.yaml:ro
    environment:
      WORKER_DOCKER_NETWORK: ${PROJECT}_ci-net
EOF
COMPOSE_FILES+=(-f "$_COMPOSE_OVERRIDE")

log "Project  : $PROJECT"
log "Worker   : $WORKER_NAME"
log "GPU mode : $GPU"
if $GPU; then
  for _y in "${GPU_TASK_YAMLS[@]}"; do log "YAML     : $_y"; done
else
  log "YAML     : $TASK_YAML"
fi
log "Timeout  : ${TIMEOUT}s"
echo

# ── 1. Pre-clean ────────────────────────────────────────────────────────────────────────────────────────────────────────
if $DO_CLEAN; then
  log "Pre-cleaning stale containers and build cache..."
  # Remove leftover worker containers from previous crashed runs of this script.
  docker ps -a --format '{{.Names}}' \
    | grep -E '^ci-worker-(cpu|gpu)-[0-9]+$' \
    | xargs -r docker rm -f 2>/dev/null || true
  # Tear down any stale ci-local-* compose stacks (e.g. from a disconnected SSH session).
  docker ps -a --format '{{.Labels}}' \
    | grep -oP 'com\.docker\.compose\.project=ci-local-\d+' \
    | sort -u \
    | sed 's/com\.docker\.compose\.project=//' \
    | xargs -r -I{} docker compose -p {} -f "$DOCKER_DIR/ci.compose.yml" down -v --remove-orphans 2>/dev/null || true
  docker image prune -f  >/dev/null
  docker volume prune -f >/dev/null
  docker builder prune -f --keep-storage 5gb 2>/dev/null \
    || docker builder prune -f --filter "until=72h" 2>/dev/null \
    || true
fi

# ── 2. Build worker image ────────────────────────────────────────────────────────────────────────────────────────────────────────────
if $DO_BUILD; then
  log "Building worker image ($WORKER_IMAGE)..."
  DOCKER_BUILDKIT=1 docker build \
    -f "$WORKER_DOCKERFILE" \
    -t "$WORKER_IMAGE" \
    .
  ok "Worker image built"
else
  if ! docker image inspect "$WORKER_IMAGE" >/dev/null 2>&1; then
    fail "--no-build specified but image '$WORKER_IMAGE' not found locally."
    fail "Run without --no-build first, or: docker build -f $WORKER_DOCKERFILE -t $WORKER_IMAGE ."
    exit 1
  fi
  log "Using cached worker image: $WORKER_IMAGE"
fi

# ── 3. Build & start services ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# --wait blocks until every healthcheck passes.
log "Starting services (redis × 2, postgres, host, supervisor)..."
if ! DOCKER_BUILDKIT=1 dc up -d --build --wait; then
  fail "Services failed to start — supervisor logs:"
  dc logs supervisor --tail=60 2>/dev/null || true
  exit 1
fi
ok "All services healthy"

# ── 4. Resolve the dynamically assigned host port ─────────────────────────────────────────────────────────────────────────────────────
# docker compose port returns 0.0.0.0:0 for 127.0.0.1-only bindings; use docker port instead.
HOST_PORT=$(docker port "$(dc ps -q host)" 8000/tcp \
  | grep '127.0.0.1:' | awk -F: '{print $NF}' | head -1)
HOST_URL="http://localhost:$HOST_PORT"
log "Host bound to $HOST_URL"

curl -sf "$HOST_URL/healthz" >/dev/null \
  || { fail "Host not reachable at $HOST_URL"; dc logs host --tail=40; exit 1; }
ok "Host healthy at $HOST_URL"

# ── 5. Confirm supervisor ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Supervisor has start_period:15s; retry up to ~45s to let it fully start.
SUPERVISOR_OK=false
for i in $(seq 1 9); do
  if dc exec -T supervisor curl -sf http://localhost:8001/healthz >/dev/null 2>&1; then
    SUPERVISOR_OK=true; break
  fi
  echo "  supervisor attempt $i/9"
  sleep 5
done
if ! $SUPERVISOR_OK; then
  fail "Supervisor never became healthy"
  dc logs supervisor --tail=40 || true
  exit 1
fi
ok "Supervisor healthy"

# ── 6. Debug snapshot ────────────────────────────────────────────────────────────────────────────────────────────────────────────
echo
log "Container state:"
dc ps
echo
log "Supervisor logs (last 20 lines):"
dc logs supervisor --tail=20
echo

# ── 7. Wait for worker to register ───────────────────────────────────────────────────────────────────────────────────────────────────────────
log "Waiting for worker to register with host..."
REGISTERED=false
for i in $(seq 1 24); do
  RESP=$(curl -sf \
    -H "Authorization: Bearer $API_KEY" \
    "$HOST_URL/api/v1/workers" 2>/dev/null || echo "CURL_FAILED")
  if echo "$RESP" | grep -qE '"worker_id"|"id":|"wkr-'; then
    REGISTERED=true
    break
  fi
  echo "  attempt $i/24 — $RESP"
  sleep 5
done

if ! $REGISTERED; then
  fail "Worker never registered.  Supervisor + worker logs:"
  dc logs supervisor --tail=40 || true
  docker logs "$WORKER_NAME" 2>&1 | tail -40 || true
  exit 1
fi
ok "Worker registered"

# ── 8. Run E2E smoke test(s) ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
echo
log "Running E2E smoke test(s)..."
log "  HOST=$HOST_URL"

if $GPU; then
  YAML_LIST=("${GPU_TASK_YAMLS[@]}")
else
  YAML_LIST=("$TASK_YAML")
fi

for _YAML in "${YAML_LIST[@]}"; do
  log "  → $(basename "$_YAML")"
  FLOWMESH_HOST_URL="$HOST_URL" \
  FLOWMESH_API_KEY="$API_KEY" \
  TASK_YAML="$_YAML" \
  E2E_TIMEOUT_SEC="$TIMEOUT" \
    uv run --with pytest --with pytest-asyncio --with requests \
      pytest tests/integration/test_e2e.py -v -s
done

# ── 9. Verify worker execution evidence ───────────────────────────────────────────────────────────────────────────────────────────────────────
echo
log "Verifying worker execution evidence..."
LOG_FILE="/tmp/flowmesh-local-worker-$$.log"
docker logs "$WORKER_NAME" 2>&1 | tee "$LOG_FILE" || true

if grep -qiE "executor|running task|dispatched|echo|inference|succeeded|TASK_SUCCEEDED|done" "$LOG_FILE"; then
  ok "Worker executed and completed the task"
else
  fail "No task execution evidence found in worker logs ($LOG_FILE)"
  exit 1
fi

if $GPU; then
  echo
  log "GPU utilisation during test:"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    || warn "nvidia-smi not available"
fi

echo
ok "All checks passed"
