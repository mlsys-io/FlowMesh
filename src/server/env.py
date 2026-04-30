import base64
import os
import tempfile
from pathlib import Path

from shared.utils import parse_bool_env, parse_float_env, parse_int_env

NODE_NAMESPACE: str = os.getenv("NODE_NAMESPACE") or "flowmesh"
NODE_CLUSTER: str = os.getenv("NODE_CLUSTER") or "cluster"
NODE_ALIAS: str = os.getenv("NODE_ALIAS") or "node"
NODE_TAGS: list[str] = [
    t.strip() for t in (os.getenv("NODE_TAGS") or "").split(",") if t.strip()
]
SERVER_BIND_HOST: str = "0.0.0.0"
SERVER_LOCAL_HOST: str = "localhost"
SERVER_HOST: str = os.getenv("SERVER_HOST") or SERVER_LOCAL_HOST
SERVER_APP_PORT: int = int(
    os.getenv("SERVER_APP_PORT")
    or os.getenv("SERVER_HTTP_PORT")
    or os.getenv("PORT")
    or "8000"
)
SERVER_GRPC_PORT: int = int(os.getenv("SERVER_GRPC_PORT") or "50051")
SERVER_TOKEN: str = os.getenv("SERVER_TOKEN") or ""

SERVER_GRPC_TLS_CA_FILE: str = (os.getenv("SERVER_GRPC_TLS_CA_FILE") or "").strip()
SERVER_GRPC_TLS_CERT_FILE: str = (os.getenv("SERVER_GRPC_TLS_CERT_FILE") or "").strip()
SERVER_GRPC_TLS_KEY_FILE: str = (os.getenv("SERVER_GRPC_TLS_KEY_FILE") or "").strip()

SUPERVISOR_GRPC_DISABLE_SERVER_TLS: bool = parse_bool_env(
    "SUPERVISOR_GRPC_DISABLE_SERVER_TLS", False
)
SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS: bool = parse_bool_env(
    "SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", True
)
SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS: int = parse_int_env(
    "SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS", 60000
)
SUPERVISOR_GRPC_EXTERNAL_PORT: int | None = parse_int_env(
    "SUPERVISOR_GRPC_EXTERNAL_PORT"
)

if SERVER_GRPC_TLS_CERT_FILE or SERVER_GRPC_TLS_KEY_FILE:
    if not (SERVER_GRPC_TLS_CERT_FILE and SERVER_GRPC_TLS_KEY_FILE):
        raise RuntimeError(
            "SERVER_GRPC_TLS_CERT_FILE and SERVER_GRPC_TLS_KEY_FILE are required"
        )
    if not SERVER_GRPC_TLS_CA_FILE:
        raise RuntimeError("SERVER_GRPC_TLS_CA_FILE is required for server TLS")
    ca_path = Path(SERVER_GRPC_TLS_CA_FILE)
    try:
        SERVER_GRPC_TLS_CA_B64 = base64.b64encode(ca_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise RuntimeError(f"Failed to read server TLS CA file: {exc}") from exc
else:
    SERVER_GRPC_TLS_CA_B64 = ""

FLOWMESH_BASE_URL: str = os.getenv("FLOWMESH_BASE_URL", "http://localhost:8000")
FLOWMESH_API_KEY: str = os.getenv("FLOWMESH_API_KEY", "")

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ACL_ENABLED = parse_bool_env("REDIS_ACL_ENABLED", False)
REDIS_USERNAME: str = os.getenv("REDIS_USERNAME", "admin")
REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
REDIS_CONTROL_URL: str = os.getenv("REDIS_CONTROL_URL", REDIS_URL)
REDIS_TELEMETRY_URL: str = os.getenv("REDIS_TELEMETRY_URL", REDIS_URL)
REDIS_TLS_CA_FILE: str = os.getenv("REDIS_TLS_CA_FILE", "").strip()

SERVER_HEARTBEAT_INTERVAL: int = int(os.getenv("SERVER_HEARTBEAT_INTERVAL") or "30")
SERVER_HEARTBEAT_TTL: int = max(SERVER_HEARTBEAT_INTERVAL * 4, 120)
ENABLE_SSH_BY_DEFAULT: bool = parse_bool_env("ENABLE_SSH_BY_DEFAULT", False)
SSH_DEFAULT_IMAGE: str | None = os.getenv("SSH_DEFAULT_IMAGE", "").strip() or None
SSH_DEFAULT_USER: str | None = os.getenv("SSH_DEFAULT_USER", "").strip() or None
SSH_DEFAULT_TTL_SEC: float | None = parse_float_env("SSH_DEFAULT_TTL_SEC")
SSH_DEFAULT_IDLE_SEC: float | None = parse_float_env("SSH_DEFAULT_IDLE_SEC")
SSH_MAX_TTL_SEC: float | None = parse_float_env("SSH_MAX_TTL_SEC")
SSH_POLL_INTERVAL_SEC: float | None = parse_float_env("SSH_POLL_INTERVAL_SEC")
SSH_STOP_TIMEOUT_SEC: float | None = parse_float_env("SSH_STOP_TIMEOUT_SEC")

LOG_FILE: str = os.getenv("LOG_FILE", "server.log")
LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 5_242_880))
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 5))
LOG_LEVEL: str = (
    os.getenv("SERVER_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
).upper()

FLOWMESH_REGISTRY: str = os.getenv("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io")
FLOWMESH_VERSION: str = os.getenv("FLOWMESH_VERSION", "latest")

WORKER_CONFIG_PATH: str = os.getenv("WORKER_CONFIG_PATH", "configs/worker_config.yaml")
CUDA_VISIBLE_DEVICES: str | None = os.getenv("CUDA_VISIBLE_DEVICES")
if CUDA_VISIBLE_DEVICES is not None:
    if CUDA_VISIBLE_DEVICES.strip().lower() == "all":
        CUDA_VISIBLE_DEVICES = None
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "flowmesh-node_results")
HF_CACHE_DIR: str | None = os.getenv("HF_CACHE_DIR") or None
PREDOWNLOAD_MODEL_LIST: str = os.getenv("PREDOWNLOAD_MODEL_LIST", "")
WORKER_TAGS: str = os.getenv("WORKER_TAGS", "")
WORKER_HB_DIR: str = os.getenv("WORKER_HB_DIR") or os.path.join(
    tempfile.gettempdir(), "flowmesh_worker_health"
)
WORKER_UPLOAD_RESULTS: bool = parse_bool_env("WORKER_UPLOAD_RESULTS", False)

VAST_SEARCH_LIMIT: int = int(os.getenv("VAST_SEARCH_LIMIT") or "10")
VAST_MAX_RETRIES: int = int(os.getenv("VAST_MAX_RETRIES") or "1")

NEBULA_API_BASE_URL: str = os.getenv("NEBULA_API_BASE_URL", "")
