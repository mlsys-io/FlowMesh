import os
import re
from collections.abc import Mapping
from pathlib import Path

from flowmesh import FlowMesh
from flowmesh_cli.core.assets import asset_path
from flowmesh_stack.env import load_env
from flowmesh_stack.node_client import NodeClient
from flowmesh_stack.paths import ensure_dir, ensure_file, resolve_path

DEFAULT_ENV_FILE = Path(".env")
STACK_PATH_KEYS = {
    "REDIS_TLS_DIR",
    "SERVER_TLS_DIR",
    "SERVER_WORKER_CONFIG",
    "FLOWMESH_PLUGIN_DIR",
}
STACK_SUFFIX_ENV = "FLOWMESH_STACK_SUFFIX"
STACK_SLUG_ENV = "FLOWMESH_STACK_SLUG"
WORKER_RESULTS_DIR_ENV = "WORKER_RESULTS_DIR"
_STACK_SLUG_BASE = "flowmesh_node"
_STACK_SUFFIX_MAX_LEN = 48


def _resolve_stack_suffix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-_.")
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)[:_STACK_SUFFIX_MAX_LEN]
    sanitized = sanitized.rstrip("-_.")
    if not sanitized and value.strip():
        raise ValueError(
            f"{STACK_SUFFIX_ENV} must contain at least one ASCII letter or digit"
        )
    return sanitized


def stack_resource_env_overrides(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = os.environ if env is None else env
    suffix = _resolve_stack_suffix(values.get(STACK_SUFFIX_ENV, ""))
    stack_slug = f"{_STACK_SLUG_BASE}_{suffix}" if suffix else _STACK_SLUG_BASE
    return {STACK_SLUG_ENV: stack_slug}


def apply_stack_resource_env() -> None:
    overrides = stack_resource_env_overrides(os.environ)
    os.environ.update(overrides)
    os.environ["COMPOSE_PROJECT_NAME"] = overrides[STACK_SLUG_ENV]
    results_volume = f"{overrides[STACK_SLUG_ENV]}_results"
    if not os.environ.get(WORKER_RESULTS_DIR_ENV, "").strip():
        os.environ[WORKER_RESULTS_DIR_ENV] = results_volume


def stack_compose_file() -> Path:
    return asset_path("flowmesh_cli_stack.assets", "compose.yml")


def stack_env_example() -> Path:
    return asset_path("flowmesh_cli_stack.assets", ".env.example")


def stack_bake_file() -> Path:
    return asset_path("flowmesh_cli_stack.assets", "docker-bake.hcl")


def stack_node_client(
    env_file: Path, base_url: str | None, token: str | None
) -> NodeClient:
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
    default_base = "http://{}:{}".format(
        os.getenv("SERVER_HOST", "localhost"),
        os.getenv("SERVER_HTTP_PORT", os.getenv("SERVER_APP_PORT", "8000")),
    )
    resolved_base = base_url or default_base
    resolved_token = token or os.getenv("FLOWMESH_API_KEY") or None
    return NodeClient(resolved_base, token=resolved_token)


def flowmesh_client(
    env_file: Path, base_url: str | None, api_key: str | None
) -> FlowMesh:
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
    return FlowMesh(base_url=base_url, api_key=api_key)


def ensure_deploy_paths(base_dir: Path) -> None:
    ensure_dir(
        resolve_path(
            os.getenv("REDIS_TLS_DIR", ""),
            default="./secrets/tls/redis",
            base_dir=base_dir,
        )
    )
    ensure_dir(
        resolve_path(
            os.getenv("SERVER_TLS_DIR", ""),
            default="./secrets/tls/server",
            base_dir=base_dir,
        )
    )
    ensure_file(
        resolve_path(
            os.getenv("SERVER_WORKER_CONFIG", ""),
            default="./configs/worker_config.yaml",
            base_dir=base_dir,
        )
    )
    ensure_dir(
        resolve_path(
            os.getenv("FLOWMESH_PLUGIN_DIR", ""),
            default="./plugins",
            base_dir=base_dir,
        )
    )
