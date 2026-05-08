import os
from typing import Any

from pydantic import SecretStr

from ..resource_manager import GpuArch


def env_to_secret_str(env_var: str) -> SecretStr | None:
    value = os.getenv(env_var, "").strip()
    if not value:
        return None
    return SecretStr(value)


def to_env_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, list):
        return ",".join(str(v) for v in val)
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    return str(val)


def get_worker_image_name(registry: str, version: str, gpu_arch: GpuArch | None) -> str:
    match gpu_arch:
        case GpuArch.HOPPER | GpuArch.BLACKWELL | GpuArch.UNKNOWN:
            variant = "gpu"
        case None:
            variant = "cpu"
        case _:
            raise ValueError(f"Unsupported GPU architecture: {gpu_arch}")
    base = f"flowmesh_worker:{version}-{variant}"
    return f"{registry}/{base}" if registry else base
