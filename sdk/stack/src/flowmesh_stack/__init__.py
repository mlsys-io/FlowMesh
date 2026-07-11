"""FlowMesh SDK — Stack management extensions."""

from .images import (
    BUILD_GROUPS,
    BUILD_TARGETS,
    get_image_ref,
    managed_repos,
    parse_image_ref,
)
from .node_client import NodeClient
from .workers import (
    create_workers,
    detect_gpu_targets,
    operate_workers,
    pull_images,
    select_worker_images,
)

__all__ = [
    "BUILD_GROUPS",
    "BUILD_TARGETS",
    "NodeClient",
    "create_workers",
    "detect_gpu_targets",
    "get_image_ref",
    "managed_repos",
    "operate_workers",
    "parse_image_ref",
    "pull_images",
    "select_worker_images",
]
