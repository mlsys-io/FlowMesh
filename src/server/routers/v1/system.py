import logging
from typing import Any

from fastapi import APIRouter, Depends

from shared._version import FLOWMESH_RELEASE_VERSION

from ...app_state import get_logger, get_metrics
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...schemas.common import VersionResponse
from ...services.metrics import MetricsRecorder

router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/version",
    summary="Get version",
    description="Server version.",
    response_description="Server version",
    tags=["Caller: anyone"],
)
async def get_version() -> VersionResponse:
    return VersionResponse(version=FLOWMESH_RELEASE_VERSION)


@router.get(
    "/metrics",
    summary="Get metrics",
    description="Get a system metrics snapshot.",
    response_description="Metrics data",
)
async def get_metrics_snapshot(
    principal: PrincipalContext = Depends(authenticate_connection),
    metrics: MetricsRecorder = Depends(get_metrics),
    logger: logging.Logger = Depends(get_logger),
) -> dict[str, Any]:
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )
    return metrics.snapshot()
