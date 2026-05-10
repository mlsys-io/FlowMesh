import logging
from typing import Any

from fastapi import APIRouter, Depends

from ...app_state import get_logger, get_metrics
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...services.metrics import MetricsRecorder

router = APIRouter(prefix="/system", tags=["System"])


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
