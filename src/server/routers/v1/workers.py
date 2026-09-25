import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ...app_state import (
    get_logger,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
    resolve_accessible_ids,
)
from ...hooks import ResourceAction, ResourceKind
from ...registries.worker import WorkerInfo, WorkerRegistry, is_cordoned
from ...schemas.worker import (
    WorkerCordon,
    WorkerCordonByAlias,
    WorkerCordonRequest,
    WorkerCordonResult,
)
from ...utils.misc import filter_models_by_queries

router = APIRouter(prefix="/workers", tags=["Workers"])


@router.get(
    "",
    summary="List workers",
    description="List all registered workers with optional filtering.",
    response_description="List of workers",
)
async def list_workers(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[WorkerInfo]:
    queries = request.query_params
    workers = await registry.list_workers_async()
    allowed = await resolve_accessible_ids(
        principal, ResourceKind.WORKER, ResourceAction.READ, logger
    )
    if allowed is not None:
        workers = [w for w in workers if w.id in allowed]
    return filter_models_by_queries(workers, queries)


# Declared before "/{worker_id}", which would otherwise match "/cordons".
@router.get(
    "/cordons",
    summary="List cordons",
    description=(
        "List the (node alias, worker alias) pairs excluded from dispatch, "
        "including cordons with no worker currently registered under them."
    ),
    response_description="Active cordons",
)
async def list_cordons(
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[WorkerCordon]:
    await require_permission(
        principal, ResourceKind.WORKER, None, ResourceAction.READ, logger
    )
    return await registry.list_cordons_async()


@router.get(
    "/{worker_id}",
    summary="Get a worker",
    description="Get worker information by ID.",
    response_description="Worker information",
)
async def get_worker(
    worker_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerInfo:
    await require_permission(
        principal, ResourceKind.WORKER, worker_id, ResourceAction.READ, logger
    )
    worker = await registry.get_worker_async(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="worker not found"
        )
    stale = await registry.is_worker_stale_async(worker.id)
    cordoned = is_cordoned(worker, await registry.cordoned_members_async())
    return WorkerInfo(**worker.model_dump(), stale=stale, cordoned=cordoned)


async def _resolve_cordon(
    request: WorkerCordonRequest,
    registry: WorkerRegistry,
    principal: PrincipalContext,
    logger: logging.Logger,
) -> WorkerCordon:
    if isinstance(request, WorkerCordonByAlias):
        await require_permission(
            principal, ResourceKind.WORKER, None, ResourceAction.WRITE, logger
        )
        return WorkerCordon(node_alias=request.node_alias, alias=request.alias)
    await require_permission(
        principal, ResourceKind.WORKER, request.worker_id, ResourceAction.WRITE, logger
    )
    worker = await registry.get_worker_async(request.worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="worker not found"
        )
    if not worker.alias:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"worker {worker.id} has no alias; reconnect it, then retry",
        )
    return WorkerCordon(node_alias=worker.node_alias, alias=worker.alias)


async def _set_cordon(
    request: WorkerCordonRequest,
    cordoned: bool,
    registry: WorkerRegistry,
    principal: PrincipalContext,
    logger: logging.Logger,
) -> WorkerCordonResult:
    cordon = await _resolve_cordon(request, registry, principal, logger)
    changed = await registry.set_cordon_async(cordon, cordoned=cordoned)
    worker_ids = await registry.live_worker_ids_async(cordon)
    logger.info(
        "%s %s/%s (workers: %s)",
        "Cordoned" if cordoned else "Uncordoned",
        cordon.node_alias,
        cordon.alias,
        ", ".join(worker_ids) or "none",
    )
    return WorkerCordonResult(
        **cordon.model_dump(), cordoned=cordoned, changed=changed, worker_ids=worker_ids
    )


@router.post(
    "/cordon",
    summary="Cordon a worker",
    description=(
        "Stop offering new tasks to a worker, selected by `worker_id` or by "
        "`node_alias` and `alias`. The worker keeps running and work already "
        "dispatched to it runs to completion. The cordon is keyed on the node "
        "alias and the worker alias and persists until uncordoned: it applies "
        "when the worker reconnects under a new id, and to a worker that "
        "registers under that key later. `worker_ids` lists the live workers "
        "the cordon currently matches."
    ),
    response_description="Cordon result",
)
async def cordon_worker(
    request: WorkerCordonRequest,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerCordonResult:
    return await _set_cordon(request, True, registry, principal, logger)


@router.post(
    "/uncordon",
    summary="Uncordon a worker",
    description=(
        "Allow a worker, selected by `worker_id` or by `node_alias` and `alias`, "
        "to receive new tasks again."
    ),
    response_description="Cordon result",
)
async def uncordon_worker(
    request: WorkerCordonRequest,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerCordonResult:
    return await _set_cordon(request, False, registry, principal, logger)
