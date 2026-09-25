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
from ...registries.worker import Worker, WorkerInfo, WorkerRegistry, is_cordoned
from ...schemas.worker import WorkerCordon, WorkerCordonResult
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


# Declared before "/{worker_id}", which would otherwise match "/cordoned".
@router.get(
    "/cordoned",
    summary="List cordons",
    description="List the (node alias, worker alias) pairs excluded from dispatch.",
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


@router.delete(
    "/cordoned/{node_alias}/{alias}",
    summary="Remove a cordon",
    description=(
        "Remove a cordon by node alias and worker alias. Works whether or not a "
        "worker with that alias is currently registered."
    ),
    response_description="Cordon result",
)
async def remove_cordon(
    node_alias: str,
    alias: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerCordonResult:
    await require_permission(
        principal, ResourceKind.WORKER, None, ResourceAction.WRITE, logger
    )
    cordon = WorkerCordon(node_alias=node_alias, alias=alias)
    changed = await registry.set_cordon_async(cordon, cordoned=False)
    logger.info("Removed cordon %s/%s", node_alias, alias)
    return WorkerCordonResult(**cordon.model_dump(), cordoned=False, changed=changed)


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


async def _worker_or_404(registry: WorkerRegistry, worker_id: str) -> Worker:
    worker = await registry.get_worker_async(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="worker not found"
        )
    return worker


def _cordon_or_409(worker: Worker) -> WorkerCordon:
    if not worker.alias:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"worker {worker.id} has no alias; reconnect it, then retry",
        )
    return WorkerCordon(node_alias=worker.node_alias, alias=worker.alias)


@router.post(
    "/{worker_id}/cordon",
    summary="Cordon a worker",
    description=(
        "Stop offering new tasks to this worker. The worker keeps running and "
        "work already dispatched to it runs to completion. The cordon is keyed "
        "on the node alias and the worker alias, so it persists when the worker "
        "reconnects under a new id."
    ),
    response_description="Cordon result",
)
async def cordon_worker(
    worker_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerCordonResult:
    await require_permission(
        principal, ResourceKind.WORKER, worker_id, ResourceAction.WRITE, logger
    )
    worker = await _worker_or_404(registry, worker_id)
    cordon = _cordon_or_409(worker)
    changed = await registry.set_cordon_async(cordon, cordoned=True)
    logger.info(
        "Cordoned worker %s (%s/%s)", worker_id, cordon.node_alias, cordon.alias
    )
    return WorkerCordonResult(**cordon.model_dump(), cordoned=True, changed=changed)


@router.post(
    "/{worker_id}/uncordon",
    summary="Uncordon a worker",
    description="Allow this worker to receive new tasks again.",
    response_description="Cordon result",
)
async def uncordon_worker(
    worker_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerCordonResult:
    await require_permission(
        principal, ResourceKind.WORKER, worker_id, ResourceAction.WRITE, logger
    )
    cordon = _cordon_or_409(await _worker_or_404(registry, worker_id))
    changed = await registry.set_cordon_async(cordon, cordoned=False)
    logger.info(
        "Uncordoned worker %s (%s/%s)", worker_id, cordon.node_alias, cordon.alias
    )
    return WorkerCordonResult(**cordon.model_dump(), cordoned=False, changed=changed)
