import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from shared.schemas.command import CommandMessage, CommandType
from shared.schemas.worker import WorkerStatus

from ...app_state import (
    get_logger,
    get_node_registry,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    register_resource,
    require_permission,
    resolve_accessible_ids,
)
from ...hooks import ResourceAction, ResourceType
from ...registries import Node, NodeRegistry, WorkerRegistry
from ...schemas.node import (
    NodeInfo,
    NodeRegisterResponse,
    NodeWorkerInfo,
    NodeWorkerStatus,
    WorkerRegisterResponse,
)
from ...utils.misc import filter_models_by_queries

router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.get(
    "",
    summary="List nodes",
    description="List all registered nodes with optional filtering.",
    response_description="List of nodes",
)
async def list_nodes(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[Node]:
    queries = request.query_params
    nodes = await node_registry.list_nodes_async()
    allowed = await resolve_accessible_ids(
        principal, ResourceType.NODE, ResourceAction.READ, logger
    )
    if allowed is not None:
        nodes = [node for node in nodes if node.id in allowed]
    return filter_models_by_queries(nodes, queries)


@router.get(
    "/workers",
    summary="List all workers across nodes",
    description=(
        "List all workers managed by all nodes with optional filtering. "
        "Worker status: "
        "STARTING (registered but not yet idle), "
        "IDLE (ready for work), "
        "BUSY (actively processing), "
        "STOPPING (node is stopping the worker), "
        "STOPPED (worker is not running)."
    ),
    response_description="List of workers",
)
async def list_all_workers(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[NodeWorkerInfo]:
    nodes = await node_registry.list_nodes_async()
    node_ids = [n.id for n in nodes]

    node_workers = await asyncio.gather(
        *[
            _fetch_node_workers(node_id, node_registry, worker_registry, logger)
            for node_id in node_ids
        ],
        return_exceptions=True,
    )
    all_workers: list[NodeWorkerInfo] = []
    for node_id, result in zip(node_ids, node_workers):
        if isinstance(result, BaseException):
            logger.warning("Failed to list workers for node %s: %s", node_id, result)
            continue
        all_workers.extend(result)

    allowed = await resolve_accessible_ids(
        principal, ResourceType.WORKER, ResourceAction.READ, logger
    )
    if allowed is not None:
        all_workers = [w for w in all_workers if w.id in allowed]
    filtered = filter_models_by_queries(all_workers, request.query_params)
    return filtered


@router.post(
    "/register",
    summary="Register a node",
    description="Register a new node.",
    response_description="Node ID",
    status_code=status.HTTP_201_CREATED,
)
async def register_node(
    node_info: NodeInfo,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    logger: logging.Logger = Depends(get_logger),
) -> NodeRegisterResponse:
    await require_permission(
        principal, ResourceType.NODE, None, ResourceAction.WRITE, logger
    )
    node_id = await node_registry.register_node_async(node_info)
    await register_resource(
        principal, ResourceType.NODE, node_id, {"alias": node_info.alias}, logger
    )
    return NodeRegisterResponse(node_id=node_id)


@router.get(
    "/{node_id}/workers",
    summary="List node workers",
    description=(
        "List workers managed by a node. "
        "Worker status: "
        "STARTING (registered but not yet idle), "
        "IDLE (ready for work), "
        "BUSY (actively processing), "
        "STOPPING (node is stopping the worker), "
        "STOPPED (worker is not running)."
    ),
    response_description="List of workers",
)
async def list_node_workers(
    node_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[NodeWorkerInfo]:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.READ, logger
    )
    workers = await _fetch_node_workers(node_id, node_registry, worker_registry, logger)
    allowed = await resolve_accessible_ids(
        principal, ResourceType.WORKER, ResourceAction.READ, logger
    )
    if allowed is not None:
        workers = [w for w in workers if w.id in allowed]
    filtered = filter_models_by_queries(workers, request.query_params)
    return filtered


@router.post(
    "/{node_id}/workers/register",
    summary="Register a worker",
    description="Register a new worker for a node.",
    response_description="Worker ID",
    status_code=status.HTTP_201_CREATED,
)
async def register_worker(
    node_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerRegisterResponse:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    node = await node_registry.get_node_async(node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="node not found"
        )
    worker_meta = await request.json()
    worker_id = await worker_registry.register_worker_async(
        node_id, node.alias, worker_meta
    )
    await register_resource(
        principal,
        ResourceType.WORKER,
        worker_id,
        {"node_id": node_id, "node_alias": node.alias},
        logger,
    )
    return WorkerRegisterResponse(worker_id=worker_id)


@router.post(
    "/{node_id}/workers/{worker_name}/start",
    summary="Start a worker",
    description="Start a worker managed by a node.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def start_node_worker(
    node_id: str,
    worker_name: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    cmd = CommandMessage(
        command=CommandType.START_WORKER, payload={"worker_name": worker_name}
    )
    try:
        resp = await node_registry.exec_node_cmd(node_id, cmd)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
        )

    if not resp.success or resp.data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=resp.message
        )
    success = resp.data["success"]
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start worker",
        )


@router.post(
    "/{node_id}/workers/{worker_name}/stop",
    summary="Stop a worker",
    description="Stop a worker managed by a node.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def stop_node_worker(
    node_id: str,
    worker_name: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    cmd = CommandMessage(
        command=CommandType.STOP_WORKER, payload={"worker_name": worker_name}
    )
    try:
        resp = await node_registry.exec_node_cmd(node_id, cmd)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
        )

    if not resp.success or resp.data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=resp.message
        )
    success = resp.data["success"]
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop worker",
        )


@router.get(
    "/{node_id}",
    summary="Get a node",
    description="Get node information by ID.",
    response_description="Node information",
)
async def get_node(
    node_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    node_registry: NodeRegistry = Depends(get_node_registry),
    logger: logging.Logger = Depends(get_logger),
) -> Node:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.READ, logger
    )
    node = await node_registry.get_node_async(node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Node not found"
        )
    return node


async def _fetch_node_workers(
    node_id: str,
    node_registry: NodeRegistry,
    worker_registry: WorkerRegistry,
    logger: logging.Logger,
) -> list[NodeWorkerInfo]:
    cmd = CommandMessage(command=CommandType.GET_WORKERS)
    try:
        resp = await node_registry.exec_node_cmd(node_id, cmd)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
        )

    if not resp.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=resp.message
        )
    if resp.data is None or "workers" not in resp.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid response from node",
        )
    raw_workers: list[dict[str, Any]] = resp.data["workers"]

    # Attach node ID to each worker
    for worker in raw_workers:
        worker["node_id"] = node_id

    # Convert registered worker status
    registered_workers: list[dict[str, Any]] = [
        worker for worker in raw_workers if worker["id"] is not None
    ]
    worker_records = await worker_registry.get_workers_async(
        [worker["id"] for worker in registered_workers]
    )
    for worker, record in zip(registered_workers, worker_records):
        if record is None:
            new_status = NodeWorkerStatus.STARTING  # Worker is not registered yet
        else:
            match record.status:
                case WorkerStatus.STARTING | WorkerStatus.UNKNOWN:
                    new_status = NodeWorkerStatus.STARTING
                case WorkerStatus.BUSY:
                    new_status = NodeWorkerStatus.BUSY
                case WorkerStatus.IDLE:
                    new_status = NodeWorkerStatus.IDLE
                case _:
                    logger.warning(
                        "Unrecognized worker status '%s' for worker ID '%s'",
                        record.status,
                        record.id,
                    )
                    new_status = NodeWorkerStatus.IDLE
        worker["status"] = new_status

    return [NodeWorkerInfo.model_validate(w) for w in raw_workers]
