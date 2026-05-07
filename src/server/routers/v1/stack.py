"""Local stack worker management"""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from shared.schemas.command import CommandMessage, CommandType

from ...app_state import get_logger, get_node_id, get_supervisor
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceType
from ...supervisor import WorkerSupervisor
from ...supervisor.manager import WorkerInitConfig
from ...supervisor.schemas import WorkerInfo
from ...utils.misc import filter_models_by_queries

router = APIRouter(prefix="/stack/workers", tags=["Stack"])

_WORKER_CREATE_TIMEOUT = 600.0


async def _exec(
    supervisor: WorkerSupervisor,
    cmd: CommandMessage,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Send a command to the local supervisor and return the response data."""
    try:
        kwargs = {"timeout": timeout} if timeout is not None else {}
        resp = await supervisor.exec_cmd(cmd, **kwargs)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
        )
    if not resp.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=resp.message or "Command failed",
        )
    return resp.data or {}


@router.get("")
async def list_workers(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> list[WorkerInfo]:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.READ, logger
    )
    cmd = CommandMessage(command=CommandType.GET_WORKERS)
    data = await _exec(supervisor, cmd)
    workers = [WorkerInfo(**w) for w in data.get("workers", [])]
    return filter_models_by_queries(workers, request.query_params)


@router.post("")
async def create_worker(
    init_config: WorkerInitConfig,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerInfo:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    cmd = CommandMessage(
        command=CommandType.CREATE_WORKER, payload=init_config.model_dump()
    )
    data = await _exec(supervisor, cmd, timeout=_WORKER_CREATE_TIMEOUT)
    return WorkerInfo(**data)


@router.get("/{name}")
async def get_worker(
    name: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> WorkerInfo:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.READ, logger
    )
    cmd = CommandMessage(command=CommandType.GET_WORKERS, payload={"worker_name": name})
    data = await _exec(supervisor, cmd)
    if workers := [WorkerInfo(**w) for w in data.get("workers", [])]:
        return workers[0]
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Worker not found"
    )


@router.post("/{name}/start")
async def start_worker(
    name: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    cmd = CommandMessage(
        command=CommandType.START_WORKER, payload={"worker_name": name}
    )
    data = await _exec(supervisor, cmd)
    if not data.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start worker '{name}'",
        )


@router.post("/{name}/stop")
async def stop_worker(
    name: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    cmd = CommandMessage(command=CommandType.STOP_WORKER, payload={"worker_name": name})
    data = await _exec(supervisor, cmd)
    if not data.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop worker '{name}'",
        )


@router.delete("/{name}")
async def destroy_worker(
    name: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    cmd = CommandMessage(
        command=CommandType.DESTROY_WORKER, payload={"worker_name": name}
    )
    await _exec(supervisor, cmd)


@router.delete("")
async def destroy_all_workers(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    supervisor: WorkerSupervisor = Depends(get_supervisor),
    node_id: str = Depends(get_node_id),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    await require_permission(
        principal, ResourceType.NODE, node_id, ResourceAction.WRITE, logger
    )
    body = await request.body()
    names: list[str] | None = None
    if body.strip():
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {exc.msg}",
            )
        if raw is None:
            names = None
        elif isinstance(raw, list):
            names = [str(n) for n in raw]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Expected request body to be an array of worker names.",
            )

    payload = None if names is None else {"worker_names": names}
    cmd = CommandMessage(command=CommandType.DESTROY_WORKERS, payload=payload)
    await _exec(supervisor, cmd)
