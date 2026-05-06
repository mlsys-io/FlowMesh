import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi import Path as ApiPath
from fastapi import Query, Request, status
from fastapi.responses import StreamingResponse

from shared.schemas.command import StopMessage
from shared.tasks import TaskType

from ...app_state import (
    get_logger,
    get_redis_client,
    get_runtime,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_request,
    require_permission,
    resolve_accessible_ids,
)
from ...clients.redis import RedisClient, task_log_closed_key, task_log_stream_key
from ...hooks import ResourceAction, ResourceType
from ...registries.worker import WorkerRegistry
from ...schemas.common import OkResponse
from ...schemas.logs import LogEntry, LogEvent, LogQueryResponse
from ...task.runtime import TaskInfo, TaskRuntime
from ...utils.misc import filter_models_by_queries

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def _strip_private_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if not key.startswith("_")}


def _sanitize_latest_update(info: TaskInfo) -> None:
    """Remove private latest_update fields from the public task response."""
    if not isinstance(info.latest_update, dict):
        return
    latest_update = _strip_private_fields(info.latest_update)
    ssh_info = latest_update.get("ssh")
    if isinstance(ssh_info, dict):
        latest_update["ssh"] = _strip_private_fields(ssh_info)
    info.latest_update = latest_update


@router.get(
    "",
    summary="List tasks",
    description="List all tasks.",
    response_description="List of task details.",
)
async def list_tasks(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> list[TaskInfo]:
    tasks = runtime.list_tasks()
    allowed = await resolve_accessible_ids(
        principal, ResourceType.TASK, ResourceAction.READ, logger
    )
    if allowed is not None:
        tasks = [task for task in tasks if task.task_id in allowed]
    for task in tasks:
        _sanitize_latest_update(task)
    return filter_models_by_queries(tasks, request.query_params)


@router.get(
    "/{task_id}",
    summary="Get a task",
    description="Get task details by ID.",
    response_description="Task details.",
)
async def get_task(
    task_id: str = ApiPath(..., min_length=1),
    principal: PrincipalContext = Depends(authenticate_request),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> TaskInfo:
    await require_permission(
        principal, ResourceType.TASK, task_id, ResourceAction.READ, logger
    )
    info = runtime.describe_task(task_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    _sanitize_latest_update(info)
    return info


@router.post(
    "/{task_id}/stop",
    summary="Stop a running task",
    description=(
        "Stop a running task. The server sends a stop command to the assigned worker, "
        "but cannot guarantee the task will stop successfully."
    ),
    response_description="Operation result.",
)
async def stop_task(
    task_id: str = ApiPath(..., min_length=1),
    principal: PrincipalContext = Depends(authenticate_request),
    runtime: TaskRuntime = Depends(get_runtime),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    logger: logging.Logger = Depends(get_logger),
) -> OkResponse:
    await require_permission(
        principal, ResourceType.TASK, task_id, ResourceAction.CANCEL, logger
    )
    record = runtime.get_record(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found"
        )
    if record.task.spec.taskType != TaskType.SSH:
        # TODO: Support stopping other task types.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Stopping is only supported for SSH tasks currently",
        )
    if record.status != "DISPATCHED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task is not running",
        )
    worker_id = record.assigned_worker
    if not worker_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task has no assigned worker",
        )
    worker = await worker_registry.get_worker_async(worker_id)
    if worker is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Assigned worker not found",
        )
    await worker_registry.publish_stop_async(
        worker, StopMessage(task_id=task_id, worker_id=worker.id)
    )
    return OkResponse(ok=True)


@router.get(
    "/{task_id}/logs",
    summary="Query task logs",
    description="Read recent task logs.",
    response_description="Task log entries.",
    response_model=LogQueryResponse,
)
async def get_task_logs(
    task_id: str = ApiPath(..., min_length=1),
    limit: int = Query(
        default=200,
        ge=1,
        le=10_000,
        description="Maximum number of log entries to return.",
    ),
    before: str | None = Query(
        default=None,
        description=(
            "Return entries strictly before this cursor. The cursor is an opaque "
            "string previously returned as `entries[].cursor` "
            '(example: `"1707349300000-0"`).'
        ),
    ),
    after: str | None = Query(
        default=None,
        description=(
            "Return entries strictly after this cursor. The cursor is an opaque string "
            "previously returned as `entries[].cursor` "
            '(example: `"1707349300000-0"`).'
        ),
    ),
    principal: PrincipalContext = Depends(authenticate_request),
    redis: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
) -> LogQueryResponse:
    await require_permission(
        principal, ResourceType.RESULT, task_id, ResourceAction.READ, logger
    )
    limit = max(1, min(10_000, int(limit)))
    if before and after:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only one of before/after may be set",
        )

    key = task_log_stream_key(task_id)
    if not await redis.asyncio.exists_telemetry(key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="log stream not found"
        )
    if after:
        raw = await redis.asyncio.xrange_telemetry(key, min_id=f"({after}", count=limit)
        ordered = raw
    else:
        max_id = f"({before}" if before else "+"
        raw = await redis.asyncio.xrevrange_telemetry(
            key, max_id=max_id, min_id="-", count=limit
        )
        ordered = list(reversed(raw))

    entries: list[LogEntry] = []
    for cursor, fields in ordered:
        payload = fields.get("payload")
        if not isinstance(payload, str) or not payload:
            continue
        workflow_id = fields.get("workflow_id", "")
        task_id_field = fields.get("task_id", "")
        event: dict[str, Any] = json.loads(payload)
        if workflow_id:
            event.setdefault("workflow_id", workflow_id)
        event.setdefault("task_id", task_id_field or task_id)
        event.setdefault("level", "INFO")
        event.setdefault("stream", "system")
        if not event.get("message"):
            event["message"] = payload
        entries.append(LogEntry(cursor=cursor, event=LogEvent.model_validate(event)))

    next_cursor = entries[-1].cursor if entries else None
    prev_cursor = entries[0].cursor if entries else None
    return LogQueryResponse(
        entries=entries, next_cursor=next_cursor, prev_cursor=prev_cursor
    )


@router.get(
    "/{task_id}/logs/stream",
    summary="Stream task logs",
    description=(
        "Stream task logs via SSE.\n\n"
        "Cursor formats:\n"
        "- `cursor` and `Last-Event-ID` are opaque stream IDs (example: "
        '`"1707349300000-0"`).\n\n'
        "Where to get cursors:\n"
        "- From SSE: each message includes an `id` field; persist the last seen `id`.\n"
        "- From query API: `GET /tasks/{task_id}/logs` returns `entries[].cursor`.\n\n"
        "End of stream:\n"
        "- When the server detects the stream has ended, it sends an `eos` event and "
        "closes the connection.\n\n"
        "Reconnection:\n"
        "- Prefer setting the standard SSE header `Last-Event-ID` on reconnect.\n"
        "- If both `cursor` and `Last-Event-ID` are set, `Last-Event-ID` takes "
        "precedence."
    ),
    response_class=StreamingResponse,
)
async def stream_task_logs(
    task_id: str = ApiPath(..., min_length=1),
    cursor: str | None = Query(
        default=None,
        description=(
            "Resume streaming strictly after this cursor. Use the last seen "
            "`entries[].cursor` from the query endpoint, or the last SSE `id` value "
            "you received."
        ),
    ),
    last_event_id: str | None = Header(
        default=None,
        alias="Last-Event-ID",
        description=(
            "SSE reconnection cursor. If set, the stream resumes strictly after this "
            "ID. When both `cursor` and `Last-Event-ID` are provided, `Last-Event-ID` "
            "takes precedence."
        ),
    ),
    principal: PrincipalContext = Depends(authenticate_request),
    redis: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
):
    await require_permission(
        principal, ResourceType.RESULT, task_id, ResourceAction.READ, logger
    )
    key = task_log_stream_key(task_id)
    if not await redis.asyncio.exists_telemetry(key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="log stream not found"
        )
    start_id = last_event_id or cursor or "$"

    async def _gen():
        current = start_id
        try:
            # Check if the stream is already closed
            if await redis.asyncio.exists(task_log_closed_key(task_id)):
                if current == "$":
                    yield b"event: eos\ndata:\n\n"
                    return
                normalized = current.lstrip("(").strip()
                if normalized:
                    newer = await redis.asyncio.xrange_telemetry(
                        key, min_id=f"({normalized}", max_id="+", count=1
                    )
                    if not newer:
                        yield b"event: eos\ndata:\n\n"
                        return

            while True:
                rows = await redis.asyncio.xread_telemetry(
                    {key: current}, count=200, block_ms=15_000
                )
                if not rows:
                    yield b": keep-alive\n\n"
                    continue
                for _, batch in rows:
                    for stream_id, fields in batch:
                        payload = fields.get("payload")
                        if not isinstance(payload, str) or not payload:
                            current = stream_id
                            continue
                        msg = f"id: {stream_id}\nevent: log\ndata: {payload}\n\n"
                        yield msg.encode()
                        current = stream_id
                        try:
                            event = json.loads(payload)
                            if (
                                isinstance(event, dict)
                                and event.get("type") == "LOG_STREAM_CLOSED"
                            ):
                                yield (
                                    f"id: {stream_id}\nevent: eos\ndata:\n\n"
                                ).encode()
                                return
                        except json.JSONDecodeError:
                            event = None
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
