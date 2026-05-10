import asyncio
import json
import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse

from shared.schemas.event import TaskEvent

from ...app_state import (
    get_logger,
    get_metrics,
    get_redis_client,
    get_runtime,
    get_workflow_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    register_resource,
    require_permission,
    resolve_accessible_ids,
)
from ...clients.redis import (
    RedisClient,
    workflow_log_closed_key,
    workflow_log_stream_key,
)
from ...hooks import SUBMISSION_GUARDS, ResourceAction, ResourceKind
from ...registries.workflow import Workflow, WorkflowRegistry
from ...schemas.logs import LogEntry, LogEvent, LogQueryResponse
from ...schemas.workflow import (
    WorkflowSubmitResponse,
    WorkflowSubmitTaskEntry,
    WorkflowValidateResponse,
    WorkflowValidateTaskEntry,
)
from ...services.metrics import MetricsRecorder
from ...task.runtime import TaskRuntime
from ...utils.misc import filter_models_by_queries

_WORKFLOW_REQUEST_BODY_FORMAT = {
    "requestBody": {
        "required": True,
        "content": {
            "text/plain": {"schema": {"type": "string"}},
            "application/json": {
                "schema": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"yaml": {"type": "string"}},
                            "required": ["yaml"],
                        },
                        {"type": "object", "additionalProperties": True},
                    ]
                }
            },
        },
    },
}
router = APIRouter(prefix="/workflows", tags=["Workflows"])


def _parse_submission_body(raw_body: bytes, content_type: str) -> str:
    if "application/json" in content_type:
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {exc}",
            ) from exc
        if isinstance(data, dict) and "yaml" in data:
            return str(data["yaml"])
        if isinstance(data, str):
            return data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Expected YAML string or {"yaml":"..."} in JSON body',
        )

    try:
        return raw_body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Body must be UTF-8 encoded text",
        ) from exc


@router.post(
    "",
    summary="Submit a workflow",
    description=(
        "Submit a workflow (YAML or JSON) for execution. "
        "For native YAML definition of workflows, accepts "
        "text/plain YAML or application/json with {'yaml': '...'} payload. "
        "For non-native formats (e.g., n8n), use the Workflow-Format "
        "header to specify the format. "
        "Now supports n8n format with application/json content type."
    ),
    response_description="Submission result.",
    status_code=status.HTTP_201_CREATED,
    openapi_extra=_WORKFLOW_REQUEST_BODY_FORMAT,
)
async def submit_workflow(
    request: Request,
    content_type: str = Header(default="", description="Request content type"),
    workflow_format: str = Header(
        default="native", description="Workflow format (native/n8n)"
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    metrics: MetricsRecorder = Depends(get_metrics),
    logger: logging.Logger = Depends(get_logger),
) -> WorkflowSubmitResponse:
    body = await request.body()
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body is required",
        )

    await require_permission(
        principal, ResourceKind.WORKFLOW, None, ResourceAction.WRITE, logger
    )
    for guard in SUBMISSION_GUARDS:
        await guard.check(principal, logger)

    payload = _get_workflow_from_request(body, content_type, workflow_format)

    try:
        workflow_id, entries = await runtime.register(
            principal.principal_id,
            principal.org_id,
            payload,
            format=workflow_format,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register workflow: {exc}",
        ) from exc

    await register_resource(
        principal,
        ResourceKind.WORKFLOW,
        workflow_id,
        {"format": workflow_format, "task_count": len(entries)},
        logger,
    )
    for entry in entries:
        await register_resource(
            principal,
            ResourceKind.TASK,
            entry.task_id,
            {"workflow_id": workflow_id},
            logger,
        )

    results: list[WorkflowSubmitTaskEntry] = []

    for entry in entries:
        task_id = entry.task_id
        info = runtime.describe_task(task_id)
        if not info:
            continue
        task_type = info.task.spec.taskType
        metrics.record_task_event(
            TaskEvent(
                type="TASK_SUBMITTED",
                task_id=task_id,
                payload={
                    "taskType": task_type,
                },
            )
        )
        results.append(
            WorkflowSubmitTaskEntry(
                task_id=task_id,
                status=info.status,
                assigned_worker=info.assigned_worker,
                topic=info.topic,
                waiting_on=info.pending_dependencies,
                depends_on=info.depends_on,
                attempts=info.attempts,
                max_attempts=info.max_attempts,
                load=info.load,
            )
        )

    return WorkflowSubmitResponse(
        ok=True, workflow_id=workflow_id, count=len(entries), tasks=results
    )


@router.post(
    "/validate",
    summary="Validate a workflow",
    description=(
        "Validate a workflow definition without submitting it for execution. "
        "Accepts the same payload formats as the submit endpoint."
    ),
    response_description="Validation result.",
    status_code=status.HTTP_200_OK,
    openapi_extra=_WORKFLOW_REQUEST_BODY_FORMAT,
)
async def validate_workflow(
    request: Request,
    content_type: str = Header(default="", description="Request content type"),
    workflow_format: str = Header(
        default="native", description="Workflow format (native/n8n)"
    ),
    _: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
) -> WorkflowValidateResponse:
    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body is required",
        )

    payload = _get_workflow_from_request(raw_body, content_type, workflow_format)

    try:
        results = runtime.validate(payload, format=workflow_format)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow validation failed: {exc}",
        ) from exc

    return WorkflowValidateResponse(
        ok=True,
        count=len(results),
        tasks=[
            WorkflowValidateTaskEntry(
                task_id=entry.task_id,
                graph_node_name=entry.graph_node_name,
                depends_on=entry.depends_on,
            )
            for entry in results
        ],
    )


@router.get(
    "/{workflow_id}",
    summary="Get a workflow",
    description=(
        "Get workflow details by ID. "
        "Workflow status: "
        "PENDING (no tasks dispatched yet), "
        "DISPATCHED (some tasks currently dispatched), "
        "FAILED (one or more tasks failed), "
        "CANCELLED (cancelled tasks remain), "
        "DONE (all tasks completed). "
    ),
    response_description="Workflow details",
)
async def get_workflow(
    workflow_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    logger: logging.Logger = Depends(get_logger),
) -> Workflow:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    workflow = await registry.get_workflow_async(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )
    return workflow


@router.get(
    "/{workflow_id}/logs",
    summary="Query workflow logs",
    description="Read recent workflow logs.",
    response_description="Workflow log entries.",
    response_model=LogQueryResponse,
)
async def get_workflow_logs(
    workflow_id: str,
    limit: int = Query(
        default=500,
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
            "Return entries strictly after this cursor. The cursor is an opaque "
            "string previously returned as `entries[].cursor` "
            '(example: `"1707349300000-0"`).'
        ),
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    redis: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
) -> LogQueryResponse:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    limit = max(1, min(10_000, int(limit)))
    if before and after:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only one of before/after may be set",
        )

    key = workflow_log_stream_key(workflow_id)
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
        workflow_id_field = fields.get("workflow_id", "")
        task_id_field = fields.get("task_id", "")
        event: dict[str, Any] = json.loads(payload)
        event.setdefault("workflow_id", workflow_id_field or workflow_id)
        if task_id_field:
            event.setdefault("task_id", task_id_field)
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
    "/{workflow_id}/logs/stream",
    summary="Stream workflow logs",
    description=(
        "Stream workflow logs via SSE.\n\n"
        "Cursor formats:\n"
        "- `cursor` and `Last-Event-ID` are opaque stream IDs (example: "
        '`"1707349300000-0"`).\n\n'
        "Where to get cursors:\n"
        "- From SSE: each message includes an `id` field; persist the last seen `id`.\n"
        "- From query API: `GET /workflows/{workflow_id}/logs` returns "
        "`entries[].cursor`.\n\n"
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
async def stream_workflow_logs(
    workflow_id: str,
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
    principal: PrincipalContext = Depends(authenticate_connection),
    redis: RedisClient = Depends(get_redis_client),
    logger: logging.Logger = Depends(get_logger),
):
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.READ, logger
    )
    key = workflow_log_stream_key(workflow_id)
    if not await redis.asyncio.exists_telemetry(key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="log stream not found"
        )
    start_id = last_event_id or cursor or "$"

    async def _gen():
        current = start_id
        try:
            # Check if the stream is already closed
            if await redis.asyncio.exists(workflow_log_closed_key(workflow_id)):
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
                            pass
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


@router.post(
    "/{workflow_id}/cancel",
    summary="Cancel a workflow",
    description="Cancel a running workflow.",
    response_description="Cancelled workflow",
)
async def cancel_workflow(
    workflow_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    logger: logging.Logger = Depends(get_logger),
) -> Workflow:
    await require_permission(
        principal, ResourceKind.WORKFLOW, workflow_id, ResourceAction.CANCEL, logger
    )
    runtime.cancel_workflow(workflow_id)
    workflow = await registry.get_workflow_async(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )
    return workflow


@router.get(
    "",
    summary="List workflows",
    description="List submitted workflows.",
    response_description="List of workflows",
)
async def list_workflows(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_connection),
    registry: WorkflowRegistry = Depends(get_workflow_registry),
    logger: logging.Logger = Depends(get_logger),
) -> list[Workflow]:
    workflow_ids = await registry.get_workflow_ids_async()
    allowed = await resolve_accessible_ids(
        principal, ResourceKind.WORKFLOW, ResourceAction.READ, logger
    )
    if allowed is not None:
        workflow_ids = workflow_ids & allowed
    workflows: list[Workflow] = []
    for workflow_id in workflow_ids:
        workflow = await registry.get_workflow_async(workflow_id)
        if workflow:
            workflows.append(workflow)
    return filter_models_by_queries(workflows, request.query_params)


def _get_workflow_from_request(
    body: bytes, content_type: str, workflow_format: str
) -> str:
    content_type = content_type.lower()
    workflow_format = workflow_format.lower()
    match workflow_format:
        case "n8n":
            if "application/json" not in content_type:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="n8n workflows require application/json content type",
                )
            try:
                payload = body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Body must be UTF-8 encoded text",
                ) from exc
        case "native":
            payload = _parse_submission_body(body, content_type)
            if not payload.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="YAML payload cannot be empty",
                )
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported workflow format: {workflow_format}",
            )
    return payload
