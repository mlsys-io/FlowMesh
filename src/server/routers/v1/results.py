import gzip
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from shared.utils.atomic import atomic_write_text
from shared.utils.manifest import ARTIFACTS_DIR, LOGS_DIR, RESULTS_NAME, sync_manifest

from ...app_state import (
    get_event_monitor,
    get_logger,
    get_results_dir,
    get_runtime,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceType
from ...schemas.common import PathResponse
from ...schemas.result import ResultPayload, read_result, result_file_path, write_result
from ...services.monitoring import EventMonitor
from ...task.models import TERMINAL_TASK_STATUSES
from ...task.runtime import TaskRuntime

# Sections the bundle endpoint can include.
_BUNDLE_SECTIONS_CONCRETE = ("results", "artifacts", "logs")
_BUNDLE_SECTIONS_ACCEPTED = (*_BUNDLE_SECTIONS_CONCRETE, "all")
_BUNDLE_SECTIONS_DEFAULT = ("results", "artifacts")

router = APIRouter(prefix="/results", tags=["Results"])


def _resolve_artifact_path(filename: str) -> Path:
    sanitized = Path(filename)
    if (
        sanitized.is_absolute()
        or filename in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in sanitized.parts)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        )
    return Path(ARTIFACTS_DIR) / sanitized


@router.post(
    "",
    summary="Submit a result",
    description="Submit a task result payload.",
    response_description="Submission status",
)
async def ingest_result(
    payload: ResultPayload,
    _: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    event_monitor: EventMonitor = Depends(get_event_monitor),
    results_dir: Path = Depends(get_results_dir),
) -> PathResponse:
    task_id = payload.task_id.strip()
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="task_id is required"
        )

    content = {
        "task_id": task_id,
        "worker_id": payload.worker_id,
        "metadata": payload.metadata,
        "received_at": payload.received_at,
        "result": payload.result,
    }

    try:
        path = write_result(results_dir, task_id, content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store result: {exc}",
        ) from exc

    expected_artifacts: list[str] = []
    record = runtime.get_record(task_id)
    if record:
        expected_artifacts = record.task.spec.get_artifacts()
    sync_manifest(path.parent, task_id, expected_artifacts)
    pending_children = event_monitor.pop_pending_clones(task_id)
    if pending_children:
        event_monitor.mirror_task_results(task_id, pending_children)
    return PathResponse(ok=True, path=str(path))


@router.get(
    "/{task_id}",
    summary="Get a result",
    description="Get a task result by task ID.",
    response_description="Task result",
)
async def get_result(
    task_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> dict[str, Any]:
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="task_id is required"
        )
    await require_permission(
        principal, ResourceType.RESULT, task_id, ResourceAction.READ, logger
    )
    try:
        raw = read_result(results_dir, task_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="result not found"
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read result: {exc}",
        ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"task_id": task_id, "raw": raw}


@router.post(
    "/{task_id}/files",
    summary="Upload a result artifact",
    description="Upload an artifact file for a task result.",
    response_description="Upload status",
)
async def upload_result_file(
    task_id: str,
    file: UploadFile = File(...),
    runtime: TaskRuntime = Depends(get_runtime),
    _: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
) -> PathResponse:
    base_dir = result_file_path(results_dir, task_id).parent
    relative_path = _resolve_artifact_path(file.filename or "")
    target_path = (base_dir / relative_path).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target_path.open("wb") as out:
            out.write(await file.read())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store artifact: {exc}",
        ) from exc

    _rewrite_jsonl_export_paths(task_id, base_dir, target_path)

    record = runtime.get_record(task_id)
    expected_artifacts: list[str] = []
    if record:
        expected_artifacts = record.task.spec.get_artifacts()
    sync_manifest(base_dir, task_id, expected_artifacts)
    return PathResponse(ok=True, path=str(target_path))


@router.get(
    "/{task_id}/files/{filename:path}",
    summary="Download a result artifact",
    description="Download an artifact file for a task result.",
    response_description="Result file",
    response_class=FileResponse,
)
async def download_result_file(
    task_id: str,
    filename: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> FileResponse:
    await require_permission(
        principal, ResourceType.RESULT, task_id, ResourceAction.READ, logger
    )
    sanitized = Path(filename)
    base_dir = result_file_path(results_dir, task_id).parent
    relative_path = _resolve_artifact_path(filename)
    target_path = (base_dir / relative_path).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        )

    if not target_path.exists() or not target_path.is_file():
        if len(sanitized.parts) != 1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
            )
        fallback = (base_dir / sanitized.name).resolve()
        try:
            fallback.relative_to(base_dir)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
            )
        if not fallback.exists() or not fallback.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
            )
        target_path = fallback

    return FileResponse(target_path)


@router.get(
    "/{task_id}/bundle",
    summary="Download a full result bundle",
    description="Download a tar archive containing the full task result directory.",
    response_description="Result bundle archive",
    response_class=FileResponse,
)
async def download_result_bundle(
    task_id: str,
    background_tasks: BackgroundTasks,
    include: list[str] = Query(default_factory=list),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> FileResponse:
    await require_permission(
        principal, ResourceType.RESULT, task_id, ResourceAction.READ, logger
    )
    sections = _resolve_bundle_sections(include)

    record = runtime.get_record(task_id)
    if record is not None and record.status not in TERMINAL_TASK_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"task {task_id} is not in a terminal state "
                f"(status={record.status}); bundle unavailable"
            ),
        )

    base_dir = result_file_path(results_dir, task_id).parent
    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="result bundle not found"
        )

    try:
        bundle_path = _create_result_bundle_archive(
            task_id, base_dir, sections=sections
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare result bundle: {exc}",
        ) from exc

    background_tasks.add_task(_cleanup_bundle_file, bundle_path)
    return FileResponse(
        bundle_path,
        media_type="application/x-tar",
        filename=f"{task_id}.tar.gz",
        headers={"Content-Encoding": "gzip"},
    )


@router.get(
    "/{task_id}/logs",
    summary="Download archived task logs",
    description="Download archived logs.jsonl for a task result.",
    response_description="Task log file",
    response_class=FileResponse,
)
async def download_task_logs(
    task_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> FileResponse:
    await require_permission(
        principal, ResourceType.RESULT, task_id, ResourceAction.READ, logger
    )
    base_dir = result_file_path(results_dir, task_id).parent
    target_path = (base_dir / LOGS_DIR / "logs.jsonl").resolve()
    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid path"
        )
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="logs not found"
        )
    return FileResponse(target_path)


def _rewrite_jsonl_export_paths(
    task_id: str, base_dir: Path, artifact_path: Path
) -> None:
    results_path = base_dir / RESULTS_NAME
    if not results_path.exists():
        return
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return

    filename = artifact_path.name
    new_abs = str(artifact_path)
    new_relative = f"{ARTIFACTS_DIR}/{filename}"
    updated = False

    def _update(entry: dict[str, Any]) -> None:
        nonlocal updated
        if not isinstance(entry, dict):
            return
        block = entry.get("jsonl_export")
        if isinstance(block, dict):
            path_value = str(block.get("path") or "")
            if path_value and Path(path_value).name == filename:
                if "worker_path" not in block and path_value != new_abs:
                    block["worker_path"] = path_value
                block["path"] = new_abs
                block["relative_path"] = new_relative
                block.setdefault("url", f"/api/v1/results/{task_id}/files/{filename}")
                updated = True
        children = entry.get("children")
        if isinstance(children, dict):
            for child_entry in children.values():
                if isinstance(child_entry, dict):
                    _update(child_entry)

    result_entry = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result_entry, dict):
        _update(result_entry)

    if updated:
        try:
            atomic_write_text(
                results_path, json.dumps(payload, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass


def _resolve_bundle_sections(include: list[str]) -> tuple[str, ...]:
    if not include:
        return _BUNDLE_SECTIONS_DEFAULT
    invalid = sorted({v for v in include if v not in _BUNDLE_SECTIONS_ACCEPTED})
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown include values: {invalid}. "
                f"Accepted: {list(_BUNDLE_SECTIONS_ACCEPTED)}"
            ),
        )
    requested = set(include)
    if "all" in requested:
        return _BUNDLE_SECTIONS_CONCRETE
    ordered = tuple(s for s in _BUNDLE_SECTIONS_CONCRETE if s in requested)
    return ordered or _BUNDLE_SECTIONS_DEFAULT


def _create_result_bundle_archive(
    task_id: str,
    base_dir: Path,
    sections: tuple[str, ...] = _BUNDLE_SECTIONS_DEFAULT,
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"flowmesh-result-{task_id}-",
        suffix=".tar.gz",
        delete=False,
    ) as tmp:
        bundle_path = Path(tmp.name)

    try:
        with (
            gzip.open(bundle_path, mode="wb") as fileobj,
            tarfile.open(fileobj=fileobj, mode="w") as archive,
        ):
            for section in sections:
                candidate = _bundle_section_path(base_dir, section)
                if candidate is None or not candidate.exists():
                    continue
                archive.add(candidate, arcname=f"{task_id}/{candidate.name}")
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise

    return bundle_path


def _bundle_section_path(base_dir: Path, section: str) -> Path | None:
    if section == "results":
        return base_dir / RESULTS_NAME
    if section == "artifacts":
        return base_dir / ARTIFACTS_DIR
    if section == "logs":
        return base_dir / LOGS_DIR
    return None


def _cleanup_bundle_file(path: Path) -> None:
    path.unlink(missing_ok=True)
