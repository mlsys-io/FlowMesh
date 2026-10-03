import asyncio
import contextlib
import gzip
import json
import logging
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

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
from pydantic import ValidationError

from shared.schemas.result import (
    AnyExecutorResult,
    ResultEnvelope,
    read_result,
    result_file_path,
    write_result,
)
from shared.utils.atomic import atomic_write_stream, is_atomic_temp
from shared.utils.manifest import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    RESULTS_NAME,
    prepare_output_dir,
    sync_manifest,
)

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
from ...hooks import ResourceAction, ResourceKind
from ...schemas.common import PathResponse
from ...services.monitoring import EventMonitor
from ...task.models import TERMINAL_TASK_STATUSES
from ...task.runtime import TaskRuntime

# Sections the bundle endpoint can include.
_BUNDLE_SECTIONS_CONCRETE = ("results", "artifacts", "logs")
_BUNDLE_SECTIONS_ACCEPTED = (*_BUNDLE_SECTIONS_CONCRETE, "all")
_BUNDLE_SECTIONS_DEFAULT = ("results", "artifacts")
# The deepest a results-tree walk descends; a deeper tree is left unarchived below it.
_MAX_WALK_DEPTH = 128

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
    envelope: ResultEnvelope,
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    event_monitor: EventMonitor = Depends(get_event_monitor),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> PathResponse:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    task_id = envelope.task_id.strip()
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="task_id is required"
        )
    envelope.task_id = task_id

    try:
        path = write_result(results_dir, envelope)
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
) -> AnyExecutorResult:
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="task_id is required"
        )
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
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
        content = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Result file is not valid JSON: {exc}",
        ) from exc
    try:
        return ResultEnvelope.model_validate(content).result
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Result file does not match ResultEnvelope: {exc}",
        ) from exc


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
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> PathResponse:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    base_dir = result_file_path(results_dir, task_id).parent
    relative_path = _resolve_artifact_path(file.filename or "")
    target_path = (base_dir / relative_path).resolve()

    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        )

    record = runtime.get_record(task_id)
    expected_artifacts = record.task.spec.get_artifacts() if record else []
    try:
        await asyncio.to_thread(_store_artifact, file, base_dir, target_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store artifact: {exc}",
        ) from exc
    await asyncio.to_thread(sync_manifest, base_dir, task_id, expected_artifacts)
    return PathResponse(ok=True, path=str(target_path))


def _store_artifact(file: UploadFile, base_dir: Path, target_path: Path) -> None:
    prepare_output_dir(base_dir)
    atomic_write_stream(target_path, file.file)


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
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
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
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
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
        bundle_path = await asyncio.to_thread(
            _create_result_bundle_archive, task_id, base_dir, sections=sections
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
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
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
                _add_tree(archive, candidate, f"{task_id}/{candidate.name}")
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise

    return bundle_path


def _add_tree(archive: tarfile.TarFile, root: Path, arcname: str) -> None:
    """Add ``root`` and everything under it, leaving out in-flight atomic writes and
    any file removed while the archive is built."""
    archive.add(root, arcname=arcname, recursive=False)
    for path in _bounded_walk(root):
        if is_atomic_temp(path.name):
            continue
        with contextlib.suppress(FileNotFoundError):
            archive.add(
                path,
                arcname=f"{arcname}/{path.relative_to(root).as_posix()}",
                recursive=False,
            )


def _bounded_walk(root: Path) -> Iterator[Path]:
    """Yield every path under ``root``, descending no deeper than
    ``_MAX_WALK_DEPTH`` so a pathological tree cannot be walked unbounded."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= _MAX_WALK_DEPTH:
            continue
        try:
            children = sorted(current.iterdir())
        except OSError:
            continue
        for child in children:
            yield child
            if child.is_dir() and not child.is_symlink():
                stack.append((child, depth + 1))


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
