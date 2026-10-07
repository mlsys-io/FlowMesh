import asyncio
import contextlib
import gzip
import json
import logging
import mimetypes
import os
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import quote

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
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

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
from shared.utils.result_delivery import (
    RECEIPT_NAME,
    add_receipt,
    artifacts_ready,
    commit_delivery,
    delivery_lock,
    extract_delivery_bundle,
    make_receipt,
    portable_link,
    read_receipt,
    result_generation,
    safe_relative,
    selection_roots,
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
    _require_current_dispatch(runtime, task_id, envelope)

    try:
        path = await asyncio.to_thread(_store_result, results_dir, envelope)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store result: {exc}",
        ) from exc

    expected_artifacts: list[str] = []
    record = runtime.get_record(task_id)
    if record:
        expected_artifacts = record.task.spec.get_artifacts()
    await asyncio.to_thread(sync_manifest, path.parent, task_id, expected_artifacts)
    pending_children = event_monitor.pop_pending_clones(task_id)
    if pending_children:
        await asyncio.to_thread(
            event_monitor.mirror_task_results, task_id, pending_children
        )
    return PathResponse(ok=True, path=str(path))


@router.post("/{task_id}/delivery", summary="Publish a complete result selection")
async def ingest_delivery(
    task_id: str,
    file: UploadFile = File(...),
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> PathResponse:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    try:
        if safe_relative(task_id).name != task_id:
            raise ValueError("Invalid task ID")
        destination = await run_in_threadpool(
            _ingest_delivery, file.file, results_dir, task_id, runtime
        )
        return PathResponse(ok=True, path=destination.as_posix())
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid result delivery: {exc}"
        ) from exc


@router.get(
    "/{task_id}/delivery",
    summary="Check whether a result selection is already held",
    description="Check whether a result snapshot selection is held.",
    response_description="Selection held (204); not held (404)",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def check_delivery(
    task_id: str,
    generation: str = Query(...),
    artifact_path: list[str] = Query(default_factory=list),
    all_artifacts: bool = Query(default=False),
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    # A worker that shares the results volume finds its own snapshot here and
    # skips the upload.
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    try:
        if safe_relative(task_id).name != task_id:
            raise ValueError("Invalid task ID")
        for path in artifact_path:
            safe_relative(path)
        if all_artifacts and artifact_path:
            raise ValueError("Pass either artifact_path or all_artifacts, not both")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    paths = None if all_artifacts else artifact_path
    if not await asyncio.to_thread(
        _holds_delivery, results_dir, task_id, paths, generation
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="selection not held"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    runtime: TaskRuntime = Depends(get_runtime),
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
    _require_terminal_result(runtime, task_id)
    try:
        raw = await run_in_threadpool(_read_result_locked, results_dir, task_id)
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
        envelope = ResultEnvelope.model_validate(content)
        _require_current_dispatch(runtime, task_id, envelope)
        return envelope.result
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
    atomic_write_stream(
        target_path, file.file, commit=partial(_invalidating_receipt, base_dir)
    )


@contextlib.contextmanager
def _invalidating_receipt(base_dir: Path) -> Iterator[None]:
    """Hold the task's snapshot lock while an upload is installed, dropping the
    delivery receipt the changed artifacts no longer match."""
    with delivery_lock(base_dir):
        (base_dir / RECEIPT_NAME).unlink(missing_ok=True)
        yield


@router.get(
    "/{task_id}/files/{filename:path}",
    summary="Download a result artifact",
    description="Download an artifact file for a task result.",
    response_description="Result file",
    response_class=StreamingResponse,
)
async def download_result_file(
    task_id: str,
    filename: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    _require_terminal_result(runtime, task_id)
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

    handle = await run_in_threadpool(
        _open_current_file, runtime, task_id, base_dir, target_path
    )
    return StreamingResponse(
        _iter_file(handle),
        media_type=mimetypes.guess_type(sanitized.name)[0]
        or "application/octet-stream",
        headers={
            "Content-Length": str(os.fstat(handle.fileno()).st_size),
            "Content-Disposition": _attachment(sanitized.name),
        },
        background=BackgroundTask(handle.close),
    )


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
    artifact_path: list[str] = Query(default_factory=list),
    generation: str | None = Query(default=None),
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
            _create_result_bundle_archive,
            task_id,
            base_dir,
            sections,
            artifact_path,
            generation,
            record.result_dispatch if record is not None else None,
        )
    except HTTPException:
        raise
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
    artifact_paths: list[str] | None = None,
    generation: str | None = None,
    dispatch_id: str | None = None,
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"flowmesh-result-{task_id}-",
        suffix=".tar.gz",
        delete=False,
    ) as tmp:
        bundle_path = Path(tmp.name)

    try:
        with delivery_lock(base_dir):
            if dispatch_id is not None and (base_dir / RESULTS_NAME).is_file():
                envelope = ResultEnvelope.model_validate_json(
                    (base_dir / RESULTS_NAME).read_text()
                )
                if (envelope.metadata or {}).get("independent_results") and (
                    envelope.metadata or {}
                ).get("result_dispatch") != dispatch_id:
                    raise HTTPException(
                        status_code=404,
                        detail="Result belongs to an earlier task attempt",
                    )
            for section in sections:
                candidate = _bundle_section_path(base_dir, section)
                if candidate is None or not candidate.exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Requested result section unavailable: {section}",
                    )
            paths = artifact_paths or None
            receipt = read_receipt(base_dir)
            if "artifacts" in sections and not artifacts_ready(
                base_dir, task_id, paths, generation, verify_content=False
            ):
                # Without a receipt, a client's full bundle carries what is
                # stored, as for a result published through its own destination;
                # dependents always ask for a receipted selection or generation.
                if receipt is not None or paths is not None or generation is not None:
                    raise HTTPException(
                        status_code=404,
                        detail="Requested artifacts have not been completely delivered",
                    )
            transferable = partial(_transferable, base_dir)
            with (
                gzip.open(bundle_path, mode="wb") as fileobj,
                tarfile.open(fileobj=fileobj, mode="w") as archive,
            ):
                for section in sections:
                    candidate = _bundle_section_path(base_dir, section)
                    if candidate is None:
                        continue
                    if section == "artifacts" and paths is not None:
                        archive.add(
                            candidate, arcname=f"{task_id}/artifacts", recursive=False
                        )
                        for name in selection_roots(base_dir, paths):
                            _add_tree(
                                archive,
                                candidate / safe_relative(name),
                                f"{task_id}/artifacts/{name}",
                                transferable,
                            )
                    else:
                        _add_tree(
                            archive,
                            candidate,
                            f"{task_id}/{candidate.name}",
                            transferable,
                        )
                if "results" in sections and "artifacts" in sections:
                    add_receipt(
                        archive, task_id, make_receipt(base_dir, task_id, paths)
                    )
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise

    return bundle_path


def _require_current_dispatch(
    runtime: TaskRuntime, task_id: str, envelope: ResultEnvelope
) -> None:
    record = runtime.get_record(task_id)
    metadata = envelope.metadata or {}
    if (
        record is not None
        and metadata.get("independent_results")
        and metadata.get("result_dispatch") != record.result_dispatch
    ):
        raise HTTPException(
            status_code=404, detail="Result belongs to an earlier task attempt"
        )


def _require_terminal_result(runtime: TaskRuntime, task_id: str) -> None:
    record = runtime.get_record(task_id)
    if record is not None and record.status not in TERMINAL_TASK_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is not terminal; result unavailable",
        )


def _store_result(results_dir: Path, envelope: ResultEnvelope) -> Path:
    base_dir = result_file_path(results_dir, envelope.task_id).parent
    with delivery_lock(base_dir):
        previous = read_receipt(base_dir)
        path = write_result(results_dir, envelope)
        if previous is not None and previous.generation != result_generation(base_dir):
            (base_dir / RECEIPT_NAME).unlink(missing_ok=True)
    return path


def _ingest_delivery(
    upload: BinaryIO, results_dir: Path, task_id: str, runtime: TaskRuntime
) -> Path:
    upload.seek(0)
    with tempfile.TemporaryDirectory(prefix="flowmesh-ingest-") as temporary:
        staging = extract_delivery_bundle(upload, Path(temporary), task_id)
        envelope = ResultEnvelope.model_validate_json(
            (staging / RESULTS_NAME).read_text()
        )
        _require_current_dispatch(runtime, task_id, envelope)
        destination = result_file_path(results_dir, task_id).parent
        commit_delivery(
            staging,
            destination,
            task_id,
            partial(_require_current_dispatch, runtime, task_id),
        )
    return destination


def _holds_delivery(
    results_dir: Path, task_id: str, paths: list[str] | None, generation: str
) -> bool:
    base_dir = result_file_path(results_dir, task_id).parent
    if not base_dir.is_dir():
        # The lock would create the directory; nothing is held there anyway.
        return False
    with delivery_lock(base_dir):
        return artifacts_ready(
            base_dir, task_id, paths, generation, verify_content=False
        )


def _read_result_locked(results_dir: Path, task_id: str) -> str:
    with delivery_lock(result_file_path(results_dir, task_id).parent):
        return read_result(results_dir, task_id)


def _open_current_file(
    runtime: TaskRuntime, task_id: str, base_dir: Path, target: Path
) -> BinaryIO:
    """Open ``target`` while the task's snapshot is locked.

    Deliveries replace files by rename, so the open handle keeps serving the
    snapshot that was current when it was opened.
    """
    with delivery_lock(base_dir):
        envelope_path = base_dir / RESULTS_NAME
        if envelope_path.is_file():
            envelope = ResultEnvelope.model_validate_json(envelope_path.read_text())
            _require_current_dispatch(runtime, task_id, envelope)
        resolved = target.resolve()
        try:
            resolved.relative_to(base_dir.resolve())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
            )
        return resolved.open("rb")


def _iter_file(handle: BinaryIO) -> Iterator[bytes]:
    while chunk := handle.read(64 * 1024):
        yield chunk


def _attachment(filename: str) -> str:
    quoted = quote(filename)
    if quoted != filename:
        return f"attachment; filename*=utf-8''{quoted}"
    return f'attachment; filename="{filename}"'


def _transferable(base_dir: Path, member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if member.issym():
        parts = PurePosixPath(member.name).parts[1:]
        if parts[:1] != ("artifacts",):
            return None
        link_path = base_dir.joinpath(*parts)
        member.linkname = portable_link(link_path, base_dir)
        return member
    return member if member.isfile() or member.isdir() else None


def _add_tree(
    archive: tarfile.TarFile,
    root: Path,
    arcname: str,
    member_filter: Callable[[tarfile.TarInfo], tarfile.TarInfo | None] | None = None,
) -> None:
    """Add ``root`` and everything under it, leaving out in-flight atomic writes and
    any file removed while the archive is built. A link is added as a link and
    never descended into."""
    archive.add(root, arcname=arcname, recursive=False, filter=member_filter)
    if root.is_symlink():
        return
    for path in _bounded_walk(root):
        if is_atomic_temp(path.name):
            continue
        with contextlib.suppress(FileNotFoundError):
            archive.add(
                path,
                arcname=f"{arcname}/{path.relative_to(root).as_posix()}",
                recursive=False,
                filter=member_filter,
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
