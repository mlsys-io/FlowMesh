import fcntl
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import ResultEnvelope
from shared.schemas.result_delivery import DeliveredFile, ResultDeliveryReceipt
from shared.utils.atomic import atomic_write_text
from shared.utils.manifest import prepare_output_dir

RECEIPT_NAME = ".delivery.json"


@contextmanager
def delivery_lock(base_dir: Path) -> Iterator[None]:
    prepare_output_dir(base_dir)
    lock_path = base_dir / ".delivery.lock"
    with lock_path.open("a+b") as lock:
        try:
            lock_path.chmod(0o666)
        except OSError:
            pass
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def artifact_path(value: Any) -> str | None:
    if isinstance(value, ArtifactRef):
        return value.path
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return value["path"]
    return None


def result_field(value: Any, selector: str) -> Any:
    for part in selector.split("."):
        if isinstance(value, BaseModel):
            value = dict(value).get(part)
        elif isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def result_generation(base_dir: Path) -> str:
    envelope = ResultEnvelope.model_validate_json(
        (base_dir / "results.json").read_text()
    )
    canonical = json.dumps(
        envelope.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def describe_file(path: Path) -> DeliveredFile:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return DeliveredFile(size=path.stat().st_size, sha256=digest.hexdigest())


def make_receipt(
    base_dir: Path, task_id: str, paths: list[str] | None
) -> ResultDeliveryReceipt:
    receipt = ResultDeliveryReceipt(
        task_id=task_id,
        generation=result_generation(base_dir),
        all_artifacts=paths is None,
        artifact_paths=paths or [],
    )
    roots = (
        [Path("artifacts")]
        if paths is None
        else [Path("artifacts") / safe_relative(p) for p in paths]
    )
    for root in roots:
        target = base_dir / root
        target.resolve().relative_to((base_dir / "artifacts").resolve())
        if not target.exists():
            raise ValueError(f"Missing artifact: {root}")
        for entry in (
            [target, *sorted(target.rglob("*"))] if target.is_dir() else [target]
        ):
            if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
                raise ValueError(f"Unsupported artifact: {entry}")
            name = entry.relative_to(base_dir).as_posix()
            if entry.is_dir():
                receipt.directories.append(name)
            else:
                receipt.files[name] = describe_file(entry)
    return receipt


def read_receipt(base_dir: Path) -> ResultDeliveryReceipt | None:
    try:
        return ResultDeliveryReceipt.model_validate_json(
            (base_dir / RECEIPT_NAME).read_text()
        )
    except (OSError, ValueError):
        return None


def validate_receipt(
    base_dir: Path,
    receipt: ResultDeliveryReceipt,
    task_id: str,
    generation: str | None = None,
) -> None:
    envelope = ResultEnvelope.model_validate_json(
        (base_dir / "results.json").read_text()
    )
    if (
        envelope.task_id != task_id
        or receipt.task_id != task_id
        or receipt.generation != result_generation(base_dir)
        or (generation is not None and generation != receipt.generation)
    ):
        raise ValueError(f"Result snapshot mismatch for {task_id}")
    for name in receipt.directories:
        if not name.startswith("artifacts/") and name != "artifacts":
            raise ValueError(f"Invalid artifact directory: {name}")
        path = base_dir / safe_relative(name)
        path.resolve().relative_to(base_dir.resolve())
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Missing artifact directory: {name}")
    for name, expected in receipt.files.items():
        if not name.startswith("artifacts/"):
            raise ValueError(f"Invalid artifact file: {name}")
        path = base_dir / safe_relative(name)
        path.resolve().relative_to(base_dir.resolve())
        if path.is_symlink() or not path.is_file() or describe_file(path) != expected:
            raise ValueError(f"Incomplete artifact: {name}")
    for name in receipt.artifact_paths:
        root = (Path("artifacts") / safe_relative(name)).as_posix()
        if root not in receipt.directories and root not in receipt.files:
            raise ValueError(f"Missing selection: {name}")
    if receipt.all_artifacts and "artifacts" not in receipt.directories:
        raise ValueError("Missing complete artifacts directory")


def artifacts_ready(
    base_dir: Path,
    task_id: str,
    paths: list[str] | None = None,
    generation: str | None = None,
) -> bool:
    receipt = read_receipt(base_dir)
    if receipt is None:
        try:
            envelope = ResultEnvelope.model_validate_json(
                (base_dir / "results.json").read_text()
            )
            if (envelope.metadata or {}).get("independent_results") or generation:
                return False
            roots = (
                [base_dir / "artifacts"]
                if paths is None
                else [base_dir / "artifacts" / safe_relative(p) for p in paths]
            )
            return all(path.exists() for path in roots)
        except (OSError, ValueError):
            return False
    try:
        validate_receipt(base_dir, receipt, task_id, generation)
        if paths is None:
            return receipt.all_artifacts
        for selection in paths:
            name = (Path("artifacts") / safe_relative(selection)).as_posix()
            if name not in receipt.files and name not in receipt.directories:
                return False
        return all(
            receipt.all_artifacts
            or any(
                Path(p) == Path(root) or Path(root) in Path(p).parents
                for root in receipt.artifact_paths
            )
            for p in paths
        )
    except (OSError, ValueError):
        return False


def write_receipt(base_dir: Path, receipt: ResultDeliveryReceipt) -> None:
    atomic_write_text(base_dir / RECEIPT_NAME, receipt.model_dump_json())


def create_delivery_bundle(
    base_dir: Path, task_id: str, paths: list[str] | None, include_traces: bool = False
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="flowmesh-delivery-", suffix=".tar", delete=False
    ) as sink:
        bundle = Path(sink.name)
    try:
        with delivery_lock(base_dir):
            receipt = make_receipt(base_dir, task_id, paths)
            with tarfile.open(bundle, "w") as archive:
                archive.add(
                    base_dir / "results.json", arcname=f"{task_id}/results.json"
                )
                for name in sorted(set(receipt.directories)):
                    archive.add(
                        base_dir / name, arcname=f"{task_id}/{name}", recursive=False
                    )
                for name in receipt.files:
                    archive.add(
                        base_dir / name, arcname=f"{task_id}/{name}", recursive=False
                    )
                if include_traces:
                    for name in ("spans.jsonl", "assets.jsonl", "lineage.jsonl"):
                        path = base_dir / "logs" / name
                        if path.is_file() and not path.is_symlink():
                            archive.add(path, arcname=f"{task_id}/logs/{name}")
                add_receipt(archive, task_id, receipt)
        return bundle
    except Exception:
        bundle.unlink(missing_ok=True)
        raise


def add_receipt(
    archive: tarfile.TarFile, task_id: str, receipt: ResultDeliveryReceipt
) -> None:
    content = receipt.model_dump_json().encode()
    info = tarfile.TarInfo(f"{task_id}/{RECEIPT_NAME}")
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def extract_delivery_bundle(bundle: Path, destination: Path, task_id: str) -> Path:
    safe_relative(task_id)
    if len(Path(task_id).parts) != 1:
        raise ValueError("Invalid task ID")
    with tarfile.open(bundle, "r:*") as archive:
        names: set[str] = set()
        for member in archive:
            relative = safe_relative(member.name)
            if (
                relative.parts[0] != task_id
                or member.name in names
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe bundle member: {member.name}")
            names.add(member.name)
            archive.extract(member, destination, filter="data")
    base_dir = destination / task_id
    receipt = read_receipt(base_dir)
    if receipt is None:
        raise ValueError("Result bundle has no delivery receipt")
    validate_receipt(base_dir, receipt, task_id)
    allowed = {
        "results.json",
        RECEIPT_NAME,
        "logs",
        "logs/spans.jsonl",
        "logs/assets.jsonl",
        "logs/lineage.jsonl",
        *receipt.directories,
        *receipt.files,
    }
    for name in list(allowed):
        allowed.update(
            parent.as_posix() for parent in Path(name).parents if parent != Path(".")
        )
    for entry in base_dir.rglob("*"):
        name = entry.relative_to(base_dir).as_posix()
        if name not in allowed:
            raise ValueError(f"Unexpected delivery member: {name}")
    return base_dir


def commit_delivery(
    staging: Path,
    destination: Path,
    task_id: str,
    validate_current: Callable[[ResultEnvelope], None] | None = None,
) -> None:
    receipt = read_receipt(staging)
    if receipt is None:
        raise ValueError("Result bundle has no delivery receipt")
    validate_receipt(staging, receipt, task_id)
    with delivery_lock(destination):
        if validate_current is not None:
            validate_current(
                ResultEnvelope.model_validate_json(
                    (staging / "results.json").read_text()
                )
            )
        existing = read_receipt(destination)
        keep_existing = False
        if existing is not None and existing.generation == receipt.generation:
            try:
                validate_receipt(destination, existing, task_id)
                keep_existing = True
            except (OSError, ValueError):
                pass
        (destination / RECEIPT_NAME).unlink(missing_ok=True)
        if not keep_existing:
            shutil.rmtree(destination / "artifacts", ignore_errors=True)
            prepare_output_dir(destination)
        for entry in staging.rglob("*"):
            name = entry.relative_to(staging)
            if name.as_posix() == RECEIPT_NAME:
                continue
            target = destination / name
            target.resolve().relative_to(destination.resolve())
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                try:
                    target.chmod(0o777)
                except OSError:
                    pass
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                for parent in [target.parent, *target.parent.parents]:
                    if parent == destination:
                        break
                    try:
                        parent.chmod(0o777)
                    except OSError:
                        pass
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".delivery-", delete=False
                ) as temporary:
                    replacement = Path(temporary.name)
                try:
                    shutil.copyfile(entry, replacement)
                    shutil.copymode(entry, replacement)
                    replacement.replace(target)
                finally:
                    replacement.unlink(missing_ok=True)
        if keep_existing and existing is not None:
            receipt.all_artifacts |= existing.all_artifacts
            receipt.artifact_paths = sorted(
                set(receipt.artifact_paths + existing.artifact_paths)
            )
            receipt.directories = sorted(
                set(receipt.directories + existing.directories)
            )
            receipt.files = existing.files | receipt.files
        validate_receipt(destination, receipt, task_id)
        write_receipt(destination, receipt)
