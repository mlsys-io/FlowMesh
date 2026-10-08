import errno
import fcntl
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Self

from pydantic import BaseModel

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import ResultEnvelope
from shared.schemas.result_delivery import DeliveredFile, ResultDeliveryReceipt
from shared.utils.atomic import atomic_write_text, is_atomic_temp
from shared.utils.manifest import prepare_output_dir

RECEIPT_NAME = ".delivery.json"
_MISSING_ERRNOS = {errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP}
_MISSING = object()


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


def dig_result_path(value: Any, parts: list[str]) -> Any:
    """Follow a stage-reference path into a result; ``None`` when it is absent.

    Raises ``ValueError`` when a list is indexed by a non-integer part.
    """
    current = value
    for part in parts:
        part = part.strip()
        if part == "":
            continue
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"List index must be integer in reference path, got '{part}'"
                ) from exc
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        elif isinstance(current, BaseModel):
            current = getattr(current, part, _MISSING)
            if current is _MISSING:
                return None
        else:
            return None
    return current


def result_field(value: Any, selector: str) -> Any:
    try:
        return dig_result_path(value, selector.split("."))
    except ValueError:
        return None


def result_generation(base_dir: Path) -> str:
    envelope = ResultEnvelope.from_file(base_dir / "results.json")
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
    artifacts_root = (base_dir / "artifacts").resolve()
    roots = (
        [Path("artifacts")]
        if paths is None
        else [
            Path("artifacts") / safe_relative(p)
            for p in selection_roots(base_dir, paths)
        ]
    )
    directories: set[str] = set()
    for root in roots:
        target = base_dir / root
        if paths is None:
            if target.is_symlink():
                raise ValueError(f"Unsupported artifact: {target}")
        else:
            target.parent.resolve().relative_to(artifacts_root)
        if not target.exists() and not target.is_symlink():
            raise ValueError(f"Missing artifact: {root}")
        entries = (
            [target, *sorted(target.rglob("*"))]
            if target.is_dir() and not target.is_symlink()
            else [target]
        )
        for entry in entries:
            if is_atomic_temp(entry.name):
                # An upload still being written is not part of the snapshot.
                continue
            name = entry.relative_to(base_dir).as_posix()
            if entry.is_symlink():
                receipt.symlinks[name] = portable_link(entry, base_dir)
            elif entry.is_dir():
                directories.add(name)
            elif entry.is_file() and name not in receipt.files:
                receipt.files[name] = describe_file(entry)
            # Sockets and FIFOs cannot be transferred and are left out.
    receipt.directories = sorted(directories)
    return receipt


def portable_link(link_path: Path, base_dir: Path) -> str:
    """Return the target of ``link_path`` in a form that survives relocation.

    An absolute target inside this task's own artifacts becomes relative. That
    path was written in the producer's mount namespace, so it is recognized by its
    ``<task_id>/artifacts/`` segment, with ``base_dir`` named after the task.
    """
    target = os.readlink(link_path)
    path = PurePosixPath(target)
    if not path.is_absolute():
        return target
    parts = path.parts
    for index in range(len(parts) - 1):
        inside = parts[index + 2 :]
        if (
            parts[index] == base_dir.name
            and parts[index + 1] == "artifacts"
            and ".." not in inside
        ):
            return os.path.relpath(
                base_dir.joinpath("artifacts", *inside), link_path.parent
            )
    return target


def relocated_links(base_dir: Path, paths: list[str] | None) -> dict[str, str]:
    """Return the links in a selection whose target changes when it is copied.

    Keys are paths relative to ``base_dir``; values are the portable targets a
    copy has to carry instead of the stored ones.
    """
    roots = (
        [base_dir / "artifacts"]
        if paths is None
        else [
            base_dir / "artifacts" / safe_relative(p)
            for p in selection_roots(base_dir, paths)
        ]
    )
    changes: dict[str, str] = {}
    for root in roots:
        entries = (
            [root, *root.rglob("*")]
            if root.is_dir() and not root.is_symlink()
            else [root]
        )
        for entry in entries:
            if not entry.is_symlink():
                continue
            link = portable_link(entry, base_dir)
            if link != os.readlink(entry):
                changes[entry.relative_to(base_dir).as_posix()] = link
    return changes


def selection_roots(base_dir: Path, paths: list[str]) -> list[str]:
    """Return ``paths`` plus the in-artifacts targets of the links they contain.

    A link travels as a link, so its target has to travel with it for the link to
    resolve on the receiving side. Links are followed one hop at a time, so a
    chain of links brings every intermediate link along.
    """
    artifacts_root = (base_dir / "artifacts").absolute()
    roots: list[str] = []
    pending = list(paths)
    while pending:
        selection = pending.pop(0)
        if selection in roots:
            continue
        roots.append(selection)
        target = base_dir / "artifacts" / safe_relative(selection)
        entries = (
            [target, *target.rglob("*")]
            if target.is_dir() and not target.is_symlink()
            else [target]
        )
        for entry in entries:
            if not entry.is_symlink():
                continue
            link = portable_link(entry, base_dir)
            hop = Path(os.path.normpath(entry.parent.absolute() / link))
            try:
                linked = hop.relative_to(artifacts_root)
            except ValueError:
                continue
            if linked.parts:
                pending.append(linked.as_posix())
    return [
        root
        for root in roots
        if not any(
            other != root and Path(other) in Path(root).parents for other in roots
        )
    ]


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
    verify_content: bool = True,
) -> None:
    """Check that ``base_dir`` holds the snapshot ``receipt`` describes.

    With ``verify_content=False`` files are matched by size instead of hash, which
    is enough to tell a complete snapshot from an in-progress one.
    """
    envelope = ResultEnvelope.from_file(base_dir / "results.json")
    # The receipt must describe this task's envelope as it is on disk now.
    if (
        envelope.task_id != task_id
        or receipt.task_id != task_id
        or receipt.generation != result_generation(base_dir)
        or (generation is not None and generation != receipt.generation)
    ):
        raise ValueError(f"Result snapshot mismatch for {task_id}")
    # Every listed entry must sit under artifacts/ and exist as recorded. Each
    # entry's type is checked before its parent is confined, and its parent is
    # confined before any file is opened.
    root = base_dir.resolve()
    confined: set[Path] = set()

    def confine(path: Path) -> None:
        # A listed entry that is not itself a link resolves inside its parent, so
        # confining each parent once confines every entry beneath it.
        if path.parent not in confined:
            path.parent.resolve().relative_to(root)
            confined.add(path.parent)

    for name in receipt.directories:
        if not name.startswith("artifacts/") and name != "artifacts":
            raise ValueError(f"Invalid artifact directory: {name}")
        path = base_dir / safe_relative(name)
        info = _lstat(path)
        if info is None or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Missing artifact directory: {name}")
        confine(path)
    for name, expected in receipt.files.items():
        if not name.startswith("artifacts/"):
            raise ValueError(f"Invalid artifact file: {name}")
        path = base_dir / safe_relative(name)
        info = _lstat(path)
        if info is None or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Incomplete artifact: {name}")
        confine(path)
        actual = (
            describe_file(path)
            if verify_content
            else DeliveredFile(size=info.st_size, sha256=expected.sha256)
        )
        if actual != expected:
            raise ValueError(f"Incomplete artifact: {name}")
    for name, link in receipt.symlinks.items():
        if not name.startswith("artifacts/"):
            raise ValueError(f"Invalid artifact link: {name}")
        path = base_dir / safe_relative(name)
        info = _lstat(path)
        if info is None or not stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Incomplete artifact: {name}")
        confine(path)
        if portable_link(path, base_dir) != link:
            raise ValueError(f"Incomplete artifact: {name}")
    # The listed entries must cover every selection the snapshot claims.
    for name in receipt.artifact_paths:
        if not receipt.has_artifact(name):
            raise ValueError(f"Missing selection: {name}")
    if receipt.all_artifacts and "artifacts" not in receipt.directories:
        raise ValueError("Missing complete artifacts directory")


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError as exc:
        if exc.errno in _MISSING_ERRNOS:
            return None
        raise


@dataclass(frozen=True)
class SnapshotCoverage:
    """The artifact selections a validated snapshot can serve."""

    generation: str
    all_artifacts: bool
    artifact_paths: tuple[str, ...]
    names: frozenset[str]

    @classmethod
    def of(cls, receipt: ResultDeliveryReceipt) -> Self:
        return cls(
            generation=receipt.generation,
            all_artifacts=receipt.all_artifacts,
            artifact_paths=tuple(receipt.artifact_paths),
            names=frozenset([*receipt.directories, *receipt.files, *receipt.symlinks]),
        )

    def covers(self, paths: list[str] | None) -> bool:
        """Whether the snapshot holds every selection in ``paths``, or all
        artifacts when ``paths`` is ``None``."""
        if paths is None:
            return self.all_artifacts
        return all(
            PurePosixPath("artifacts", selection).as_posix() in self.names
            and (
                self.all_artifacts
                or any(
                    Path(selection) == Path(root)
                    or Path(root) in Path(selection).parents
                    for root in self.artifact_paths
                )
            )
            for selection in paths
        )


# Device, inode, mtime_ns and size of one file.
type _FileIdentity = tuple[int, int, int, int]
# The envelope's and the receipt's file identities; compare for equality only.
type SnapshotIdentity = tuple[_FileIdentity, _FileIdentity]


def snapshot_identity(base_dir: Path) -> SnapshotIdentity | None:
    """Identify the envelope and receipt files in ``base_dir``; ``None`` when either
    is missing. Any change to either file changes the identity."""
    try:
        envelope = os.stat(base_dir / "results.json")
        receipt = os.stat(base_dir / RECEIPT_NAME)
    except OSError:
        return None
    return _file_identity(envelope), _file_identity(receipt)


def _file_identity(info: os.stat_result) -> _FileIdentity:
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)


def validated_snapshot(
    base_dir: Path,
    task_id: str,
    generation: str | None = None,
    verify_content: bool = True,
) -> tuple[ResultDeliveryReceipt, SnapshotIdentity] | None:
    """Validate the snapshot in ``base_dir`` against its receipt.

    Returns the receipt with the ``snapshot_identity`` of the files it was read
    from, ``None`` when there is no readable receipt, and raises ``ValueError`` or
    ``OSError`` when the snapshot does not match it.
    """
    try:
        with (base_dir / RECEIPT_NAME).open("rb") as source:
            receipt_info = os.fstat(source.fileno())
            receipt = ResultDeliveryReceipt.model_validate_json(source.read())
    except (OSError, ValueError):
        return None
    # Taken before validation reads the envelope, so a concurrent rewrite leaves an
    # identity that no later stat matches.
    envelope_info = os.stat(base_dir / "results.json")
    validate_receipt(base_dir, receipt, task_id, generation, verify_content)
    return receipt, (_file_identity(envelope_info), _file_identity(receipt_info))


def artifacts_ready(
    base_dir: Path,
    task_id: str,
    paths: list[str] | None = None,
    generation: str | None = None,
    verify_content: bool = True,
) -> bool:
    try:
        snapshot = validated_snapshot(base_dir, task_id, generation, verify_content)
    except (OSError, ValueError):
        return False
    if snapshot is None:
        try:
            envelope = ResultEnvelope.from_file(base_dir / "results.json")
            if (envelope.metadata or {}).get("independent_results") or generation:
                return False
            roots = (
                [base_dir / "artifacts"]
                if paths is None
                else [base_dir / "artifacts" / safe_relative(p) for p in paths]
            )
            return all(path.exists() or path.is_symlink() for path in roots)
        except (OSError, ValueError):
            return False
    return SnapshotCoverage.of(snapshot[0]).covers(paths)


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
                for name in [*sorted(set(receipt.directories)), *receipt.files]:
                    archive.add(
                        base_dir / name, arcname=f"{task_id}/{name}", recursive=False
                    )
                for name, link in receipt.symlinks.items():
                    info = archive.gettarinfo(base_dir / name, f"{task_id}/{name}")
                    info.linkname = link
                    archive.addfile(info)
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


def extract_delivery_bundle(
    bundle: Path | BinaryIO, destination: Path, task_id: str
) -> Path:
    """Extract a delivery bundle under ``destination`` and verify its receipt.

    Symlinks are recreated as links after every other member is written, and no
    member may sit beneath one, so a link can never redirect a write.
    """
    safe_relative(task_id)
    if len(Path(task_id).parts) != 1:
        raise ValueError("Invalid task ID")
    with (
        tarfile.open(bundle, "r:*")
        if isinstance(bundle, Path)
        else tarfile.open(fileobj=bundle, mode="r:*")
    ) as archive:
        members = archive.getmembers()
        names: set[str] = set()
        links: dict[str, str] = {}
        # Members are unique, normalized paths under the task directory; only
        # entries under artifacts/ may be links.
        for member in members:
            relative = safe_relative(member.name)
            if (
                member.name != PurePosixPath(member.name).as_posix()
                or relative.parts[0] != task_id
                or member.name in names
                or not (member.isfile() or member.isdir() or member.issym())
            ):
                raise ValueError(f"Unsafe bundle member: {member.name}")
            names.add(member.name)
            if member.issym():
                if relative.parts[1:2] != ("artifacts",) or len(relative.parts) < 3:
                    raise ValueError(f"Unsafe bundle member: {member.name}")
                links[member.name] = member.linkname
        # Nothing may be written through a link.
        for member in members:
            if any(parent.as_posix() in links for parent in Path(member.name).parents):
                raise ValueError(f"Unsafe bundle member: {member.name}")
        for member in members:
            if not member.issym():
                archive.extract(member, destination, filter="data")
    # Links are created only after every other member is in place.
    root = destination.resolve()
    for name, link in links.items():
        path = destination / name
        path.parent.resolve().relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(link)
    base_dir = destination / task_id
    receipt = read_receipt(base_dir)
    if receipt is None:
        raise ValueError("Result bundle has no delivery receipt")
    validate_receipt(base_dir, receipt, task_id)
    # Reject any entry the receipt does not account for, other than the envelope,
    # the receipt itself, trace logs, and the parents of listed entries.
    allowed = {
        "results.json",
        RECEIPT_NAME,
        "logs",
        "logs/spans.jsonl",
        "logs/assets.jsonl",
        "logs/lineage.jsonl",
        *receipt.directories,
        *receipt.files,
        *receipt.symlinks,
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
    """Install a snapshot unpacked by ``extract_delivery_bundle`` at ``destination``.

    The extraction already verified the staged content, so it is matched here by
    size only.
    """
    receipt = read_receipt(staging)
    if receipt is None:
        raise ValueError("Result bundle has no delivery receipt")
    validate_receipt(staging, receipt, task_id, verify_content=False)
    with delivery_lock(destination):
        if validate_current is not None:
            validate_current(ResultEnvelope.from_file(staging / "results.json"))
        # An intact snapshot of the same generation is extended; anything else is
        # replaced.
        existing = read_receipt(destination)
        if existing is not None and existing.generation == receipt.generation:
            try:
                validate_receipt(destination, existing, task_id)
            except (OSError, ValueError):
                existing = None
        else:
            existing = None
        # Readers treat a missing receipt as incomplete while files change.
        (destination / RECEIPT_NAME).unlink(missing_ok=True)
        if existing is None:
            shutil.rmtree(destination / "artifacts", ignore_errors=True)
            prepare_output_dir(destination)
        # Install each staged entry, skipping files the snapshot already holds.
        root = destination.resolve()
        for entry in staging.rglob("*"):
            name = entry.relative_to(staging)
            if name.as_posix() == RECEIPT_NAME:
                continue
            target = destination / name
            target.parent.resolve().relative_to(root)
            if entry.is_symlink():
                if name.parts[0] != "artifacts":
                    raise ValueError(f"Unsupported delivery link: {name}")
                _commit_symlink(target, os.readlink(entry))
            elif entry.is_dir():
                if target.is_symlink():
                    target.unlink()
                target.mkdir(parents=True, exist_ok=True)
                try:
                    target.chmod(0o777)
                except OSError:
                    pass
            elif not _holds_same_file(existing, receipt, name.as_posix(), target):
                _commit_file(entry, target, destination)
        # The new receipt covers both the existing and the delivered selections.
        if existing is not None:
            receipt.all_artifacts |= existing.all_artifacts
            receipt.artifact_paths = sorted(
                set(receipt.artifact_paths + existing.artifact_paths)
            )
            receipt.directories = sorted(
                set(receipt.directories + existing.directories)
            )
            receipt.files = existing.files | receipt.files
            receipt.symlinks = existing.symlinks | receipt.symlinks
        validate_receipt(destination, receipt, task_id, verify_content=False)
        write_receipt(destination, receipt)


def _holds_same_file(
    existing: ResultDeliveryReceipt | None,
    receipt: ResultDeliveryReceipt,
    name: str,
    target: Path,
) -> bool:
    if existing is None or name not in existing.files:
        return False
    return (
        existing.files[name] == receipt.files.get(name)
        and target.is_file()
        and not target.is_symlink()
    )


def _commit_file(source: Path, target: Path, destination: Path) -> None:
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
        shutil.copyfile(source, replacement)
        shutil.copymode(source, replacement)
        replacement.replace(target)
    finally:
        replacement.unlink(missing_ok=True)


def _commit_symlink(target: Path, link: str) -> None:
    if target.is_symlink() and os.readlink(target) == link:
        return
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".delivery-"))
    try:
        (staging / "link").symlink_to(link)
        (staging / "link").replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
