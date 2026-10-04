"""Result resource operations."""

import json
import shutil
import tarfile
import tempfile
from collections.abc import AsyncIterable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import TypeAdapter

from ..exceptions import FlowMeshError
from ..models.result import AnyExecutorResult, ResultEnvelope
from ._base import AsyncResource, SyncResource

type BundleSection = Literal["results", "artifacts", "logs", "all"]

_DEFAULT_INCLUDE: tuple[BundleSection, ...] = ("results", "artifacts")

_RESULT_ADAPTER: TypeAdapter[AnyExecutorResult] = TypeAdapter(AnyExecutorResult)


class Results(SyncResource):
    """Synchronous result operations."""

    def retrieve(self, task_id: str) -> AnyExecutorResult:
        """Retrieve the result for a completed task."""
        data = self._client._request("GET", f"/results/{task_id}")
        return _RESULT_ADAPTER.validate_python(data)

    def get_bundle(
        self,
        task_id: str,
        output_path: Path,
        include: Iterable[BundleSection] | None = None,
    ) -> None:
        """Download a tar bundle of the task result."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(_bundle_path(task_id, include), output_path)

    def download_file(
        self,
        task_id: str,
        filename: str,
        output_path: Path,
    ) -> None:
        """Download a specific artifact file from a task result."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(f"/results/{task_id}/files/{filename}", output_path)

    def download_logs(self, task_id: str, output_path: Path) -> None:
        """Download archived logs.jsonl for a task."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(f"/results/{task_id}/logs", output_path)

    def materialize(
        self,
        task_id: str,
        output_dir: Path,
        include: Iterable[BundleSection] | None = None,
    ) -> tuple[dict[str, Any], Path, list[Path]]:
        """Fetch the task bundle and extract it under `output_dir/<task_id>/`.

        `include` defaults to `("results", "artifacts")`. Returns
        `(payload, json_path, extracted_paths)`."""
        sections = _normalize_include(include)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix=f"flowmesh-bundle-{task_id}-", suffix=".tar", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._client._download(_bundle_path(task_id, sections), tmp_path)
            extracted = _extract_bundle(tmp_path, output_dir, task_id, sections)
        finally:
            tmp_path.unlink(missing_ok=True)

        return _finalize_materialize(output_dir, task_id, sections, extracted)

    def download_files(
        self,
        task_id: str,
        file_paths: list[str],
        output_dir: Path,
    ) -> Iterable[Path]:
        """Download specific result files into an output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in file_paths:
            out_path = output_dir / Path(file_path).name
            self.download_file(task_id, file_path, out_path)
            yield out_path


class AsyncResults(AsyncResource):
    """Asynchronous result operations."""

    async def retrieve(self, task_id: str) -> AnyExecutorResult:
        """Retrieve the result for a completed task."""
        data = await self._client._request("GET", f"/results/{task_id}")
        return _RESULT_ADAPTER.validate_python(data)

    async def get_bundle(
        self,
        task_id: str,
        output_path: Path,
        include: Iterable[BundleSection] | None = None,
    ) -> None:
        """Download a tar bundle of the task result. See sync variant."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(_bundle_path(task_id, include), output_path)

    async def download_file(
        self,
        task_id: str,
        filename: str,
        output_path: Path,
    ) -> None:
        """Download a specific artifact file from a task result."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(
            f"/results/{task_id}/files/{filename}", output_path
        )

    async def download_logs(self, task_id: str, output_path: Path) -> None:
        """Download archived logs.jsonl for a task."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(f"/results/{task_id}/logs", output_path)

    async def materialize(
        self,
        task_id: str,
        output_dir: Path,
        include: Iterable[BundleSection] | None = None,
    ) -> tuple[dict[str, Any], Path, list[Path]]:
        """Fetch the task bundle and extract it. See sync variant."""
        sections = _normalize_include(include)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix=f"flowmesh-bundle-{task_id}-", suffix=".tar", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            await self._client._download(_bundle_path(task_id, sections), tmp_path)
            extracted = _extract_bundle(tmp_path, output_dir, task_id, sections)
        finally:
            tmp_path.unlink(missing_ok=True)

        return _finalize_materialize(output_dir, task_id, sections, extracted)

    async def download_files(
        self,
        task_id: str,
        file_paths: list[str],
        output_dir: Path,
    ) -> AsyncIterable[Path]:
        """Download specific result files into an output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in file_paths:
            out_path = output_dir / Path(file_path).name
            await self.download_file(task_id, file_path, out_path)
            yield out_path


def _normalize_include(
    include: Iterable[BundleSection] | None,
) -> tuple[BundleSection, ...]:
    return tuple(dict.fromkeys(include)) if include else _DEFAULT_INCLUDE


def _bundle_path(task_id: str, include: Iterable[BundleSection] | None) -> str:
    sections = _normalize_include(include)
    query = "&".join(f"include={s}" for s in sections)
    return f"/results/{task_id}/bundle?{query}"


def _extract_bundle(
    bundle_path: Path,
    output_dir: Path,
    task_id: str,
    sections: tuple[BundleSection, ...],
) -> list[Path]:
    extracted: list[Path] = []
    dest_root = output_dir.resolve()
    with tarfile.open(bundle_path, mode="r:*") as archive:
        members = archive.getmembers()
        required = ("results", "artifacts", "logs") if "all" in sections else sections
        names = {
            member.name.rstrip("/")
            for member in members
            if member.isfile() or member.isdir()
        }
        for section in required:
            expected = (
                f"{task_id}/results.json"
                if section == "results"
                else f"{task_id}/{section}"
            )
            present = (
                any(member.name == expected and member.isfile() for member in members)
                if section == "results"
                else any(
                    name == expected or name.startswith(expected + "/")
                    for name in names
                )
            )
            if not present:
                raise FlowMeshError(
                    f"Result bundle for {task_id} is missing "
                    f"requested section '{section}'"
                )
        links = _bundle_links(members, task_id)
        for member in members:
            member_path = dest_root / member.name
            try:
                member_path.parent.resolve().relative_to(dest_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe member path in result bundle: {member.name}"
                ) from exc
            if member.issym():
                continue
            archive.extract(member, dest_root, filter="data")
            if member.isfile():
                extracted.append(member_path.resolve())
    for name, target in links.items():
        link_path = dest_root / name
        link_path.parent.resolve().relative_to(dest_root)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
        link_path.symlink_to(target)
    return extracted


def _bundle_links(members: list[tarfile.TarInfo], task_id: str) -> dict[str, str]:
    """Return the bundle's artifact links, rejecting any that could redirect a write.

    Links are created after every other member, only under the task's
    ``artifacts`` directory, and no member may sit beneath one.
    """
    links: dict[str, str] = {}
    for member in members:
        name = PurePosixPath(member.name)
        if member.name != name.as_posix() or name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe member path in result bundle: {member.name}")
        if member.issym():
            if name.parts[:2] != (task_id, "artifacts") or len(name.parts) < 3:
                raise ValueError(f"Unsafe link in result bundle: {member.name}")
            links[member.name] = member.linkname
    for member in members:
        if any(
            parent.as_posix() in links for parent in PurePosixPath(member.name).parents
        ):
            raise ValueError(f"Unsafe member path in result bundle: {member.name}")
    return links


def _finalize_materialize(
    output_dir: Path,
    task_id: str,
    sections: tuple[BundleSection, ...],
    extracted: list[Path],
) -> tuple[dict[str, Any], Path, list[Path]]:
    """Validate the envelope and point _artifacts at the local extracted dir."""
    json_path = output_dir / task_id / "results.json"
    if "results" not in sections and "all" not in sections:
        return {}, json_path, extracted
    if not json_path.is_file():
        raise FlowMeshError(f"Result bundle for {task_id} is missing results.json")

    envelope = ResultEnvelope.model_validate_json(json_path.read_text())
    if _wants_artifacts(sections) and (ctx := envelope.result.artifacts_):
        ctx.base_dir = (output_dir / task_id).resolve().as_posix()
        ctx.base_url = None
    payload = envelope.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2))
    return payload, json_path, extracted


def _wants_artifacts(sections: tuple[BundleSection, ...]) -> bool:
    return "artifacts" in sections or "all" in sections
