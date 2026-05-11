"""Result resource operations."""

import json
import tarfile
import tempfile
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ._base import AsyncResource, SyncResource


class _ResultEnvelope(BaseModel):
    """Local mirror of server-side ``ResultPayload`` for SDK validation."""

    task_id: str
    result: dict[str, Any]
    worker_id: str | None = None
    metadata: dict[str, Any] | None = None
    received_at: str | None = Field(default=None)


type BundleSection = Literal["results", "artifacts", "logs", "all"]

_DEFAULT_INCLUDE: tuple[BundleSection, ...] = ("results", "artifacts")


class Results(SyncResource):
    """Synchronous result operations."""

    def retrieve(self, task_id: str) -> dict[str, Any]:
        """Retrieve the JSON result for a completed task."""
        return self._client._request("GET", f"/results/{task_id}")

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
            extracted = _extract_bundle(tmp_path, output_dir)
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

    async def retrieve(self, task_id: str) -> dict[str, Any]:
        """Retrieve the JSON result for a completed task."""
        return await self._client._request("GET", f"/results/{task_id}")

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
            extracted = _extract_bundle(tmp_path, output_dir)
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


def _extract_bundle(bundle_path: Path, output_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    dest_root = output_dir.resolve()
    with tarfile.open(bundle_path, mode="r:*") as archive:
        for member in archive:
            member_path = (dest_root / member.name).resolve()
            try:
                member_path.relative_to(dest_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe member path in result bundle: {member.name}"
                ) from exc
            archive.extract(member, dest_root, filter="data")
            if member.isfile():
                extracted.append(member_path)
    return extracted


def _finalize_materialize(
    output_dir: Path,
    task_id: str,
    sections: tuple[BundleSection, ...],
    extracted: list[Path],
) -> tuple[dict[str, Any], Path, list[Path]]:
    """Unwrap the envelope and point _artifacts at the local extracted dir."""
    json_path = output_dir / task_id / "results.json"
    if not json_path.is_file():
        return {}, json_path, extracted

    result = _ResultEnvelope.model_validate_json(json_path.read_text()).result
    if _wants_artifacts(sections):
        ctx = result["_artifacts"]
        ctx["base_dir"] = (output_dir / task_id).resolve().as_posix()
        ctx.pop("base_url", None)
    json_path.write_text(json.dumps(result, indent=2))
    return result, json_path, extracted


def _wants_artifacts(sections: tuple[BundleSection, ...]) -> bool:
    return "artifacts" in sections or "all" in sections
