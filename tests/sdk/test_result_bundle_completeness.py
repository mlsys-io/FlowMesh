import io
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest
from flowmesh.exceptions import FlowMeshError
from flowmesh.resources.results import AsyncResults, Results


def incomplete_bundle() -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w") as archive:
        info = tarfile.TarInfo("tsk-test/artifacts")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
    return sink.getvalue()


def test_old_extraction_target_cannot_hide_missing_result(tmp_path: Path) -> None:
    target = tmp_path / "tsk-test"
    target.mkdir()
    (target / "results.json").write_text('{"task_id":"tsk-test","result":{}}')

    class Client:
        def _download(self, url: str, destination: Path) -> None:
            destination.write_bytes(incomplete_bundle())

    with pytest.raises(FlowMeshError, match="missing requested section 'results'"):
        Results(cast(Any, Client())).materialize("tsk-test", tmp_path)


@pytest.mark.anyio
async def test_async_materialize_rejects_missing_result(tmp_path: Path) -> None:
    class Client:
        async def _download(self, url: str, destination: Path) -> None:
            destination.write_bytes(incomplete_bundle())

    with pytest.raises(FlowMeshError, match="missing requested section 'results'"):
        await AsyncResults(cast(Any, Client())).materialize("tsk-test", tmp_path)


def linked_bundle(link_name: str = "tsk-test/artifacts/venv/bin/python") -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w") as archive:
        content = b'{"task_id":"tsk-test","result":{}}'
        envelope = tarfile.TarInfo("tsk-test/results.json")
        envelope.size = len(content)
        archive.addfile(envelope, io.BytesIO(content))
        for name in ("tsk-test/artifacts", "tsk-test/artifacts/venv"):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
        link = tarfile.TarInfo(link_name)
        link.type = tarfile.SYMTYPE
        link.linkname = "/usr/bin/python3"
        archive.addfile(link)
    return sink.getvalue()


def test_materialize_keeps_absolute_artifact_links(tmp_path: Path) -> None:
    class Client:
        def _download(self, url: str, destination: Path) -> None:
            destination.write_bytes(linked_bundle())

    Results(cast(Any, Client())).materialize("tsk-test", tmp_path)
    link = tmp_path / "tsk-test" / "artifacts" / "venv" / "bin" / "python"
    assert link.is_symlink() and link.readlink() == Path("/usr/bin/python3")


@pytest.mark.anyio
async def test_async_materialize_keeps_absolute_artifact_links(tmp_path: Path) -> None:
    class Client:
        async def _download(self, url: str, destination: Path) -> None:
            destination.write_bytes(linked_bundle())

    await AsyncResults(cast(Any, Client())).materialize("tsk-test", tmp_path)
    link = tmp_path / "tsk-test" / "artifacts" / "venv" / "bin" / "python"
    assert link.readlink() == Path("/usr/bin/python3")


@pytest.mark.parametrize(
    "link_name", ["tsk-test/logs", "tsk-test/artifacts/./venv/bin/python"]
)
def test_materialize_rejects_links_that_could_redirect_writes(
    tmp_path: Path, link_name: str
) -> None:
    class Client:
        def _download(self, url: str, destination: Path) -> None:
            destination.write_bytes(linked_bundle(link_name))

    with pytest.raises(ValueError, match="Unsafe"):
        Results(cast(Any, Client())).materialize("tsk-test", tmp_path)
