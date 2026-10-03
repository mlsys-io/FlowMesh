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
