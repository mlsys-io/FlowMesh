"""Stage-reference artifact rendering in the dispatcher.

A cross-stage reference like ``${lora-train.final_lora_archive}`` resolves to an
``ArtifactRef`` result field. The dispatcher renders it to a URL; a bare
``{"path": ...}`` dict renders the same way. An unrenderable value fails the
downstream task's spec validation.
"""

from server.dispatcher.base import Dispatcher
from shared.schemas.artifact import ArtifactContext, ArtifactRef
from shared.schemas.result import LoRAResult, ResultEnvelope


def _envelope(result: LoRAResult) -> ResultEnvelope:
    return ResultEnvelope(task_id="tsk-abc", result=result)


def test_render_typed_artifact_ref_to_url() -> None:
    result = LoRAResult(
        final_lora_archive=ArtifactRef(path="final_lora.tar"),
        _artifacts=ArtifactContext(
            base_dir="/data/results/tsk-abc", base_url="http://srv:8000"
        ),
    )
    rendered = Dispatcher._render_artifact_ref(
        result.final_lora_archive, _envelope(result)
    )
    assert rendered == "http://srv:8000/api/v1/results/tsk-abc/files/final_lora.tar"


def test_render_typed_artifact_ref_to_filesystem_path() -> None:
    result = LoRAResult(
        final_lora_archive=ArtifactRef(path="final_lora.tar"),
        _artifacts=ArtifactContext(base_dir="/data/results/tsk-abc"),
    )
    rendered = Dispatcher._render_artifact_ref(
        result.final_lora_archive, _envelope(result)
    )
    assert rendered == "/data/results/tsk-abc/artifacts/final_lora.tar"


def test_render_dict_ref_to_url() -> None:
    result = LoRAResult(
        _artifacts=ArtifactContext(
            base_dir="/data/results/tsk-abc", base_url="http://srv:8000"
        ),
    )
    rendered = Dispatcher._render_artifact_ref({"path": "x.bin"}, _envelope(result))
    assert rendered == "http://srv:8000/api/v1/results/tsk-abc/files/x.bin"


def test_render_non_artifact_value_returns_none() -> None:
    result = LoRAResult(
        _artifacts=ArtifactContext(base_dir="/data/results/tsk-abc"),
    )
    assert Dispatcher._render_artifact_ref("plain-text", _envelope(result)) is None
