# ruff: noqa: E402
"""Tests for the vLLM text-embedding path (``EMBEDDING`` task type).

The vLLM engine is mocked: the executor's ``encode`` call is exercised against
a fake pooling engine so the input-collection, tensor-artifact writing, and
fail-fast behavior can be verified without a GPU or model download.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

pytest.importorskip("torch", reason="torch not installed (needs worker runtime)")
pytest.importorskip("pandas", reason="pandas not installed (needs worker runtime)")
pytest.importorskip("datasets", reason="datasets not installed (needs worker runtime)")

import torch
from safetensors.torch import load_file

from shared.schemas.result import EmbeddingResult
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import EmbeddingSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_embedding_executor import VLLMEmbeddingExecutor
from worker.runner import Runner


class _FakeEmbeddingLLM:
    """Stand-in for a vLLM ``LLM`` initialized in pooling mode."""

    def __init__(self, vectors: list[list[float]], token_counts: list[int]) -> None:
        self._vectors = vectors
        self._token_counts = token_counts
        self.encoded: list[list[str]] = []
        self.pooling_tasks: list[str | None] = []

    def encode(self, prompts: list[str], pooling_task: str | None = None) -> list[Any]:
        self.encoded.append(list(prompts))
        self.pooling_tasks.append(pooling_task)
        return [
            SimpleNamespace(
                outputs=SimpleNamespace(data=torch.tensor(self._vectors[idx])),
                prompt_token_ids=[0] * self._token_counts[idx],
            )
            for idx in range(len(prompts))
        ]


def _embedding_spec(
    items: list[str],
    metadata: list[dict[str, Any]] | None = None,
    vllm: dict[str, Any] | None = None,
):
    data: dict[str, Any] = {"type": "list", "items": items}
    if metadata is not None:
        data["metadata"] = metadata
    return EmbeddingSpecStrict(
        taskType=TaskType.EMBEDDING,
        model=ModelConfig(source=ModelSource(identifier="org/embed"), vllm=vllm or {}),
        data=data,
    )


def _executor_with_fake_engine(fake: _FakeEmbeddingLLM) -> VLLMEmbeddingExecutor:
    executor = VLLMEmbeddingExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    executor._llm = cast(Any, fake)
    executor._model_name = "org/embed"
    return executor


def test_embedding_writes_tensor_artifact_and_metadata(tmp_path: Path) -> None:
    fake = _FakeEmbeddingLLM(
        vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        token_counts=[3, 5],
    )
    executor = _executor_with_fake_engine(fake)
    spec = _embedding_spec(
        ["hello world", "second input"],
        metadata=[{"row_id": "a"}, {"row_id": "b"}],
    )
    task = make_worker_task_message(
        spec, task_type=TaskType.EMBEDDING, task_id="tsk-embed"
    )

    with patch.object(executor, "_ensure_embedding_llm") as mock_ensure:
        result = executor.run(task, tmp_path)
    mock_ensure.assert_called_once()

    assert isinstance(result, EmbeddingResult)
    assert fake.encoded == [["hello world", "second input"]]
    assert fake.pooling_tasks == ["embed"]

    # Result payload carries metadata + the one artifact ref ONLY.
    assert result.model == "org/embed"
    assert result.embedding_file is not None
    assert result.embedding_file.path == "embeddings.safetensors"
    dumped = result.model_dump()
    assert "items" not in dumped
    # count / image_group_sizes belong to the unified EmbeddingResult but the
    # vLLM embedding path does not populate them, so they drop from the payload.
    assert "count" not in dumped
    assert "image_group_sizes" not in dumped
    assert "dim" not in dumped
    assert "prompts_file" not in dumped
    assert not any(
        isinstance(value, list) and value and isinstance(value[0], float)
        for value in dumped.values()
    )
    assert result.usage is not None
    assert result.usage.num_requests == 2
    assert result.usage.prompt_tokens == 8
    assert result.usage.embedding_dim == 3
    assert result.usage.latency_sec >= 0.0

    # The embedding matrix is written as a [count, dim] float32 tensor.
    artifacts_dir = tmp_path / "artifacts"
    matrix = load_file(artifacts_dir / "embeddings.safetensors")["embeddings"]
    assert matrix.shape == (2, 3)
    assert matrix.dtype == torch.float32
    assert torch.allclose(
        matrix, torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), atol=1e-6
    )

    # No row->prompt sidecar is written; the tensor is row-aligned to inputs.
    assert not (artifacts_dir / "prompts.json").exists()


def test_embedding_engine_kwargs_select_pooling(tmp_path: Path) -> None:
    executor = VLLMEmbeddingExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    spec = _embedding_spec(["a"], vllm={"convert": "embed"})

    with patch.object(executor, "_init_vllm_engine") as mock_init:
        executor._ensure_embedding_llm(spec, ["tsk-embed"])

    mock_init.assert_called_once()
    kwargs = mock_init.call_args.kwargs
    assert kwargs["extra_llm_kwargs"] == {"runner": "pooling", "convert": "embed"}
    assert kwargs["checkpoint_cfg"] == {}
    # `convert` is consumed by the pooling engine, not left as a stray vllm arg.
    assert "convert" not in kwargs["vllm_cfg"]


def test_embedding_reloads_engine_on_model_change() -> None:
    executor = VLLMEmbeddingExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    built: list[str] = []
    shutdowns: list[int] = []

    def fake_llm(**kwargs: Any) -> Any:
        built.append(kwargs["model"])
        return SimpleNamespace()

    def fake_shutdown() -> None:
        shutdowns.append(1)
        executor._llm = None

    spec_a = _embedding_spec(["x"], vllm={"convert": "embed"})
    spec_b = EmbeddingSpecStrict(
        taskType=TaskType.EMBEDDING,
        model=ModelConfig(
            source=ModelSource(identifier="org/other"), vllm={"convert": "embed"}
        ),
        data={"type": "list", "items": ["x"]},
    )

    with (
        patch("worker.executors.vllm_executor.LLM", side_effect=fake_llm),
        patch.object(executor, "_shutdown_llm", side_effect=fake_shutdown),
    ):
        executor._ensure_embedding_llm(spec_a, ["t1"])
        # Same identifier + config: the loaded engine is reused, no reload.
        executor._ensure_embedding_llm(spec_a, ["t2"])
        assert built == ["org/embed"]
        assert shutdowns == []
        # Different identifier: the reuse key differs, forcing a reload.
        executor._ensure_embedding_llm(spec_b, ["t3"])
        assert shutdowns == [1]
        assert built == ["org/embed", "org/other"]


def test_embedding_fails_fast_on_empty_input(tmp_path: Path) -> None:
    fake = _FakeEmbeddingLLM(vectors=[], token_counts=[])
    executor = _executor_with_fake_engine(fake)
    spec = _embedding_spec([])
    task = make_worker_task_message(
        spec, task_type=TaskType.EMBEDDING, task_id="tsk-empty"
    )

    with patch.object(executor, "_ensure_embedding_llm"):
        with pytest.raises(ExecutionError, match="No inputs prepared"):
            executor.run(task, tmp_path)
    assert fake.encoded == []
    assert not (tmp_path / "artifacts" / "embeddings.safetensors").exists()


def test_embedding_rejects_inconsistent_dimensions(tmp_path: Path) -> None:
    fake = _FakeEmbeddingLLM(
        vectors=[[0.1, 0.2, 0.3], [0.4, 0.5]],
        token_counts=[3, 3],
    )
    executor = _executor_with_fake_engine(fake)
    spec = _embedding_spec(["a", "b"])
    task = make_worker_task_message(
        spec, task_type=TaskType.EMBEDDING, task_id="tsk-dim"
    )

    with patch.object(executor, "_ensure_embedding_llm"):
        with pytest.raises(ExecutionError, match="inconsistent dimensionality"):
            executor.run(task, tmp_path)


def test_embedding_executor_advertises_embedding_task() -> None:
    assert TaskType.EMBEDDING in VLLMEmbeddingExecutor.supported_task_types


def test_embedding_executor_routing_prefers_vllm_when_configured() -> None:
    runner = cast(Runner, object.__new__(Runner))

    vllm_spec = EmbeddingSpecStrict(
        taskType=TaskType.EMBEDDING,
        model=ModelConfig(source=ModelSource(identifier="org/embed"), vllm={}),
    )
    transformers_spec = EmbeddingSpecStrict(
        taskType=TaskType.EMBEDDING,
        model=ModelConfig(source=ModelSource(identifier="org/embed")),
    )

    assert runner._select_embedding_executor_key(vllm_spec) == "vllm_embedding"
    assert runner._select_embedding_executor_key(transformers_spec) == "default"
