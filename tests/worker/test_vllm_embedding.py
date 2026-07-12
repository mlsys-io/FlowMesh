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

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import EmbeddingSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.vllm_embedding_executor import (
    VLLMEmbeddingExecutor,
    VLLMEmbeddingResult,
)
from worker.runner import Runner


class _FakeEmbeddingLLM:
    """Stand-in for a vLLM ``LLM`` initialized in pooling mode.

    ``vectors`` / ``token_counts`` are indexed by the input's position in the
    original (unsplit) text list, so assertions hold regardless of how many
    ``encode`` calls the executor issues.
    """

    def __init__(
        self,
        vectors: list[list[float]],
        token_counts: list[int],
        tokenizer: Any | None = None,
        budget: int = 1_000_000,
    ) -> None:
        self._vectors = vectors
        self._token_counts = token_counts
        self._tokenizer = tokenizer
        self._offset = 0
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=budget)
            )
        )
        self.encoded: list[list[str]] = []
        self.pooling_tasks: list[str | None] = []

    def get_tokenizer(self) -> Any | None:
        return self._tokenizer

    def encode(self, prompts: list[str], pooling_task: str | None = None) -> list[Any]:
        self.encoded.append(list(prompts))
        self.pooling_tasks.append(pooling_task)
        start = self._offset
        self._offset += len(prompts)
        return [
            SimpleNamespace(
                outputs=SimpleNamespace(data=torch.tensor(self._vectors[start + i])),
                prompt_token_ids=[0] * self._token_counts[start + i],
            )
            for i in range(len(prompts))
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
    token_counts = {"hello world": 3, "second input": 5}
    fake = _FakeEmbeddingLLM(
        vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        token_counts=[3, 5],
        tokenizer=SimpleNamespace(encode=lambda text: [0] * token_counts[text]),
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

    assert isinstance(result, VLLMEmbeddingResult)
    assert fake.encoded == [["hello world", "second input"]]
    assert fake.pooling_tasks == ["embed"]

    # Result payload carries metadata + the one artifact ref ONLY.
    assert result.model == "org/embed"
    assert result.embedding_file is not None
    assert result.embedding_file.path == "embeddings.safetensors"
    dumped = result.model_dump()
    assert "items" not in dumped
    assert "count" not in dumped
    assert "dim" not in dumped
    assert "prompts_file" not in dumped
    assert not any(
        isinstance(value, list) and value and isinstance(value[0], float)
        for value in dumped.values()
    )
    assert result.usage is not None
    assert result.usage["num_requests"] == 2
    assert result.usage["prompt_tokens"] == 8
    assert result.usage["embedding_dim"] == 3
    assert "latency_sec" in result.usage

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
        tokenizer=SimpleNamespace(encode=lambda text: [0]),
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


def test_encode_in_batches_splits_by_token_budget_and_preserves_order() -> None:
    texts = ["a", "b", "c", "d"]
    token_counts = {"a": 3, "b": 3, "c": 3, "d": 3}
    fake = _FakeEmbeddingLLM(
        vectors=[[float(idx)] for idx in range(len(texts))],
        token_counts=[3, 3, 3, 3],
        tokenizer=SimpleNamespace(encode=lambda text: [0] * token_counts[text]),
        budget=10,
    )
    executor = _executor_with_fake_engine(fake)

    outputs = executor._encode_in_batches(texts, task_id="tsk-batch")

    # budget = int(10 * 0.9) = 9; three 3-token items pack into one chunk
    # (3+3+3=9), the fourth cannot join (9+3>9) and starts a new chunk.
    assert fake.encoded == [["a", "b", "c"], ["d"]]
    assert [out.outputs.data.item() for out in outputs] == [0.0, 1.0, 2.0, 3.0]


def test_encode_in_batches_accepts_item_between_pack_and_raw_budget() -> None:
    # raw_budget=10, pack_budget=int(10*0.9)=9; a 10-token item exceeds
    # pack_budget but fits the raw scheduler budget, so it is encoded alone.
    texts = ["short", "a heavy single input"]
    token_counts = {texts[0]: 3, texts[1]: 10}
    fake = _FakeEmbeddingLLM(
        vectors=[[0.0], [1.0]],
        token_counts=[3, 10],
        tokenizer=SimpleNamespace(encode=lambda text: [0] * token_counts[text]),
        budget=10,
    )
    executor = _executor_with_fake_engine(fake)

    outputs = executor._encode_in_batches(texts, task_id="tsk-edge")

    assert fake.encoded == [["short"], ["a heavy single input"]]
    assert [out.outputs.data.item() for out in outputs] == [0.0, 1.0]


def test_encode_in_batches_raises_for_oversized_single_item() -> None:
    texts = ["short", "an input requiring far too many tokens to schedule"]
    token_counts = {texts[0]: 3, texts[1]: 50}
    fake = _FakeEmbeddingLLM(
        vectors=[[0.0], [0.0]],
        token_counts=[3, 50],
        tokenizer=SimpleNamespace(encode=lambda text: [0] * token_counts[text]),
        budget=10,
    )
    executor = _executor_with_fake_engine(fake)

    with pytest.raises(
        ExecutionError, match="Input 1 requires 50 tokens.*10-token scheduler budget"
    ):
        executor._encode_in_batches(texts, task_id="tsk-oversized")
    assert fake.encoded == []


def test_encode_in_batches_uses_engine_config_budget() -> None:
    texts = ["a", "b", "c"]
    token_counts = {"a": 40, "b": 40, "c": 40}
    fake = _FakeEmbeddingLLM(
        vectors=[[0.0], [1.0], [2.0]],
        token_counts=[40, 40, 40],
        tokenizer=SimpleNamespace(encode=lambda text: [0] * token_counts[text]),
        budget=100,
    )
    executor = _executor_with_fake_engine(fake)

    outputs = executor._encode_in_batches(texts, task_id="tsk-engine")

    # raw_budget=100 (from engine config), pack_budget=90; two 40-token items
    # pack (80<=90), the third cannot join (120>90) and starts a new chunk.
    assert fake.encoded == [["a", "b"], ["c"]]
    assert [out.outputs.data.item() for out in outputs] == [0.0, 1.0, 2.0]


def test_encode_in_batches_wraps_encode_failure() -> None:
    class _FailingLLM:
        llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=1_000_000)
            )
        )

        def get_tokenizer(self) -> Any:
            return SimpleNamespace(encode=lambda text: [0])

        def encode(self, prompts: list[str], pooling_task: str | None = None) -> Any:
            raise RuntimeError("scheduler blew up")

    executor = VLLMEmbeddingExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)
    executor._llm = cast(Any, _FailingLLM())

    with pytest.raises(
        ExecutionError, match=r"vLLM encode failed for inputs 0-1"
    ) as exc_info:
        executor._encode_in_batches(["a", "b"], task_id="tsk-fail")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
