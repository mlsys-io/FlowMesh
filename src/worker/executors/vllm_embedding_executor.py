#!/usr/bin/env python3
"""VLLMEmbeddingExecutor: text embeddings via a vLLM pooling model."""

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import torch
except Exception:
    if TYPE_CHECKING:
        import torch
    else:
        torch = None

try:
    from safetensors.torch import save_file as save_safetensors
except Exception:
    if TYPE_CHECKING:
        from safetensors.torch import save_file as save_safetensors
    else:
        save_safetensors = None

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import EmbeddingResult, EmbeddingUsage
from shared.tasks.specs import EmbeddingSpecStrict
from shared.tasks.task_type import TaskType

from .base_executor import ExecutionError, ExecutorTask
from .vllm_executor import VLLMExecutor

logger = logging.getLogger(__name__)


class VLLMEmbeddingExecutor(VLLMExecutor):
    """vLLM executor that pools inputs into an embedding tensor artifact."""

    name = "vllm_embedding"
    supported_task_types = frozenset({TaskType.EMBEDDING})

    _EMBEDDINGS_ARTIFACT = "embeddings.safetensors"

    def _ensure_embedding_llm(
        self, spec: EmbeddingSpecStrict, task_ids: Iterable[str] | None = None
    ) -> None:
        ident = self._resolve_model_ident(spec)
        vllm_cfg = self._vllm_cfg(spec)
        convert = vllm_cfg.pop("convert", None)
        extra_llm_kwargs: dict[str, Any] = {"runner": "pooling"}
        if convert:
            extra_llm_kwargs["convert"] = str(convert)
        self._init_vllm_engine(
            ident=ident,
            vllm_cfg=vllm_cfg,
            checkpoint_cfg={},
            new_inference_spec={
                **self._base_reuse_key(spec),
                "embedding": extra_llm_kwargs,
            },
            requested_gpu_count=self._requested_gpu_count(spec),
            revision=spec.model_revision,
            extra_llm_kwargs=extra_llm_kwargs,
            adjust_tp=lambda size: size,
            task_ids=task_ids,
        )

    def _run_inner(self, task: ExecutorTask, out_dir: Path) -> EmbeddingResult:
        task_id = task.task_id.strip()
        spec = self.require_spec(task, EmbeddingSpecStrict)

        raw_entry = self._collect_prompts_for_spec(spec, task_id=task_id)
        texts: list[str] = []
        for prompt in raw_entry.prompts:
            if not isinstance(prompt, str):
                raise ExecutionError(
                    "vLLM text embedding requires plain-string inputs; got a "
                    "chat-style message payload. Provide spec.data.items as strings."
                )
            texts.append(prompt)
        if not texts:
            raise ExecutionError(
                "No inputs prepared for embedding. Check spec.data configuration."
            )

        self._ensure_embedding_llm(spec, [task_id])
        assert self._llm is not None

        t0 = time.time()
        with self._span(
            "embedding",
            span_type=SpanType.COMPUTE,
            attributes={"task_ids": [task_id], "input_count": len(texts)},
        ):
            outputs = self._llm.encode(texts, pooling_task="embed")
        latency = time.time() - t0
        if len(outputs) != len(texts):
            raise ExecutionError(
                f"vLLM returned {len(outputs)} embeddings for {len(texts)} inputs "
                f"(task={task_id})."
            )

        matrix = self._stack_embeddings(outputs, task_id=task_id)
        total_prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
        embedding_file = self._write_embedding_artifact(matrix=matrix, out_dir=out_dir)

        count, dim = matrix.shape
        result = EmbeddingResult(
            model=self._model_name,
            embedding_file=embedding_file,
            usage=EmbeddingUsage(
                prompt_tokens=total_prompt_tokens,
                total_tokens=total_prompt_tokens,
                num_requests=count,
                latency_sec=latency,
                embedding_dim=dim,
            ),
        )
        self._dump_to_governance(
            task_id=task_id,
            result=result,
            dependencies_by_task={task_id: self._extract_source_data_ids(spec)},
        )
        return result

    @staticmethod
    def _stack_embeddings(outputs: list[Any], *, task_id: str) -> "torch.Tensor":
        """Read ``out.outputs.data`` (the raw tensor) rather than ``.embedding``
        (a list view) to avoid a per-element float round-trip."""
        if torch is None:
            raise ExecutionError(
                f"torch is required to persist embeddings (task={task_id})."
            )
        tensors: list[torch.Tensor] = []
        dim: int | None = None
        for idx, out in enumerate(outputs):
            data = out.outputs.data
            if data.ndim != 1 or data.shape[0] == 0:
                raise ExecutionError(
                    "vLLM produced an empty embedding for input "
                    f"{idx} (task={task_id})."
                )
            if dim is None:
                dim = data.shape[0]
            elif data.shape[0] != dim:
                raise ExecutionError(
                    "vLLM produced embeddings with inconsistent dimensionality "
                    f"(task={task_id}, expected={dim}, got={data.shape[0]})."
                )
            tensors.append(data)
        return torch.stack(tensors).to(torch.float32).cpu().contiguous()

    def _write_embedding_artifact(
        self, *, matrix: "torch.Tensor", out_dir: Path
    ) -> ArtifactRef:
        if save_safetensors is None:
            raise ExecutionError("safetensors is required to persist embeddings.")
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = artifacts_dir / self._EMBEDDINGS_ARTIFACT
        save_safetensors({"embeddings": matrix}, embedding_path.as_posix())
        return ArtifactRef(path=self._EMBEDDINGS_ARTIFACT)
