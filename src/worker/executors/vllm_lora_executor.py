#!/usr/bin/env python3
"""VLLMLoRAExecutor: LoRA-enabled vLLM inference."""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from vllm.lora.request import LoRARequest

    # vLLM configures its own "vllm" logger via dictConfig and sets propagate=False
    # by default. Disable that configuration early so vLLM logs can be captured by
    # task-level log streaming (root handlers / QueueHandler).
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
except Exception:  # pragma: no cover - optional dependency at import time
    LoRARequest = None  # type: ignore

from shared.tasks.components import AdapterConfig
from shared.tasks.components.model import AdapterApplyMode
from shared.tasks.specs import InferenceSpecStrict

from .base_executor import ExecutionError
from .utils.checkpoints import (
    detect_archive_format,
    download_and_unpack,
    select_extracted_subdir,
)
from .vllm_executor import VLLMExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoRAAdapterSpec:
    name: str
    id: int
    apply: AdapterApplyMode
    path: str | None
    url: str | None
    task_id: str | None
    headers: dict[str, str]
    archive_format: str


class VLLMLoRAExecutor(VLLMExecutor):
    """vLLM executor with explicit LoRA adapter handling."""

    name = "vllm_lora"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._adapter_specs: list[LoRAAdapterSpec] = []
        self._runtime_specs: list[LoRAAdapterSpec] = []

    def _build_inference_spec(
        self,
        vllm_cfg: dict[str, Any],
        checkpoint_cfg: dict[str, Any],
        spec: InferenceSpecStrict,
    ) -> dict[str, Any]:
        base_spec = super()._build_inference_spec(vllm_cfg, checkpoint_cfg, spec)
        base_spec["adapters"] = self._extract_adapter_specs(spec)
        return base_spec

    def _extra_llm_kwargs(self, spec: InferenceSpecStrict) -> dict[str, Any]:
        adapters = self._extract_adapter_specs(spec)
        # LoRA merge also needs LoRA-enabled engine so load_lora_adapter exists.
        if adapters:
            return {"enable_lora": True}
        return {}

    def _adjust_tensor_parallel_size(self, spec: InferenceSpecStrict, size: int) -> int:
        if self._extract_adapter_specs(spec) and size > 1:
            logger.info(
                "LoRA adapters detected; forcing tensor_parallel_size=1 (was %s)",
                size,
            )
            return 1
        return size

    def _build_generate_kwargs(
        self, spec: InferenceSpecStrict, out_dir: Path
    ) -> dict[str, Any]:
        self._prepare_adapters(spec, out_dir)
        lora_payload = self._build_lora_requests(out_dir)
        if lora_payload is None:
            return {}
        return {"lora_request": lora_payload}

    def cleanup_after_run(self) -> None:
        super().cleanup_after_run()
        self._adapter_specs = []
        self._runtime_specs = []

    # ------------------------------------------------------------------ #
    # Adapter utilities
    # ------------------------------------------------------------------ #
    def _extract_adapter_specs(
        self, spec: InferenceSpecStrict
    ) -> list[LoRAAdapterSpec]:
        adapters = spec.adapters or []
        if not isinstance(adapters, list):
            raise ExecutionError("spec.model.adapters must be a list when provided")

        normalized: list[LoRAAdapterSpec] = []
        for idx, adapter in enumerate(adapters):
            if not isinstance(adapter, AdapterConfig):
                raise ExecutionError(
                    "Each entry in spec.model.adapters must be AdapterConfig"
                )
            if adapter.type.lower() != "lora":
                continue

            name = adapter.name or f"lora_{idx}"
            path_value = adapter.path
            url_value = adapter.url
            task_id = adapter.task_id
            if not path_value and not url_value and not task_id:
                raise ExecutionError(
                    f"LoRA adapter '{name}' must provide path, url, or task_id"
                )

            normalized.append(
                LoRAAdapterSpec(
                    name=str(name),
                    id=adapter.id if adapter.id is not None else idx + 1,
                    apply=adapter.apply,
                    path=path_value,
                    url=url_value,
                    task_id=task_id,
                    headers=adapter.headers or {},
                    archive_format=adapter.archive_format,
                )
            )

        self._adapter_specs = normalized
        return normalized

    def _prepare_adapters(self, spec: InferenceSpecStrict, out_dir: Path) -> None:
        runtime_specs: list[LoRAAdapterSpec] = []
        merge_specs: list[LoRAAdapterSpec] = []
        for adapter in self._extract_adapter_specs(spec):
            if adapter.apply == "merge":
                merge_specs.append(adapter)
            else:
                runtime_specs.append(adapter)
        self._runtime_specs = runtime_specs

        if not merge_specs:
            return
        if self._llm is None:
            raise ExecutionError("vLLM must be initialized before LoRA merge")

        load_fn = getattr(self._llm, "load_lora_adapter", None)
        merge_fn = getattr(self._llm, "merge_lora_weights", None)
        unload_fn = getattr(self._llm, "unload_lora_adapter", None)
        if load_fn is None or merge_fn is None:
            raise ExecutionError(
                "vLLM load_lora_adapter/merge_lora_weights API required for apply=merge"
            )

        for adapter in merge_specs:
            adapter_dir = self._resolve_adapter_directory(adapter, out_dir)
            load_fn(adapter_name=adapter.name, adapter_path=adapter_dir.as_posix())
            merge_fn(adapter_name=adapter.name)
            if unload_fn is not None:
                unload_fn(adapter_name=adapter.name)

    def _build_lora_requests(self, out_dir: Path) -> Any | None:
        if not self._runtime_specs:
            return None
        if LoRARequest is None:
            raise ExecutionError("vLLM LoRARequest helper is unavailable")

        requests: list[Any] = []
        for adapter in self._runtime_specs:
            adapter_dir = self._resolve_adapter_directory(adapter, out_dir)
            request = LoRARequest(adapter.name, adapter.id, adapter_dir.as_posix())
            requests.append(request)

        if len(requests) == 1:
            return requests[0]
        return requests

    def _resolve_adapter_directory(
        self, adapter: LoRAAdapterSpec, out_dir: Path
    ) -> Path:
        name = adapter.name
        path_value = adapter.path
        url_value = adapter.url
        archive_format = adapter.archive_format

        if path_value:
            path = Path(path_value).expanduser()
            # `path` is treated as a local hint: if it resolves to an existing
            # file or directory, use it directly.
            if path.exists():
                return self._ensure_or_extract(path, out_dir, name, archive_format)

        if task_id := adapter.task_id:
            base_dir = (
                Path(os.getenv("RESULTS_DIR", "").strip() or "./results")
                .expanduser()
                .resolve()
            )
            artifacts_base = base_dir / str(task_id) / "artifacts"
            candidates = [
                artifacts_base / "final_lora",
                artifacts_base / "final_lora.tar.gz",
                artifacts_base / "final_model",
                artifacts_base / "final_model.tar.gz",
                artifacts_base,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return self._ensure_or_extract(
                        candidate, out_dir, name, archive_format
                    )

        if not url_value:
            raise ExecutionError(
                f"LoRA adapter '{name}' must include a valid url "
                "when path/taskId is missing"
            )

        download_root = out_dir / "_lora_cache" / name
        download_root.mkdir(parents=True, exist_ok=True)
        load_cfg = {
            "url": url_value,
            "archive_format": archive_format,
            "headers": adapter.headers,
        }
        extracted = download_and_unpack(load_cfg, download_root)
        return self._ensure_adapter_root(extracted)

    def _ensure_or_extract(
        self, path: Path, out_dir: Path, name: str, archive_format: str
    ) -> Path:
        if path.is_dir():
            return self._ensure_adapter_root(path)

        target_root = out_dir / "_lora_cache" / name
        if target_root.exists():
            for child in target_root.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child, ignore_errors=True)
        target_root.mkdir(parents=True, exist_ok=True)

        fmt = detect_archive_format(archive_format, path.name)
        if fmt != "tar":
            raise ExecutionError(
                f"Unsupported LoRA archive format for {path}; "
                "expected .tar(.gz) or a directory"
            )

        import tarfile

        with tarfile.open(path, "r:*") as tf:
            tf.extractall(target_root, filter="data")

        return self._ensure_adapter_root(select_extracted_subdir(target_root, None))

    def _ensure_adapter_root(self, path: Path) -> Path:
        if path.is_file():
            raise ExecutionError(
                f"Expected LoRA adapter directory but found file: {path}"
            )
        if (path / "adapter_config.json").exists():
            return path

        candidates = list(path.rglob("adapter_config.json"))
        if not candidates:
            raise ExecutionError(
                f"Could not locate adapter_config.json under {path} for LoRA adapter"
            )
        return candidates[0].parent
