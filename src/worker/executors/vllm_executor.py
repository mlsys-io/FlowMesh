#!/usr/bin/env python3
"""
VLLMExecutor (YAML schema specific)

- Reads model config from spec.model.vllm / spec.model.source
- Reads sampling params and input prompts from spec.inference & spec.data
- Writes generation results to out_dir/results.json
"""

import atexit
import copy
import datetime
import gc
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import as_completed
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field, create_model

from shared.utils.json import to_json_serializable

try:
    import torch
    import torch.distributed as dist
except Exception:
    if TYPE_CHECKING:
        import torch
        import torch.distributed as dist
    else:
        torch = None
        dist = None

try:
    # vLLM is optional at import time; errors are raised in prepare()
    from vllm import LLM, SamplingParams, TextPrompt
    from vllm.distributed.parallel_state import destroy_distributed_environment
    from vllm.sampling_params import StructuredOutputsParams

    # Ensure vLLM uses the 'spawn' multiprocessing start method when CUDA is
    # initialized. Setting this early avoids runtime notices about overriding the
    # method after CUDA init.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # vLLM configures its own "vllm" logger via dictConfig and sets propagate=False
    # by default. Disable that configuration early so vLLM logs can be captured by
    # task-level log streaming (root handlers / QueueHandler).
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

    _HAS_VLLM = True
except Exception:
    if TYPE_CHECKING:
        from vllm import LLM, SamplingParams, TextPrompt
        from vllm.distributed.parallel_state import destroy_distributed_environment
        from vllm.sampling_params import StructuredOutputsParams
    else:
        LLM = None  # type: ignore
        SamplingParams = None  # type: ignore
        TextPrompt = None  # type: ignore
        destroy_distributed_environment = None  # type: ignore
        _HAS_VLLM = False
        StructuredOutputsParams = None  # type: ignore

from shared.governance.spans import FlowMeshSpanKind
from shared.tasks.specs import InferenceSpecStrict
from worker.config import WorkerConfig
from worker.lifecycle import Lifecycle

from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.data import InferenceEntry
from .mixins.inference import InferenceMixin, PreparedInferenceEntry
from .utils.checkpoints import maybe_upload_artifacts, resolve_checkpoint_load

logger = logging.getLogger(__name__)


class _RawJsonSchema:
    """Tag wrapping a JSON schema so ``_build_sampling_params`` can
    ``isinstance``-dispatch raw schemas vs. named-fields pydantic kwargs."""

    __slots__ = ("schema",)

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def __repr__(self) -> str:
        return f"_RawJsonSchema({self.schema!r})"


# Ensure distributed process groups are destroyed at process exit to avoid
# warnings from libtorch/NCCL when the interpreter exits before cleanup.
def _ensure_destroy_torch_process_group() -> None:
    if dist is not None and dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            logger.debug("Failed to destroy torch process group at exit", exc_info=True)


atexit.register(_ensure_destroy_torch_process_group)


class VLLMExecutor(InferenceMixin, Executor):
    """Executor that runs text generation using vLLM based on a YAML spec."""

    name = "vllm"

    summarization_template = """Summarize the following document concisely in 2-3 \
sentences. Focus on the main topic and key information.

Document:
{content}

Summary:"""

    def __init__(
        self, config: WorkerConfig, lifecycle: Lifecycle | None = None
    ) -> None:
        super().__init__(config, lifecycle)
        self._llm: LLM | None = None
        self._model_name: str | None = None
        self._batched_inputs: list[str | TextPrompt] = []
        self._prompt_owners: list[str] = []
        self._batched_metadata: list[dict[str, Any]] = []
        self._llm_kwargs: dict[str, Any] = {}
        self._inference_spec: dict[str, Any] = {}
        self._base_inference: dict[str, Any] = {}

        self._task_id: str | None = None

    @staticmethod
    def _detect_available_gpus() -> int:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            candidates = [dev.strip() for dev in str(vis).split(",") if dev.strip()]
            if candidates:
                return len(candidates)
        try:
            if torch is not None and torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 1

    @staticmethod
    def _compute_safe_utilization(requested: float) -> tuple[float, float | None]:
        if torch is None or not torch.cuda.is_available():
            return requested, None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            return requested, None
        if total_bytes <= 0:
            return requested, None
        free_ratio = float(free_bytes) / float(total_bytes)
        safety_margin = 0.05  # keep at least 5% headroom when possible
        min_util_floor = 0.02
        max_ceiling = min(0.98, max(min_util_floor, free_ratio))

        headroom = free_ratio - safety_margin
        # If there is already enough free memory (plus margin), keep the requested
        # utilization instead of pessimistically lowering it.
        if headroom >= requested:
            adjusted = min(max_ceiling, max(min_util_floor, requested))
            return adjusted, free_ratio

        if headroom <= 0:
            adjusted = free_ratio * 0.8
        else:
            adjusted = min(requested, headroom)

        if adjusted <= 0:
            adjusted = max(min_util_floor, free_ratio * 0.8)

        if adjusted < min_util_floor:
            adjusted = min_util_floor

        adjusted = min(adjusted, max_ceiling)
        return adjusted, free_ratio

    @staticmethod
    def _requested_gpu_count(spec: InferenceSpecStrict) -> int | None:
        gpu_count = cast(
            int | None,
            (resources := spec.resources)
            and (hardware := resources.hardware)
            and (gpu := hardware.gpu)
            and gpu.count,
        )
        if gpu_count is not None and gpu_count > 0:
            return gpu_count
        return None

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def prepare(self) -> None:  # type: ignore[override]
        if not _HAS_VLLM:
            raise ExecutionError("vLLM is not installed (`pip install vllm`).")

    def _build_inference_spec(
        self,
        vllm_cfg: dict[str, Any],
        checkpoint_cfg: dict[str, Any],
        spec: InferenceSpecStrict,
    ) -> dict[str, Any]:
        return {
            "model": copy.deepcopy(vllm_cfg),
            "checkpoint": copy.deepcopy(checkpoint_cfg),
        }

    def _extra_llm_kwargs(self, spec: InferenceSpecStrict) -> dict[str, Any]:
        return {}

    def _adjust_tensor_parallel_size(self, spec: InferenceSpecStrict, size: int) -> int:
        return size

    def _build_generate_kwargs(
        self, spec: InferenceSpecStrict, out_dir: Path
    ) -> dict[str, Any]:
        return {}

    def _ensure_llm(
        self, spec: InferenceSpecStrict, task_ids: Iterable[str] | None = None
    ) -> None:
        ident = spec.model_name or os.getenv("VLLM_MODEL")
        if not ident:
            raise ExecutionError(
                "spec.model.source.identifier (or VLLM_MODEL) is required."
            )

        vllm_cfg = ((model_cfg := spec.model) and model_cfg.vllm) or {}
        checkpoint_cfg = (spec.checkpoint or {}).get("load") or {}
        new_inference_spec = self._build_inference_spec(vllm_cfg, checkpoint_cfg, spec)

        if self._llm:
            if self._inference_spec == new_inference_spec:
                logger.info("Reusing existing vLLM instance for model %s", ident)
                return
            else:
                logger.info(
                    "Releasing previous vLLM instance for model %s", self._model_name
                )
                self._shutdown_llm()
        self._inference_spec = new_inference_spec
        vllm_cfg = copy.deepcopy(vllm_cfg)

        requested_tp = vllm_cfg.pop("tensor_parallel_size", None)
        requested_gpu_count = self._requested_gpu_count(spec)
        try:
            tensor_parallel_size = int(requested_tp) if requested_tp is not None else 0
        except Exception:
            tensor_parallel_size = 0
        if tensor_parallel_size <= 0:
            available_gpus = self._detect_available_gpus()
            if requested_gpu_count is not None:
                tensor_parallel_size = max(1, min(available_gpus, requested_gpu_count))
            else:
                tensor_parallel_size = max(1, available_gpus)
        else:
            if requested_gpu_count is not None:
                tensor_parallel_size = max(
                    1, min(tensor_parallel_size, requested_gpu_count)
                )
        tensor_parallel_size = self._adjust_tensor_parallel_size(
            spec, tensor_parallel_size
        )

        local_checkpoint_dir: Path | None = None
        if checkpoint_cfg:
            load_cfg_local = dict(checkpoint_cfg)
            load_cfg_local["_logger"] = logger
            cfg_type = str(load_cfg_local.get("type", "http")).lower()
            if cfg_type == "http" and not (
                load_cfg_local.get("path")
                or load_cfg_local.get("taskId")
                or load_cfg_local.get("task_id")
            ):
                local_checkpoint_dir = None
            else:
                if cfg_type == "http":
                    load_cfg_local["type"] = "local"
                try:
                    path = resolve_checkpoint_load(
                        load_cfg_local, Path(tempfile.gettempdir()), logger=logger
                    )
                    candidate_path = Path(path) if path else None
                    if (
                        candidate_path
                        and candidate_path.exists()
                        and candidate_path.is_dir()
                    ):
                        local_checkpoint_dir = candidate_path
                        logger.info(
                            "Resolved model checkpoint locally: %s",
                            local_checkpoint_dir,
                        )
                except Exception as exc:
                    logger.debug("Local checkpoint resolution failed: %s", exc)
                    local_checkpoint_dir = None

        if local_checkpoint_dir is not None:
            logger.info(
                "Initializing vLLM from local checkpoint directory %s",
                local_checkpoint_dir,
            )
        else:
            logger.info("Initializing vLLM from identifier %s", ident)

        requested_util = float(vllm_cfg.pop("gpu_memory_utilization", 0.9))

        accepted_engine_args = {
            "max_model_len": int,
            "dtype": str,
            "download_dir": str,
            "max_num_batched_tokens": int,
            "max_cudagraph_capture_size": int,
            "enable_mm_embeds": bool,
            "limit_mm_per_prompt": dict,
            "quantization": str,
            "kv_cache_dtype": str,
            "enforce_eager": bool,
            "hf_token": str,
            "revision": str,
            "tokenizer_revision": str,
            "cpu_offload_gb": float,
            "swap_space": float,
        }

        kwargs_base: dict[str, Any] = dict(
            model=str(local_checkpoint_dir or ident),
            trust_remote_code=bool(vllm_cfg.pop("trust_remote_code", False)),
            seed=vllm_cfg.pop("seed", 42),
        )
        kwargs_base.update(self._extra_llm_kwargs(spec))
        for arg, arg_type in accepted_engine_args.items():
            if arg in vllm_cfg:
                kwargs_base[arg] = arg_type(vllm_cfg.pop(arg))
        hf_overrides: dict[str, Any] = {}
        if "rope_scaling" in vllm_cfg:
            hf_overrides["rope_scaling"] = vllm_cfg.pop("rope_scaling")
        if "rope_theta" in vllm_cfg:
            hf_overrides["rope_theta"] = float(vllm_cfg.pop("rope_theta"))
        if hf_overrides:
            kwargs_base["hf_overrides"] = hf_overrides
        if "env_vars" in vllm_cfg:
            env_vars = vllm_cfg.pop("env_vars")
            assert isinstance(env_vars, dict)
            os.environ.update(env_vars)
        if len(vllm_cfg) > 0:
            logger.info(
                "Ignored unrecognized vLLM config fields: %s", list(vllm_cfg.keys())
            )

        tp_candidates: list[int] = []
        seen_tp: set[int] = set()
        initial_tp = max(1, tensor_parallel_size)
        if initial_tp not in seen_tp:
            tp_candidates.append(initial_tp)
            seen_tp.add(initial_tp)
        if initial_tp != 1 and 1 not in seen_tp:
            tp_candidates.append(1)
            seen_tp.add(1)

        last_exc: Exception | None = None
        success = False
        chosen_kwargs: dict[str, Any] = {}

        for tp_idx, tp_value in enumerate(tp_candidates, start=1):
            kwargs = dict(kwargs_base)
            kwargs["tensor_parallel_size"] = tp_value

            safe_util, free_ratio = self._compute_safe_utilization(requested_util)
            if safe_util < requested_util - 1e-3:
                if free_ratio is not None:
                    logger.warning(
                        "Requested gpu_memory_utilization=%.3f but only %.2f%% of GPU "
                        "memory is free; reducing utilization to %.3f",
                        requested_util,
                        free_ratio * 100.0,
                        safe_util,
                    )
                else:
                    logger.warning(
                        "Requested gpu_memory_utilization=%.3f but available GPU "
                        "memory is constrained; reducing utilization to %.3f",
                        requested_util,
                        safe_util,
                    )

            util_candidates: list[float] = []
            tried_utils: set[float] = set()
            util_candidates.append(safe_util)
            for delta in (0.05, 0.1, 0.15):
                candidate = max(0.3, safe_util - delta)
                if (
                    candidate < util_candidates[0] - 1e-3
                    and candidate not in util_candidates
                ):
                    util_candidates.append(candidate)

            for util_idx, util in enumerate(util_candidates, start=1):
                if util in tried_utils:
                    continue
                tried_utils.add(util)

                attempt_kwargs = dict(kwargs)
                attempt_kwargs["gpu_memory_utilization"] = util
                self._llm_kwargs = dict(attempt_kwargs)
                logger.info(
                    "Initializing vLLM (TP candidate %d/%d, attempt %d/%d) "
                    "with tensor_parallel_size=%d, gpu_memory_utilization=%.3f",
                    tp_idx,
                    len(tp_candidates),
                    util_idx,
                    len(util_candidates),
                    tp_value,
                    util,
                )
                try:
                    with self._span(
                        "model load",
                        kind=FlowMeshSpanKind.COMPUTE,
                        attributes={
                            "task_ids": list(task_ids or ()),
                            "tensor_parallel_size": tp_value,
                            "gpu_memory_utilization": util,
                        },
                    ):
                        self._llm = LLM(**attempt_kwargs)  # type: ignore[call-arg]
                    chosen_kwargs = dict(attempt_kwargs)
                    success = True
                    break
                except TypeError as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "memory profiling" in msg and tp_value != 1:
                        logger.warning(
                            "vLLM initialization failed due to memory profiling under "
                            "tensor_parallel_size=%d; retrying with smaller "
                            "parallelism.",
                            tp_value,
                        )
                        break
                    continue
                except (ValueError, RuntimeError) as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "memory profiling" in msg and tp_value != 1:
                        logger.warning(
                            "vLLM initialization encountered memory profiling race "
                            "under tensor_parallel_size=%d; retrying with smaller "
                            "parallelism.",
                            tp_value,
                        )
                        break
                    logger.warning(
                        "vLLM initialization attempt failed (tp=%d, util=%.3f): %s",
                        tp_value,
                        util,
                        exc,
                    )
                    continue

            if success:
                break

        if not success or self._llm is None:
            message = (
                "Failed to initialize vLLM after trying tensor_parallel_size "
                f"candidates {tp_candidates} and gpu_memory_utilization adjustments "
                f"(last error: {last_exc})"
            )
            raise ExecutionError(message)

        self._llm_kwargs = chosen_kwargs
        self._model_name = ident

    def _get_tokenizer(self) -> Any | None:
        if self._llm is None:
            return None
        return self._llm.get_tokenizer()

    def _template_params_cfg(
        self, inference_cfg: dict[str, Any]
    ) -> dict[str, Any] | _RawJsonSchema | None:
        """Resolve ``spec.inference.templates``.

        Returns a dict with template, params, and placeholders if using named fields,
        or a _RawJsonSchema if using a raw JSON schema.
        """
        templates = inference_cfg.get("templates")
        if templates is None:
            return None
        if not isinstance(templates, dict | list) or not templates:
            raise ExecutionError(
                "spec.inference.templates must be a non-empty dict "
                "(raw JSON schema) or list (named fields)."
            )
        if isinstance(templates, dict):
            return _RawJsonSchema(templates)

        params = {}
        placeholders = []
        for item in templates:
            if not isinstance(item, dict) or "name" not in item:
                raise ExecutionError(
                    "Each template item must be a mapping with a 'name'"
                )
            name = item["name"]
            params[name] = item
            placeholders.append(name)

        # Auto-generate a JSON template string for structured outputs
        template_str = (
            "{" + ", ".join(f'"{name}": "{{{name}}}"' for name in placeholders) + "}"
        )
        return {
            "template": template_str,
            "params": params,
            "placeholders": set(placeholders),
        }

    def _construct_template_param_schema(
        self, inference_cfg: dict[str, Any]
    ) -> dict[str, Any] | _RawJsonSchema | None:
        cfg = self._template_params_cfg(inference_cfg)
        if not isinstance(cfg, dict):
            return cfg

        params_cfg: dict[str, Any] = cfg["params"]
        schema: dict[str, Any] = {}
        for param_name, param_cfg in params_cfg.items():
            if not isinstance(param_cfg, dict):
                raise ExecutionError(
                    f"Template parameter '{param_name}' must be a mapping"
                )

            field_type: type
            min_val: Any | None = None
            max_val: Any | None = None
            if candidates_spec := param_cfg.get("candidates"):
                if not isinstance(candidates_spec, list):
                    raise ExecutionError("Template parameter candidates must be a list")
                if len(candidates_spec) == 0:
                    raise ExecutionError(
                        "Template parameter candidates must not be empty"
                    )
                candidates = list(set(to_json_serializable(v) for v in candidates_spec))
                members = {f"v{idx}": value for idx, value in enumerate(candidates)}
                field_type = Enum(f"{param_name}Candidates", members)  # type: ignore
            else:
                if "type" not in param_cfg:
                    raise ExecutionError(
                        f"Template parameter '{param_name}' must have a 'type' "
                        "if candidates are not specified"
                    )
                type_spec = param_cfg["type"]
                min_val = param_cfg.get("min")
                max_val = param_cfg.get("max")

                field_type_resolved = self._resolve_param_type(type_spec)
                if field_type_resolved is None:
                    raise ExecutionError(
                        f"Unsupported template parameter type: {type_spec}"
                    )
                field_type = field_type_resolved

            field_type, field = self._build_param_field(field_type, min_val, max_val)
            schema[param_name] = (field_type, field)
        return schema

    @staticmethod
    def _build_param_field(
        field_type: type,
        min_val: Any | None,
        max_val: Any | None,
    ) -> tuple[type, Any]:
        field_kwargs: dict[str, Any] = {}

        def _coerce(val: Any, target_type: type) -> Any:
            if val is None or isinstance(val, target_type):
                return val
            if target_type is datetime.date and isinstance(val, str):
                return datetime.date.fromisoformat(val)
            if target_type is datetime.datetime and isinstance(val, str):
                return datetime.datetime.fromisoformat(val)
            return val

        if min_val is not None or max_val is not None:
            if field_type in {int, float, datetime.date, datetime.datetime}:
                if min_val is not None:
                    field_kwargs["ge"] = _coerce(min_val, field_type)
                if max_val is not None:
                    field_kwargs["le"] = _coerce(max_val, field_type)
            else:
                logger.warning(
                    f"{field_type} doesn't support min/max validation, "
                    f"ignoring min_val={min_val}, max_val={max_val}"
                )

        return field_type, Field(..., **field_kwargs)

    def _build_sampling_params(
        self,
        inference_cfg: dict[str, Any],
        schema: dict[str, Any] | _RawJsonSchema | None = None,
    ) -> SamplingParams:
        optional_sampling_fields: dict[str, Any] = {}
        if isinstance(schema, _RawJsonSchema):
            optional_sampling_fields["structured_outputs"] = StructuredOutputsParams(
                json=schema.schema
            )
        elif schema:
            optional_sampling_fields["structured_outputs"] = StructuredOutputsParams(
                json=create_model("Template", **schema).model_json_schema()
            )
        if "logprobs" in inference_cfg:
            optional_sampling_fields["logprobs"] = int(inference_cfg["logprobs"])
        if "seed" in inference_cfg:
            optional_sampling_fields["seed"] = int(inference_cfg["seed"])
        if "stop_token_ids" in inference_cfg:
            optional_sampling_fields["stop_token_ids"] = inference_cfg["stop_token_ids"]
        if "bad_words" in inference_cfg:
            optional_sampling_fields["bad_words"] = inference_cfg["bad_words"]
        if "n" in inference_cfg:
            optional_sampling_fields["n"] = int(inference_cfg["n"])
        return SamplingParams(  # type: ignore[call-arg]
            temperature=float(inference_cfg.get("temperature", 0.7)),
            top_p=float(inference_cfg.get("top_p", 0.95)),
            top_k=int(inference_cfg.get("top_k", -1)),
            min_p=float(inference_cfg.get("min_p", 0.0)),
            max_tokens=int(inference_cfg.get("max_tokens", 512)),
            min_tokens=int(inference_cfg.get("min_tokens", 0)),
            presence_penalty=float(inference_cfg.get("presence_penalty", 0.0)),
            frequency_penalty=float(inference_cfg.get("frequency_penalty", 0.0)),
            repetition_penalty=float(inference_cfg.get("repetition_penalty", 1.0)),
            stop=inference_cfg.get("stop"),
            skip_special_tokens=bool(inference_cfg.get("skip_special_tokens", True)),
            **optional_sampling_fields,
        )

    def _remap_grouped_outputs(
        self,
        *,
        task_id: str,
        items: list[dict[str, Any]],
        group_sizes: list[int],
        base_prompts: list[str],
        base_metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(group_sizes) != len(base_prompts):
            raise ExecutionError(
                "image_group_sizes / base prompt mismatch for grouped output remap "
                f"(task={task_id}, sizes={len(group_sizes)}, "
                f"prompts={len(base_prompts)})."
            )
        if len(base_metadata) != len(base_prompts):
            raise ExecutionError(
                "base metadata / base prompt mismatch for grouped output remap "
                f"(task={task_id}, metadata={len(base_metadata)}, "
                f"prompts={len(base_prompts)})."
            )
        expected = sum(group_sizes)
        if len(items) != expected:
            raise ExecutionError(
                "Grouped output remap item mismatch "
                f"(task={task_id}, expected={expected}, got={len(items)})."
            )

        remapped: list[dict[str, Any]] = []
        cursor = 0
        for idx, group_size in enumerate(group_sizes):
            grouped_items = items[cursor : cursor + group_size]
            outputs: list[str] = []
            finish_reasons: list[str | None] = []
            for item in grouped_items:
                output = item.get("output")
                if not isinstance(output, str):
                    raise ExecutionError(
                        "Grouped output remap expects string outputs before remap "
                        f"(task={task_id}, row={idx})."
                    )
                outputs.append(output)
                finish_reason = item.get("finish_reason")
                if finish_reason is not None and not isinstance(finish_reason, str):
                    raise ExecutionError(
                        "Grouped output remap expects finish_reason to be str|None "
                        f"(task={task_id}, row={idx})."
                    )
                finish_reasons.append(finish_reason)
            payload: dict[str, Any] = {
                "index": idx,
                "prompt": base_prompts[idx],
                "output": outputs,
                "finish_reason": finish_reasons,
            }
            metadata = base_metadata[idx]
            if metadata:
                payload["metadata"] = metadata
            remapped.append(payload)
            cursor += group_size
        return remapped

    def _flatten_image_embedding_chunks(
        self,
        embedding_chunks: list[torch.Tensor],
        *,
        task_id: str,
        group_sizes: list[int] | None,
    ) -> torch.Tensor:
        if not embedding_chunks:
            raise ExecutionError(
                "Loaded image embedding list must be non-empty " f"(task={task_id})."
            )
        if group_sizes is not None:
            if len(embedding_chunks) != len(group_sizes):
                raise ExecutionError(
                    "Grouped image embedding chunk count mismatch "
                    f"(task={task_id}, chunks={len(embedding_chunks)}, "
                    f"group_sizes={len(group_sizes)})."
                )
            for idx, (chunk, expected_size) in enumerate(
                zip(embedding_chunks, group_sizes)
            ):
                if len(chunk) != expected_size:
                    raise ExecutionError(
                        "Grouped image embedding chunk size mismatch "
                        f"(task={task_id}, chunk={idx}, expected={expected_size}, "
                        f"got={len(chunk)})."
                    )
        try:
            return torch.cat(embedding_chunks, dim=0)
        except Exception as exc:
            raise ExecutionError(
                "Failed to combine image embedding chunks into a single tensor "
                f"(task={task_id})."
            ) from exc

    def _postprocess_prompts(self, parsed: InferenceEntry) -> PreparedInferenceEntry:
        task_id = parsed.task_id
        prepared = self._prepare_inference_entry(
            parsed, has_images=(parsed.image_embedding_path is not None)
        )

        group_sizes: list[int] | None = None
        if prepared.image_group_sizes is not None:
            group_sizes = self._validate_image_group_sizes(
                prepared.image_group_sizes, task_id=task_id
            )

        image_embedding_list: list[torch.Tensor] | None = None
        if embedding_path := prepared.image_embedding_path:
            if (
                isinstance(embedding_path, (str, Path))
                and Path(embedding_path).exists()
            ):
                try:
                    loaded = torch.load(embedding_path, map_location="cpu")
                    if not isinstance(loaded, list):
                        raise TypeError(
                            "Loaded image embedding is not a list of tensors"
                        )
                    if not all(isinstance(chunk, torch.Tensor) for chunk in loaded):
                        raise TypeError(
                            "Loaded image embedding list contains non-tensor values"
                        )
                    image_embedding_list = loaded
                except Exception as exc:
                    logger.warning(
                        "Failed to load image embedding from %s: %s",
                        embedding_path,
                        exc,
                    )
                else:
                    prepared.image_embedding = self._flatten_image_embedding_chunks(
                        image_embedding_list,
                        task_id=task_id,
                        group_sizes=group_sizes,
                    )

        if group_sizes is not None:
            image_embedding = prepared.image_embedding
            if image_embedding is None:
                raise ExecutionError(
                    "image_group_sizes was provided but image_embedding is missing "
                    f"(task={task_id})."
                )
            if sum(group_sizes) != len(image_embedding):
                raise ExecutionError(
                    "image_group_sizes does not match image embedding count "
                    f"(task={task_id}, sum(group_sizes)={sum(group_sizes)}, "
                    f"embedding_count={len(image_embedding)})."
                )
            metadata_rows = prepared.metadata
            rendered_prompts = prepared.prompts
            grouped_prompt_count = len(group_sizes)
            expanded_prompt_count = sum(group_sizes)
            base_prompts: list[str] = []
            base_metadata: list[dict[str, Any]] = []
            if len(rendered_prompts) == grouped_prompt_count:
                base_prompts = list(rendered_prompts)
                base_metadata = list(metadata_rows)
                expanded_prompts: list[str] = []
                expanded_metadata: list[dict[str, Any]] = []
                for prompt, metadata, group_size in zip(
                    base_prompts, base_metadata, group_sizes
                ):
                    expanded_prompts.extend([prompt] * group_size)
                    expanded_metadata.extend([metadata] * group_size)
                rendered_prompts = expanded_prompts
                metadata_rows = expanded_metadata
            elif len(rendered_prompts) == expanded_prompt_count:
                base_prompts = []
                base_metadata = []
                cursor = 0
                for group_size in group_sizes:
                    base_prompts.append(rendered_prompts[cursor])
                    base_metadata.append(metadata_rows[cursor])
                    cursor += group_size
            else:
                raise ExecutionError(
                    "image_group_sizes length must match grouped prompt count or "
                    "expanded prompt count "
                    f"(task={task_id}, group_sizes={grouped_prompt_count}, "
                    f"prompts={len(rendered_prompts)}, "
                    f"sum_group_sizes={expanded_prompt_count})."
                )
            prepared.prompts = rendered_prompts
            prepared.metadata = metadata_rows
            prepared.image_group_sizes = group_sizes
            prepared.image_group_base_prompts = base_prompts
            prepared.image_group_base_metadata = base_metadata

        return prepared

    # --------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------- #
    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:  # type: ignore[override]
        spec = self.require_spec(task, InferenceSpecStrict)
        task_id = task.task_id.strip()
        if not task_id:
            raise ExecutionError("task_id is required for inference execution")

        with self._task_span(task_id, task.workflow_id, out_dir):
            result = self._run_body(task, spec, task_id, out_dir)
        maybe_upload_artifacts(task, out_dir, logger=logger)
        return result

    def _run_body(
        self,
        task: ExecutorTask,
        spec: InferenceSpecStrict,
        task_id: str,
        out_dir: Path,
    ) -> dict[str, Any]:
        merge_children = task.merged_children or []
        entries: list[PreparedInferenceEntry] = []
        collection_jobs: list[dict[str, Any]] = [
            {"task_id": task_id, "spec": spec, "is_parent": True}
        ]

        for child in merge_children:
            child_id = child.task_id
            child_spec = child.spec
            if not isinstance(child_spec, InferenceSpecStrict):
                raise ExecutionError(
                    "Merged child spec must be inference for merged vLLM execution"
                )
            self._log_event("queuing for execution", data_id=child_id)
            collection_jobs.append(
                {"task_id": child_id, "spec": child_spec, "is_parent": False}
            )

        task_ids = [task_id] + [child.task_id for child in merge_children]
        self._ensure_llm(spec, task_ids)
        assert self._llm is not None

        dependencies_by_task: dict[str, list[str]] = {}
        results: dict[str, tuple[PreparedInferenceEntry, list[str]]] = {}

        def _collect(
            job: dict[str, Any],
        ) -> tuple[str, PreparedInferenceEntry, list[str]]:
            job_task_id = job["task_id"]
            job_spec = job["spec"]
            raw_entry: InferenceEntry = self._collect_prompts_for_spec(
                job_spec, task_id=job_task_id
            )
            postprocessed_entry = self._postprocess_prompts(raw_entry)
            deps = self._extract_source_data_ids(job_spec)
            return job_task_id, postprocessed_entry, deps

        if len(collection_jobs) == 1:
            job_task_id, entry, deps = _collect(collection_jobs[0])
            results[job_task_id] = (entry, deps)
        else:
            logger.info(
                "Collecting prompts for %d merged tasks in parallel",
                len(collection_jobs),
            )
            future_map = {
                self._submit_in_context(_collect, job): job for job in collection_jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                job_task_id = job["task_id"]
                try:
                    job_task_id, entry, deps = future.result()
                    results[job_task_id] = (entry, deps)
                except Exception as exc:
                    if job["is_parent"]:
                        raise
                    raise ExecutionError(
                        f"Failed to prepare merged child task {job_task_id}: {exc}"
                    ) from exc

        parent_entry, parent_deps = results[task_id]
        entries.append(parent_entry)
        dependencies_by_task[task_id] = parent_deps
        entry_by_task_id: dict[str, PreparedInferenceEntry] = {task_id: parent_entry}
        child_entry: PreparedInferenceEntry | None
        for child in merge_children:
            child_id = child.task_id
            child_entry, child_deps = results[child_id]
            entries.append(child_entry)
            dependencies_by_task[child_id] = child_deps
            entry_by_task_id[child_id] = child_entry

        self._batched_inputs = []
        self._prompt_owners = []
        self._batched_metadata = []
        for entry in entries:
            owner = entry.task_id
            prompts_for_owner: list[str] = entry.prompts
            image_embedding: torch.Tensor | None = entry.image_embedding
            if image_embedding is not None:
                if len(image_embedding) != len(prompts_for_owner):
                    raise ExecutionError(
                        f"spec.data.image_embedding length mismatch for task {owner}: "
                        f"{len(image_embedding)} (image_embedding) vs "
                        f"{len(prompts_for_owner)} (prompts)"
                    )
            metadata_for_owner = entry.metadata
            if metadata_for_owner and len(metadata_for_owner) != len(prompts_for_owner):
                raise ExecutionError(
                    f"spec.data.metadata length mismatch for task {owner}: "
                    f"{len(metadata_for_owner)} (metadata) vs {len(prompts_for_owner)} "
                    "(prompts)"
                )
            image_embedding_list = (
                torch.split(image_embedding, 1) if image_embedding is not None else None
            )
            for idx, prompt in enumerate(prompts_for_owner):
                input: str | TextPrompt = prompt
                if image_embedding is not None:
                    assert image_embedding_list is not None
                    input = TextPrompt(
                        prompt=prompt,
                        multi_modal_data={"image": image_embedding_list[idx]},
                    )
                self._batched_inputs.append(input)
                self._prompt_owners.append(owner)
                if metadata_for_owner:
                    meta_entry = metadata_for_owner[idx]
                else:
                    meta_entry = {"prompt": prompt}
                self._batched_metadata.append(meta_entry)

        if not self._batched_inputs:
            raise ExecutionError("No prompts prepared. Check spec.data configuration.")

        self._base_inference = copy.deepcopy(parent_entry.inference_cfg)
        base_sampling_cfg = self._normalize_inference_for_sampling(self._base_inference)
        for entry in entries[1:]:
            other_cfg = self._normalize_inference_for_sampling(entry.inference_cfg)
            if other_cfg != base_sampling_cfg:
                raise ExecutionError(
                    "Merged tasks must share inference parameters (excluding "
                    "system_prompt)."
                )

        template_param_schema = self._construct_template_param_schema(
            self._base_inference
        )
        sampling_params = self._build_sampling_params(
            self._base_inference, schema=template_param_schema
        )

        generate_kwargs = self._build_generate_kwargs(spec, out_dir)

        t0 = time.time()
        with self._span(
            "generation",
            kind=FlowMeshSpanKind.COMPUTE,
            attributes={
                "task_ids": list(task_ids),
                "prompt_count": len(self._batched_inputs),
            },
        ):
            outputs = self._llm.generate(
                self._batched_inputs,
                sampling_params=sampling_params,
                **generate_kwargs,
            )  # type: ignore[attr-defined]
        latency = time.time() - t0

        with self._span(
            "output postprocessing",
            kind=FlowMeshSpanKind.COMPUTE,
            attributes={"task_ids": list(task_ids)},
        ):
            per_task_items: dict[str, list[dict[str, Any]]] = {}
            usage_by_task: dict[str, dict[str, int | float]] = {}
            counts_by_task: dict[str, int] = {}

            total_prompt_tokens = 0
            total_completion_tokens = 0

            for idx, out in enumerate(outputs):
                owner = (
                    self._prompt_owners[idx]
                    if idx < len(self._prompt_owners)
                    else task_id
                )
                owner_items = per_task_items.setdefault(owner, [])
                local_index = len(owner_items)
                prompt_text: str = ""
                if idx < len(self._batched_inputs):
                    prompt_payload = self._batched_inputs[idx]
                    if isinstance(prompt_payload, dict):
                        prompt_text = prompt_payload["prompt"]
                    else:
                        prompt_text = prompt_payload
                metadata_entry = (
                    self._batched_metadata[idx]
                    if idx < len(self._batched_metadata)
                    else {"prompt": prompt_text}
                )

                out_outputs = getattr(out, "outputs", None)
                if not out_outputs:
                    payload = {
                        "index": local_index,
                        "prompt": prompt_text,
                        "output": "",
                        "finish_reason": None,
                    }
                    if metadata_entry:
                        payload["metadata"] = metadata_entry
                    owner_items.append(payload)
                    usage_by_task.setdefault(
                        owner, {"prompt_tokens": 0, "completion_tokens": 0}
                    )
                    counts_by_task[owner] = counts_by_task.get(owner, 0) + 1
                    continue

                best = out_outputs[0]
                text = getattr(best, "text", "") or ""

                output_value: Any = text
                if template_param_schema:
                    try:
                        output_value = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Task %s: failed to parse structured output as JSON: %r",
                            owner,
                            text,
                        )

                finish_reason = getattr(best, "finish_reason", None)
                payload = {
                    "index": local_index,
                    "prompt": prompt_text,
                    "output": output_value,
                    "finish_reason": finish_reason,
                }
                if metadata_entry:
                    payload["metadata"] = metadata_entry
                owner_items.append(payload)

                prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
                best_token_ids = getattr(best, "token_ids", None) or []
                prompt_len = len(prompt_token_ids)
                completion_len = len(best_token_ids)

                total_prompt_tokens += prompt_len
                total_completion_tokens += completion_len

                usage_entry = usage_by_task.setdefault(
                    owner, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                usage_entry["prompt_tokens"] += prompt_len
                usage_entry["completion_tokens"] += completion_len
                counts_by_task[owner] = counts_by_task.get(owner, 0) + 1

            for owner, entry in entry_by_task_id.items():
                group_sizes: list[int] | None = entry.image_group_sizes
                if group_sizes is None:
                    continue
                base_prompts = entry.image_group_base_prompts
                base_metadata = entry.image_group_base_metadata
                if base_prompts is None or base_metadata is None:
                    raise ExecutionError(
                        "Grouped image outputs require base prompts and metadata "
                        f"(task={owner})."
                    )
                owner_items = per_task_items.get(owner, [])
                per_task_items[owner] = self._remap_grouped_outputs(
                    task_id=owner,
                    items=owner_items,
                    group_sizes=self._validate_image_group_sizes(
                        group_sizes,
                        task_id=owner,
                    ),
                    base_prompts=base_prompts,
                    base_metadata=base_metadata,
                )

            for owner, usage in usage_by_task.items():
                usage["total_tokens"] = (
                    usage["prompt_tokens"] + usage["completion_tokens"]
                )
                usage["latency_sec"] = latency
                usage["num_requests"] = counts_by_task.get(owner, 0)

            parent_usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "latency_sec": latency,
                "num_requests": len(self._batched_inputs),
            }

            result: dict[str, Any] = {
                "ok": True,
                "model": self._model_name,
                "items": per_task_items.get(task_id, []),
                "usage": parent_usage,
            }

            child_results: dict[str, Any] = {}
            for child in merge_children:
                child_id = child.task_id.strip()
                if not child_id:
                    continue
                child_payload: dict[str, Any] = {
                    "items": per_task_items.get(child_id, []),
                }
                maybe_usage = usage_by_task.get(child_id)
                if maybe_usage:
                    child_payload["usage"] = maybe_usage
                child_results[child_id] = child_payload

            if parent_tables := parent_entry.tables:
                result = self._populate_table(result, parent_tables)
            if child_results:
                for child_id, child_payload in list(child_results.items()):
                    if (child_entry := entry_by_task_id.get(child_id)) and (
                        child_tables := child_entry.tables
                    ):
                        child_results[child_id] = self._populate_table(
                            child_payload, child_tables
                        )

            if child_results:
                result["children"] = child_results

        with self._span(
            "JSONL export",
            kind=FlowMeshSpanKind.COMPUTE,
            attributes={"task_ids": list(task_ids)},
        ):
            self._maybe_export_jsonl(spec, task_id, result, out_dir)

        self._dump_to_governance(
            governance_spec=spec.governance,
            task_id=task_id,
            result=result,
            dependencies_by_task=dependencies_by_task,
        )

        return result

    def cleanup_after_run(self) -> None:
        if self._llm:
            self._shutdown_llm()
        self._llm_kwargs = {}
        self._batched_inputs = []
        self._prompt_owners = []
        self._batched_metadata = []
        self._base_inference = {}

        logger.debug("Shutting down I/O executor")
        self.io_executor.shutdown(wait=True)
        logger.debug("I/O executor shut down successfully")

    def _shutdown_llm(self) -> None:
        if self._llm is None:
            logger.warning("No vLLM instance to shutdown")
            return
        try:
            destroy_distributed_environment()
            if dist.is_initialized():
                dist.destroy_process_group()
            self._llm.llm_engine.engine_core.shutdown()
        except Exception:
            logger.debug("Failed to shutdown vLLM instance cleanly", exc_info=True)
        finally:
            self._llm = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                for idx in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass
        gc.collect()
