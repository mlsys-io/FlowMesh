#!/usr/bin/env python3
"""
HFTransformersExecutor

- Uses Hugging Face Transformers to run LLM text generation.
- Mirrors the YAML-reading behavior of VLLMExecutor to ease swapping.
- Supports CPU/GPU execution with optional device_map/quantization.

YAML spec expectations (compatible with your existing schema):

spec:
  model:
    source:
      identifier: meta-llama/Llama-3-8b-instruct   # or local path
    transformers:                                  # optional
      device_map: auto | cuda | cpu                # default: auto if accelerate
                                                     present else single device
      dtype: auto | float16 | bfloat16 | float32   # default: auto
      trust_remote_code: false                     # default: false
      low_cpu_mem_usage: true                      # default: true
      load_in_8bit: false                          # optional (requires bitsandbytes)
      load_in_4bit: false                          # optional (requires bitsandbytes)
      use_flash_attention_2: false                 # optional (if supported by model)
      mode: text-generation | visual-embedding     # default: text-generation
  inference:
    temperature: 0.7
    top_p: 0.95
    top_k: 50
    max_tokens: 512                                # alias of max_new_tokens
    max_new_tokens: 512                            # takes precedence if provided
    do_sample: true                                # optional, will be inferred from
                                                     temperature/top_p/top_k if missing
    repetition_penalty: 1.0                        # transformers-specific
    stop: ["\n\nUser:", "</s>"]                    # optional stop strings
                                                     (post-process truncation)
  data:
    type: dataset | list
    # if dataset:
    url: glue                                      # HF hub name or path
    name: sst2                                     # optional config name
    split: validation                              # default train
    column: text                                   # default text
    shuffle: false                                 # optional
    seed: 42                                       # optional
    buffer_size: 1000                              # optional
    # if list:
    items: ["Hello, world!"]

Output file: out_dir/results.json
"""

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import (
    EmbeddingSpecStrict,
    InferenceSpecStrict,
)

from ..utils.logging import configure_hf_library_logging
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.data import InferenceEntry
from .mixins.inference import InferenceMixin
from .utils.checkpoints import maybe_upload_artifacts, maybe_upload_traces

try:
    import torch
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        GenerationConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )

    _HAS_TRANSFORMERS = True
except Exception:
    if TYPE_CHECKING:
        import torch
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            GenerationConfig,
            PreTrainedModel,
            PreTrainedTokenizerBase,
        )
    else:
        torch = None
        AutoConfig = None
        AutoModelForImageTextToText = None
        AutoImageProcessor = None
        AutoModelForCausalLM = None
        AutoTokenizer = None
        GenerationConfig = None
        PreTrainedModel = None
        PreTrainedTokenizerBase = None

    _HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class TransformersResult(BaseExecutorResult):
    ok: bool = True
    model: str | None = None
    items: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    count: int | None = None
    embedding_file: ArtifactRef | None = None
    image_group_sizes: list[int] | None = None


class HFTransformersExecutor(InferenceMixin, Executor):
    """Executor that runs text generation via Hugging Face Transformers."""

    name = "transformers"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tok: PreTrainedTokenizerBase | None = None
        self._image_processor: Any | None = None
        self._model: PreTrainedModel | None = None
        self._model_config: dict[str, Any] | None = None
        self._device: str | None = None
        self._model_name: str | None = None
        self._mode: str = "text-generation"
        self._prompts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._inf: dict[str, Any] = {}
        self._applied_chat_template = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def prepare(self) -> None:  # type: ignore[override]
        if not _HAS_TRANSFORMERS:
            raise ExecutionError(
                "transformers/torch is not installed (`pip install transformers "
                "torch`)."
            )
        configure_hf_library_logging()

    def _pick_device(self, cfg: dict[str, Any]) -> str:
        # Explicit device_map overrides simple device if provided
        device_map = cfg.get("device_map")
        if device_map in {
            "auto",
            "balanced",
            "balanced_low_0",
        }:  # acceptable values for accelerate
            return "auto"
        # Simple single-device selection
        if device_map == "cuda":
            if torch and torch.cuda.is_available():
                return "cuda"
            raise ExecutionError("Requested CUDA but no GPU is available.")
        if device_map == "cpu":
            return "cpu"
        # Default preference: CUDA if available
        return "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    def _to_torch_dtype(self, s: str | None):
        if not s or s == "auto":
            return "auto"
        s = str(s).lower()
        if s in {"float16", "fp16"}:
            return torch.float16
        if s in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if s in {"float32", "fp32"}:
            return torch.float32
        raise ExecutionError(f"Unsupported dtype: {s}")

    def _ensure_model(self, spec: InferenceSpecStrict | EmbeddingSpecStrict) -> None:
        ident = spec.model_name or os.getenv("HF_MODEL")
        if not ident:
            raise ExecutionError(
                "spec.model.source.identifier (or HF_MODEL) is required."
            )

        model_cfg = spec.model
        tcfg = (model_cfg and model_cfg.transformers) or {}
        self._mode = tcfg.get("mode", "text-generation")

        device = self._pick_device(tcfg)
        dtype = self._to_torch_dtype(tcfg.get("dtype", "auto"))
        trust_remote_code = spec.model_trust_remote_code or bool(
            tcfg.get("trust_remote_code", False)
        )
        revision = spec.model_revision or tcfg.get("revision")

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": bool(tcfg.get("low_cpu_mem_usage", True)),
        }
        if revision:
            load_kwargs["revision"] = revision
        # quantization options (optional, require bitsandbytes)
        if bool(tcfg.get("load_in_8bit", False)):
            load_kwargs["load_in_8bit"] = True
        if bool(tcfg.get("load_in_4bit", False)):
            load_kwargs["load_in_4bit"] = True
        # AWQ / GPTQ quantization (require auto-awq / auto-gptq)
        if tcfg.get("quantization"):
            quant_config = AutoConfig.from_pretrained(
                ident, revision=revision, trust_remote_code=trust_remote_code
            ).quantization_config
            load_kwargs["quantization_config"] = quant_config
        if tcfg.get("use_flash_attention_2") is True:
            load_kwargs["use_flash_attention_2"] = True

        # Device placement
        if device == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["dtype"] = dtype
        else:
            load_kwargs["dtype"] = dtype

        model_cfg_payload = (
            model_cfg.model_dump(mode="python", exclude_none=True)
            if model_cfg is not None
            else None
        )
        if self._model is not None and self._model_config == model_cfg_payload:
            logger.info("Model already loaded and matches spec; reusing.")
            return

        self._model_config = model_cfg_payload

        try:
            match self._mode:
                case "visual-embedding":
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        ident, **load_kwargs
                    )
                    tok_kwargs: dict[str, Any] = {
                        "use_fast": True,
                        "trust_remote_code": trust_remote_code,
                    }
                    if revision:
                        tok_kwargs["revision"] = revision
                    self._image_processor = AutoImageProcessor.from_pretrained(
                        ident, **tok_kwargs
                    )
                    self._tok = None

                case "text-generation":
                    tok_kwargs = {
                        "use_fast": True,
                        "trust_remote_code": trust_remote_code,
                    }
                    if revision:
                        tok_kwargs["revision"] = revision
                    tokenizer = AutoTokenizer.from_pretrained(ident, **tok_kwargs)
                    self._tok = tokenizer
                    # Ensure we have a pad token for batch generation
                    if tokenizer.pad_token_id is None:
                        # Fallback to eos token, common for decoder-only LMs
                        tokenizer.pad_token = tokenizer.eos_token

                    self._model = AutoModelForCausalLM.from_pretrained(
                        ident, **load_kwargs
                    )
                    self._image_processor = None

                case _:
                    raise ExecutionError(f"Unsupported task type: {self._mode}")

            if device != "auto":  # single-device path
                self._model.to(device)  # type: ignore[arg-type]
            self._model.eval()
        except Exception as e:
            raise ExecutionError(f"Failed to load model/tokenizer: {e}", retryable=True)

        self._device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model_name = ident

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _normalize_stops(self, stop_val: Any) -> list[str]:
        if stop_val is None:
            return []
        if isinstance(stop_val, str):
            return [stop_val]
        if isinstance(stop_val, Sequence):
            return [str(x) for x in stop_val]
        return []

    def _get_tokenizer(self) -> Any | None:
        return self._tok

    def _build_generation_config(
        self, inference_cfg: dict[str, Any], stop_strings: list[str]
    ) -> GenerationConfig:
        assert self._tok is not None

        if "max_new_tokens" in inference_cfg and "max_tokens" in inference_cfg:
            logger.warning(
                "Both max_new_tokens and max_tokens are set in inference config; "
                "max_new_tokens will take precedence."
            )
        if ((raw_max_tokens := inference_cfg.get("max_new_tokens")) is not None) or (
            (raw_max_tokens := inference_cfg.get("max_tokens")) is not None
        ):
            max_new_tokens = int(raw_max_tokens)
        else:
            max_new_tokens = 512
        temperature = float(inference_cfg.get("temperature", 0.7))
        top_p = float(inference_cfg.get("top_p", 0.95))
        top_k = int(inference_cfg.get("top_k", 50))
        do_sample = inference_cfg.get("do_sample")
        if do_sample is None:
            do_sample = (temperature > 0.0) or (top_p < 1.0) or (top_k > 0)

        config_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "do_sample": bool(do_sample),
            "repetition_penalty": float(inference_cfg.get("repetition_penalty", 1.0)),
            "pad_token_id": self._tok.pad_token_id,
            "eos_token_id": self._tok.eos_token_id,
        }

        if "min_new_tokens" in inference_cfg or "min_tokens" in inference_cfg:
            config_kwargs["min_new_tokens"] = int(
                inference_cfg.get("min_new_tokens", inference_cfg.get("min_tokens", 0))
            )
        if "min_p" in inference_cfg:
            config_kwargs["min_p"] = float(inference_cfg["min_p"])
        if "num_beams" in inference_cfg:
            config_kwargs["num_beams"] = int(inference_cfg["num_beams"])
        if "early_stopping" in inference_cfg:
            config_kwargs["early_stopping"] = inference_cfg["early_stopping"]
        if "length_penalty" in inference_cfg:
            config_kwargs["length_penalty"] = float(inference_cfg["length_penalty"])
        if "no_repeat_ngram_size" in inference_cfg:
            config_kwargs["no_repeat_ngram_size"] = int(
                inference_cfg["no_repeat_ngram_size"]
            )
        if stop_strings:
            config_kwargs["stop_strings"] = stop_strings

        bad_words = inference_cfg.get("bad_words")
        if bad_words:
            bad_word_list = (
                [bad_words] if isinstance(bad_words, str) else list(bad_words)
            )
            config_kwargs["bad_words_ids"] = [
                self._tok(str(word), add_special_tokens=False).input_ids
                for word in bad_word_list
            ]

        return GenerationConfig(**config_kwargs)

    def _detect_finish_reason(
        self,
        raw_text: str,
        final_text: str,
        gen_ids: Any,
        max_new_tokens: int,
        stop_strings: list[str],
    ) -> str | None:
        if stop_strings and raw_text != final_text:
            return "stop"
        eos_token_id = self._tok.eos_token_id if self._tok is not None else None
        eos_ids = (
            {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
        )
        if eos_ids and len(gen_ids) > 0 and int(gen_ids[-1]) in eos_ids:
            return "stop"
        if len(gen_ids) >= max_new_tokens:
            return "length"
        return None

    def run(self, task: ExecutorTask, out_dir: Path) -> TransformersResult:
        configure_hf_library_logging()
        spec = task.spec
        if not isinstance(spec, (InferenceSpecStrict, EmbeddingSpecStrict)):
            raise ExecutionError(
                "Unsupported spec type for transformers executor: "
                f"{spec.__class__.__name__}"
            )
        task_id = task.task_id
        with self._task_span(
            task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            result = self._run_inner(spec, task_id, out_dir)
        maybe_upload_artifacts(task, out_dir, logger=logger)
        maybe_upload_traces(task, out_dir, logger=logger)
        return result

    def _run_inner(
        self,
        spec: "InferenceSpecStrict | EmbeddingSpecStrict",
        task_id: str,
        out_dir: Path,
    ) -> TransformersResult:
        with self._span("model load", span_type=SpanType.COMPUTE):
            self._ensure_model(spec)

        deps = self._extract_source_data_ids(spec)
        dependencies_by_task = {task_id: deps}

        # Prepare data
        fetch_images = self._mode == "visual-embedding"

        # Use DataMixin to collect prompts and optionally fetch images
        raw_entry: InferenceEntry = self._collect_prompts_for_spec(
            spec, task_id=task_id, fetch_images=fetch_images
        )

        assert self._model is not None

        if self._mode == "visual-embedding":
            assert self._image_processor is not None
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            images = raw_entry.images
            image_group_sizes = raw_entry.image_group_sizes
            if not images:
                raise ExecutionError("No valid images found for visual-embedding task.")
            inputs = self._image_processor(images=images, return_tensors="pt").to(
                self._model.device
            )
            if isinstance(self._model.vision_tower, torch.Tensor):
                raise ExecutionError(
                    "The vision_tower attribute of the model is a Tensor, "
                    "expected a nn.Module."
                )
            if isinstance(self._model.multi_modal_projector, torch.Tensor):
                raise ExecutionError(
                    "The multi_modal_projector attribute of the model is a Tensor, "
                    "expected a nn.Module."
                )
            with torch.no_grad():
                vision_outputs = self._model.vision_tower(
                    inputs.pixel_values, output_hidden_states=True
                )
                selected_features = vision_outputs.hidden_states[
                    self._model.config.vision_feature_layer
                ]
                visual_embeddings = self._model.multi_modal_projector(selected_features)

            grouped_visual_embeddings: list[torch.Tensor]
            if image_group_sizes is not None:
                self._validate_image_group_sizes(
                    image_group_sizes,
                    task_id=task_id,
                )
                group_ranges = self._build_group_ranges(
                    image_group_sizes,
                    total_count=len(images),
                    task_id=task_id,
                    value_name="image embeddings",
                )
                grouped_visual_embeddings = [
                    visual_embeddings[start:end] for start, end in group_ranges
                ]
            else:
                grouped_visual_embeddings = list(
                    torch.split(visual_embeddings, 1, dim=0)
                )

            artifacts_dir = out_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            emb_path = artifacts_dir / "visual_embeddings.pt"
            torch.save(grouped_visual_embeddings, emb_path)

            result = TransformersResult(
                model=self._model_name,
                items=[],
                count=len(grouped_visual_embeddings),
                embedding_file=ArtifactRef(path="visual_embeddings.pt"),
                image_group_sizes=image_group_sizes,
            )

            self._dump_to_governance(
                task_id=task_id,
                result=result,
                dependencies_by_task=dependencies_by_task,
            )

            return result

        assert self._tok is not None
        prepared = self._prepare_inference_entry(raw_entry)
        self._prompts = prepared.prompts
        self._metadata = prepared.metadata
        self._applied_chat_template = prepared.applied_chat_template
        self._inf = (
            spec.inference or {} if isinstance(spec, InferenceSpecStrict) else {}
        )

        stops = self._normalize_stops(self._inf.get("stop"))
        skip_special_tokens = bool(self._inf.get("skip_special_tokens", True))
        gen_cfg = self._build_generation_config(self._inf, stop_strings=stops)

        if not self._prompts:
            raise ExecutionError("No prompts prepared. Check spec.data configuration.")

        # Tokenize batch
        enc = self._tok(
            self._prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=not self._applied_chat_template,
        )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        enc = {k: v.to(device) for k, v in enc.items()}  # type: ignore[arg-type]

        t0 = time.time()
        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={"prompt_count": len(self._prompts)},
        ):
            with torch.no_grad():
                try:
                    outputs = self._model.generate(  # type: ignore
                        **enc, generation_config=gen_cfg
                    )
                except ValueError as exc:
                    if not (stops and "stop" in str(exc).lower()):
                        raise
                    logger.warning(
                        "Falling back to decoded stop-string truncation after "
                        "native stop configuration failed: %s",
                        exc,
                    )
                    gen_cfg = self._build_generation_config(self._inf, stop_strings=[])
                    outputs = self._model.generate(  # type: ignore
                        **enc, generation_config=gen_cfg
                    )
        latency = time.time() - t0

        items: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        # For each sequence in the batch, split prompt vs generated part
        input_ids = enc["input_ids"]
        max_new_tokens = gen_cfg.max_new_tokens
        for i, (inp_ids, seq, prompt_text, metadata_entry) in enumerate(
            zip(input_ids, outputs, self._prompts, self._metadata, strict=True)
        ):
            input_len = int(inp_ids.shape[0])
            gen_part = seq[input_len:]
            raw_text = self._tok.decode(
                gen_part, skip_special_tokens=skip_special_tokens
            )
            text = raw_text

            # Apply simple stop-string truncation on decoded text
            if stops:
                cut = len(text)
                for s in stops:
                    idx = text.find(s)
                    if idx != -1:
                        cut = min(cut, idx)
                text = text[:cut]

            payload = {
                "index": i,
                "prompt": prompt_text,
                "output": text,
                "finish_reason": self._detect_finish_reason(
                    raw_text=raw_text,
                    final_text=text,
                    gen_ids=gen_part,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stops,
                ),
            }
            if metadata_entry:
                payload["metadata"] = metadata_entry
            items.append(payload)
            prompt_tokens += int(input_len)
            completion_tokens += int(gen_part.shape[0])

        result = TransformersResult(
            model=self._model_name,
            items=items,
            usage={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
                "num_requests": len(self._prompts),
                "latency_sec": latency,
            },
        )

        if isinstance(spec, InferenceSpecStrict):
            self._maybe_export_jsonl(spec, task_id, items, out_dir)

        self._dump_to_governance(
            task_id=task_id,
            result=result,
            dependencies_by_task=dependencies_by_task,
        )

        return result
