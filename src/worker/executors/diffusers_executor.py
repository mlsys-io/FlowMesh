#!/usr/bin/env python3
"""
DiffusersExecutor

- Uses Hugging Face Diffusers to run Text-to-Image generation.
- Supports explicit caching of system prompt embeddings to optimize performance.
"""

import gc
import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from PIL import Image

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import DiffusionResult
from shared.tasks.specs import DiffusionSpecStrict
from shared.tasks.task_type import TaskType

from ..utils.logging import configure_hf_library_logging
from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    RunControl,
    TaskReference,
)
from .mixins.data import DataMixin
from .utils.checkpoints import maybe_upload_artifacts, maybe_upload_traces

try:
    import torch
    from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
    from transformers import BitsAndBytesConfig

    _HAS_DIFFUSERS = True
except Exception:
    if TYPE_CHECKING:
        import torch
        from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
        from transformers import BitsAndBytesConfig
    else:
        torch = None
        AutoPipelineForText2Image = None
        BitsAndBytesConfig = None
        _HAS_DIFFUSERS = False

logger = logging.getLogger(__name__)


class _Member(NamedTuple):
    """A task rendered in one batch: the parent or one of its merged children."""

    task_id: str
    ref: TaskReference
    spec: DiffusionSpecStrict
    out_dir: Path
    prompts: list[str]


def _pack_calls(members: list[_Member], num_images: int, cap: int) -> list[slice]:
    """Image ranges of the pipeline calls: whole members, in order, packed into
    calls of at most ``cap`` images. A member larger than ``cap`` gets a call
    of its own."""
    calls: list[slice] = []
    begin = end = 0
    for member in members:
        count = len(member.prompts) * num_images
        if end > begin and end + count - begin > cap:
            calls.append(slice(begin, end))
            begin = end
        end += count
    calls.append(slice(begin, end))
    return calls


class DiffusersExecutor(DataMixin, Executor):
    """Executor that runs text-to-image generation via Hugging Face Diffusers."""

    name = "diffusers"
    supported_task_types = frozenset({TaskType.DIFFUSION})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pipe: Any | None = None
        self._device: str | None = None
        self._model_name: str | None = None

        # Caching for system prompt embeddings
        self._cached_system_prompt: str | None = None
        self._cached_system_embeds: Any | None = None
        self._cached_system_neg_embeds: Any | None = None
        # We also cache the pooler_output if available (e.g. for SDXL)
        self._cached_system_pooled: Any | None = None

    def prepare(self) -> None:
        if not _HAS_DIFFUSERS:
            raise ExecutionError(
                "diffusers/torch is not installed "
                "(`pip install diffusers transformers torch accelerate`)."
            )
        configure_hf_library_logging()

    def _ensure_pipeline(self, spec: DiffusionSpecStrict) -> None:
        model_cfg = spec.model
        model_src = model_cfg and model_cfg.source
        ident = (model_src and model_src.identifier) or os.getenv("DIFFUSERS_MODEL")
        if not ident:
            raise ExecutionError(
                "spec.model.source.identifier (or DIFFUSERS_MODEL) is required."
            )

        if self._pipe is not None and self._model_name == ident:
            return

        # If switching models, clear old one
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        diffusion_config = (model_cfg and model_cfg.diffusers) or {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        match dtype_str := diffusion_config.get("dtype", "auto"):
            case "auto":
                torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            case "fp16":
                torch_dtype = torch.float16
            case "bf16":
                torch_dtype = torch.bfloat16
            case "fp32":
                torch_dtype = torch.float32
            case _:
                raise ExecutionError(
                    f"Unsupported dtype: '{dtype_str}'. "
                    "Only 'auto', 'fp16', 'bf16', 'fp32' are supported."
                )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "use_safetensors": bool(diffusion_config.get("use_safetensors", True)),
        }
        revision = spec.model_revision or diffusion_config.get("revision")
        if revision:
            load_kwargs["revision"] = revision
        if spec.model_trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        quant_method = diffusion_config.get("quantization")
        if quant_method == "bitsandbytes_8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_method == "bitsandbytes_4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        logger.info(f"Loading Diffusers pipeline: {ident} on {device}")
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(ident, **load_kwargs)
            pipe.to(device)
            self._pipe = pipe
        except Exception as e:
            raise ExecutionError(
                f"Failed to load Diffusers pipeline: {e}", retryable=True
            )
        logger.info(f"Loaded Diffusers pipeline: {ident} on {device}")

        self._device = device
        self._model_name = ident

        # Invalidate cache when model changes
        self._cached_system_prompt = None
        self._cached_system_embeds = None
        self._cached_system_neg_embeds = None
        self._cached_system_pooled = None

    def _encode_prompt_compat(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None,
        num_images_per_prompt: int,
    ):
        """
        Wrapper around pipeline's prompt encoder to handle different architectures.
        """
        assert self._pipe is not None, "Pipeline must be initialized"
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "device": self._device,
            "num_images_per_prompt": num_images_per_prompt,
        }

        # Check for SD3 architecture to pass required positional/keyword arguments
        if "StableDiffusion3" in self._pipe.__class__.__name__:
            kwargs["prompt_2"] = prompt
            kwargs["prompt_3"] = prompt
            if negative_prompt is not None:
                kwargs["negative_prompt_2"] = negative_prompt
                kwargs["negative_prompt_3"] = negative_prompt

        # SD1.x/2.x/XL encoders require do_classifier_free_guidance (no default in
        # diffusers >=0.38); request it so negative embeddings are returned. SD3
        # has no such parameter, so only pass it when the signature accepts it.
        encode_params = inspect.signature(self._pipe.encode_prompt).parameters
        if "do_classifier_free_guidance" in encode_params:
            kwargs["do_classifier_free_guidance"] = True

        return self._pipe.encode_prompt(**kwargs)

    def _encode_and_combine_prompts(
        self,
        user_prompts: list[str],
        system_prompt: str | None,
        negative_prompt: str | None,
        num_images_per_prompt: int,
    ):
        """
        Encodes user prompts and optionally concatenates with system prompt embeddings.
        Ensures positive and negative embeddings have matching sequence lengths.
        """
        assert self._pipe is not None, "Pipeline must be initialized"

        # 1. Encode/Get System Prompt Embeddings (if present)
        sys_pos = None
        sys_neg = None

        if system_prompt:
            if (
                self._cached_system_prompt != system_prompt
                or self._cached_system_embeds is None
            ):
                logger.info("Encoding and caching system prompt...")
                # Encode system prompt (batch size 1)
                # We use empty string for system negative prompt to match shape
                encoded_sys = self._encode_prompt_compat(
                    prompt=system_prompt,
                    negative_prompt="",
                    num_images_per_prompt=1,
                )
                self._cached_system_embeds = encoded_sys[0]
                self._cached_system_neg_embeds = encoded_sys[1]
                self._cached_system_prompt = system_prompt

            sys_pos = self._cached_system_embeds
            sys_neg = self._cached_system_neg_embeds

        # 2. Encode User Prompts (Batched)
        neg_prompts = [negative_prompt or ""] * len(user_prompts)
        encoded_user = self._encode_prompt_compat(
            prompt=user_prompts,
            negative_prompt=neg_prompts,
            num_images_per_prompt=num_images_per_prompt,
        )
        user_pos = encoded_user[0]  # [batch_size * num_images, user_seq_len, dim]
        user_neg = encoded_user[1]
        user_pos_pooled = encoded_user[2] if len(encoded_user) > 2 else None
        user_neg_pooled = encoded_user[3] if len(encoded_user) > 3 else None

        if sys_pos is None:
            # No system prompt, return user embeddings directly
            return user_pos, user_neg, user_pos_pooled, user_neg_pooled

        # 3. Concatenate System + User Embeddings
        # Repeat system embeddings to match user batch size
        batch_total = user_pos.shape[0]
        sys_pos_rep = sys_pos.repeat(batch_total, 1, 1)
        sys_neg_rep = sys_neg.repeat(batch_total, 1, 1) if sys_neg is not None else None

        # Concatenate along sequence dimension (dim 1)
        combined_pos = torch.cat([sys_pos_rep, user_pos], dim=1)
        if sys_neg_rep is not None and user_neg is not None:
            combined_neg = torch.cat([sys_neg_rep, user_neg], dim=1)
        else:
            combined_neg = user_neg

        return combined_pos, combined_neg, user_pos_pooled, user_neg_pooled

    def run(
        self, task: ExecutorTask, out_dir: Path, control: RunControl
    ) -> DiffusionResult:
        configure_hf_library_logging()
        spec = self.require_spec(task, DiffusionSpecStrict)
        task_id = task.task_id.strip()
        with self._task_span(
            task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            members = self._members(task, spec, task_id, out_dir)
            result = self._run_inner(spec, task_id, members)
        maybe_upload_artifacts(task, out_dir, logger=logger)
        maybe_upload_traces(task, out_dir, logger=logger)
        for member in members[1:]:
            maybe_upload_artifacts(member.ref, member.out_dir, logger=logger)
        return result

    def _members(
        self,
        task: ExecutorTask,
        spec: DiffusionSpecStrict,
        task_id: str,
        out_dir: Path,
    ) -> list[_Member]:
        """The task and its merged children. Children share the parent's model
        and inference settings (the merge key); only their prompts differ, so
        they render on the same pipeline."""
        system_prompt = (spec.inference or {}).get("system_prompt")
        members = [
            _Member(task_id, task, spec, out_dir, self._member_prompts(spec, task_id))
        ]
        for child in task.merged_children or []:
            if not isinstance(child.spec, DiffusionSpecStrict):
                raise ExecutionError(
                    "Merged child spec must be diffusion for merged execution."
                )
            if (child.spec.inference or {}).get("system_prompt") != system_prompt:
                raise ExecutionError(
                    f"Merged child {child.task_id} has a different system_prompt."
                )
            members.append(
                _Member(
                    child.task_id,
                    child,
                    child.spec,
                    out_dir.parent / child.task_id,
                    self._member_prompts(child.spec, child.task_id),
                )
            )
        return members

    def _run_inner(
        self,
        spec: DiffusionSpecStrict,
        task_id: str,
        members: list[_Member],
    ) -> DiffusionResult:
        self._ensure_pipeline(spec)
        assert self._pipe is not None

        inference_config = spec.inference or {}
        system_prompt = inference_config.get("system_prompt")
        prompts = [p for member in members for p in member.prompts]
        num_inference_steps = int(inference_config.get("num_inference_steps", 50))
        guidance_scale = float(inference_config.get("guidance_scale", 7.5))
        height = int(inference_config.get("height", 768))
        width = int(inference_config.get("width", 768))
        num_images = int(inference_config.get("num_images_per_prompt", 1))
        if num_images <= 0:
            raise ExecutionError(
                "spec.inference.num_images_per_prompt must be greater than 0 "
                f"(got {num_images})."
            )
        seed = inference_config.get("seed")
        negative_prompt = inference_config.get("negative_prompt")

        # One generator per member, repeated for each of its images, so a
        # member's latents do not depend on which tasks it was merged with.
        generators: list[Any] | None = None
        if seed is not None:
            generators = []
            for member in members:
                generator = torch.Generator(device=self._device).manual_seed(int(seed))
                generators += [generator] * (len(member.prompts) * num_images)

        logger.info(f"Starting generation for {len(prompts)} prompts...")

        # Encode prompts (with optional system prompt concatenation)
        logger.info("Encoding and combining prompts...")
        prompt_embeds, neg_embeds, pooled_embeds, neg_pooled_embeds = (
            self._encode_and_combine_prompts(
                prompts, system_prompt, negative_prompt, num_images
            )
        )
        logger.info("Prompts encoded.")

        # Diffusers denoises the whole batch at once, so cap the images per call.
        expected_count = len(prompts) * num_images
        images: list[Image.Image] = []
        logger.info(f"Running pipeline with {num_inference_steps} steps...")
        for batch in _pack_calls(
            members, num_images, self._config.diffusion_max_images_per_call
        ):
            kwargs = {
                "prompt_embeds": prompt_embeds[batch],
                "negative_prompt_embeds": (
                    neg_embeds[batch] if neg_embeds is not None else None
                ),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "generator": generators[batch] if generators is not None else None,
            }
            if pooled_embeds is not None:
                kwargs["pooled_prompt_embeds"] = pooled_embeds[batch]
            if neg_pooled_embeds is not None:
                kwargs["negative_pooled_prompt_embeds"] = neg_pooled_embeds[batch]
            images += self._pipe(**kwargs).images
        logger.info("Pipeline generation completed.")
        actual_count = len(images)
        if actual_count != expected_count:
            raise ExecutionError(
                "Diffusers output image count mismatch: "
                f"expected={expected_count} got={actual_count}. "
                "No fallback/coercion is allowed."
            )

        # Images come back grouped per prompt, members in order.
        results: list[DiffusionResult] = []
        offset = 0
        for member in members:
            count = len(member.prompts) * num_images
            results.append(
                self._save_images(member.out_dir, images[offset : offset + count])
            )
            offset += count
        result = results[0]
        for member, child_result in zip(members[1:], results[1:]):
            result.children[member.task_id] = child_result
        self._dump_to_governance(
            task_id=task_id,
            result=result,
            dependencies_by_task={
                member.task_id: self._extract_source_data_ids(member.spec)
                for member in members
            },
            owners_by_task={member.task_id: member.ref.owner_id for member in members},
        )
        return result

    def _member_prompts(self, spec: DiffusionSpecStrict, task_id: str) -> list[str]:
        """The text prompts of one task (parent or merged child)."""
        data_entry = self._collect_prompts_for_spec(spec, task_id=task_id)
        if len(data_entry.prompts) == 0:
            raise ExecutionError(f"No prompts provided for task {task_id}.")
        texts: list[str] = []
        for prompt in data_entry.prompts:
            if isinstance(prompt, str):
                text: str | None = prompt
            else:
                text = prompt[-1].get("content") if prompt else None
            if not isinstance(text, str):
                raise ExecutionError(f"Task {task_id} has a prompt without text.")
            texts.append(text)
        return texts

    def _save_images(self, out_dir: Path, images: list[Image.Image]) -> DiffusionResult:
        """Write one task's images under ``out_dir/artifacts/images``."""
        image_dir = out_dir / "artifacts" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        generated_images: list[ArtifactRef] = []
        for idx, img in enumerate(images):
            img.save(image_dir / f"image_{idx}.png", format="PNG")
            generated_images.append(ArtifactRef(path=f"images/image_{idx}.png"))
        return DiffusionResult(model=self._model_name, images=generated_images)

    def cleanup_after_run(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None

        self._cached_system_prompt = None
        self._cached_system_embeds = None
        self._cached_system_neg_embeds = None
        self._cached_system_pooled = None

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
