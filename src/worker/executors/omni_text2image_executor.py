"""Omni executor for image generation via vllm_omni."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import OmniImageItem, OmniText2ImageResult
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2ImageSpecStrict
from shared.tasks.task_type import TaskType

from .base_executor import ExecutionError, ExecutorTask
from .omni_executor_base import (
    _HAS_OMNI,
    Omni,
    OmniExecutorBase,
    OmniRequestOutput,
)

logger = logging.getLogger(__name__)
EXECUTOR_NAME = "omni_text2image"


class OmniText2ImageExecutor(OmniExecutorBase):
    """Generate images using vllm_omni.Omni."""

    name = EXECUTOR_NAME
    supported_task_types = frozenset({TaskType.OMNI_TEXT2IMAGE})
    _TASK_SPEC_TYPE = OmniText2ImageSpecStrict

    def prepare(self) -> None:
        if not _HAS_OMNI:
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2image executor."
            )

    def _run_inner(
        self,
        task: ExecutorTask,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniText2ImageResult:
        assert isinstance(spec, OmniText2ImageSpecStrict)
        prompts = self._collect_text_inputs(spec, task.task_id)

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            self._ensure_omni(spec_dict)
        cfg = self.omni_cfg(spec_dict, "omni:image generation", "omni_text2image")
        fmt = str(cfg.get("output_format") or "").strip().lower() or "png"

        artifacts_dir = out_dir / "artifacts"
        items: list[OmniImageItem] = []
        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            images = self._generate_images(prompts)
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "item_count": len(prompts)},
        ):
            for idx, (prompt, image) in enumerate(zip(prompts, images)):
                save_path = self.resolve_save_path(
                    cfg,
                    out_dir,
                    index=idx,
                    ext=fmt,
                    multi=len(prompts) > 1,
                    default_prefix="generated_image",
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(save_path.as_posix())
                items.append(
                    OmniImageItem(
                        index=idx,
                        prompt=prompt,
                        image=ArtifactRef(
                            path=self.relative_to(save_path, artifacts_dir)
                        ),
                    )
                )

        return OmniText2ImageResult(
            model=self.model_name,
            image=items[0].image if items else None,
            items=items,
        )

    # ── model ────────────────────────────────────────────────────────────

    def _ensure_omni(self, spec_dict: dict[str, Any]) -> None:
        cfg = self.omni_cfg(spec_dict, "omni:image generation", "omni_text2image")
        model_name = self.resolve_model_identifier(
            spec_dict,
            cfg,
            env_keys=("OMNI_MODEL",),
            default="Qwen/Qwen-Image-2512",
        )
        new_spec = self._build_omni_spec(model_name, cfg)
        if self._omni is not None:
            if self._omni_spec == new_spec:
                logger.info("Reusing existing Omni instance for model %s", model_name)
                return
            logger.info(
                "Releasing previous Omni instance for model %s (spec changed)",
                self._model_name,
            )
            self._close_omni()
        self._omni = Omni(**self.build_omni_init_kwargs(model_name, cfg))
        self._model_name = model_name
        self._omni_spec = new_spec

    # ── generation ───────────────────────────────────────────────────────

    def _generate_images(self, prompts: list[str]) -> list[Image.Image]:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        if len(prompts) == 1:
            return [self._generate_single(prompts[0])]
        outputs = self._omni.generate(prompts, use_tqdm=False)
        images = _extract_images(outputs)
        if len(images) != len(prompts):
            raise ExecutionError(
                f"Omni image batch returned {len(images)} images "
                f"for {len(prompts)} prompts."
            )
        return images

    def _generate_single(self, prompt: str) -> Image.Image:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        outputs = self._omni.generate(prompt, use_tqdm=False)
        images = _extract_images(outputs)
        if not images:
            raise ExecutionError("Omni image generation returned no image.")
        return images[0]


# ── image extraction helpers ─────────────────────────────────────────────────


def _extract_images(outputs: Iterable[OmniRequestOutput]) -> list[Image.Image]:
    return [imgs[0] for stage_output in outputs if (imgs := stage_output.images)]
