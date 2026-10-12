"""Omni executor for image generation via vllm_omni."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from shared.schemas.artifact import ArtifactRef
from shared.schemas.governance import SpanType
from shared.schemas.result import OmniImageItem, OmniResult, OmniText2ImageResult
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.omni import OmniText2ImageSpecStrict
from shared.tasks.task_type import TaskType

from .base_executor import ExecutionError
from .omni_executor_base import (
    _HAS_OMNI,
    Omni,
    OmniExecutorBase,
    OmniMember,
    OmniRequestOutput,
    prompt_index_from_request_id,
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
        task_id: str,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniResult:
        return self._run_members([OmniMember(task_id, spec, out_dir)])[0]

    def _run_members(self, members: list[OmniMember]) -> list[OmniResult]:
        """Render the prompts of every member in one ``Omni.generate`` call;
        each member's images go to its own ``artifacts/``."""
        task_id = members[0].task_id
        spec_dict = members[0].spec.model_dump(by_alias=True)
        prompts_by_member = [
            self._collect_text_inputs(member.spec, member.task_id) for member in members
        ]
        prompts = [p for member_prompts in prompts_by_member for p in member_prompts]

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task_id, "prompt_count": len(prompts)},
        ):
            self._ensure_omni(spec_dict)
        cfg = self.omni_cfg(spec_dict, "omni:image generation", "omni_text2image")

        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={
                "task_id": task_id,
                "prompt_count": len(prompts),
                "merged_tasks": len(members),
            },
        ):
            images = self._generate_images(prompts)

        results: list[OmniResult] = []
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task_id, "item_count": len(prompts)},
        ):
            offset = 0
            for member, member_prompts in zip(members, prompts_by_member):
                member_images = images[offset : offset + len(member_prompts)]
                offset += len(member_prompts)
                results.append(
                    self._save_images(
                        cfg, member.out_dir, member_prompts, member_images
                    )
                )
        return results

    def _save_images(
        self,
        cfg: dict[str, Any],
        out_dir: Path,
        prompts: list[str],
        images: list[Image.Image],
    ) -> OmniText2ImageResult:
        """Write one task's images under ``out_dir/artifacts`` and describe them."""
        fmt = str(cfg.get("output_format") or "").strip().lower() or "png"
        artifacts_dir = out_dir / "artifacts"
        items: list[OmniImageItem] = []
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
                    image=ArtifactRef(path=self.relative_to(save_path, artifacts_dir)),
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
        by_index: dict[int, Image.Image] = {}
        for output in outputs:
            if not output.images:
                continue
            idx = prompt_index_from_request_id(str(output.request_id))
            if idx is None or idx >= len(prompts) or idx in by_index:
                raise ExecutionError(
                    f"Omni image batch returned an unexpected request id "
                    f"{output.request_id!r}."
                )
            by_index[idx] = output.images[0]
        if len(by_index) != len(prompts):
            raise ExecutionError(
                f"Omni image batch returned {len(by_index)} images "
                f"for {len(prompts)} prompts."
            )
        return [by_index[i] for i in range(len(prompts))]

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
