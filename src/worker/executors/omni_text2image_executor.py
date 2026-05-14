"""Omni executor for image generation via vllm_omni."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from vllm_omni.entrypoints.omni import Omni

    _HAS_OMNI = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.entrypoints.omni import Omni
    else:
        Omni = None
    _HAS_OMNI = False

from shared.schemas.governance import SpanType
from shared.tasks.specs.omni import OmniText2ImageSpecStrict
from shared.utils.parsing import as_list

from .base_executor import ExecutionError, ExecutorTask
from .omni_executor_base import OmniExecutorBase
from .utils.checkpoints import artifact_ref, maybe_upload_artifacts, maybe_upload_traces

logger = logging.getLogger(__name__)


class OmniText2ImageExecutor(OmniExecutorBase):
    """Generate images using vllm_omni.Omni."""

    name = "omni_text2image"

    def prepare(self) -> None:
        if not _HAS_OMNI:
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2image executor."
            )

    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:
        spec = self.require_spec(task, OmniText2ImageSpecStrict)
        spec_dict = spec.model_dump(by_alias=True)
        out_dir = Path(out_dir).resolve()

        with self._task_span(
            task.task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            result = self._run_inner(task, spec, spec_dict, out_dir)
        maybe_upload_artifacts(task, out_dir, logger=logger)
        maybe_upload_traces(task, out_dir, logger=logger)
        return result

    def _run_inner(
        self,
        task: ExecutorTask,
        spec: OmniText2ImageSpecStrict,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> dict[str, Any]:
        prompts: list[str] = []
        for p in self._collect_prompts_for_spec(spec, task.task_id).prompts:
            if not isinstance(p, str):
                raise ExecutionError("omni_text2image prompts must be strings.")
            prompts.append(p)
        if not prompts:
            raise ExecutionError(
                "omni_text2image requires prompts "
                "in spec.data.prompt or spec.data.items."
            )

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            self._ensure_omni(spec_dict)
        cfg = self.omni_cfg(spec_dict, "omni:image generation", "omni_text2image")
        fmt = str(cfg.get("output_format") or "").strip().lower() or "png"

        artifacts_dir = out_dir / "artifacts"
        items: list[dict[str, Any]] = []
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
                _save_image(image, save_path)
                items.append(
                    {
                        "index": idx,
                        "prompt": prompt,
                        "image": artifact_ref(
                            self.relative_to(save_path, artifacts_dir)
                        ),
                    }
                )

        first = items[0]["image"] if items else {}
        result: dict[str, Any] = {
            "ok": True,
            "executor": self.name,
            "mode": "image",
            "model": self._model_name,
            "image": first,
            "items": items,
        }
        return result

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

    def _generate_images(self, prompts: list[str]) -> list[Any]:
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

    def _generate_single(self, prompt: str) -> Any:
        if self._omni is None:
            raise ExecutionError("Omni model not initialized.")
        outputs = self._omni.generate(prompt, use_tqdm=False)
        images = _extract_images(outputs)
        if not images:
            raise ExecutionError("Omni image generation returned no image.")
        return images[0]


# ── image extraction helpers ─────────────────────────────────────────────────


def _extract_images(outputs: Any) -> list[Any]:
    collected: list[Any] = []
    for stage_output in as_list(outputs):
        request_outputs = as_list(getattr(stage_output, "request_output", None))
        if request_outputs:
            for ro in request_outputs:
                if ro is None:
                    continue
                imgs = getattr(ro, "images", None)
                if isinstance(imgs, list) and imgs:
                    collected.extend(imgs[:1])
                    continue
                if isinstance(ro, dict):
                    imgs = ro.get("images")
                    if isinstance(imgs, list) and imgs:
                        collected.extend(imgs[:1])
            continue
        imgs = getattr(stage_output, "images", None)
        if isinstance(imgs, list) and imgs:
            collected.extend(imgs[:1])
        if isinstance(stage_output, dict):
            imgs = stage_output.get("images")
            if isinstance(imgs, list) and imgs:
                collected.extend(imgs[:1])
    return collected


def _save_image(image: Any, path: Path) -> None:
    if hasattr(image, "save"):
        image.save(path.as_posix())
        return
    raise ExecutionError("Omni image object does not support save().")
