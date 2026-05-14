"""Omni executor for background music generation via vllm_omni."""

import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
    import torch
except Exception:
    if TYPE_CHECKING:
        import numpy as np
        import torch
    else:
        np = None
        torch = None

try:
    from vllm_omni.entrypoints.omni import Omni

    _HAS_OMNI = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.entrypoints.omni import Omni
    else:
        Omni = None
    _HAS_OMNI = False

try:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

    _HAS_OMNI_DIFFUSION = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
    else:
        OmniDiffusionSamplingParams = None
        OmniTextPrompt = None
    _HAS_OMNI_DIFFUSION = False

try:
    from vllm_omni.platforms import current_omni_platform

    _HAS_OMNI_PLATFORM = True
except Exception:
    if TYPE_CHECKING:
        from vllm_omni.platforms import current_omni_platform
    else:
        current_omni_platform = None
    _HAS_OMNI_PLATFORM = False

from shared.schemas.governance import SpanType
from shared.tasks.specs.omni import OmniText2AudioSpecStrict
from shared.utils.parsing import to_float, to_int

from .base_executor import ExecutionError, ExecutorTask
from .omni_executor_base import OmniExecutorBase, extract_multimodal_output
from .utils.checkpoints import artifact_ref, maybe_upload_artifacts, maybe_upload_traces

logger = logging.getLogger(__name__)


class OmniText2AudioExecutor(OmniExecutorBase):
    """Generate background music with Omni diffusion sampling."""

    name = "omni_text2audio"

    def prepare(self) -> None:
        if torch is None:
            raise ExecutionError("omni_text2audio requires torch.")
        if np is None:
            raise ExecutionError("omni_text2audio requires numpy.")
        if not (_HAS_OMNI and _HAS_OMNI_DIFFUSION):
            raise ExecutionError(
                "vllm_omni is not installed; cannot use omni_text2audio executor."
            )

    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:
        spec = self.require_spec(task, OmniText2AudioSpecStrict)
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
        spec: OmniText2AudioSpecStrict,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> dict[str, Any]:
        prompts: list[str] = []
        for p in self._collect_prompts_for_spec(spec, task.task_id).prompts:
            if not isinstance(p, str):
                raise ExecutionError("omni_text2audio prompts must be strings.")
            prompts.append(p)
        if not prompts:
            raise ExecutionError(
                "omni_text2audio requires prompt text in spec.data.items."
            )

        cfg = _bgm_cfg(spec_dict)
        output_format = str(cfg.get("output_format", "wav")).strip().lower()
        if output_format != "wav":
            raise ExecutionError(
                "omni_text2audio currently supports output_format='wav' only."
            )

        sample_rate = to_int(cfg.get("sample_rate"), default=44100)
        num_waveforms = max(1, to_int(cfg.get("num_waveforms"), default=1))
        guidance_scale = to_float(cfg.get("guidance_scale"), default=7.0)
        num_inference_steps = max(
            1, to_int(cfg.get("num_inference_steps"), default=100)
        )
        audio_start = to_float(cfg.get("audio_start"), default=0.0)
        audio_length = max(0.1, to_float(cfg.get("audio_length"), default=10.0))
        audio_end = audio_start + audio_length
        base_seed = to_int(cfg.get("seed"), default=42)
        negative_prompt = str(cfg.get("negative_prompt") or "Low quality.").strip()

        with self._span(
            "model load",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            self._ensure_omni(spec_dict)
        omni = self._omni
        if omni is None:
            raise ExecutionError("Omni BGM model failed to initialize.")

        generator_device = _resolve_generator_device()
        per_prompt_outputs: list[tuple[int, str, Any]] = []
        with self._span(
            "generation",
            span_type=SpanType.COMPUTE,
            attributes={
                "task_id": task.task_id,
                "prompt_count": len(prompts),
                "num_waveforms": num_waveforms,
                "num_inference_steps": num_inference_steps,
            },
        ):
            for prompt_idx, prompt in enumerate(prompts):
                seed = base_seed + prompt_idx
                torch_generator = torch.Generator(device=generator_device).manual_seed(
                    seed
                )

                omni_prompt: OmniTextPrompt = {"prompt": prompt}
                if negative_prompt:
                    omni_prompt["negative_prompt"] = negative_prompt

                sampling = OmniDiffusionSamplingParams(
                    generator=torch_generator,
                    generator_device=generator_device,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_outputs_per_prompt=num_waveforms,
                    extra_args={
                        "audio_start_in_s": audio_start,
                        "audio_end_in_s": audio_end,
                    },
                )
                try:
                    outputs = omni.generate(omni_prompt, sampling)
                except Exception as exc:
                    raise ExecutionError(
                        f"omni_text2audio generation failed: {exc}"
                    ) from exc
                per_prompt_outputs.append((prompt_idx, prompt, outputs))

        artifacts_dir = out_dir / "artifacts"
        items: list[dict[str, Any]] = []
        global_index = 0
        with self._span(
            "output postprocessing",
            span_type=SpanType.COMPUTE,
            attributes={"task_id": task.task_id, "prompt_count": len(prompts)},
        ):
            for prompt_idx, prompt, outputs in per_prompt_outputs:
                extracted = _extract_audio_waveforms(outputs)
                if not extracted:
                    raise ExecutionError(
                        "omni_text2audio completed but returned no audio output."
                    )

                for local_idx, audio_entry in enumerate(extracted):
                    multi = len(prompts) * len(extracted) > 1
                    save_path = _resolve_bgm_save_path(
                        cfg,
                        out_dir,
                        index=global_index,
                        ext=output_format,
                        multi=multi,
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_waveform(
                        audio_entry["waveform"], save_path, sample_rate=sample_rate
                    )
                    items.append(
                        {
                            "index": global_index,
                            "prompt_index": prompt_idx,
                            "waveform_index": local_idx,
                            "prompt": prompt,
                            "audio": artifact_ref(
                                self.relative_to(save_path, artifacts_dir)
                            ),
                        }
                    )
                    global_index += 1

        if not items:
            raise ExecutionError("omni_text2audio produced no savable waveforms.")

        first = items[0]["audio"] if items else {}
        result: dict[str, Any] = {
            "ok": True,
            "executor": self.name,
            "mode": "bgm",
            "model": self._model_name,
            "audio": first,
            "items": items,
            "sample_rate": sample_rate,
            "num_waveforms": len(items),
            "audio_length": audio_length,
        }
        storyboard = spec_dict.get("storyboard")
        if isinstance(storyboard, dict):
            result["storyboard"] = dict(storyboard)
        return result

    # ── model ────────────────────────────────────────────────────────────

    def _ensure_omni(self, spec_dict: dict[str, Any]) -> None:
        model_name = self.resolve_model_identifier(
            spec_dict,
            _bgm_cfg(spec_dict),
            env_keys=("STABLE_AUDIO_MODEL", "OMNI_BGM_MODEL"),
            default="stabilityai/stable-audio-open-1.0",
        )
        cfg = _bgm_cfg(spec_dict)
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


# ── BGM-specific helpers ─────────────────────────────────────────────────────


def _bgm_cfg(spec_dict: dict[str, Any]) -> dict[str, Any]:
    return OmniExecutorBase.omni_cfg(
        spec_dict, "omni:bgm", "stable_audio", "omni_text2audio"
    )


def _resolve_generator_device() -> str:
    if _HAS_OMNI_PLATFORM and current_omni_platform is not None:
        device_type = (
            str(getattr(current_omni_platform, "device_type", "")).strip().lower()
        )
        if device_type in {"cuda", "cpu", "mps", "xpu", "hpu", "npu"}:
            return device_type
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_audio_waveforms(outputs: Any) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for stage_output in _iter_outputs(outputs):
        for req in _extract_request_outputs(stage_output):
            request_id = _request_id(req)
            mm = extract_multimodal_output(req)
            audio_obj = mm.get("audio") if isinstance(mm, dict) else None
            if audio_obj is None and isinstance(stage_output, dict):
                mm2 = stage_output.get("multimodal_output")
                if isinstance(mm2, dict):
                    audio_obj = mm2.get("audio")
            if audio_obj is None:
                continue
            for waveform in _split_waveforms(audio_obj):
                collected.append({"request_id": request_id, "waveform": waveform})
    return collected


def _iter_outputs(outputs: Any) -> list[Any]:
    if outputs is None:
        return []
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    if hasattr(outputs, "__iter__") and not isinstance(
        outputs, (dict, str, bytes, bytearray)
    ):
        try:
            return list(outputs)
        except Exception:
            return [outputs]
    return [outputs]


def _extract_request_outputs(stage_output: Any) -> list[Any]:
    if stage_output is None:
        return []
    if isinstance(stage_output, dict):
        ro = stage_output.get("request_output")
        if isinstance(ro, list):
            return ro
        return [ro] if ro is not None else [stage_output]
    ro = getattr(stage_output, "request_output", None)
    if isinstance(ro, list):
        return ro
    return [ro] if ro is not None else [stage_output]


def _request_id(value: Any) -> str:
    rid = getattr(value, "request_id", None)
    if rid in (None, "") and isinstance(value, dict):
        rid = value.get("request_id")
    return str(rid) if rid not in (None, "") else "req"


def _split_waveforms(audio_obj: Any) -> list[Any]:
    arr = _to_numpy(audio_obj)
    if arr is None:
        return []
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        return [arr]
    if arr.ndim == 3:
        waveforms: list[Any] = []
        for idx in range(arr.shape[0]):
            sample = arr[idx]
            if (
                sample.ndim == 2
                and sample.shape[0] <= 8
                and sample.shape[1] > sample.shape[0]
            ):
                sample = sample.T
            waveforms.append(sample.astype(np.float32, copy=False))
        return waveforms
    return [arr.reshape(-1)]


def _to_numpy(value: Any) -> Any:
    if np is None or value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if hasattr(value, "__array__"):
        try:
            return np.asarray(value)
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            return None
    return None


def _save_waveform(waveform: Any, path: Path, sample_rate: int) -> None:
    data = waveform
    if data.ndim == 2 and data.shape[0] <= 8 and data.shape[1] > data.shape[0]:
        data = data.T
    if data.ndim == 1:
        channels = 1
        samples = np.clip(data, -1.0, 1.0)
    elif data.ndim == 2:
        channels = int(data.shape[1])
        samples = np.clip(data, -1.0, 1.0)
    else:
        channels = 1
        samples = np.clip(data.reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(path.as_posix(), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _resolve_bgm_save_path(
    cfg: dict[str, Any],
    out_dir: Path,
    index: int,
    ext: str,
    multi: bool,
) -> Path:
    artifacts_dir = (out_dir / "artifacts").resolve()
    raw_path = cfg.get("save_path")
    if raw_path:
        text = str(raw_path)
        if "{index}" in text:
            text = text.format(index=index + 1)
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            raise ExecutionError(
                f"omni_text2audio save_path must be relative to artifacts/ "
                f"(got {raw_path})"
            )
        candidate = (artifacts_dir / candidate).resolve()
        if candidate.suffix == "":
            candidate = candidate.with_suffix(f".{ext}")
        if multi and "{" not in str(raw_path) and index > 0:
            candidate = candidate.with_name(
                f"{candidate.stem}_{index + 1}{candidate.suffix}"
            )
        try:
            candidate.relative_to(artifacts_dir)
        except ValueError as exc:
            raise ExecutionError(
                f"omni_text2audio save_path {candidate} must stay under "
                f"artifacts/ (out_dir={out_dir})"
            ) from exc
        return candidate
    base_name = "bgm" if not multi else f"bgm_{index + 1}"
    return (artifacts_dir / f"{base_name}.{ext}").resolve()
