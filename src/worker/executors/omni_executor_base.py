"""Shared base class and utilities for Omni-family executors.

All four Omni executors (text2image, text2speech, text2audio, text2general) inherit from
``OmniExecutorBase`` which provides model lifecycle management, config
resolution, and common helpers.  This keeps each concrete executor
focused on its generation logic.
"""

import gc
import json
import logging
import os
import struct
import tempfile
import wave
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import yaml

from shared.schemas.result import OmniResult
from shared.tasks.specs import TaskSpecStrictBase
from shared.utils.parsing import to_bool, to_int
from worker.config import WorkerConfig

from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.inference import InferenceMixin
from .utils.checkpoints import maybe_upload_artifacts, maybe_upload_traces

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
    from vllm import RequestOutput as RequestOutput
    from vllm_omni.entrypoints.omni import Omni as Omni
    from vllm_omni.outputs import OmniRequestOutput as OmniRequestOutput

    _HAS_OMNI = True
except Exception:
    if TYPE_CHECKING:
        from vllm import RequestOutput as RequestOutput
        from vllm_omni.entrypoints.omni import Omni as Omni
        from vllm_omni.outputs import OmniRequestOutput as OmniRequestOutput
    else:
        Omni = object
        OmniRequestOutput = object
        RequestOutput = object

    _HAS_OMNI = False

logger = logging.getLogger(__name__)


class OmniExecutorBase(InferenceMixin, Executor):
    """Shared base for Omni-family executors.

    Manages the ``_omni`` model handle, runs the generic ``run()`` shape
    (task span + artifact / trace upload), and delegates the task body to
    each subclass's ``_run_inner``. Subclasses set ``_TASK_SPEC_TYPE`` so
    the base can call ``require_spec`` without knowing the concrete type.
    """

    _TASK_SPEC_TYPE: ClassVar[type[TaskSpecStrictBase]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._omni: Omni | None = None
        self._model_name: str | None = None
        self._omni_spec: tuple[Any, ...] | None = None
        self._stage_configs_tmp: Path | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return _HAS_OMNI

    def run(self, task: ExecutorTask, out_dir: Path) -> OmniResult:
        spec = self.require_spec(task, self._TASK_SPEC_TYPE)
        spec_dict = spec.model_dump(by_alias=True)
        out_dir = Path(out_dir).resolve()
        with self._task_span(
            task.task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            result = self._run_inner(task, spec, spec_dict, out_dir)
        maybe_upload_artifacts(task, out_dir, logger=logger)
        maybe_upload_traces(task, out_dir, logger=logger)
        return result

    @abstractmethod
    def _run_inner(
        self,
        task: ExecutorTask,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniResult:
        """Run the executor-specific body. ``spec`` is the concrete strict
        spec; subclasses ``assert isinstance(spec, ...)`` to narrow."""
        raise NotImplementedError

    # ── model lifecycle ──────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Identifier of the loaded model; raises when no model is loaded."""
        if self._model_name is None:
            raise ExecutionError("Omni model not initialized.")
        return self._model_name

    def _close_omni(self) -> None:
        """Release the Omni model handle and free GPU memory."""
        omni = self._omni
        self._omni = None
        self._model_name = None
        self._omni_spec = None
        if omni is not None:
            try:
                omni.close()
            except Exception:
                logger.debug(
                    "Failed to close Omni model handle gracefully", exc_info=True
                )
                pass
            release_gpu_memory()
        self._cleanup_stage_configs_tmp()

    @staticmethod
    def _build_omni_spec(model_name: str, cfg: dict[str, Any]) -> tuple[Any, ...]:
        """Build a hashable spec capturing all init-relevant settings.

        Two tasks with matching specs can safely share an ``Omni`` instance.
        """
        stage_configs = cfg.get("stage_configs")
        stage_configs_key = (
            json.dumps(stage_configs, sort_keys=True)
            if isinstance(stage_configs, dict)
            else None
        )
        return (
            model_name,
            stage_configs_key,
            cfg.get("log_stats"),
            cfg.get("stage_init_timeout"),
            cfg.get("init_timeout"),
            cfg.get("async_chunk"),
        )

    def _materialize_stage_configs(self, cfg: dict[str, Any]) -> str | None:
        """Write inline ``stage_configs`` dict to a temp YAML file."""
        self._cleanup_stage_configs_tmp()
        stage_configs = cfg.get("stage_configs")
        if not isinstance(stage_configs, dict):
            return None
        fd, path = tempfile.mkstemp(prefix="omni_stage_configs_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(stage_configs, f)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        self._stage_configs_tmp = Path(path)
        return path

    def _cleanup_stage_configs_tmp(self) -> None:
        path = self._stage_configs_tmp
        self._stage_configs_tmp = None
        if path is not None:
            path.unlink(missing_ok=True)

    def teardown(self) -> None:
        self._close_omni()

    def cleanup_after_run(self) -> None:
        self._close_omni()

    # ── config helpers ───────────────────────────────────────────────────

    @staticmethod
    def omni_cfg(spec_dict: dict[str, Any], *sections: str) -> dict[str, Any]:
        """Merge ``spec.omni`` with any additional config sections."""
        merged: dict[str, Any] = {}
        base_cfg = spec_dict.get("omni")
        if isinstance(base_cfg, dict):
            merged.update(base_cfg)
        for key in sections:
            cfg = spec_dict.get(key)
            if isinstance(cfg, dict):
                merged.update(cfg)
        return merged

    @staticmethod
    def resolve_model_identifier(
        spec_dict: dict[str, Any],
        cfg: dict[str, Any],
        env_keys: tuple[str, ...],
        default: str,
    ) -> str:
        model_cfg = spec_dict.get("model") or {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        source_cfg = model_cfg.get("source") or {}
        if not isinstance(source_cfg, dict):
            source_cfg = {}
        value = source_cfg.get("identifier") or cfg.get("model")
        if value in (None, ""):
            for env_key in env_keys:
                env_value = os.getenv(env_key)
                if env_value:
                    value = env_value
                    break
        if value in (None, ""):
            value = default
        return str(value).strip()

    def _collect_text_inputs(self, spec: TaskSpecStrictBase, task_id: str) -> list[str]:
        prompts = self._collect_prompts_for_spec(spec, task_id).prompts
        if not prompts:
            raise ExecutionError(f"{self.name} requires text input in spec.data.items.")
        if not all(isinstance(p, str) for p in prompts):
            raise ExecutionError(f"{self.name} prompts must be strings.")
        return cast(list[str], prompts)

    @staticmethod
    def resolve_save_path(
        cfg: dict[str, Any],
        out_dir: Path,
        index: int,
        ext: str,
        multi: bool,
        default_prefix: str,
    ) -> Path:
        """Resolve the save path for an omni-produced artifact."""
        artifacts_dir = (out_dir / "artifacts").resolve()
        raw_path = cfg.get("save_path")
        if raw_path:
            text = str(raw_path)
            if "{index}" in text:
                text = text.format(index=index + 1)
            candidate = Path(text).expanduser()
            if candidate.is_absolute():
                raise ExecutionError(
                    f"omni save_path must be relative to artifacts/ (got {raw_path})"
                )
            candidate = (artifacts_dir / candidate).resolve()
            if multi and "{index}" not in str(raw_path) and index > 0:
                candidate = candidate.with_name(
                    f"{candidate.stem}_{index + 1}{candidate.suffix}"
                )
            try:
                candidate.relative_to(artifacts_dir)
            except ValueError as exc:
                raise ExecutionError(
                    f"omni save_path {candidate} must stay under "
                    f"artifacts/ (out_dir={out_dir})"
                ) from exc
            return candidate
        base_name = default_prefix if not multi else f"{default_prefix}_{index + 1}"
        return (artifacts_dir / f"{base_name}.{ext}").resolve()

    def build_omni_init_kwargs(
        self, model_name: str, cfg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build kwargs for instantiating ``vllm_omni.Omni``."""
        init_kwargs: dict[str, Any] = {"model": model_name}
        stage_configs_path = self._materialize_stage_configs(cfg)
        if stage_configs_path is not None:
            init_kwargs["stage_configs_path"] = stage_configs_path
        if cfg.get("log_stats") is not None:
            init_kwargs["log_stats"] = to_bool(cfg.get("log_stats"), default=False)
        if cfg.get("async_chunk") is not None:
            init_kwargs["async_chunk"] = to_bool(cfg.get("async_chunk"), default=False)
        for key in ("stage_init_timeout", "init_timeout"):
            val = to_int(cfg.get(key))
            if val is not None:
                init_kwargs[key] = val
        return init_kwargs

    # ── common small helpers ─────────────────────────────────────────────

    @staticmethod
    def relative_to(path: Path, base: Path) -> str:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except Exception as exc:
            raise ExecutionError(
                f"Artifact path {path} must stay within {base}"
            ) from exc


# ── module-level utilities (used by multiple executors) ──────────────────────


def release_gpu_memory() -> None:
    gc.collect()
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ── audio I/O (shared by tts, bgm, narration) ───────────────────────────────


def save_audio(audio: Any, path: Path, sample_rate: int) -> None:
    """Save an audio object (various types) to *path*."""
    if hasattr(audio, "save"):
        audio.save(path.as_posix())
        return
    if isinstance(audio, (bytes, bytearray)):
        path.write_bytes(bytes(audio))
        return
    extracted = coerce_audio_samples(audio, sample_rate=sample_rate)
    if extracted is None:
        raise ExecutionError(
            f"Audio object is not supported for saving (type={type(audio).__name__})."
        )
    samples, sr = extracted
    write_wav(path, samples, sample_rate=sr)


def coerce_audio_samples(
    audio: Any, sample_rate: int
) -> tuple[list[float], int] | None:
    sr = sample_rate
    values = audio
    if isinstance(audio, tuple) and len(audio) == 2:
        values, sr_raw = audio
        sr = to_int(sr_raw, default=sample_rate)
    elif isinstance(audio, dict):
        values = audio.get("audio")
        sr_raw = (
            audio.get("sample_rate")
            or audio.get("sampling_rate")
            or audio.get("audio_sample_rate")
            or audio.get("sr")
        )
        if sr_raw is not None:
            sr = to_int(sr_raw, default=sample_rate)
    if values is None:
        return None
    flattened = flatten_audio_values(values)
    return (flattened, sr) if flattened is not None else None


def flatten_audio_values(values: Any) -> list[float] | None:
    if values is None:
        return None
    if np is not None and isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False)
        while arr.ndim > 1 and arr.size > 0:
            arr = arr[0]
        return arr.reshape(-1).tolist()
    if torch is not None and isinstance(values, torch.Tensor):
        try:
            arr = values.detach().cpu().float().numpy()
            if np is not None and isinstance(arr, np.ndarray):
                arr = arr.astype(float, copy=False)
                while arr.ndim > 1 and arr.size > 0:
                    arr = arr[0]
                return arr.reshape(-1).tolist()
        except Exception as exc:
            logger.warning("Failed to convert tensor to audio samples: %s", exc)
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, (list, tuple)):
        seq = list(values)
        if not seq:
            return []
        if all(isinstance(i, (int, float)) for i in seq):
            return [float(i) for i in seq]
        merged: list[float] = []
        for item in seq:
            nested = flatten_audio_values(item)
            if nested:
                merged.extend(nested)
        return merged if merged else None
    return None


def write_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    clipped = [max(-1.0, min(1.0, float(v))) for v in samples]
    pcm = b"".join(struct.pack("<h", int(v * 32767.0)) for v in clipped)
    with wave.open(path.as_posix(), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)


# ── multimodal output extraction (shared by tts, narration) ─────────────────


def extract_multimodal_output(
    output: OmniRequestOutput | None,
) -> Mapping[str, Any] | None:
    if output is None:
        return None
    return mm if isinstance(mm := output.multimodal_output, Mapping) else None


def extract_audio_from_mm(mm: Mapping[str, Any] | None) -> Any:
    if mm is None:
        return None
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if audio is None:
        return None
    sr = (
        mm.get("audio_sample_rate")
        or mm.get("sample_rate")
        or mm.get("sampling_rate")
        or mm.get("sr")
    )
    if isinstance(sr, (list, tuple)) and sr:
        sr = sr[0]
    return {"audio": audio, "sample_rate": sr} if sr is not None else audio
