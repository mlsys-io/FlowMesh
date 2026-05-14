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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from shared.utils.parsing import to_bool, to_int

from ..config import WorkerConfig
from ..lifecycle import Lifecycle
from .base_executor import ExecutionError, Executor
from .mixins.inference import InferenceMixin

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

logger = logging.getLogger(__name__)


class OmniExecutorBase(InferenceMixin, Executor):
    """Shared base for Omni-family executors.

    Manages the ``_omni`` model handle and provides config / audio helpers.
    Concrete subclasses implement ``prepare()`` and ``run()`` as usual.
    """

    def __init__(
        self, config: WorkerConfig, lifecycle: Lifecycle | None = None
    ) -> None:
        super().__init__(config, lifecycle)
        self._omni: Any | None = None
        self._model_name: str | None = None
        self._omni_spec: tuple[Any, ...] | None = None
        self._stage_configs_tmp: Path | None = None

    # ── model lifecycle ──────────────────────────────────────────────────

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


def extract_multimodal_output(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        mm = value.get("multimodal_output")
        return mm if isinstance(mm, dict) else None
    outputs = getattr(value, "outputs", None)
    if isinstance(outputs, list) and outputs:
        mm = getattr(outputs[0], "multimodal_output", None)
        if isinstance(mm, dict):
            return mm
    mm = getattr(value, "multimodal_output", None)
    return mm if isinstance(mm, dict) else None


def extract_audio_from_mm(mm: dict[str, Any] | None) -> Any:
    if not isinstance(mm, dict):
        return None
    audio = mm.get("audio")
    if audio is None:
        return None
    sr = (
        mm.get("audio_sample_rate")
        or mm.get("sample_rate")
        or mm.get("sampling_rate")
        or mm.get("sr")
    )
    return {"audio": audio, "sample_rate": sr} if sr is not None else audio
