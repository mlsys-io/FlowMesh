"""Shared base class and utilities for Omni-family executors.

All Omni executors inherit from ``OmniExecutorBase`` which provides model lifecycle
management, config resolution, and common helpers.  This keeps each concrete
executor focused on its generation logic.
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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    model_validator,
)

from shared.schemas.result import OmniResult
from shared.tasks.specs import TaskSpecStrictBase
from shared.utils.parsing import to_bool, to_int
from worker.config import WorkerConfig

from .base_executor import ExecutionError, Executor, ExecutorTask, RunControl
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


class OmniWeightOptions(BaseModel):
    """``Omni`` options for how model weights are held: where they live
    (``enable_cpu_offload`` / ``enable_layerwise_offload``) and how they are
    stored (``quantization``, a method name or a quantization-config mapping)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    enable_cpu_offload: bool | None = None
    enable_layerwise_offload: bool | None = None
    quantization: str | dict[str, Any] | None = None


class OmniExecutionOptions(BaseModel):
    """``Omni`` options for how a diffusion stage executes: request batching
    (``step_execution``), sequence parallelism (``ulysses_degree`` /
    ``ring_degree``, which need that many GPUs) and step caching
    (``cache_backend`` with an optional ``cache_config`` mapping)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    step_execution: bool | None = None
    ulysses_degree: PositiveInt | None = None
    ring_degree: PositiveInt | None = None
    cache_backend: str | None = Field(default=None, min_length=1)
    cache_config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _cache_config_needs_backend(self) -> Self:
        if self.cache_config is not None and self.cache_backend is None:
            raise ValueError("cache_config needs cache_backend")
        return self


@dataclass(frozen=True)
class OmniMember:
    """One task of a dispatch: the dispatched task or one of its merged children."""

    task_id: str
    spec: TaskSpecStrictBase
    out_dir: Path


class OmniExecutorBase(InferenceMixin, Executor):
    """Shared base for Omni-family executors.

    Manages the ``_omni`` model handle, runs the generic ``run()`` shape
    (task span + artifact / trace upload, merged children's results under
    ``result.children``), and delegates the task body to each subclass's
    ``_run_inner``. Subclasses set ``_TASK_SPEC_TYPE`` so the base can call
    ``require_spec`` without knowing the concrete type.
    """

    _TASK_SPEC_TYPE: ClassVar[type[TaskSpecStrictBase]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._omni: Omni | None = None
        self._model_name: str | None = None
        self._omni_spec: tuple[Any, ...] | None = None
        self._deploy_config_tmp: Path | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return _HAS_OMNI

    def run(self, task: ExecutorTask, out_dir: Path, control: RunControl) -> OmniResult:
        spec = self.require_spec(task, self._TASK_SPEC_TYPE)
        out_dir = Path(out_dir).resolve()
        children = task.merged_children or []
        members = [OmniMember(task.task_id, spec, out_dir)]
        for child in children:
            if not isinstance(child.spec, self._TASK_SPEC_TYPE):
                raise ExecutionError(
                    f"Merged child spec must be {self.name} for merged execution."
                )
            members.append(
                OmniMember(child.task_id, child.spec, out_dir.parent / child.task_id)
            )
        with self._task_span(
            task.task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            results = self._run_members(members)
        maybe_upload_artifacts(task, out_dir, logger=logger)
        maybe_upload_traces(task, out_dir, logger=logger)
        for child, member, result in zip(children, members[1:], results[1:]):
            maybe_upload_artifacts(child, member.out_dir, logger=logger)
            results[0].children[child.task_id] = result
        return results[0]

    def _run_members(self, members: list[OmniMember]) -> list[OmniResult]:
        """Run a task and its merged children, returning one result per member
        in order. Members share a merge key, so they share the model and its
        settings; this runs them one after another on that model. Executors
        whose engine call takes a prompt batch override it to render every
        member in one call."""
        return [
            self._run_inner(
                member.task_id,
                member.spec,
                member.spec.model_dump(by_alias=True),
                member.out_dir,
            )
            for member in members
        ]

    @abstractmethod
    def _run_inner(
        self,
        task_id: str,
        spec: TaskSpecStrictBase,
        spec_dict: dict[str, Any],
        out_dir: Path,
    ) -> OmniResult:
        """Run the executor-specific body for one task. ``spec`` is the
        concrete strict spec; subclasses ``assert isinstance(spec, ...)`` to
        narrow."""
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
        self._cleanup_deploy_config_tmp()

    @classmethod
    def _build_omni_spec(cls, model_name: str, cfg: dict[str, Any]) -> tuple[Any, ...]:
        """Build a hashable spec capturing all init-relevant settings.

        Two tasks with matching specs can safely share an ``Omni`` instance.
        """
        deploy_config = cfg.get("deploy_config")
        deploy_config_key = (
            json.dumps(deploy_config, sort_keys=True)
            if isinstance(deploy_config, dict)
            else deploy_config
        )
        stage_overrides = cfg.get("stage_overrides")
        stage_overrides_key = (
            json.dumps(stage_overrides, sort_keys=True)
            if isinstance(stage_overrides, dict)
            else stage_overrides
        )
        return (
            model_name,
            deploy_config_key,
            stage_overrides_key,
            cfg.get("log_stats"),
            cfg.get("stage_init_timeout"),
            cfg.get("init_timeout"),
            cfg.get("async_chunk"),
            json.dumps(cls.engine_weight_kwargs(cfg), sort_keys=True),
            json.dumps(cls.engine_execution_kwargs(cfg), sort_keys=True),
        )

    @staticmethod
    def engine_weight_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
        """``Omni`` options for how model weights are held, validated by
        ``OmniWeightOptions``. Only the options set in the spec are passed."""
        options = validate_options(OmniWeightOptions, cfg, "omni weight options")
        kwargs = options.model_dump(exclude_none=True, exclude={"quantization"})
        if options.quantization:
            kwargs["quantization_config"] = options.quantization
        return kwargs

    @staticmethod
    def engine_execution_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
        """``Omni`` options for how a diffusion stage executes, validated by
        ``OmniExecutionOptions``. Only the options set in the spec are passed."""
        options = validate_options(OmniExecutionOptions, cfg, "omni execution options")
        return options.model_dump(exclude_none=True)

    def _materialize_deploy_config(self, cfg: dict[str, Any]) -> str | None:
        """Resolve ``deploy_config`` to a path.

        A string is treated as an existing deploy-YAML path; an inline dict is
        written to a temp YAML so a workflow can carry a custom pipeline layout
        without shipping a file into the worker.
        """
        self._cleanup_deploy_config_tmp()
        deploy_config = cfg.get("deploy_config")
        if deploy_config is None:
            return None
        if isinstance(deploy_config, str):
            return deploy_config or None
        if not isinstance(deploy_config, dict):
            raise ExecutionError(
                "omni deploy_config must be a deploy-YAML path or an inline "
                f"mapping, got {type(deploy_config).__name__}."
            )
        fd, path = tempfile.mkstemp(prefix="omni_deploy_config_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(deploy_config, f)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        self._deploy_config_tmp = Path(path)
        return path

    def _cleanup_deploy_config_tmp(self) -> None:
        path = self._deploy_config_tmp
        self._deploy_config_tmp = None
        if path is not None:
            path.unlink(missing_ok=True)

    @staticmethod
    def _resolve_stage_overrides(cfg: dict[str, Any]) -> dict[str, Any] | str | None:
        """Normalize ``stage_overrides`` to the mapping or JSON string that
        ``vllm_omni``'s ``parse_stage_overrides`` accepts (mapping keys to str)."""
        stage_overrides = cfg.get("stage_overrides")
        if stage_overrides is None:
            return None
        if isinstance(stage_overrides, str):
            return stage_overrides or None
        if isinstance(stage_overrides, dict):
            return {str(k): v for k, v in stage_overrides.items()} or None
        raise ExecutionError(
            "omni stage_overrides must be a per-stage mapping or JSON string, "
            f"got {type(stage_overrides).__name__}."
        )

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
        deploy_config_path = self._materialize_deploy_config(cfg)
        if deploy_config_path is not None:
            init_kwargs["deploy_config"] = deploy_config_path
        stage_overrides = self._resolve_stage_overrides(cfg)
        if stage_overrides is not None:
            init_kwargs["stage_overrides"] = stage_overrides
        if cfg.get("log_stats") is not None:
            init_kwargs["log_stats"] = to_bool(cfg.get("log_stats"), default=False)
        if cfg.get("async_chunk") is not None:
            init_kwargs["async_chunk"] = to_bool(cfg.get("async_chunk"), default=False)
        for key in ("stage_init_timeout", "init_timeout"):
            val = to_int(cfg.get(key))
            if val is not None:
                init_kwargs[key] = val
        init_kwargs.update(self.engine_weight_kwargs(cfg))
        init_kwargs.update(self.engine_execution_kwargs(cfg))
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


def prompt_index_from_request_id(request_id: str) -> int | None:
    """Parse the prompt index from a vllm_omni ``{index}_{uuid}`` request id."""
    head = request_id.split("_", 1)[0]
    return int(head) if head.isdecimal() else None


def validate_options[M: BaseModel](
    model: type[M], cfg: dict[str, Any], label: str
) -> M:
    """Validate the keys of ``cfg`` that ``model`` declares; other keys are
    left to their own readers. Raises ``ExecutionError`` naming each bad key."""
    try:
        return model.model_validate(
            {key: cfg[key] for key in model.model_fields if key in cfg}
        )
    except ValidationError as exc:
        raise ExecutionError(
            f"invalid {label}: {_describe_validation_error(exc)}"
        ) from exc


def _describe_validation_error(exc: ValidationError) -> str:
    """One line naming each invalid field and why it was rejected."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"])
        problems.append(f"{field}: {err['msg']}" if field else err["msg"])
    return "; ".join(problems)


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
