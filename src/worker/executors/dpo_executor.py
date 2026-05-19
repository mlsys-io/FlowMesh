#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Executor using TRL's simple approach

Simple implementation using TRL's DPOTrainer with built-in train() method.
"""

import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import DPOSpecStrict
from shared.utils.manifest import scratch_dir

from ..utils.logging import configure_hf_library_logging
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.training import TrainingMixin
from .utils.checkpoints import (
    archive_model_dir,
    get_http_destination,
    maybe_upload_artifacts,
    write_executor_result,
)
from .utils.data_utils import resolve_jsonl_path
from .utils.distributed import run_torchrun
from .utils.huggingface import build_hf_load_kwargs, pick_torch_dtype

logger = logging.getLogger("worker.dpo")


class DPOResult(BaseExecutorResult):
    training_successful: bool = True
    training_time_seconds: float | None = None
    error_message: str | None = None
    model_name: str | None = None
    dataset_size: int = 0
    output_dir: str | None = None
    checkpoints_dir: ArtifactRef | None = None
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class DPOExecutor(TrainingMixin, Executor):
    """DPO training executor using TRL library."""

    name = "dpo_executor"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_name: str | None = None
        self._current_model: PreTrainedModel | None = None
        self._current_ref_model: PreTrainedModel | None = None
        self._current_tokenizer: PreTrainedTokenizerBase | None = None
        self._current_trainer: DPOTrainer | None = None
        self._task_out_dir: Path | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> DPOResult:
        configure_hf_library_logging()
        logger.info("Starting DPO training task")
        spec = self.require_spec(task, DPOSpecStrict)
        training_config = spec.training or {}
        launcher_flag = "KV_DPO_DISTRIBUTED"
        already_spawned = os.environ.get(launcher_flag) == "1"
        gpu_count = self._detect_gpu_count(training_config)
        allow_multi_cfg = training_config.get("allow_multi_gpu")
        allow_multi = (
            bool(allow_multi_cfg) if allow_multi_cfg is not None else gpu_count > 1
        )

        self._task_out_dir = out_dir
        try:
            if allow_multi and not already_spawned and gpu_count > 1:
                self._spawn_distributed(
                    task, out_dir, gpu_count, launcher_flag, training_config
                )
                ipc_path = scratch_dir(out_dir) / "distributed_result.json"
                if ipc_path.exists():
                    return DPOResult.model_validate(self.load_json(ipc_path))
                return DPOResult(
                    training_successful=True,
                    spawned_torchrun=True,
                    model_name=(
                        spec.model.source.identifier
                        if spec.model and spec.model.source
                        else None
                    ),
                    output_dir=out_dir.as_posix(),
                )

            result = self._execute_training(task, out_dir)
            logger.info(
                "DPO training task completed in %.2f seconds",
                result.training_time_seconds or 0.0,
            )
            return result
        finally:
            self._task_out_dir = None

    def _load_dataset(self, spec: DPOSpecStrict) -> Dataset:
        """Load training dataset in DPO format"""
        data_config = spec.data or {}

        def _ensure_jsonl_local(jsonl_cfg: dict[str, Any]) -> Path:
            headers_cfg = (
                jsonl_cfg.get("download_headers") or jsonl_cfg.get("headers") or {}
            )
            headers = (
                {str(k): str(v) for k, v in headers_cfg.items()}
                if headers_cfg
                else None
            )
            timeout = float(
                jsonl_cfg.get("download_timeoutSec")
                or jsonl_cfg.get("timeoutSec")
                or jsonl_cfg.get("timeout")
                or 60
            )

            candidates: list[str] = []
            for key in ("download_url", "url", "path", "worker_path"):
                value = str(jsonl_cfg.get(key) or "").strip()
                if value and value not in candidates:
                    candidates.append(value)

            base_dir = (
                self._task_out_dir
                or Path(tempfile.mkdtemp(prefix="flowmesh_jsonl_")).resolve()
            )
            last_error: Exception | None = None

            for candidate in candidates:
                try:
                    resolved = resolve_jsonl_path(
                        candidate,
                        out_dir=base_dir,
                        headers=headers,
                        timeout=timeout,
                        logger=logger,
                    )
                    jsonl_cfg["path"] = str(resolved)
                    return resolved
                except ExecutionError as exc:
                    last_error = exc
                    continue

            if last_error is not None:
                raise ExecutionError(str(last_error)) from last_error
            if candidates:
                raise ExecutionError(f"JSONL dataset not found: {candidates[-1]}")
            raise ExecutionError("data.jsonl.path is required when using JSONL input")

        jsonl_cfg = data_config.get("jsonl")
        jsonl_path = data_config.get("jsonl_path")
        if jsonl_cfg or jsonl_path:
            if jsonl_cfg is None:
                jsonl_cfg = {}
            else:
                jsonl_cfg = dict(jsonl_cfg)
            if jsonl_path:
                jsonl_cfg.setdefault("path", jsonl_path)

            jsonl_file = _ensure_jsonl_local(jsonl_cfg)

            prompt_field = (
                jsonl_cfg.get("prompt_field")
                or data_config.get("prompt_field")
                or "prompt"
            )
            chosen_field = (
                jsonl_cfg.get("chosen_field")
                or data_config.get("chosen_field")
                or "chosen"
            )
            rejected_field = (
                jsonl_cfg.get("rejected_field")
                or data_config.get("rejected_field")
                or "rejected"
            )

            prompts: list[str] = []
            chosen: list[str] = []
            rejected: list[str] = []

            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line_number, raw in enumerate(fh, start=1):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ExecutionError(
                            f"Invalid JSON on line {line_number} of {jsonl_file}: {exc}"
                        ) from exc

                    if (
                        prompt_field not in record
                        or chosen_field not in record
                        or rejected_field not in record
                    ):
                        raise ExecutionError(
                            "JSONL record missing required fields. Expected keys: "
                            f"'{prompt_field}', '{chosen_field}', '{rejected_field}'"
                        )

                    prompts.append(str(record[prompt_field]))
                    chosen.append(str(record[chosen_field]))
                    rejected.append(str(record[rejected_field]))

            if not prompts:
                raise ExecutionError(f"JSONL dataset at {jsonl_file} is empty")

            dataset = Dataset.from_dict(
                {
                    "prompt": prompts,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

        elif "dataset_name" in data_config:
            # Load from Hugging Face datasets (like ultrafeedback_binarized)
            from datasets import load_dataset

            dataset_name = data_config["dataset_name"]
            split = data_config.get("split", "train")

            config_name = data_config.get("config_name")
            revision = data_config.get("revision")
            load_kwargs: dict[str, Any] = {"split": split}
            if config_name:
                load_kwargs["name"] = config_name
            if revision:
                load_kwargs["revision"] = revision

            try:
                dataset = load_dataset(dataset_name, **load_kwargs)
            except ValueError as exc:
                should_retry = False
                if config_name and "config" in str(exc):
                    should_retry = True
                if config_name and "BuilderConfig" in str(exc):
                    should_retry = True
                if should_retry:
                    logger.warning(
                        "Failed to load %s with config %s (split=%s). Retrying with "
                        "default config.",
                        dataset_name,
                        config_name,
                        split,
                    )
                    fallback_kwargs = {
                        k: v for k, v in load_kwargs.items() if k != "name"
                    }
                    fallback_split = split
                    prefix = split.split("[", 1)[0]
                    if prefix != config_name and config_name:
                        fallback_split = split.replace(prefix, config_name, 1)
                    fallback_kwargs["split"] = fallback_split
                    try:
                        dataset = load_dataset(dataset_name, **fallback_kwargs)
                    except ValueError:
                        fallback_kwargs["split"] = split
                        dataset = load_dataset(dataset_name, **fallback_kwargs)
                else:
                    raise

            if data_config.get("max_samples"):
                dataset = cast(Dataset, dataset)
                dataset = dataset.select(
                    range(min(len(dataset), data_config["max_samples"]))
                )

            prompt_col = data_config.get("prompt_column", "prompt")
            chosen_col = data_config.get("chosen_column", "chosen")
            rejected_col = data_config.get("rejected_column", "rejected")

            required_source_columns = {prompt_col, chosen_col, rejected_col}
            column_names = dataset.column_names or []
            missing_columns = required_source_columns - set(column_names)
            if missing_columns:
                raise ExecutionError(
                    f"Dataset {dataset_name} missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            rename_map = {}
            if prompt_col != "prompt":
                rename_map[prompt_col] = "prompt"
            if chosen_col != "chosen":
                rename_map[chosen_col] = "chosen"
            if rejected_col != "rejected":
                rename_map[rejected_col] = "rejected"

            for old_name, new_name in rename_map.items():
                dataset = dataset.rename_column(old_name, new_name)

            extra_columns = set(column_names) - {"prompt", "chosen", "rejected"}
            if extra_columns:
                logger.info(
                    "Removing unused columns from dataset %s: %s",
                    dataset_name,
                    ", ".join(sorted(extra_columns)),
                )
                dataset = dataset.remove_columns(sorted(extra_columns))

        elif "preferences" in data_config:
            # Use provided preference data directly
            preferences = data_config["preferences"]
            dataset = Dataset.from_dict(
                {
                    "prompt": [item["prompt"] for item in preferences],
                    "chosen": [item["chosen"] for item in preferences],
                    "rejected": [item["rejected"] for item in preferences],
                }
            )
        else:
            # Default demo dataset with preference pairs
            demo_data = [
                {
                    "prompt": "Write a helpful response about Python programming:",  # noqa: E501
                    "chosen": "Python is a versatile programming language known for its readability and extensive libraries. It's great for beginners and widely used in data science, web development, and automation.",  # noqa: E501
                    "rejected": "Python is just another programming language. Nothing special about it.",  # noqa: E501
                },
                {
                    "prompt": "Explain machine learning in simple terms:",  # noqa: E501
                    "chosen": "Machine learning is like teaching computers to recognize patterns and make predictions from data, similar to how humans learn from experience.",  # noqa: E501
                    "rejected": "Machine learning is complicated math stuff that computers do.",  # noqa: E501
                },
                {
                    "prompt": "Give advice on learning to code:",  # noqa: E501
                    "chosen": "Start with a beginner-friendly language like Python, practice regularly with small projects, and don't be afraid to make mistakes - they're part of learning!",  # noqa: E501
                    "rejected": "Just memorize syntax and you'll be fine.",  # noqa: E501
                },
                {
                    "prompt": "Describe the importance of documentation:",  # noqa: E501
                    "chosen": "Good documentation is essential for code maintainability, team collaboration, and helping future developers (including yourself) understand what the code does and why.",  # noqa: E501
                    "rejected": "Documentation is just extra work that slows down development.",  # noqa: E501
                },
            ]

            dataset = Dataset.from_dict(
                {
                    "prompt": [item["prompt"] for item in demo_data],
                    "chosen": [item["chosen"] for item in demo_data],
                    "rejected": [item["rejected"] for item in demo_data],
                }
            )

        return dataset  # type: ignore[return-value]

    def _execute_training(self, task: ExecutorTask, out_dir: Path) -> DPOResult:
        spec = self.require_spec(task, DPOSpecStrict)
        training_config = spec.training or {}
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = artifacts_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        dataset: Dataset | None = None

        final_model_path: Path | None = None
        final_archive_path: Path | None = None

        try:
            model_name = spec.model_name or "microsoft/DialoGPT-small"
            self._model_name = model_name

            torch_dtype = pick_torch_dtype(training_config)
            tok_kwargs, model_kwargs = build_hf_load_kwargs(
                revision=spec.model_revision,
                trust_remote_code=spec.model_trust_remote_code,
                training_cfg=training_config,
                torch_dtype=torch_dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            self._current_tokenizer = tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            ref_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self._current_model = model
            self._current_ref_model = ref_model

            logger.info("Models loaded: %s", model_name)

            dataset = self._load_dataset(spec)
            logger.info("Dataset loaded with %d samples", len(dataset))

            dpo_config = DPOConfig(
                learning_rate=float(training_config.get("learning_rate", 5e-7)),
                per_device_train_batch_size=int(training_config.get("batch_size", 4)),
                gradient_accumulation_steps=int(
                    training_config.get("gradient_accumulation_steps", 1)
                ),
                num_train_epochs=int(training_config.get("num_train_epochs", 1)),
                output_dir=str(checkpoint_dir),
                save_steps=int(training_config.get("save_freq", 500)),
                logging_steps=10,
            )

            try:
                dpo_trainer = DPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=dpo_config,
                    train_dataset=dataset,
                    tokenizer=tokenizer,  # type: ignore[call-arg]
                )
                self._current_trainer = dpo_trainer
            except TypeError as exc:
                if "unexpected keyword argument 'tokenizer'" in str(exc):
                    try:
                        dpo_trainer = DPOTrainer(
                            model=model,
                            ref_model=ref_model,
                            args=dpo_config,
                            train_dataset=dataset,
                            processing_class=tokenizer,
                        )
                        self._current_trainer = dpo_trainer
                    except TypeError:
                        dpo_trainer = DPOTrainer(
                            model=model,
                            ref_model=ref_model,
                            args=dpo_config,
                            train_dataset=dataset,
                        )
                        self._current_trainer = dpo_trainer
                else:
                    raise

            logger.info("Starting DPO training...")
            dpo_trainer.train()
            logger.info("DPO training completed")

            if training_config.get("save_model", True):
                try:
                    model_save_path = checkpoint_dir / "final_model"
                    dpo_trainer.save_model(str(model_save_path))
                    logger.info("Model saved to: %s", model_save_path)
                    final_model_path = model_save_path
                    destination = get_http_destination(task.spec)
                    if destination:
                        try:
                            final_archive_path = archive_model_dir(model_save_path)
                            logger.info(
                                "Archived DPO model to %s for HTTP delivery",
                                final_archive_path,
                            )
                        except Exception as arch_exc:
                            logger.warning(
                                "Failed to archive DPO model for upload: %s", arch_exc
                            )
                except Exception as exc:
                    logger.warning("Failed to save model: %s", exc)

            training_time = time.time() - start_time
            result = DPOResult(
                training_successful=True,
                training_time_seconds=training_time,
                error_message=None,
                model_name=self._model_name,
                dataset_size=len(dataset) if dataset is not None else 0,
                output_dir=out_dir.as_posix(),
                checkpoints_dir=ArtifactRef(path="checkpoints"),
            )
            if final_model_path is not None:
                result.final_model = ArtifactRef(
                    path=final_model_path.relative_to(artifacts_dir).as_posix()
                )
            if final_archive_path is not None:
                result.final_model_archive = ArtifactRef(
                    path=final_archive_path.relative_to(artifacts_dir).as_posix()
                )

            maybe_upload_artifacts(task, out_dir, logger=logger)

            self._cleanup_local_artifacts(
                task,
                checkpoint_dir,
                final_model_path,
                final_archive_path,
            )
            return result
        except Exception as exc:
            training_time = time.time() - start_time
            result = DPOResult(
                training_successful=False,
                training_time_seconds=training_time,
                error_message=str(exc),
                model_name=self._model_name,
                dataset_size=len(dataset) if dataset is not None else 0,
                output_dir=out_dir.as_posix(),
            )
            write_executor_result(
                out_dir / "results.json", task.task_id, task.spec, result
            )
            logger.exception("DPO training failed: %s", exc)
            raise ExecutionError(result.error_message or "DPO training failed") from exc

    def _spawn_distributed(
        self,
        task: ExecutorTask,
        out_dir: Path,
        n_gpus: int,
        launcher_flag: str,
        training_config: dict[str, Any],
    ) -> None:
        launcher_dir = scratch_dir(out_dir) / "launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        task_file = launcher_dir / "task_spec.json"
        with task_file.open("w", encoding="utf-8") as fh:
            json.dump(task.model_dump(mode="json", by_alias=True), fh)

        nproc = int(training_config.get("nproc_per_node", n_gpus))
        logger.info(
            "Launching torchrun for DPO (nproc=%d, CUDA_VISIBLE_DEVICES=%s)",
            nproc,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        run_torchrun(
            nproc_per_node=nproc,
            module="worker.executors.dpo_dist_entry",
            module_args=[task_file.as_posix(), out_dir.as_posix()],
            launcher_env_flag=launcher_flag,
        )

    @staticmethod
    def _detect_gpu_count(training_config: dict[str, Any]) -> int:
        vis = training_config.get("visible_devices") or os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
        if vis:
            tokens = [dev.strip() for dev in str(vis).split(",") if dev.strip()]
            if tokens:
                return len(tokens)
        try:
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0

    def cleanup_after_run(self) -> None:
        dropped_objects = []
        for attr in (
            "_current_trainer",
            "_current_model",
            "_current_ref_model",
            "_current_tokenizer",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                dropped_objects.append(obj)
            setattr(self, attr, None)

        for obj in dropped_objects:
            try:
                del obj
            except Exception:
                pass

        try:
            dist = getattr(torch, "distributed", None)
            if dist is not None:
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.destroy_process_group()
                        logger.debug(
                            "Destroyed torch distributed process group during cleanup"
                        )
                except Exception:
                    logger.debug(
                        "Failed to destroy torch distributed process group",
                        exc_info=True,
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    logger.debug("torch.cuda.ipc_collect failed", exc_info=True)
                if hasattr(torch.cuda, "reset_peak_memory_stats"):
                    try:
                        for idx in range(torch.cuda.device_count()):
                            torch.cuda.reset_peak_memory_stats(idx)
                    except Exception:
                        logger.debug(
                            "Failed to reset CUDA peak memory stats", exc_info=True
                        )
        except Exception:
            pass

        gc.collect()
