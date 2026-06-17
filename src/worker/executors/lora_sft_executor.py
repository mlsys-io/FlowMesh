#!/usr/bin/env python3
"""LoRA fine-tuning executor built on TRL's SFTTrainer with PEFT support."""

import gc
import logging
import time
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import LoRASFTSpecStrict

from ..utils.logging import configure_hf_library_logging
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.training import TrainingMixin
from .sft_executor import SFTExecutor
from .utils.checkpoints import (
    archive_model_dir,
    determine_resume_path,
    maybe_upload_artifacts,
    write_executor_result,
)
from .utils.huggingface import build_hf_load_kwargs, pick_torch_dtype

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    if TYPE_CHECKING:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    else:
        LoraConfig = None
        TaskType = None
        get_peft_model = None
        PeftModel = None

logger = logging.getLogger("worker.sft.lora")


class LoRAResult(BaseExecutorResult):
    training_time_seconds: float | None = None
    error_message: str | None = None
    model_name: str | None = None
    dataset_size: int = 0
    output_dir: str | None = None
    checkpoints_dir: ArtifactRef | None = None
    resume_from_path: str | None = None
    final_lora: ArtifactRef | None = None
    final_lora_archive: ArtifactRef | None = None


class LoRASFTExecutor(TrainingMixin, Executor):
    """Execute LoRA-based supervised fine-tuning using TRL's SFTTrainer."""

    name = "lora_sft_executor"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_name: str | None = None
        self._current_model: Any | None = None
        self._current_trainer: Any | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> LoRAResult:
        configure_hf_library_logging()
        spec = self.require_spec(task, LoRASFTSpecStrict)
        start_time = time.time()
        ok = False
        error_msg: str | None = None

        if (
            LoraConfig is None
            or TaskType is None
            or get_peft_model is None
            or PeftModel is None
        ):
            raise ExecutionError(
                "peft is required for LoRA SFT tasks. Install the 'peft' package in "
                "the worker environment."
            )

        raw_training_cfg = spec.training or {}
        training_cfg = dict(raw_training_cfg)

        if bool(training_cfg.get("allow_multi_gpu", False)):
            logger.warning(
                "LoRA SFT currently runs on a single GPU; ignoring allow_multi_gpu "
                "request"
            )
            training_cfg["allow_multi_gpu"] = False

        SFTExecutor._configure_devices(training_cfg)
        if training_cfg.get("deepspeed"):
            logger.info(
                "DeepSpeed configuration detected for LoRA run; forwarding to trainer"
            )
        lora_cfg = spec.lora or {}

        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = artifacts_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        resume_path: Path | None = None
        train_dataset: Dataset | None = None
        try:
            model_name = spec.model_name or "gpt2"
            self._model_name = model_name

            logger.info("Loading tokenizer and model for LoRA SFT: %s", model_name)
            torch_dtype = pick_torch_dtype(training_cfg)
            tok_kwargs, model_kwargs = build_hf_load_kwargs(
                revision=spec.model_revision,
                trust_remote_code=spec.model_trust_remote_code,
                training_cfg=training_cfg,
                torch_dtype=torch_dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            model.config.use_cache = False
            self._current_model = model

            resume_path = determine_resume_path(
                spec, training_cfg, out_dir, logger=logger
            )
            resume_str = resume_path.as_posix() if resume_path else None

            if bool(training_cfg.get("gradient_checkpointing", False)):
                model.gradient_checkpointing_enable()

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            deepspeed_config = SFTExecutor._resolve_deepspeed_config(
                training_cfg, logger
            )

            peft_model: PeftModel
            if resume_path:
                logger.info(
                    "Resuming LoRA training from local checkpoint %s", resume_path
                )
                peft_model = PeftModel.from_pretrained(
                    model, resume_path.as_posix(), is_trainable=True
                )
                logger.info("Loaded existing LoRA adapters; continuing fine-tuning")
            else:
                logger.info(
                    "No existing LoRA adapters detected; starting from base model"
                )
                lora_target_modules = lora_cfg.get("target_modules") or [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]
                if not isinstance(lora_target_modules, (list, tuple)):
                    raise ValueError(
                        "lora.target_modules must be a list of module names"
                    )
                lora_target_modules = [str(mod) for mod in lora_target_modules]

                task_type_raw = str(lora_cfg.get("task_type", "CAUSAL_LM")).upper()
                try:
                    task_type = TaskType[task_type_raw]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported LoRA task_type '{task_type_raw}'"
                    ) from exc

                peft_config = LoraConfig(
                    r=int(lora_cfg.get("r", 16)),
                    lora_alpha=int(lora_cfg.get("alpha", 32)),
                    lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                    bias=str(lora_cfg.get("bias", "none")),  # type: ignore
                    target_modules=lora_target_modules,
                    task_type=task_type,
                    use_rslora=bool(lora_cfg.get("use_rslora", False)),
                )

                created_model = get_peft_model(model, peft_config)
                if not isinstance(created_model, PeftModel):
                    raise ExecutionError(
                        "LoRA SFT requires PeftModel; "
                        f"got {type(created_model).__name__} instead"
                    )
                peft_model = created_model
                logger.info("Initialized new LoRA adapters: %s", lora_target_modules)

            sft_config = SFTConfig(
                output_dir=checkpoint_dir.as_posix(),
                num_train_epochs=float(training_cfg.get("num_train_epochs", 1.0)),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(
                    training_cfg.get("gradient_accumulation_steps", 1)
                ),
                learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
                warmup_steps=int(training_cfg.get("warmup_steps", 0)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_strategy=str(training_cfg.get("save_strategy", "steps")),
                report_to=[],
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                dataset_text_field=text_field,
                max_length=int(training_cfg.get("max_seq_length", 1024)),
                packing=bool(training_cfg.get("packing", False)),
                pad_token=tokenizer.pad_token,
                eos_token=tokenizer.eos_token,
                deepspeed=deepspeed_config,
            )

            trainer = SFTTrainer(
                model=peft_model,
                args=sft_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )

            orig_compute_loss = trainer.compute_loss
            self._current_trainer = trainer

            def _compute_loss_with_guard(
                self, model, inputs, return_outputs=False, num_items_in_batch=None
            ):
                try:
                    return orig_compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )
                except RuntimeError as exc:
                    if "size of tensor" not in str(exc):
                        raise
                    logger.warning(
                        "LoRA SFT entropy metric mismatch encountered; falling back to "
                        "baseline loss computation: %s",
                        exc,
                    )
                    safe_inputs = dict(inputs)
                    safe_inputs["use_cache"] = False
                    loss, outputs = Trainer.compute_loss(
                        self,
                        model,
                        safe_inputs,
                        return_outputs=True,
                        num_items_in_batch=num_items_in_batch,
                    )
                    return (loss, outputs) if return_outputs else loss

            trainer.compute_loss = MethodType(  # type: ignore[method-assign]
                _compute_loss_with_guard, trainer
            )

            logger.info("Starting LoRA supervised fine-tuning run")
            trainer.train()
            ok = True
            logger.info("LoRA SFT training completed")

            final_adapter_path: Path | None = None
            archive_path: Path | None = None
            if bool(training_cfg.get("save_model", True)):
                model_path = artifacts_dir / "final_lora"
                trainer.save_model(model_path.as_posix())
                tokenizer.save_pretrained(model_path)
                final_adapter_path = model_path
                logger.info("Saved LoRA-adapted weights to %s", model_path)

            training_time = time.time() - start_time
            final_lora: ArtifactRef | None = None
            final_lora_archive: ArtifactRef | None = None
            if final_adapter_path:
                final_lora = ArtifactRef(
                    path=final_adapter_path.relative_to(artifacts_dir).as_posix()
                )
                archive_path = archive_model_dir(final_adapter_path)
                final_lora_archive = ArtifactRef(
                    path=archive_path.relative_to(artifacts_dir).as_posix()
                )
                logger.info("Prepared LoRA archive at %s", archive_path)
            result = LoRAResult(
                ok=ok,
                training_time_seconds=training_time,
                error_message=error_msg,
                model_name=self._model_name,
                dataset_size=len(train_dataset) if train_dataset is not None else 0,
                output_dir=out_dir.as_posix(),
                checkpoints_dir=ArtifactRef(path="checkpoints"),
                resume_from_path=resume_str,
                final_lora=final_lora,
                final_lora_archive=final_lora_archive,
            )

            maybe_upload_artifacts(task, out_dir, logger=logger)

            self._cleanup_local_artifacts(
                task,
                checkpoint_dir,
                final_adapter_path,
                archive_path,
            )
            return result

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = str(exc)
            ok = False
            logger.exception("LoRA SFT training failed: %s", exc)

        training_time = time.time() - start_time

        result = LoRAResult(
            ok=ok,
            training_time_seconds=training_time,
            error_message=error_msg,
            model_name=self._model_name,
            dataset_size=len(train_dataset) if train_dataset is not None else 0,
            output_dir=out_dir.as_posix(),
            checkpoints_dir=ArtifactRef(path="checkpoints"),
            resume_from_path=resume_path.as_posix() if resume_path else None,
        )

        if ok:
            return result

        write_executor_result(out_dir / "results.json", task.task_id, task.spec, result)
        message = error_msg or "LoRA SFT training failed"
        raise ExecutionError(message)

    def cleanup_after_run(self) -> None:
        self._current_trainer = None
        self._current_model = None
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

    def _prepare_dataset(self, spec: LoRASFTSpecStrict) -> tuple[Dataset, str]:
        data_cfg = spec.data or {}

        dataset_name = data_cfg.get("dataset_name")
        prompt_col = data_cfg.get("prompt_column")
        response_col = data_cfg.get("response_column")

        if dataset_name:
            split = data_cfg.get("split", "train")
            config_name = data_cfg.get("config_name")
            trust_remote_code = data_cfg.get("trust_remote_code")
            revision = data_cfg.get("revision")
            dataset_kwargs = {
                "split": split,
                "revision": revision,
            }
            if trust_remote_code is not None:
                dataset_kwargs["trust_remote_code"] = bool(trust_remote_code)
            dataset = load_dataset(
                dataset_name,
                config_name,
                **{k: v for k, v in dataset_kwargs.items() if v is not None},
            )

            if prompt_col and response_col:
                missing = [
                    col
                    for col in (prompt_col, response_col)
                    if col not in dataset.column_names
                ]
                if missing:
                    raise ValueError(f"Dataset missing columns {missing}")

                separator = data_cfg.get("separator", "\n\n")

                def _combine(example: dict[str, Any]) -> dict[str, Any]:
                    return {
                        "text": (
                            f"{example[prompt_col]}{separator}"
                            f"{example[response_col]}"
                        )
                    }

                dataset = dataset.map(_combine, remove_columns=dataset.column_names)
                text_field = "text"
            else:
                text_field = data_cfg.get("text_field", "text")
                if text_field not in dataset.column_names:
                    raise ValueError(
                        f"Dataset missing text field '{text_field}'. "
                        f"Columns: {dataset.column_names}"
                    )

            max_samples = data_cfg.get("max_samples")
            if max_samples is not None:
                max_samples = int(max_samples)
                dataset = dataset.select(range(min(len(dataset), max_samples)))

            return dataset, text_field

        if "prompts" in data_cfg:
            prompts = data_cfg["prompts"]
            if not isinstance(prompts, list) or not prompts:
                raise ValueError(
                    "spec.data.prompts must be a non-empty list of strings"
                )
            dataset = Dataset.from_dict({"text": [str(p) for p in prompts]})
            return dataset, "text"

        raise ValueError(
            "spec.data must define dataset_name or prompts for LoRA SFT tasks"
        )
