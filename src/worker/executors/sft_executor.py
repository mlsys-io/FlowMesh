#!/usr/bin/env python3
"""SFT executor powered by TRL's SFTTrainer/SFTConfig.

Single-GPU runs execute in-process. Multi-GPU runs go through
``torch.distributed.run.main`` (the same entry point ``torchrun`` calls),
or through ``deepspeed.launcher.runner.main`` when a DeepSpeed configuration
is supplied and the ``deepspeed`` package is importable.
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
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from shared.tasks.specs import SFTSpecStrict, TaskSpecStrictBase
from shared.utils.manifest import scratch_dir

from ..utils.logging import configure_hf_library_logging
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.training import TrainingMixin
from .utils.checkpoints import (
    archive_model_dir,
    artifact_ref,
    determine_resume_path,
    get_http_destination,
    maybe_upload_artifacts,
    write_executor_result,
)
from .utils.data_utils import resolve_jsonl_path
from .utils.distributed import deepspeed_available, run_deepspeed, run_torchrun
from .utils.huggingface import build_hf_load_kwargs, pick_torch_dtype

logger = logging.getLogger("worker.sft")


class SFTExecutor(TrainingMixin, Executor):
    name = "sft_executor"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_name: str | None = None
        self._current_model: Any | None = None
        self._current_trainer: Any | None = None
        self._current_tokenizer: Any | None = None
        self._final_model_dir: Path | None = None
        self._task_out_dir: Path | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> dict[str, Any]:
        configure_hf_library_logging()
        spec = self.require_spec(task, SFTSpecStrict)
        requested_gpu_count = self._requested_gpu_count(spec)
        start_time = time.time()
        training_successful = False
        error_msg: str | None = None
        caught_exc: Exception | None = None
        self._task_out_dir = out_dir

        training_cfg = spec.training or {}
        self._configure_devices(
            training_cfg
        )  # only constrain visible devices when explicit
        deepspeed_cfg = self._resolve_deepspeed_config(training_cfg, logger)
        # Under torchrun/accelerate, set CUDA device from LOCAL_RANK to avoid
        # NCCL warnings
        try:
            if torch.cuda.is_available():
                local_rank = int(
                    os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1"))
                )
                if local_rank >= 0:
                    torch.cuda.set_device(local_rank)
                    logger.info("Set CUDA device by LOCAL_RANK/RANK=%d", local_rank)
        except Exception as _e:
            logger.debug("Skipping cuda.set_device: %s", _e)
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = artifacts_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Internal distributed launcher: spawn multi-GPU training as subprocesses
        try:
            allow_multi_cfg = training_cfg.get("allow_multi_gpu")
            launcher_env_flag = "KV_SFT_DISTRIBUTED"
            already_spawned = os.environ.get(launcher_env_flag) == "1"
            # Determine requested GPU count
            vis = os.environ.get("CUDA_VISIBLE_DEVICES") or training_cfg.get(
                "visible_devices"
            )
            available_gpus = None
            if vis:
                available_gpus = len(
                    [dev for dev in str(vis).split(",") if dev.strip()]
                )
            if available_gpus is None:
                try:
                    if torch.cuda.is_available():
                        available_gpus = torch.cuda.device_count()
                except Exception:
                    available_gpus = None
            if available_gpus is None:
                available_gpus = 0

            if requested_gpu_count and requested_gpu_count > 1:
                n_gpus = min(requested_gpu_count, available_gpus)
            else:
                n_gpus = available_gpus

            # If a DeepSpeed config is supplied, assume multi-GPU intent
            deepspeed_intent = bool(deepspeed_cfg)
            if allow_multi_cfg is None:
                allow_multi = n_gpus > 1
            else:
                allow_multi = bool(allow_multi_cfg)

            if allow_multi and not deepspeed_intent and n_gpus > 1:
                auto_ds = self._build_auto_deepspeed_config(training_cfg, n_gpus)
                if auto_ds:
                    training_cfg["deepspeed"] = auto_ds
                    deepspeed_cfg = auto_ds
                    deepspeed_intent = True
                    logger.info(
                        "Auto-generated DeepSpeed config for multi-GPU SFT "
                        "(world size=%d, stage=%s).",
                        n_gpus,
                        auto_ds["zero_optimization"]["stage"],
                    )

            logger.info(
                "SFT spawn decision: allow_multi=%s deepspeed_intent=%s "
                "already_spawned=%s n_gpus=%s",
                allow_multi,
                deepspeed_intent,
                already_spawned,
                n_gpus,
            )
            if (
                (allow_multi or deepspeed_intent)
                and not already_spawned
                and (n_gpus or 0) > 1
            ):
                launcher_dir = scratch_dir(out_dir) / "launcher"
                launcher_dir.mkdir(parents=True, exist_ok=True)
                task_file = launcher_dir / "task_spec.json"
                with task_file.open("w", encoding="utf-8") as fh:
                    fh.write(task.model_dump_json(by_alias=True))

                nproc = int(training_cfg.get("nproc_per_node", n_gpus))
                use_deepspeed = deepspeed_intent and deepspeed_available()
                if deepspeed_intent and not use_deepspeed:
                    logger.warning(
                        "DeepSpeed configuration provided but the `deepspeed` "
                        "package is not importable; falling back to torchrun."
                    )
                if use_deepspeed:
                    logger.info(
                        "Launching DeepSpeed for SFT "
                        "(num_gpus=%d, CUDA_VISIBLE_DEVICES=%s)",
                        nproc,
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                    )
                    run_deepspeed(
                        num_gpus=nproc,
                        module="worker.executors.sft_dist_entry",
                        module_args=[task_file.as_posix(), out_dir.as_posix()],
                        launcher_env_flag=launcher_env_flag,
                    )
                else:
                    logger.info(
                        "Launching torchrun for SFT "
                        "(nproc=%d, CUDA_VISIBLE_DEVICES=%s)",
                        nproc,
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                    )
                    run_torchrun(
                        nproc_per_node=nproc,
                        module="worker.executors.sft_dist_entry",
                        module_args=[task_file.as_posix(), out_dir.as_posix()],
                        launcher_env_flag=launcher_env_flag,
                    )
                ipc_path = scratch_dir(out_dir) / "distributed_result.json"
                if ipc_path.exists():
                    distributed_result = self.load_json(ipc_path)
                    self._task_out_dir = None
                    return distributed_result
                self._task_out_dir = None
                return {
                    "training_successful": True,
                    "spawned_torchrun": True,
                    "model_name": spec.model_name,
                    "output_dir": out_dir.as_posix(),
                }
        except Exception as spawn_exc:
            logger.exception("Failed to launch distributed SFT: %s", spawn_exc)
            raise ExecutionError(
                "Failed to launch distributed SFT subprocess"
            ) from spawn_exc

        try:
            # Proceed with in-process training (single GPU or inside torchrun)
            model_name = spec.model_name or "gpt2"
            self._model_name = model_name

            resume_path = determine_resume_path(
                spec, training_cfg, out_dir, logger=logger
            )
            resume_str = str(resume_path) if resume_path else None

            if resume_path:
                logger.info(
                    "Loading tokenizer and model from local checkpoint: %s", resume_path
                )
            else:
                logger.info(
                    "Loading tokenizer and model from identifier: %s", model_name
                )
            # When resuming from a local checkpoint, the revision is baked in
            # already — don't pass it (would confuse from_pretrained).
            torch_dtype = pick_torch_dtype(training_cfg)
            tok_kwargs, model_kwargs = build_hf_load_kwargs(
                revision=spec.model_revision if not resume_str else None,
                trust_remote_code=spec.model_trust_remote_code,
                training_cfg=training_cfg,
                torch_dtype=torch_dtype,
            )
            tok_kwargs["use_fast"] = True
            tokenizer = AutoTokenizer.from_pretrained(
                resume_str or model_name, **tok_kwargs
            )
            self._current_tokenizer = tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if getattr(tokenizer, "padding_side", None) != "right":
                tokenizer.padding_side = "right"

            model = AutoModelForCausalLM.from_pretrained(
                resume_str or model_name,
                **model_kwargs,
            )
            model.config.use_cache = False
            if bool(training_cfg.get("gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()

            self._current_model = model

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            # Determine distributed context
            world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
            try:
                import torch.distributed as dist

                dist_initialized = dist.is_available() and dist.is_initialized()
            except Exception:
                dist_initialized = False
            expected_distributed = (
                dist_initialized or bool(deepspeed_cfg) or world_size_env > 1
            )

            # If DeepSpeed is requested but we are not in a distributed context, keep
            # going. Hugging Face will still initialize DeepSpeed on the current rank
            # (often rank 0 only).
            if deepspeed_cfg and not dist_initialized:
                if os.environ.get("KV_SFT_DISTRIBUTED") == "1":
                    logger.info(
                        "DeepSpeed runtime will initialize torch.distributed "
                        "(local_rank=%s)",
                        os.environ.get("LOCAL_RANK", "0"),
                    )
                else:
                    logger.warning(
                        "DeepSpeed config provided but torch.distributed is not "
                        "initialized; training continues on a single rank."
                    )

            # Optional DDP knobs
            ddp_kwargs: dict[str, Any] = {}
            if "ddp_find_unused_parameters" in training_cfg:
                ddp_kwargs["ddp_find_unused_parameters"] = bool(
                    training_cfg["ddp_find_unused_parameters"]
                )

            sft_config = SFTConfig(
                output_dir=str(checkpoint_dir),
                num_train_epochs=float(training_cfg.get("num_train_epochs", 1.0)),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(
                    training_cfg.get("gradient_accumulation_steps", 1)
                ),
                learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
                warmup_steps=int(training_cfg.get("warmup_steps", 0)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_strategy=str(training_cfg.get("save_strategy", "steps")),
                report_to=[],  # disable default wandb integration
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                dataset_text_field=text_field,
                max_length=int(training_cfg.get("max_seq_length", 1024)),
                packing=bool(training_cfg.get("packing", False)),
                gradient_checkpointing=bool(
                    training_cfg.get("gradient_checkpointing", True)
                ),
                pad_token=tokenizer.pad_token,
                eos_token=tokenizer.eos_token,
                deepspeed=deepspeed_cfg,
                **ddp_kwargs,
            )

            # Let the distributed backend handle placement when running under
            # torchrun/deepspeed
            if expected_distributed:
                logger.info(
                    "Distributed mode detected - device placement handled by "
                    "DeepSpeed/DDP backend."
                )
            if torch.cuda.is_available() and not expected_distributed:
                target_device = torch.device("cuda:0")
                model = model.to(target_device)  # type: ignore
                if any(p.device != target_device for _, p in model.named_parameters()):
                    logger.warning(
                        "Some parameters not moved to %s; check environment.",
                        target_device,
                    )
            elif expected_distributed:
                # placement handled by backend
                pass

            # Construct the trainer while handling TRL signature variations
            trainer = None
            tried = []
            for variant in ("tokenizer", "processing_class", "none"):
                try:
                    if variant == "tokenizer":
                        trainer = SFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                            tokenizer=tokenizer,  # type: ignore[call-arg]
                        )
                    elif variant == "processing_class":
                        trainer = SFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                            processing_class=tokenizer,
                        )
                    else:
                        trainer = SFTTrainer(
                            model=model, args=sft_config, train_dataset=train_dataset
                        )
                    break
                except TypeError as e:
                    tried.append(str(e))

            if trainer is None:
                raise TypeError(
                    "Failed to construct SFTTrainer. Tried variants: "
                    + " | ".join(tried)
                )

            self._current_trainer = trainer

            # Dry-run dataloader shapes to surface obvious padding mistakes early
            try:
                dl = trainer.get_train_dataloader()
                first_batch = next(iter(dl))
                ids = first_batch["input_ids"]
                mask = first_batch["attention_mask"]
                logger.info(
                    "Dry-run local batch -> input_ids=%s, attention_mask=%s",
                    tuple(ids.shape),
                    tuple(mask.shape),
                )
            except Exception as e:
                logger.warning("Dry-run dataloader check failed (non-fatal): %s", e)

            logger.info(
                "Effective batch -> per_device=%d, grad_accum=%d, deepspeed=%s",
                sft_config.per_device_train_batch_size,
                sft_config.gradient_accumulation_steps,
                bool(deepspeed_cfg),
            )

            logger.info("Starting supervised fine-tuning")
            trainer.train()
            training_successful = True
            logger.info("Training finished")

            final_model_path: Path | None = None
            final_archive_path: Path | None = None
            if bool(training_cfg.get("save_model", True)):
                model_path = artifacts_dir / "final_model"
                trainer.save_model(str(model_path))
                tokenizer.save_pretrained(model_path)
                final_model_path = model_path
                logger.info("Saved fine-tuned model to %s", model_path)
                if get_http_destination(spec):
                    final_archive_path = archive_model_dir(model_path)
                    logger.info(
                        "Archived fine-tuned model to %s for HTTP delivery",
                        final_archive_path,
                    )
                else:
                    logger.info(
                        "No HTTP destination detected; skipping archive generation"
                    )
            else:
                logger.info(
                    "save_model flag is false; skipping model serialization and "
                    "archive upload"
                )

            training_time = time.time() - start_time
            result_payload: dict[str, Any] = {
                "task_id": task.task_id,
                "training_successful": training_successful,
                "training_time_seconds": training_time,
                "error_message": error_msg,
                "model_name": self._model_name,
                "dataset_size": len(train_dataset),
                "output_dir": out_dir.as_posix(),
                "checkpoints_dir": artifact_ref("checkpoints"),
                "resume_from_path": resume_str,
            }

            if final_model_path is not None:
                resolved_model_path = Path(final_model_path)
                self._final_model_dir = (
                    resolved_model_path if resolved_model_path.exists() else None
                )
                result_payload["final_model"] = artifact_ref(
                    final_model_path.relative_to(artifacts_dir).as_posix()
                )
                if final_archive_path is not None:
                    result_payload["final_model_archive"] = artifact_ref(
                        final_archive_path.relative_to(artifacts_dir).as_posix()
                    )

            maybe_upload_artifacts(task, out_dir, logger=logger)

            self._cleanup_local_artifacts(
                task,
                checkpoint_dir,
                final_model_path,
                final_archive_path,
            )

            # Drop heavy references before runner-level cleanup
            trainer = None
            model = None  # type: ignore[assignment]
            tokenizer = None
            train_dataset = None
            self._current_trainer = None
            self._current_model = None
            self._current_tokenizer = None
            self._final_model_dir = None

            self._task_out_dir = None
            return result_payload

        except Exception as exc:
            error_msg = str(exc)
            training_successful = False
            caught_exc = exc
            logger.exception("SFT training failed: %s", exc)
            trainer = None
            model = None
            tokenizer = None
            train_dataset = None
            self._current_trainer = None
            self._current_model = None
            self._current_tokenizer = None
            self._final_model_dir = None

        training_time = time.time() - start_time
        result: dict[str, Any] = {
            "task_id": task.task_id,
            "training_successful": training_successful,
            "training_time_seconds": training_time,
            "error_message": error_msg,
            "model_name": self._model_name,
            "dataset_size": (
                len(train_dataset)
                if "train_dataset" in locals() and train_dataset is not None
                else 0
            ),
            "output_dir": out_dir.as_posix(),
            "checkpoints_dir": artifact_ref("checkpoints"),
            "resume_from_path": (
                str(resume_path) if "resume_path" in locals() and resume_path else None
            ),
        }
        write_executor_result(out_dir / "results.json", task.task_id, task.spec, result)
        if caught_exc is not None:
            self._task_out_dir = None
            raise ExecutionError(error_msg or "SFT training failed") from caught_exc
        self._task_out_dir = None
        raise ExecutionError(error_msg or "SFT training failed")

    def cleanup_after_run(self) -> None:
        dropped_objects = []
        for attr in (
            "_current_trainer",
            "_current_model",
            "_current_tokenizer",
            "_final_model_dir",
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
            import torch

            dist = getattr(torch, "distributed", None)
            if dist is not None:
                try:
                    if dist.is_available() and dist.is_initialized():
                        shutdown_fn = getattr(dist, "shutdown", None)
                        if callable(shutdown_fn):
                            shutdown_fn()
                            logger.debug(
                                "torch.distributed.shutdown() invoked during cleanup"
                            )
                        else:
                            dist.destroy_process_group()
                            logger.debug(
                                "Destroyed torch distributed process group during "
                                "cleanup"
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

    def _ensure_jsonl_local(self, jsonl_cfg: dict[str, Any]) -> Path:
        headers_cfg = (
            jsonl_cfg.get("download_headers") or jsonl_cfg.get("headers") or {}
        )
        headers = (
            {str(k): str(v) for k, v in headers_cfg.items()} if headers_cfg else None
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
            raise ValueError(str(last_error)) from last_error
        if candidates:
            raise ValueError(f"JSONL dataset not found: {candidates[-1]}")
        raise ValueError("data.jsonl.path is required when using JSONL inputs for SFT")

    def _prepare_dataset(self, spec: SFTSpecStrict) -> tuple[Dataset, str]:
        data_cfg = spec.data or {}
        dataset_name = data_cfg.get("dataset_name")
        prompt_col = data_cfg.get("prompt_column")
        response_col = data_cfg.get("response_column")

        jsonl_cfg = data_cfg.get("jsonl")
        jsonl_path = data_cfg.get("jsonl_path")
        if jsonl_cfg or jsonl_path:
            if jsonl_cfg is None:
                jsonl_cfg = {}
            else:
                jsonl_cfg = dict(jsonl_cfg)
            if jsonl_path:
                jsonl_cfg.setdefault("path", jsonl_path)

            jsonl_file = self._ensure_jsonl_local(jsonl_cfg)

            text_field = jsonl_cfg.get("text_field") or data_cfg.get("text_field")
            prompt_field = jsonl_cfg.get("prompt_field") or data_cfg.get("prompt_field")
            response_field = jsonl_cfg.get("response_field") or data_cfg.get(
                "response_field"
            )
            separator = (
                jsonl_cfg.get("separator") or data_cfg.get("separator") or "\n\n"
            )

            texts: list[str] = []

            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line_number, raw in enumerate(fh, start=1):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_number} of {jsonl_file}: {exc}"
                        ) from exc

                    if text_field and text_field in record:
                        value = record[text_field]
                    elif prompt_field and response_field:
                        if prompt_field not in record:
                            raise ValueError(
                                f"JSONL record missing prompt field '{prompt_field}' "
                                f"on line {line_number}"
                            )
                        if response_field not in record:
                            raise ValueError(
                                f"JSONL record missing response field "
                                f"'{response_field}' on line {line_number}"
                            )
                        value = (
                            f"{record[prompt_field]}{separator}{record[response_field]}"
                        )
                    elif prompt_field:
                        if prompt_field not in record:
                            raise ValueError(
                                f"JSONL record missing field '{prompt_field}' on "
                                f"line {line_number}"
                            )
                        value = record[prompt_field]
                    elif response_field:
                        if response_field not in record:
                            raise ValueError(
                                f"JSONL record missing field '{response_field}' on "
                                f"line {line_number}"
                            )
                        value = record[response_field]
                    else:
                        raise ValueError(
                            "data.jsonl must specify text_field or prompt_field/"
                            "response_field"
                        )

                    texts.append(str(value))

            if not texts:
                raise ValueError(f"JSONL dataset at {jsonl_file} is empty")

            dataset = Dataset.from_dict({"text": texts})
            max_samples = data_cfg.get("max_samples")
            if max_samples is not None:
                max_samples = int(max_samples)
                dataset = dataset.select(range(min(len(dataset), max_samples)))

            return dataset, "text"

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
                    c
                    for c in (prompt_col, response_col)
                    if c not in dataset.column_names
                ]
                if missing:
                    raise ValueError(f"Dataset missing columns {missing}")
                sep = data_cfg.get("separator", "\n\n")

                def _combine(ex):
                    return {"text": f"{ex[prompt_col]}{sep}{ex[response_col]}"}

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

        raise ValueError("spec.data must define dataset_name or prompts for SFT tasks")

    @staticmethod
    def _requested_gpu_count(spec: TaskSpecStrictBase) -> int | None:
        gpu_count = cast(
            int | None,
            (resources := spec.resources)
            and (hardware := resources.hardware)
            and (gpu := hardware.gpu)
            and gpu.count,
        )
        if gpu_count is not None and gpu_count > 0:
            return gpu_count
        return None

    @staticmethod
    def _configure_devices(training_cfg: dict[str, Any]) -> None:
        """Control CUDA_VISIBLE_DEVICES only; no model.to() here."""
        if not torch.cuda.is_available():
            return
        requested = training_cfg.get("visible_devices")
        allow_multi_cfg = training_cfg.get("allow_multi_gpu")
        try:
            n_devices = torch.cuda.device_count()
        except Exception:
            n_devices = 0
        if allow_multi_cfg is None:
            allow_multi = n_devices > 1
        else:
            allow_multi = bool(allow_multi_cfg)

        if requested:
            devices = (
                ",".join(str(x) for x in requested)
                if isinstance(requested, (list, tuple))
                else str(requested)
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            logger.info("Using user-specified CUDA_VISIBLE_DEVICES=%s", devices)
            return
        if allow_multi:
            logger.info("Multi-GPU allowed; using all visible GPUs.")
            return
        # Default to a single GPU when multiple devices are visible but not
        # explicitly allowed
        if n_devices > 1:
            preferred = training_cfg.get("primary_gpu", 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(preferred)
            logger.info(
                "Multiple GPUs detected (%d); restrict to device %s (set "
                "training.allow_multi_gpu=false to override).",
                n_devices,
                preferred,
            )

    @staticmethod
    def _resolve_deepspeed_config(training_cfg: dict[str, Any], log) -> Any | None:
        cfg = training_cfg.get("deepspeed")
        if not cfg:
            return None
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, (str, Path)):
            candidate = Path(str(cfg)).expanduser()
            if candidate.exists():
                log.info("Using DeepSpeed config at %s", candidate)
                return str(candidate)
            # Allow literal identifiers (e.g., 'auto') without file presence.
            log.info("Using DeepSpeed literal configuration '%s'", cfg)
            return str(cfg)
        raise ValueError("training.deepspeed must be a dict, path string, or falsy")

    @staticmethod
    def _build_auto_deepspeed_config(
        training_cfg: dict[str, Any], world_size: int
    ) -> dict[str, Any] | None:
        if world_size <= 1:
            return None
        if not bool(training_cfg.get("auto_deepspeed", True)):
            return None
        try:
            per_device = max(
                1,
                int(
                    training_cfg.get(
                        "batch_size", training_cfg.get("per_device_batch_size", 2)
                    )
                ),
            )
        except Exception:
            per_device = 2
        try:
            grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
        except Exception:
            grad_accum = 1
        total_batch = per_device * grad_accum * max(1, world_size)
        try:
            stage = int(
                training_cfg.get(
                    "auto_deepspeed_stage", training_cfg.get("zero_stage", 2)
                )
            )
        except Exception:
            stage = 2
        stage = max(1, min(stage, 3))
        fp16_enabled = bool(training_cfg.get("fp16", False))
        bf16_enabled = bool(training_cfg.get("bf16", False))
        return {
            "train_batch_size": total_batch,
            "train_micro_batch_size_per_gpu": per_device,
            "train_micro_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "zero_optimization": {
                "stage": stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            },
            "bf16": {"enabled": bf16_enabled},
            "fp16": {"enabled": fp16_enabled and not bf16_enabled},
            "steps_per_print": 2000,
        }
