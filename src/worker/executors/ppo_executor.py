#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Executor using TRL's simple approach

Simplified implementation using TRL's built-in training methods.
"""

import gc
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from trl.experimental.ppo.modeling_value_head import (
    AutoModelForCausalLMWithValueHead,
)
from trl.experimental.ppo.ppo_config import PPOConfig
from trl.experimental.ppo.ppo_trainer import PPOTrainer

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import PPOResult
from shared.tasks.specs import PPOSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.manifest import scratch_dir
from shared.utils.parsing import safe_float, safe_int, to_bool

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

logger = logging.getLogger("worker.ppo")


class _ExternalRewardModel(torch.nn.Module):
    """Wraps a sequence classification model to score decoded PPO responses."""

    def __init__(
        self, reward_cfg: dict[str, Any], policy_tokenizer: PreTrainedTokenizerBase
    ):
        super().__init__()
        identifier = reward_cfg.get("identifier")
        if not identifier:
            raise ValueError(
                "reward_model.identifier is required for external reward models"
            )

        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(identifier)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            identifier
        )
        self.reward_model.to("cpu")
        self.base_model_prefix = "reward_model"
        self._patch_reward_forward()
        for param in self.reward_model.parameters():
            param.requires_grad_(False)
        self.reward_model.eval()

        self.reward_type = str(reward_cfg.get("type", "classification")).lower()
        self.scale = float(reward_cfg.get("scale", 1.0))
        self.max_length = int(
            reward_cfg.get(
                "max_length",
                min(
                    getattr(self.reward_tokenizer, "model_max_length", 512) or 512, 512
                ),
            )
        )
        self.positive_label = reward_cfg.get("positive_label")
        self.negative_label = reward_cfg.get("negative_label")

        id2label = getattr(self.reward_model.config, "id2label", {}) or {}
        self._id2label = {int(k): str(v) for k, v in id2label.items()}
        self._label2id = {v.lower(): k for k, v in self._id2label.items()}

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "External reward model is only used via compute_reward_from_tokens."
        )

    def to(self, *args, **kwargs):  # type: ignore[override]
        # Keep the reward model on CPU to avoid GPU-side driver asserts with
        # classification heads.
        self.reward_model.to("cpu")
        return self

    def _patch_reward_forward(self) -> None:
        try:
            original_forward = self.reward_model.forward
        except AttributeError:
            return

        def wrapped_forward(*args, **kwargs):
            kwargs.pop("use_cache", None)
            kwargs.pop("output_hidden_states", None)
            return original_forward(*args, **kwargs)

        try:
            self.reward_model.forward = wrapped_forward  # type: ignore[assignment]
        except Exception:
            pass

        try:
            if hasattr(self.reward_model.config, "use_cache"):
                self.reward_model.config.use_cache = False
        except Exception:
            pass

        try:
            backbone = getattr(self.reward_model, "base_model", None)
            if (
                backbone is not None
                and hasattr(backbone, "config")
                and hasattr(backbone.config, "use_cache")
            ):
                backbone.config.use_cache = False
        except Exception:
            pass

    def compute_reward_from_tokens(
        self,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ):
        device = query_responses.device
        batch_size, seq_len = query_responses.shape

        response_texts = []
        for sample in query_responses:
            response_tokens = sample[context_length:]
            if pad_token_id is not None:
                response_tokens = response_tokens[response_tokens != pad_token_id]
            text = self.policy_tokenizer.decode(
                response_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            assert isinstance(text, str)
            response_texts.append(text.strip())

        reward_scores = self._score_texts(response_texts).to(device)

        reward_logits = (
            reward_scores.view(batch_size, 1, 1)
            .expand(batch_size, seq_len, 1)
            .contiguous()
        )
        non_pad = (query_responses != pad_token_id).int()
        sequence_lengths = non_pad.sum(dim=1) - 1
        sequence_lengths = torch.clamp(sequence_lengths, min=0)

        return reward_logits, reward_scores, sequence_lengths

    def _score_texts(self, texts):
        model_device = next(self.reward_model.parameters()).device
        if not texts:
            return torch.zeros(0, dtype=torch.float32, device=model_device)

        encoded = self.reward_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.reward_model(**encoded)
            logits = outputs.logits.float()

        if logits.shape[-1] == 1:
            rewards = logits.squeeze(-1)
        else:
            if self.reward_type == "sentiment":
                pos_idx = self._resolve_label(self.positive_label, "positive")
                neg_idx = self._resolve_label(self.negative_label, "negative")
                pos_score = (
                    logits[:, pos_idx]
                    if pos_idx is not None
                    else logits.max(dim=-1).values
                )
                neg_score = (
                    logits[:, neg_idx]
                    if neg_idx is not None
                    else torch.zeros_like(pos_score)
                )
                rewards = pos_score - neg_score
            else:
                label_idx = self._resolve_label(self.positive_label, None)
                rewards = (
                    logits[:, label_idx]
                    if label_idx is not None
                    else logits.max(dim=-1).values
                )

        return rewards.float().to(model_device) * self.scale

    def _resolve_label(self, label, fallback_keyword: str | None) -> int | None:
        if label is None and fallback_keyword:
            return next(
                (
                    idx
                    for idx, name in self._id2label.items()
                    if fallback_keyword in name.lower()
                ),
                None,
            )

        if label is None:
            return None

        if isinstance(label, int):
            return label if label in self._id2label else None

        key = str(label).lower()
        if key in self._label2id:
            return self._label2id[key]
        return None


@contextmanager
def _patched_reward_dispatch():
    from trl.experimental import utils as trl_utils
    from trl.experimental.ppo import ppo_trainer as trl_ppo

    original_get_reward = trl_utils.get_reward
    original_get_reward_ppo = getattr(trl_ppo, "get_reward", None)

    def _wrapped_get_reward(model, query_responses, pad_token_id, context_length):
        if hasattr(model, "compute_reward_from_tokens"):
            return model.compute_reward_from_tokens(
                query_responses, pad_token_id, context_length
            )
        return original_get_reward(model, query_responses, pad_token_id, context_length)

    trl_utils.get_reward = _wrapped_get_reward
    if original_get_reward_ppo is not None:
        trl_ppo.get_reward = _wrapped_get_reward  # type: ignore
    try:
        yield
    finally:
        trl_utils.get_reward = original_get_reward
        if original_get_reward_ppo is not None:
            trl_ppo.get_reward = original_get_reward_ppo  # type: ignore


class _PPOCollator(DataCollatorWithPadding):
    """Pad tokenized rows; pass raw string fields (e.g. ``query``) through."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}
        f0 = features[0]
        # If already tokenized, pad into tensors
        if "input_ids" in f0:
            items = []
            for f in features:
                item = {"input_ids": f["input_ids"]}
                if "attention_mask" in f:
                    item["attention_mask"] = f["attention_mask"]
                items.append(item)
            return dict(self.tokenizer.pad(items, padding=True, return_tensors="pt"))
        # Otherwise, pass raw queries through
        if "query" in f0:
            return {"query": [f["query"] for f in features]}
        # Fallback: return all fields as lists
        return {k: [f[k] for f in features] for k in f0.keys()}


class _RewardAdapter(torch.nn.Module):
    """Fallback reward adapter that reuses the policy value head."""

    def __init__(self, value_head_model: torch.nn.Module):
        super().__init__()
        self._lm = value_head_model
        head = getattr(value_head_model, "v_head", None) or getattr(
            value_head_model, "value_head", None
        )
        if head is None:
            hidden = None
            try:
                hidden = getattr(value_head_model.config, "hidden_size", None)
            except Exception:
                pass
            if hidden is None:
                try:
                    if hasattr(value_head_model, "base_model_prefix") and hasattr(
                        value_head_model,
                        cast(str, value_head_model.base_model_prefix),
                    ):
                        backbone = getattr(
                            value_head_model,
                            cast(str, value_head_model.base_model_prefix),
                        )
                        hidden = getattr(
                            getattr(backbone, "config", None), "hidden_size", None
                        )
                except Exception:
                    pass
            if hidden is None:
                raise AttributeError("Cannot infer hidden_size for reward head")
            head = torch.nn.Linear(int(hidden), 1, bias=False)
        self.v_head = head
        self.base_model_prefix = getattr(value_head_model, "base_model_prefix", "model")
        if hasattr(value_head_model, self.base_model_prefix):
            setattr(
                self,
                self.base_model_prefix,
                getattr(value_head_model, self.base_model_prefix),
            )
        for attr in ("config", "generation_config"):
            if hasattr(value_head_model, attr):
                setattr(self, attr, getattr(value_head_model, attr))

    def score(self, hidden_states):
        return self.v_head(hidden_states)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__.get("_lm", object()), name)


class _EarlyStopSignal(Exception):
    """Internal signal: KL exceeded ``target_kl``; unwind ``PPOTrainer.train``."""


class _EarlyStopPPOTrainer(PPOTrainer):
    """``PPOTrainer`` subclass that halts when ``objective/kl`` exceeds a threshold.

    TRL's PPO loop calls ``self.log(metrics)`` once per update step but
    never checks ``control.should_training_stop``, so a stock
    ``TrainerCallback`` cannot end training. Overriding ``log`` and
    raising ``_EarlyStopSignal`` instead lets the exception unwind the
    loop cleanly; the executor catches it at the ``train()`` call site.

    ``target_kl`` defaults to ``None`` so the subclass is a safe
    drop-in when early stopping is disabled.
    """

    target_kl: float | None = None

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        super().log(logs, *args, **kwargs)
        if self.target_kl is None:
            return
        kl = logs.get("objective/kl")
        if kl is None:
            return
        assert isinstance(kl, float), f"TRL logged non-float objective/kl: {kl!r}"
        if kl > self.target_kl:
            logger.info(
                "PPO early stop: objective/kl=%.4f > target_kl=%.4f", kl, self.target_kl
            )
            raise _EarlyStopSignal()


def _resolve_report_to(value: Any) -> str | list[str]:
    """Translate ``training.report_to`` into the value PPOConfig expects.

    ``None`` (YAML ``null``) and empty strings/lists disable integrations
    via TRL's ``"none"`` sentinel; strings and lists pass through.
    """
    if value is None:
        return "none"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "none"
    if isinstance(value, list):
        cleaned = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        return cleaned if cleaned else "none"
    raise ExecutionError(
        f"training.report_to must be null, str, or list[str]; got {value!r}"
    )


class PPOExecutor(TrainingMixin, Executor):
    """PPO training executor using TRL library."""

    name = "ppo_executor"
    supported_task_types = frozenset({TaskType.PPO})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_name: str | None = None
        self._policy_model: PreTrainedModel | None = None
        self._ref_model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._ppo_trainer: PPOTrainer | None = None
        self._reward_module: _ExternalRewardModel | _RewardAdapter | None = None
        self._task_out_dir: Path | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> PPOResult:
        configure_hf_library_logging()
        logger.info("Starting PPO training task")
        spec = self.require_spec(task, PPOSpecStrict)
        training_config = spec.training or {}
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = artifacts_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._task_out_dir = out_dir

        launcher_flag = "KV_PPO_DISTRIBUTED"
        already_spawned = os.environ.get(launcher_flag) == "1"
        gpu_count = self._detect_gpu_count(training_config)
        allow_multi_cfg = training_config.get("allow_multi_gpu")
        allow_multi = (
            bool(allow_multi_cfg) if allow_multi_cfg is not None else gpu_count > 1
        )

        if allow_multi and not already_spawned and gpu_count > 1:
            self._spawn_distributed(
                task, out_dir, gpu_count, launcher_flag, training_config
            )
            ipc_path = scratch_dir(out_dir) / "distributed_result.json"
            if ipc_path.exists():
                self._task_out_dir = None
                return PPOResult.model_validate(self.load_json(ipc_path))
            self._task_out_dir = None
            return PPOResult(
                spawned_torchrun=True,
                model_name=spec.model_name,
                output_dir=out_dir.as_posix(),
            )

        start_time = time.time()

        final_model_path: Path | None = None
        final_archive_path: Path | None = None

        try:
            # Get model configuration
            model_name = spec.model_name or "microsoft/DialoGPT-small"
            self._model_name = model_name

            logger.info("Loading model and tokenizer...")

            # Load tokenizer + models (PPO requires policy + reference model)
            torch_dtype = pick_torch_dtype(training_config)
            tok_kwargs, model_kwargs = build_hf_load_kwargs(
                revision=spec.model_revision,
                trust_remote_code=spec.model_trust_remote_code,
                training_cfg=training_config,
                torch_dtype=torch_dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            self._tokenizer = tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name, **model_kwargs
            )
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name, **model_kwargs
            )
            self._policy_model = model
            self._ref_model = ref_model

            # Some TRL versions expect `generation_config` on the policy/ref models.
            # AutoModelForCausalLMWithValueHead may not set it; ensure presence.
            def _ensure_generation_config(m):
                if not hasattr(m, "generation_config") or m.generation_config is None:
                    try:
                        gen_cfg = GenerationConfig.from_pretrained(model_name)
                    except Exception:
                        try:
                            gen_cfg = GenerationConfig.from_model_config(
                                getattr(m, "config")
                            )
                        except Exception:
                            gen_cfg = GenerationConfig()
                    try:
                        m.generation_config = gen_cfg
                    except Exception:
                        # As a fallback, set eos_token_id on config if available
                        if hasattr(m, "config") and hasattr(gen_cfg, "eos_token_id"):
                            m.config.eos_token_id = getattr(
                                gen_cfg, "eos_token_id", None
                            )

            _ensure_generation_config(model)
            _ensure_generation_config(ref_model)

            # Ensure TRL's PolicyAndValueWrapper can locate the backbone via
            # `<model>.base_model_prefix` attribute and a matching attribute on
            # the instance.
            def _ensure_backbone(m):
                # Common attribute names for the wrapped transformer backbone
                candidates = [
                    getattr(m, "pretrained_model", None),
                    getattr(m, "transformer", None),
                    getattr(m, "model", None),
                    getattr(m, "base_model", None),
                ]
                backbone = next((c for c in candidates if c is not None), None)
                if backbone is None:
                    return
                # Prefer the underlying module's own base_model_prefix if present
                prefix = getattr(backbone, "base_model_prefix", None) or "model"
                try:
                    setattr(m, prefix, backbone)
                    m.base_model_prefix = prefix
                except Exception:
                    # Fallback to a neutral attribute name
                    setattr(m, "backbone", backbone)
                    m.base_model_prefix = "backbone"

            _ensure_backbone(model)
            _ensure_backbone(ref_model)

            # Ensure flag expected by TRL's wrapper exists on policy/value models
            def _ensure_grad_ckpt_flag(m):
                try:
                    if hasattr(m, "is_gradient_checkpointing"):
                        return
                    # Derive from backbone or config if possible
                    val = False
                    try:
                        backbone = None
                        if hasattr(m, "base_model_prefix") and hasattr(
                            m, m.base_model_prefix
                        ):
                            backbone = getattr(m, m.base_model_prefix)
                        if backbone is not None and hasattr(
                            backbone, "is_gradient_checkpointing"
                        ):
                            val = bool(getattr(backbone, "is_gradient_checkpointing"))
                        elif hasattr(m, "config") and hasattr(
                            m.config, "gradient_checkpointing"
                        ):
                            val = bool(getattr(m.config, "gradient_checkpointing"))
                    except Exception:
                        pass
                    setattr(m, "is_gradient_checkpointing", val)
                except Exception:
                    pass

            _ensure_grad_ckpt_flag(model)
            _ensure_grad_ckpt_flag(ref_model)

            # As a last resort, monkey-patch forward to always expose `.logits`
            try:

                def _patch_forward_returns_logits(m):
                    orig_forward = m.forward

                    def wrapped_forward(*args, **kwargs):
                        # Normalize calls to avoid duplicate bindings of input_ids.
                        # Extract potential input_ids/attention_mask from positional[0]
                        # if present.
                        new_kwargs = dict(kwargs)
                        if len(args) > 0:
                            first = args[0]
                            if isinstance(first, torch.Tensor):
                                new_kwargs["input_ids"] = first
                            elif isinstance(first, dict):
                                if "input_ids" in first:
                                    new_kwargs["input_ids"] = first["input_ids"]
                                if (
                                    "attention_mask" in first
                                    and "attention_mask" not in new_kwargs
                                ):
                                    new_kwargs["attention_mask"] = first[
                                        "attention_mask"
                                    ]
                        # Ignore all positional args; call forward with kwargs only.
                        out = orig_forward(**new_kwargs)
                        if isinstance(out, tuple):
                            return SimpleNamespace(logits=out[0])
                        return out

                    m.forward = wrapped_forward.__get__(m, m.__class__)

                _patch_forward_returns_logits(model)
                _patch_forward_returns_logits(ref_model)
            except Exception:
                pass

            logger.info("Models loaded: %s", self._model_name)

            # Ensure value head exposes score() for TRL reward helpers
            self._ensure_value_head_score(model, ref_model)

            # Load dataset
            logger.info("Loading dataset...")
            dataset = self._load_dataset(spec)
            dataset_size = len(dataset)
            logger.info("Dataset loaded with %d samples", dataset_size)

            per_device_batch = safe_int(
                training_config.get("per_device_train_batch_size"), minimum=1
            )
            if per_device_batch is None:
                per_device_batch = safe_int(
                    training_config.get("batch_size"), default=1, minimum=1
                )
            grad_acc_steps: int | None = safe_int(
                training_config.get("gradient_accumulation_steps"), default=1, minimum=1
            )
            num_mini_batches: int | None = safe_int(
                training_config.get("num_mini_batches"), default=1, minimum=1
            )
            per_device_batch, grad_acc_steps = self._normalize_ppo_batch_settings(
                dataset_size,
                per_device_batch,
                grad_acc_steps,
            )
            num_mini_batches = self._normalize_ppo_num_mini_batches(
                per_device_batch,
                grad_acc_steps,
                num_mini_batches,
            )

            # Some TRL versions expect tokenized inputs in the dataset and will
            # route through a padding collator. Ensure input_ids/attention_mask exist.
            try:
                max_len = int(training_config.get("max_seq_length", 512))

                def _tok_fn(batch):
                    texts = batch.get("query") or []
                    enc = tokenizer(
                        texts,
                        padding=False,
                        truncation=True,
                        max_length=max_len,
                    )
                    return enc

                if "input_ids" not in dataset.column_names:
                    dataset = dataset.map(_tok_fn, batched=True, remove_columns=[])
                    logger.info("Tokenized dataset with max_length=%d", max_len)
            except Exception as _e:
                logger.warning("Skipping dataset tokenization step: %s", _e)

            reward_module, reward_is_external, reward_ctx = self._prepare_reward_model(
                spec,
                tokenizer,
                model,
                ref_model,
            )
            self._reward_module = reward_module
            if hasattr(reward_module, "eval"):
                reward_module.eval()

            logger.info("Creating PPOConfig...")
            response_cfg = spec.generation or {}
            ppo_config = self._build_ppo_config(
                training_config,
                response_cfg,
                checkpoint_dir,
                per_device_batch=per_device_batch,
                grad_acc_steps=grad_acc_steps,
                num_mini_batches=num_mini_batches,
                dataset_size=dataset_size,
            )
            logger.info("PPOConfig created successfully")

            logger.info("Creating PPOTrainer...")
            ppo_trainer = _EarlyStopPPOTrainer(
                args=ppo_config,
                processing_class=tokenizer,
                model=model,
                ref_model=ref_model,
                reward_model=reward_module,
                value_model=model,
                train_dataset=dataset,
                eval_dataset=dataset,
                data_collator=_PPOCollator(tokenizer),
            )
            self._ppo_trainer = ppo_trainer
            self._install_trainer_save_overrides(ppo_trainer)
            logger.info("PPOTrainer created successfully")

            if reward_is_external:
                logger.info("External reward model enabled for PPO training")

            self._install_kl_early_stopping(ppo_trainer, training_config)

            logger.info("Starting PPO training...")
            try:
                with reward_ctx:
                    ppo_trainer.train()
            except _EarlyStopSignal:
                pass
            logger.info("PPO training completed")

            ok = True
            error_msg = None

            # Save model if requested
            if training_config.get("save_model", True):
                try:
                    logger.info("Saving trained model...")
                    model_save_path = checkpoint_dir / "final_model"
                    # Prefer save_model to avoid safetensors shared-tensor errors
                    ppo_trainer.save_model(model_save_path.as_posix())
                    logger.info("Model saved to: %s", model_save_path)
                    final_model_path = model_save_path
                    destination = get_http_destination(task.spec)
                    if destination:
                        try:
                            final_archive_path = archive_model_dir(model_save_path)
                            logger.info(
                                "Archived PPO model to %s for HTTP delivery",
                                final_archive_path,
                            )
                        except Exception as arch_exc:
                            logger.warning(
                                "Failed to archive PPO model for upload: %s", arch_exc
                            )
                except Exception as exc:
                    logger.warning("Failed to save model: %s", exc)

        except Exception as exc:
            ok = False
            error_msg = str(exc)
            logger.exception("PPO training failed: %s", exc)
            training_time = time.time() - start_time
            dataset_size = len(dataset) if "dataset" in locals() else 0  # type: ignore
            result = PPOResult(
                ok=ok,
                training_time_seconds=training_time,
                error_message=error_msg,
                model_name=self._model_name,
                dataset_size=dataset_size,
                output_dir=out_dir.as_posix(),
            )
            write_executor_result(
                out_dir / "results.json", task.task_id, task.spec, result
            )
            self._task_out_dir = None
            raise ExecutionError(error_msg or "PPO training failed") from exc

        training_time = time.time() - start_time

        result = PPOResult(
            ok=ok,
            training_time_seconds=training_time,
            error_message=error_msg,
            model_name=self._model_name,
            dataset_size=len(dataset),
            output_dir=out_dir.as_posix(),
            checkpoints_dir=ArtifactRef(path="checkpoints"),
            final_model=(
                ArtifactRef(path=final_model_path.relative_to(artifacts_dir).as_posix())
                if final_model_path
                else None
            ),
            final_model_archive=(
                ArtifactRef(
                    path=final_archive_path.relative_to(artifacts_dir).as_posix()
                )
                if final_archive_path
                else None
            ),
        )

        maybe_upload_artifacts(task, out_dir, logger=logger)

        self._cleanup_local_artifacts(
            task,
            checkpoint_dir,
            final_model_path,
            final_archive_path,
        )

        logger.info("PPO training task completed in %.2f seconds", training_time)
        self._task_out_dir = None
        return result

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
                jsonl_cfg["path"] = resolved.as_posix()
                return resolved
            except ExecutionError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise ExecutionError(str(last_error)) from last_error
        if candidates:
            raise ExecutionError(f"JSONL dataset not found: {candidates[-1]}")
        raise ExecutionError("data.jsonl.path is required when using JSONL input")

    def _load_dataset(self, spec: PPOSpecStrict) -> Dataset:
        """Load training dataset"""
        data_config = spec.data or {}

        jsonl_cfg = data_config.get("jsonl")
        jsonl_path = data_config.get("jsonl_path")
        if jsonl_cfg or jsonl_path:
            if jsonl_cfg is None:
                jsonl_cfg = {}
            else:
                jsonl_cfg = dict(jsonl_cfg)
            if jsonl_path:
                jsonl_cfg.setdefault("path", jsonl_path)

            jsonl_file = self._ensure_jsonl_local(jsonl_cfg)

            query_field = (
                jsonl_cfg.get("query_field")
                or data_config.get("query_field")
                or "query"
            )
            response_field = jsonl_cfg.get("response_field") or data_config.get(
                "response_field"
            )

            prompts: list[str] = []
            references: list[str] | None = [] if response_field else None

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

                    if query_field not in record:
                        raise ExecutionError(
                            f"JSONL record missing required field '{query_field}' on "
                            f"line {line_number}"
                        )
                    prompts.append(str(record[query_field]))
                    if references is not None:
                        references.append(str(record.get(response_field, "")))

            if not prompts:
                raise ExecutionError(f"JSONL dataset at {jsonl_file} is empty")

            dataset_dict: dict[str, Any] = {"query": prompts}
            if references is not None:
                dataset_dict["reference_response"] = references
            dataset = Dataset.from_dict(dataset_dict)

        elif "prompts" in data_config:
            # Use provided prompts directly
            prompts = data_config["prompts"]
            # Convert to the format expected by TRL PPO
            dataset_dict = {"query": prompts}
            dataset = Dataset.from_dict(dataset_dict)
        else:
            # Default demo dataset
            prompts = [
                "Write a positive review for a restaurant:",
                "Describe a beautiful sunset:",
                "Tell me about a helpful AI assistant:",
                "Write a motivational quote:",
            ]
            dataset_dict = {"query": prompts}
            dataset = Dataset.from_dict(dataset_dict)

        return dataset

    def _prepare_reward_model(
        self,
        spec: PPOSpecStrict,
        tokenizer: PreTrainedTokenizerBase,
        value_model: AutoModelForCausalLMWithValueHead,
        ref_model: AutoModelForCausalLMWithValueHead,
    ):
        reward_cfg = spec.reward_model or {}
        if reward_cfg:
            identifier = reward_cfg.get("identifier")
            try:
                external_model = _ExternalRewardModel(reward_cfg, tokenizer)
                external_model.base_model_prefix = "reward_model"
                logger.info(
                    "Using external reward model '%s' (type=%s)",
                    identifier,
                    external_model.reward_type,
                )
                return external_model, True, _patched_reward_dispatch()
            except Exception as exc:
                logger.warning(
                    "Failed to initialize reward model '%s' (type=%s); falling back "
                    "to value-head rewards. Error: %s",
                    identifier,
                    reward_cfg.get("type"),
                    exc,
                )
        reward_adapter = _RewardAdapter(value_model)
        self._ensure_value_head_score(value_model, ref_model)
        return reward_adapter, False, nullcontext()

    def _build_ppo_config(
        self,
        training_config: dict[str, Any],
        response_cfg: dict[str, Any],
        checkpoint_dir: Path,
        per_device_batch: int | None,
        grad_acc_steps: int | None,
        num_mini_batches: int | None,
        dataset_size: int,
    ) -> PPOConfig:
        learning_rate = safe_float(
            training_config.get("learning_rate"), default=1.41e-5, minimum=0
        )
        batch_size = safe_int(
            training_config.get("batch_size"), default=per_device_batch or 1, minimum=1
        )
        mini_batch_size = safe_int(
            training_config.get("mini_batch_size"),
            default=num_mini_batches or 1,
            minimum=1,
        )
        seed = safe_int(training_config.get("seed"), default=42, minimum=0)
        ppo_epochs = safe_int(training_config.get("ppo_epochs"), minimum=1)
        num_train_epochs = safe_float(
            training_config.get("num_train_epochs"), default=1.0, minimum=1.0
        )
        kl_coef = safe_float(training_config.get("kl_coef"), minimum=0)

        max_seq_length = safe_int(
            training_config.get("max_seq_length"), default=64, minimum=1
        )
        response_length = safe_int(
            response_cfg.get("max_new_tokens"),
            default=max_seq_length,
            minimum=1,
        )
        if response_length is None:
            response_length = max_seq_length

        temperature = safe_float(response_cfg.get("temperature"), minimum=0)
        if temperature is None:
            temperature = safe_float(training_config.get("temperature"), minimum=0)
        if temperature == 0:
            logger.warning(
                "Non-positive temperature is capped to 0, which may cause issues "
                "during PPO training; consider using a small positive value instead."
            )

        save_strategy = str(training_config.get("save_strategy", "steps")).lower()

        save_steps = training_config.get("save_steps")
        if save_steps is None:
            save_steps = training_config.get("save_freq")
        save_steps = safe_int(save_steps, minimum=1)
        save_total_limit = training_config.get("save_total_limit")
        save_total_limit = safe_int(save_total_limit, minimum=1)
        save_only_model = training_config.get("save_only_model")

        steps_requested = safe_int(training_config.get("steps"), minimum=1)
        total_episodes = None
        if steps_requested is not None and per_device_batch:
            total_episodes = steps_requested * max(1, per_device_batch)

        ppo_ctor_kwargs: dict[str, Any] = {}
        if per_device_batch is not None:
            ppo_ctor_kwargs["per_device_train_batch_size"] = per_device_batch
        if grad_acc_steps is not None:
            ppo_ctor_kwargs["gradient_accumulation_steps"] = grad_acc_steps
        if num_mini_batches is not None:
            ppo_ctor_kwargs["num_mini_batches"] = num_mini_batches
        if total_episodes is not None:
            ppo_ctor_kwargs["total_episodes"] = total_episodes
        if ppo_epochs is not None:
            ppo_ctor_kwargs["num_ppo_epochs"] = ppo_epochs
        if kl_coef is not None:
            ppo_ctor_kwargs["kl_coef"] = kl_coef
        if temperature is not None:
            ppo_ctor_kwargs["temperature"] = temperature
        if save_steps is not None:
            ppo_ctor_kwargs["save_steps"] = save_steps
        if save_total_limit is not None:
            ppo_ctor_kwargs["save_total_limit"] = save_total_limit
        if save_only_model is not None:
            ppo_ctor_kwargs["save_only_model"] = bool(save_only_model)

        if "report_to" in training_config:
            ppo_ctor_kwargs["report_to"] = _resolve_report_to(
                training_config["report_to"]
            )
        project = training_config.get("project")
        if isinstance(project, str) and project:
            ppo_ctor_kwargs["project"] = project

        ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            output_dir=checkpoint_dir.as_posix(),
            seed=seed,
            num_train_epochs=num_train_epochs,
            response_length=response_length,
            save_strategy=save_strategy,
            remove_unused_columns=False,
            fp16=bool(training_config.get("fp16", False)),
            bf16=bool(training_config.get("bf16", False)),
            **ppo_ctor_kwargs,
        )

        stop_token = response_cfg.get("stop")
        if isinstance(stop_token, str):
            ppo_config.stop_token = stop_token  # type: ignore[assignment]

        logger.info(
            "Final PPO batch parameters: per_device=%s, grad_acc=%s, "
            "num_mini_batches=%s",
            per_device_batch,
            grad_acc_steps,
            num_mini_batches,
        )

        if total_episodes is not None:
            logger.info(
                "Configuring PPO to run %d update steps (~%d episodes)",
                steps_requested,
                total_episodes,
            )
        else:
            logger.info(
                "Using num_train_epochs=%.2f over %d samples "
                "(per_device_batch=%s, grad_acc=%s)",
                num_train_epochs,
                dataset_size,
                per_device_batch,
                grad_acc_steps,
            )

        return ppo_config

    @staticmethod
    def _ppo_world_size() -> int:
        world_size_raw = os.environ.get("WORLD_SIZE")
        if world_size_raw:
            try:
                world_size = int(world_size_raw)
                if world_size > 0:
                    return world_size
            except ValueError:
                pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                if world_size > 0:
                    return world_size
        except Exception:
            pass
        return 1

    def _normalize_ppo_batch_settings(
        self,
        dataset_size: int,
        per_device_batch: int | None,
        grad_acc_steps: int | None,
    ) -> tuple[int | None, int | None]:
        """Ensure PPO batch settings are compatible with dataset size and world size,
        adjusting if necessary."""
        if dataset_size <= 0 or per_device_batch is None or grad_acc_steps is None:
            return per_device_batch, grad_acc_steps

        world_size = self._ppo_world_size()
        max_local_batch = dataset_size // world_size
        if max_local_batch < 1:
            raise ExecutionError(
                "PPO dataset is too small for distributed training: "
                f"{dataset_size} samples for world_size={world_size}. "
                "TRL PPO requires at least one full local batch per rank."
            )

        local_batch_size = per_device_batch * grad_acc_steps
        if local_batch_size <= max_local_batch:
            return per_device_batch, grad_acc_steps

        original_per_device = per_device_batch
        original_grad_acc = grad_acc_steps

        max_grad_acc_steps = max_local_batch // per_device_batch
        if max_grad_acc_steps >= 1:
            grad_acc_steps = max_grad_acc_steps
        else:
            per_device_batch = max_local_batch
            grad_acc_steps = 1

        logger.warning(
            "Clipping PPO batch settings from per_device=%d, grad_acc=%d "
            "(local_batch=%d) to per_device=%d, grad_acc=%d "
            "for dataset_size=%d and world_size=%d. "
            "TRL PPO uses drop_last=True and requires at least one full "
            "local batch per rank.",
            original_per_device,
            original_grad_acc,
            local_batch_size,
            per_device_batch,
            grad_acc_steps,
            dataset_size,
            world_size,
        )
        return per_device_batch, grad_acc_steps

    @staticmethod
    def _normalize_ppo_num_mini_batches(
        per_device_batch: int | None,
        grad_acc_steps: int | None,
        num_mini_batches: int | None,
    ) -> int | None:
        """Ensure num_mini_batches is compatible with local batch size, adjusting if
        necessary."""
        if (
            per_device_batch is None
            or grad_acc_steps is None
            or num_mini_batches is None
            or num_mini_batches < 1
        ):
            return num_mini_batches

        local_batch_size = per_device_batch * grad_acc_steps
        adjusted = min(num_mini_batches, local_batch_size)
        while adjusted > 1 and local_batch_size % adjusted != 0:
            adjusted -= 1
        if adjusted < 1:
            adjusted = 1
        if adjusted != num_mini_batches:
            logger.warning(
                "Adjusting PPO num_mini_batches from %d to %d so local_batch=%d "
                "divides evenly.",
                num_mini_batches,
                adjusted,
                local_batch_size,
            )
        return adjusted

    @staticmethod
    def _ensure_value_head_score(
        value_model: AutoModelForCausalLMWithValueHead,
        ref_model: AutoModelForCausalLMWithValueHead | None,
    ) -> None:
        import types

        def _score_impl(self, hidden_states):
            head = getattr(self, "v_head", None) or getattr(self, "value_head", None)
            if head is None:
                raise AttributeError("Model lacks v_head/value_head for reward scoring")
            return head(hidden_states)

        for target in (value_model, ref_model):
            if target is None:
                continue
            if not hasattr(target, "score"):
                try:
                    target.score = types.MethodType(_score_impl, target)
                except Exception:
                    pass

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
            "Launching torchrun for PPO (nproc=%d, CUDA_VISIBLE_DEVICES=%s)",
            nproc,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        run_torchrun(
            nproc_per_node=nproc,
            module="worker.executors.ppo_dist_entry",
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

    @staticmethod
    def _resolve_model_for_save(model: Any) -> Any:
        """Return the policy model that should be serialized."""
        policy_model = getattr(model, "policy", None)
        if policy_model is not None:
            return policy_model

        module = getattr(model, "module", None)
        if module is not None:
            policy_model = getattr(module, "policy", None)
            if policy_model is not None:
                return policy_model

        return model

    def _install_kl_early_stopping(
        self, ppo_trainer: _EarlyStopPPOTrainer, training_config: dict[str, Any]
    ) -> None:
        """Set ``target_kl`` on the trainer when ``training.early_stopping`` is on.

        The trainer is already an ``_EarlyStopPPOTrainer``; we just stamp the
        threshold so its ``log`` override starts watching KL.
        Enforces that ``early_stopping=True`` is paired with a positive
        ``target_kl``; if ``early_stopping`` is off, ``target_kl`` is ignored
        but logged so users notice the gap.
        """
        enabled = to_bool(training_config.get("early_stopping"), default=False)
        target_kl = safe_float(training_config.get("target_kl"))
        if not enabled:
            if target_kl is not None and target_kl > 0:
                logger.info(
                    "PPO training.target_kl=%.4f set without early_stopping=true; "
                    "no early-stop hook attached",
                    target_kl,
                )
            return
        if target_kl is None or target_kl <= 0:
            raise ExecutionError(
                "training.early_stopping requires a positive training.target_kl"
            )
        ppo_trainer.target_kl = target_kl
        logger.info("PPO KL early-stop enabled at target_kl=%.4f", target_kl)

    def _install_trainer_save_overrides(self, ppo_trainer: PPOTrainer) -> None:
        """Patch PPO trainer saves to avoid TRL's DDP-unsafe checkpoint wrapper.

        TRL's PPO checkpoint path assumes ``self.model`` exposes policy/config
        attributes directly. Under DDP, ``self.model`` is wrapped, so that path
        can fail on rank 0 while other ranks continue into checkpoint
        collectives, hanging the run.
        """

        def _wrapped_save_model(
            output_dir: str | None = None, _internal_call: bool = False
        ) -> None:
            backup_model = ppo_trainer.model
            backup_deepspeed: Any = None
            ppo_trainer.model = self._resolve_model_for_save(backup_model)
            if ppo_trainer.is_deepspeed_enabled:
                backup_deepspeed = ppo_trainer.deepspeed
                ppo_trainer.deepspeed = ppo_trainer.model  # type: ignore[assignment]
            try:
                Trainer.save_model(ppo_trainer, output_dir, _internal_call)
            finally:
                ppo_trainer.model = backup_model
                if ppo_trainer.is_deepspeed_enabled:
                    ppo_trainer.deepspeed = backup_deepspeed  # type: ignore[assignment]

        def _wrapped_save_checkpoint(model: Any, trial: Any) -> None:
            Trainer._save_checkpoint(ppo_trainer, model, trial)

        setattr(ppo_trainer, "save_model", _wrapped_save_model)
        setattr(ppo_trainer, "_save_checkpoint", _wrapped_save_checkpoint)

    def cleanup_after_run(self) -> None:
        dropped_objects = []
        for attr in (
            "_ppo_trainer",
            "_policy_model",
            "_ref_model",
            "_tokenizer",
            "_reward_module",
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
