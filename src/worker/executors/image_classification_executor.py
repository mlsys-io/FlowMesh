#!/usr/bin/env python3
"""Image classification executor powered by Hugging Face Transformers.

Loads ``AutoModelForImageClassification`` + ``AutoImageProcessor``, prepares
the dataset via the ``datasets`` library, and trains with the standard
``transformers.Trainer``. Single-GPU runs execute in-process; multi-GPU
support can be added later by spawning ``image_classification_dist_entry``
through ``run_torchrun`` / ``run_deepspeed`` in the same way ``SFTExecutor``
does.
"""

import gc
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import ClassLabel, Dataset, load_dataset
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import ImageClassificationTrainingSpecStrict

from ..utils.logging import configure_hf_library_logging
from .base_executor import ExecutionError, Executor, ExecutorTask
from .mixins.training import TrainingMixin
from .utils.checkpoints import (
    archive_model_dir,
    determine_resume_path,
    get_http_destination,
    maybe_upload_artifacts,
    write_executor_result,
)
from .utils.huggingface import pick_torch_dtype

logger = logging.getLogger("worker.image_classification")


class ImageClassificationTrainingResult(BaseExecutorResult):
    training_time_seconds: float | None = None
    error_message: str | None = None
    model_name: str | None = None
    dataset_size: int = 0
    num_labels: int = 0
    eval_accuracy: float | None = None
    train_losses: list[float] = []
    output_dir: str | None = None
    checkpoints_dir: ArtifactRef | None = None
    resume_from_path: str | None = None
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None


class ImageClassificationTrainingExecutor(TrainingMixin, Executor):
    name = "image_classification_training_executor"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_name: str | None = None
        self._current_model: Any | None = None
        self._current_trainer: Any | None = None
        self._current_processor: Any | None = None
        self._final_model_dir: Path | None = None

    def run(
        self, task: ExecutorTask, out_dir: Path
    ) -> ImageClassificationTrainingResult:
        configure_hf_library_logging()
        spec = self.require_spec(task, ImageClassificationTrainingSpecStrict)
        start_time = time.time()
        ok = False
        error_msg: str | None = None
        caught_exc: Exception | None = None

        training_cfg = spec.training.copy() if spec.training else {}
        if bool(training_cfg.get("allow_multi_gpu", False)):
            logger.warning(
                "ImageClassificationTrainingExecutor currently runs on a single GPU; "
                "ignoring allow_multi_gpu request"
            )
            training_cfg["allow_multi_gpu"] = False

        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = artifacts_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        resume_path: Path | None = None
        train_dataset: Dataset | None = None
        eval_dataset: Dataset | None = None
        try:
            model_name = spec.model_name
            if not model_name:
                raise ExecutionError(
                    "image_classification_training requires model.source.identifier"
                )
            self._model_name = model_name

            resume_path = determine_resume_path(
                spec, training_cfg, out_dir, logger=logger
            )
            resume_str = resume_path.as_posix() if resume_path else None

            (
                train_dataset,
                eval_dataset,
                label2id,
                id2label,
                image_field,
                label_field,
                label_to_id,
            ) = self._prepare_dataset(spec)
            num_labels = len(id2label)
            logger.info(
                "Loaded training dataset with %d rows; %d labels",
                len(train_dataset),
                num_labels,
            )

            if resume_path:
                logger.info(
                    "Loading processor and model from local checkpoint: %s",
                    resume_path,
                )
            else:
                logger.info(
                    "Loading processor and model from identifier: %s", model_name
                )

            processor = AutoImageProcessor.from_pretrained(
                resume_str or model_name,
                trust_remote_code=spec.model_trust_remote_code,
            )
            self._current_processor = processor

            torch_dtype = pick_torch_dtype(training_cfg)
            model_kwargs: dict[str, Any] = {
                "num_labels": num_labels,
                "id2label": id2label,
                "label2id": label2id,
                "ignore_mismatched_sizes": True,
                "trust_remote_code": spec.model_trust_remote_code,
            }
            if spec.model_revision and not resume_str:
                model_kwargs["revision"] = spec.model_revision
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            model = AutoModelForImageClassification.from_pretrained(
                resume_str or model_name, **model_kwargs
            )
            if bool(training_cfg.get("gradient_checkpointing", False)):
                model.gradient_checkpointing_enable()
            self._current_model = model

            augment = bool(training_cfg.get("augmentation", False))
            train_dataset = self._apply_transform(
                train_dataset,
                processor,
                image_field,
                label_field,
                label_to_id,
                augment=augment,
            )
            if eval_dataset is not None:
                eval_dataset = self._apply_transform(
                    eval_dataset,
                    processor,
                    image_field,
                    label_field,
                    label_to_id,
                    augment=False,
                )

            eval_strategy = "epoch" if eval_dataset is not None else "no"
            training_args = TrainingArguments(
                output_dir=checkpoint_dir.as_posix(),
                num_train_epochs=float(training_cfg.get("num_train_epochs", 3.0)),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 16)),
                per_device_eval_batch_size=int(
                    training_cfg.get(
                        "eval_batch_size", training_cfg.get("batch_size", 16)
                    )
                ),
                gradient_accumulation_steps=int(
                    training_cfg.get("gradient_accumulation_steps", 1)
                ),
                learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
                optim=_resolve_optim(training_cfg.get("optimizer")),
                weight_decay=float(training_cfg.get("weight_decay", 0.0)),
                warmup_steps=int(training_cfg.get("warmup_steps", 0)),
                warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_strategy=str(training_cfg.get("save_strategy", "steps")),
                eval_strategy=eval_strategy,
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                gradient_checkpointing=bool(
                    training_cfg.get("gradient_checkpointing", False)
                ),
                remove_unused_columns=False,
                report_to=[],
                label_names=["labels"],
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=_image_collate,
                processing_class=processor,
                compute_metrics=_compute_accuracy if eval_dataset is not None else None,
            )
            self._current_trainer = trainer

            logger.info(
                "Starting image classification training "
                "(rows=%d, labels=%d, epochs=%s)",
                len(train_dataset),
                num_labels,
                training_args.num_train_epochs,
            )
            trainer.train(resume_from_checkpoint=resume_str)
            ok = True
            logger.info("Training finished")

            train_losses = [
                round(float(entry["loss"]), 6)
                for entry in trainer.state.log_history
                if "loss" in entry
            ]

            eval_accuracy: float | None = None
            if eval_dataset is not None:
                metrics = trainer.evaluate()
                acc = metrics.get("eval_accuracy")
                eval_accuracy = float(acc) if acc is not None else None
                logger.info("Eval metrics: %s", metrics)

            final_model_path: Path | None = None
            final_archive_path: Path | None = None
            if bool(training_cfg.get("save_model", True)):
                model_path = artifacts_dir / "final_model"
                trainer.save_model(model_path.as_posix())
                processor.save_pretrained(model_path)
                final_model_path = model_path
                logger.info("Saved fine-tuned model to %s", model_path)
                if get_http_destination(spec):
                    final_archive_path = archive_model_dir(model_path)
                    logger.info(
                        "Archived fine-tuned model to %s for HTTP delivery",
                        final_archive_path,
                    )

            training_time = time.time() - start_time
            final_model: ArtifactRef | None = None
            final_model_archive: ArtifactRef | None = None
            if final_model_path:
                self._final_model_dir = (
                    final_model_path if final_model_path.exists() else None
                )
                final_model = ArtifactRef(
                    path=final_model_path.relative_to(artifacts_dir).as_posix()
                )
                if final_archive_path:
                    final_model_archive = ArtifactRef(
                        path=final_archive_path.relative_to(artifacts_dir).as_posix()
                    )

            result = ImageClassificationTrainingResult(
                ok=ok,
                training_time_seconds=training_time,
                error_message=error_msg,
                model_name=self._model_name,
                dataset_size=len(train_dataset),
                num_labels=num_labels,
                eval_accuracy=eval_accuracy,
                train_losses=train_losses,
                output_dir=out_dir.as_posix(),
                checkpoints_dir=ArtifactRef(path="checkpoints"),
                resume_from_path=resume_str,
                final_model=final_model,
                final_model_archive=final_model_archive,
            )

            maybe_upload_artifacts(task, out_dir, logger=logger)
            self._cleanup_local_artifacts(
                task,
                checkpoint_dir,
                final_model_path,
                final_archive_path,
            )

            self._current_trainer = None
            self._current_model = None
            self._current_processor = None
            self._final_model_dir = None

            return result

        except Exception as exc:
            error_msg = str(exc)
            ok = False
            caught_exc = exc
            logger.exception("Image classification training failed: %s", exc)
            self._current_trainer = None
            self._current_model = None
            self._current_processor = None
            self._final_model_dir = None

        training_time = time.time() - start_time
        rows = len(train_dataset) if train_dataset is not None else 0
        result = ImageClassificationTrainingResult(
            ok=ok,
            training_time_seconds=training_time,
            error_message=error_msg,
            model_name=self._model_name,
            dataset_size=rows,
            output_dir=out_dir.as_posix(),
            checkpoints_dir=ArtifactRef(path="checkpoints"),
            resume_from_path=resume_path.as_posix() if resume_path else None,
        )
        write_executor_result(out_dir / "results.json", task.task_id, task.spec, result)
        if caught_exc is not None:
            raise ExecutionError(
                error_msg or "Image classification training failed"
            ) from caught_exc
        raise ExecutionError(error_msg or "Image classification training failed")

    def cleanup_after_run(self) -> None:
        for attr in (
            "_current_trainer",
            "_current_model",
            "_current_processor",
            "_final_model_dir",
        ):
            setattr(self, attr, None)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to empty CUDA cache", exc_info=True)
        gc.collect()

    def _prepare_dataset(self, spec: ImageClassificationTrainingSpecStrict) -> tuple[
        Dataset,
        Dataset | None,
        dict[str, int],
        dict[int, str],
        str,
        str,
        Callable[[Any], int],
    ]:
        data_cfg = spec.data or {}
        dataset_name = data_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError(
                "spec.data.dataset_name is required for image classification"
            )

        split = data_cfg.get("split", "train")
        eval_split = data_cfg.get("eval_split")
        config_name = data_cfg.get("config_name")
        revision = data_cfg.get("revision")
        trust_remote_code = data_cfg.get("trust_remote_code")
        image_field = data_cfg.get("image_column", "image")
        label_field = data_cfg.get("label_column", "label")

        load_kwargs: dict[str, Any] = {"split": split}
        if revision is not None:
            load_kwargs["revision"] = revision
        if trust_remote_code is not None:
            load_kwargs["trust_remote_code"] = bool(trust_remote_code)

        train_dataset = load_dataset(dataset_name, config_name, **load_kwargs)

        missing = [
            c for c in (image_field, label_field) if c not in train_dataset.column_names
        ]
        if missing:
            raise ValueError(
                f"Dataset missing required columns {missing}; "
                f"available: {train_dataset.column_names}"
            )

        max_samples = data_cfg.get("max_samples")
        if max_samples:
            max_samples = int(max_samples)
            train_dataset = train_dataset.select(
                range(min(len(train_dataset), max_samples))
            )

        eval_dataset: Dataset | None = None
        if eval_split:
            eval_kwargs = dict(load_kwargs)
            eval_kwargs["split"] = eval_split
            eval_dataset = load_dataset(dataset_name, config_name, **eval_kwargs)
            if max_eval := (data_cfg.get("max_eval_samples") or max_samples):
                max_eval = int(max_eval)
                eval_dataset = eval_dataset.select(
                    range(min(len(eval_dataset), max_eval))
                )

        label2id, id2label, label_to_id = self._resolve_label_maps(
            train_dataset, data_cfg, label_field
        )
        return (
            train_dataset,
            eval_dataset,
            label2id,
            id2label,
            image_field,
            label_field,
            label_to_id,
        )

    @staticmethod
    def _resolve_label_maps(
        dataset: Dataset, data_cfg: dict[str, Any], label_field: str
    ) -> tuple[dict[str, int], dict[int, str], Callable[[Any], int]]:
        """Build the label maps and a function that turns a raw dataset label
        into its target id.

        A ``ClassLabel`` column already stores integer indices aligned with
        ``feature.names``, so its remap is the identity; string / arbitrary
        labels are looked up by name through ``label2id``.
        """
        explicit = data_cfg.get("labels")
        if explicit:
            names = [str(n) for n in explicit]
        else:
            feature = dataset.features[label_field]
            if isinstance(feature, ClassLabel):
                names = [str(name) for name in feature.names]
                label2id = {name: idx for idx, name in enumerate(names)}
                id2label = {idx: name for idx, name in enumerate(names)}
                return label2id, id2label, lambda raw: int(raw)
            names = sorted({str(row[label_field]) for row in dataset})

        label2id = {name: idx for idx, name in enumerate(names)}
        id2label = {idx: name for idx, name in enumerate(names)}
        return label2id, id2label, lambda raw: label2id[str(raw)]

    @staticmethod
    def _apply_transform(
        dataset: Dataset,
        processor: Any,
        image_field: str,
        label_field: str,
        label_to_id: Callable[[Any], int],
        augment: bool = False,
    ) -> Dataset:
        def _transform(batch: dict[str, Any]) -> dict[str, Any]:
            images = [img.convert("RGB") for img in batch[image_field]]
            if augment:
                images = [_augment_image(img) for img in images]
            encoded = processor(images=images, return_tensors="pt")
            encoded["labels"] = [label_to_id(raw) for raw in batch[label_field]]
            return encoded

        dataset.set_transform(_transform)
        return dataset


_OPTIM_ALIASES = {"adam": "adamw_torch", "adamw": "adamw_torch", "sgd": "sgd"}


def _resolve_optim(optimizer: Any) -> str:
    """Map a spec optimizer name onto a Transformers ``TrainingArguments.optim``."""
    if not optimizer:
        return "adamw_torch"
    name = str(optimizer).lower()
    return _OPTIM_ALIASES.get(name, name)


def _augment_image(img: Image.Image) -> Image.Image:
    """Horizontal flip (p=0.5) + 4px-pad random crop, matching the loop's aug."""
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    pad = 4
    padded = ImageOps.expand(img, border=pad, fill=0)
    left = random.randint(0, 2 * pad)
    top = random.randint(0, 2 * pad)
    return padded.crop((left, top, left + w, top + h))


def _image_collate(features: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def _compute_accuracy(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    correct = int((preds == labels).sum())
    total = len(labels)
    return {"accuracy": correct / total if total else 0.0}
