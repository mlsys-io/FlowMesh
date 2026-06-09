from typing import Any, Literal

from pydantic import model_validator

from ..placeholders import TemplateInt
from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class TrainingSpecStrict(ModelSpecStrict):
    data: dict[str, Any] | None = None
    training: dict[str, Any] | None = None


class TrainingSpecTemplate(ModelSpecTemplate):
    data: dict[str, Any] | None = None
    training: dict[str, Any] | None = None


class SFTSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.SFT]

    checkpoint: dict[str, Any] | None = None


class SFTSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.SFT]

    checkpoint: dict[str, Any] | None = None


class LoRASFTSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.LORA_SFT]

    lora: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sloSeconds: int | None = None


class LoRASFTSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.LORA_SFT]

    lora: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sloSeconds: TemplateInt | None = None


class PPOSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.PPO]

    reward_model: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None


class PPOSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.PPO]

    reward_model: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None


class DPOSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.DPO]


class DPOSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.DPO]


def _require_image_classification_model(model_name: str | None) -> None:
    if not model_name:
        raise ValueError(
            "image_classification_training requires model.source.identifier"
        )


class ImageClassificationTrainingSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING]

    checkpoint: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_model(self) -> "ImageClassificationTrainingSpecStrict":
        _require_image_classification_model(self.model_name)
        return self


class ImageClassificationTrainingSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING]

    checkpoint: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_model(self) -> "ImageClassificationTrainingSpecTemplate":
        _require_image_classification_model(self.model_name)
        return self
