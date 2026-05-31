from typing import Any, Literal

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


class ImageClassificationSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION]

    checkpoint: dict[str, Any] | None = None


class ImageClassificationSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION]

    checkpoint: dict[str, Any] | None = None
