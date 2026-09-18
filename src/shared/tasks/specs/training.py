from typing import Any, Literal, Self

from pydantic import model_validator

from ...utils.redact import has_redacted_credential_fields, redact_credential_fields
from ..placeholders import TemplateInt
from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class TrainingSpecStrict(ModelSpecStrict):
    data: dict[str, Any] | None = None
    training: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "data": redact_credential_fields(spec.data),
                "training": redact_credential_fields(spec.training),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"data": self.data, "training": self.training}
        )


class TrainingSpecTemplate(ModelSpecTemplate):
    data: dict[str, Any] | None = None
    training: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "data": redact_credential_fields(spec.data),
                "training": redact_credential_fields(spec.training),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"data": self.data, "training": self.training}
        )


class SFTSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.SFT]

    checkpoint: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={"checkpoint": redact_credential_fields(spec.checkpoint)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.checkpoint
        )


class SFTSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.SFT]

    checkpoint: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={"checkpoint": redact_credential_fields(spec.checkpoint)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.checkpoint
        )


class LoRASFTSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.LORA_SFT]

    lora: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sloSeconds: int | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "lora": redact_credential_fields(spec.lora),
                "checkpoint": redact_credential_fields(spec.checkpoint),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"lora": self.lora, "checkpoint": self.checkpoint}
        )


class LoRASFTSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.LORA_SFT]

    lora: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sloSeconds: TemplateInt | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "lora": redact_credential_fields(spec.lora),
                "checkpoint": redact_credential_fields(spec.checkpoint),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"lora": self.lora, "checkpoint": self.checkpoint}
        )


class PPOSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.PPO]

    reward_model: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "reward_model": redact_credential_fields(spec.reward_model),
                "generation": redact_credential_fields(spec.generation),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"reward_model": self.reward_model, "generation": self.generation}
        )


class PPOSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.PPO]

    reward_model: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "reward_model": redact_credential_fields(spec.reward_model),
                "generation": redact_credential_fields(spec.generation),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"reward_model": self.reward_model, "generation": self.generation}
        )


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

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={"checkpoint": redact_credential_fields(spec.checkpoint)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.checkpoint
        )

    @model_validator(mode="after")
    def _require_model(self) -> "ImageClassificationTrainingSpecStrict":
        _require_image_classification_model(self.model_name)
        return self


class ImageClassificationTrainingSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING]

    checkpoint: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={"checkpoint": redact_credential_fields(spec.checkpoint)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.checkpoint
        )

    @model_validator(mode="after")
    def _require_model(self) -> "ImageClassificationTrainingSpecTemplate":
        _require_image_classification_model(self.model_name)
        return self
