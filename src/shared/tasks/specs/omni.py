from typing import Any, Literal, Self

from ...utils.redact import has_redacted_credential_fields, redact_credential_fields
from ..task_type import TaskType
from .common import ModelInferSpecStrict, ModelInferSpecTemplate


class OmniSpecStrict(ModelInferSpecStrict):
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "omni": redact_credential_fields(spec.omni),
                "storyboard": redact_credential_fields(spec.storyboard),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"omni": self.omni, "storyboard": self.storyboard}
        )


class OmniSpecTemplate(ModelInferSpecTemplate):
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return spec.model_copy(
            update={
                "omni": redact_credential_fields(spec.omni),
                "storyboard": redact_credential_fields(spec.storyboard),
            }
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            {"omni": self.omni, "storyboard": self.storyboard}
        )


# ── Text-to-Image ────────────────────────────────────────────────────────────


class OmniText2ImageSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2IMAGE]


class OmniText2ImageSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2IMAGE]


# ── Text-to-Speech ───────────────────────────────────────────────────────────


class OmniText2SpeechSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2SPEECH]


class OmniText2SpeechSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2SPEECH]


# ── Text-to-Audio (BGM) ─────────────────────────────────────────────────────


class OmniText2AudioSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2AUDIO]


class OmniText2AudioSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2AUDIO]


# ── Text-to-General (Narration) ──────────────────────────────────────────────


class OmniText2GeneralSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2GENERAL]


class OmniText2GeneralSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2GENERAL]
