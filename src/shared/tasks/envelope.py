from typing import Annotated

from pydantic import BaseModel, Field

from ._base import StrictBaseModel, TemplateBaseModel
from .components import TaskMetadata
from .placeholders import PLACEHOLDER_PATTERN
from .specs import (
    AgentSpecStrict,
    AgentSpecTemplate,
    ApiSpecStrict,
    ApiSpecTemplate,
    DataProfilingSpecStrict,
    DataProfilingSpecTemplate,
    DataRetrievalSpecStrict,
    DataRetrievalSpecTemplate,
    DiffusionSpecStrict,
    DiffusionSpecTemplate,
    DPOSpecStrict,
    DPOSpecTemplate,
    EchoSpecStrict,
    EchoSpecTemplate,
    EmbeddingSpecStrict,
    EmbeddingSpecTemplate,
    ImageClassificationSpecStrict,
    ImageClassificationSpecTemplate,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    LoRASFTSpecStrict,
    LoRASFTSpecTemplate,
    OmniText2AudioSpecStrict,
    OmniText2AudioSpecTemplate,
    OmniText2GeneralSpecStrict,
    OmniText2GeneralSpecTemplate,
    OmniText2ImageSpecStrict,
    OmniText2ImageSpecTemplate,
    OmniText2SpeechSpecStrict,
    OmniText2SpeechSpecTemplate,
    PPOSpecStrict,
    PPOSpecTemplate,
    RagSpecStrict,
    RagSpecTemplate,
    SFTSpecStrict,
    SFTSpecTemplate,
    SSHSpecStrict,
    SSHSpecTemplate,
)

type TaskSpecStrict = Annotated[
    InferenceSpecStrict
    | DiffusionSpecStrict
    | RagSpecStrict
    | ApiSpecStrict
    | SFTSpecStrict
    | LoRASFTSpecStrict
    | PPOSpecStrict
    | DPOSpecStrict
    | ImageClassificationSpecStrict
    | EchoSpecStrict
    | AgentSpecStrict
    | DataProfilingSpecStrict
    | DataRetrievalSpecStrict
    | EmbeddingSpecStrict
    | SSHSpecStrict
    | OmniText2ImageSpecStrict
    | OmniText2SpeechSpecStrict
    | OmniText2AudioSpecStrict
    | OmniText2GeneralSpecStrict,
    Field(discriminator="taskType"),
]

type TaskSpecTemplate = Annotated[
    InferenceSpecTemplate
    | DiffusionSpecTemplate
    | RagSpecTemplate
    | ApiSpecTemplate
    | SFTSpecTemplate
    | LoRASFTSpecTemplate
    | PPOSpecTemplate
    | DPOSpecTemplate
    | ImageClassificationSpecTemplate
    | EchoSpecTemplate
    | AgentSpecTemplate
    | DataProfilingSpecTemplate
    | DataRetrievalSpecTemplate
    | EmbeddingSpecTemplate
    | SSHSpecTemplate
    | OmniText2ImageSpecTemplate
    | OmniText2SpeechSpecTemplate
    | OmniText2AudioSpecTemplate
    | OmniText2GeneralSpecTemplate,
    Field(discriminator="taskType"),
]


class TaskEnvelopeStrict(StrictBaseModel):
    apiVersion: str
    kind: str
    metadata: TaskMetadata | None = None
    spec: TaskSpecStrict


class TaskEnvelopeTemplate(TemplateBaseModel):
    apiVersion: str
    kind: str
    metadata: TaskMetadata | None = None
    spec: TaskSpecTemplate

    def has_placeholder(self) -> bool:
        def walk(value: object) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(PLACEHOLDER_PATTERN.search(value))
            if isinstance(value, dict):
                return any(walk(v) for v in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(walk(v) for v in value)
            if isinstance(value, BaseModel):
                for _, sub in value:
                    if walk(sub):
                        return True
                return False
            return False

        return walk(self)


type TaskEnvelope = TaskEnvelopeStrict | TaskEnvelopeTemplate
