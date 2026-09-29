from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, AliasChoices, Field, StrictBool, StrictInt

from ...utils.pydantic_utils import copy_preserving_fields_set
from ...utils.redact import has_redacted_credential_fields, redact_credential_fields
from .._base import StrictBaseModel, TemplateBaseModel
from ..placeholders import PlaceholderString, TemplateBool
from ..task_type import TaskType
from .common import (
    ModelSpecStrict,
    ModelSpecTemplate,
    TaskSpecStrictBase,
    TaskSpecTemplateBase,
)

# Upper bounds on spec.api.concurrency and spec.api.retries.
_MAX_CONCURRENCY = 8
_MAX_RETRIES = 10

# Bounds shared by the Strict and Template models. The Template unions each
# bounded literal with a placeholder so a ``${...}`` reference stays accepted.
Retries = Annotated[StrictInt, Field(ge=0, le=_MAX_RETRIES)]
Concurrency = Annotated[StrictInt, Field(ge=1, le=_MAX_CONCURRENCY)]
MaxBodyBytes = Annotated[StrictInt, Field(gt=0)]
TimeoutSec = Annotated[float, Field(gt=0)]


def _upper_method(value: str) -> str:
    return value.upper()


Method = Annotated[str, AfterValidator(_upper_method)]


class ApiResponseConfig(StrictBaseModel):
    parse_json: StrictBool = True
    return_body: StrictBool = True
    include_headers: StrictBool = False
    max_body_bytes: MaxBodyBytes = 200000
    raise_for_status: StrictBool = True


class ApiResponseConfigTemplate(TemplateBaseModel):
    parse_json: TemplateBool = True
    return_body: TemplateBool = True
    include_headers: TemplateBool = False
    max_body_bytes: MaxBodyBytes | PlaceholderString = 200000
    raise_for_status: TemplateBool = True


class ApiConfig(StrictBaseModel):
    url: str | None = None
    method: Method = "POST"
    headers: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    json_body: Any | None = Field(
        default=None,
        validation_alias=AliasChoices("json_body", "json"),
        serialization_alias="json",
    )
    body: Any | None = None
    data: Any | None = None
    timeout_sec: TimeoutSec = 60.0
    verify_tls: StrictBool = True
    follow_redirects: StrictBool = True
    retries: Retries = 0
    concurrency: Concurrency = _MAX_CONCURRENCY
    response: ApiResponseConfig | None = None


class ApiConfigTemplate(TemplateBaseModel):
    url: str | None = None
    method: str = "POST"
    headers: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    json_body: Any | None = Field(
        default=None,
        validation_alias=AliasChoices("json_body", "json"),
        serialization_alias="json",
    )
    body: Any | None = None
    data: Any | None = None
    timeout_sec: TimeoutSec | PlaceholderString = 60.0
    verify_tls: TemplateBool = True
    follow_redirects: TemplateBool = True
    retries: Retries | PlaceholderString = 0
    concurrency: Concurrency | PlaceholderString = _MAX_CONCURRENCY
    response: ApiResponseConfigTemplate | None = None


def _redact_api_config(
    api: ApiConfig | ApiConfigTemplate | None,
) -> ApiConfig | ApiConfigTemplate | None:
    if api is None:
        return None
    return copy_preserving_fields_set(
        api,
        {
            "headers": redact_credential_fields(api.headers),
            "params": redact_credential_fields(api.params),
            "json_body": redact_credential_fields(api.json_body),
            "body": redact_credential_fields(api.body),
            "data": redact_credential_fields(api.data),
        },
    )


def _api_has_redacted_credentials(
    api: ApiConfig | ApiConfigTemplate | None,
) -> bool:
    if api is None:
        return False
    return has_redacted_credential_fields(
        {
            "headers": api.headers,
            "params": api.params,
            "json_body": api.json_body,
            "body": api.body,
            "data": api.data,
        }
    )


class ApiSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.API]
    api: ApiConfig | None = None
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(spec, {"api": _redact_api_config(spec.api)})

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or _api_has_redacted_credentials(
            self.api
        )


class ApiSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.API]
    api: ApiConfigTemplate | None = None
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(spec, {"api": _redact_api_config(spec.api)})

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or _api_has_redacted_credentials(
            self.api
        )


class EchoSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.ECHO]
    data: dict[str, Any] | None = None


class EchoSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.ECHO]
    data: dict[str, Any] | None = None


class AgentSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.AGENT]

    configName: str | None = None
    task: str | None = None
    agent: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


class AgentSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.AGENT]

    configName: str | None = None
    task: str | None = None
    agent: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


class DataProfilingSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class DataProfilingSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_PROFILING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class DataRetrievalSpecStrict(TaskSpecStrictBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class DataRetrievalSpecTemplate(TaskSpecTemplateBase):
    taskType: Literal[TaskType.DATA_RETRIEVAL]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )


class EmbeddingSpecStrict(ModelSpecStrict):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )

    def uses_gpu(self) -> bool:
        model = self.model
        if model is not None and model.vllm is not None:
            return True
        return self.model_uses_gpu()


class EmbeddingSpecTemplate(ModelSpecTemplate):
    taskType: Literal[TaskType.EMBEDDING]
    data: dict[str, Any] | None = None

    def redact_credentials(self) -> Self:
        spec = super().redact_credentials()
        return copy_preserving_fields_set(
            spec, {"data": redact_credential_fields(spec.data)}
        )

    def has_redacted_credentials(self) -> bool:
        return super().has_redacted_credentials() or has_redacted_credential_fields(
            self.data
        )

    def uses_gpu(self) -> bool:
        model = self.model
        if model is not None and model.vllm is not None:
            return True
        return self.model_uses_gpu()
