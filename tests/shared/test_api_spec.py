"""Tests for the typed ``spec.api`` model."""

import os
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from shared.tasks import TaskEnvelopeStrict, TaskEnvelopeTemplate
from shared.tasks.specs import ApiSpecStrict, ApiSpecTemplate
from shared.tasks.specs.misc import (
    _MAX_CONCURRENCY,
    _MAX_RETRIES,
    ApiConfig,
    ApiConfigTemplate,
)
from shared.utils.redact import REDACTED


def _strict(**fields: Any) -> ApiSpecStrict:
    return ApiSpecStrict.model_validate({"taskType": "api", **fields})


def _template(**fields: Any) -> ApiSpecTemplate:
    return ApiSpecTemplate.model_validate({"taskType": "api", **fields})


def _api(**fields: Any) -> ApiConfig:
    api = _strict(api=fields).api
    assert api is not None
    return api


class TestRetries:
    def test_maximum_accepted(self) -> None:
        assert _api(retries=_MAX_RETRIES).retries == _MAX_RETRIES

    def test_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"retries": _MAX_RETRIES + 1})

    def test_zero_accepted(self) -> None:
        assert _api(retries=0).retries == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"retries": -1})

    def test_bool_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"retries": True})

    def test_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"retries": "2"})


class TestConcurrency:
    def test_maximum_accepted(self) -> None:
        assert _api(concurrency=_MAX_CONCURRENCY).concurrency == _MAX_CONCURRENCY

    def test_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"concurrency": _MAX_CONCURRENCY + 1})

    def test_one_accepted(self) -> None:
        assert _api(concurrency=1).concurrency == 1

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"concurrency": 0})

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"concurrency": -1})

    def test_bool_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"concurrency": True})

    def test_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"concurrency": "2"})


class TestMethod:
    def test_upper_cased(self) -> None:
        assert _api(method="get").method == "GET"

    def test_default_is_post(self) -> None:
        assert ApiConfig.model_validate({}).method == "POST"

    def test_template_keeps_placeholder(self) -> None:
        api = _template(api={"method": "${stage.method}"}).api
        assert api is not None
        assert api.method == "${stage.method}"


class TestTemplatePlaceholders:
    def test_retries_placeholder_accepted(self) -> None:
        api = _template(api={"retries": "${stage.items.0.text}"}).api
        assert api is not None
        assert api.retries == "${stage.items.0.text}"

    def test_concurrency_placeholder_accepted(self) -> None:
        api = _template(api={"concurrency": "${stage.items.0.text}"}).api
        assert api is not None
        assert api.concurrency == "${stage.items.0.text}"

    def test_timeout_placeholder_accepted(self) -> None:
        api = _template(api={"timeout_sec": "${stage.items.0.text}"}).api
        assert api is not None
        assert api.timeout_sec == "${stage.items.0.text}"

    def test_max_body_bytes_placeholder_accepted(self) -> None:
        api = _template(
            api={"response": {"max_body_bytes": "${stage.items.0.text}"}}
        ).api
        assert api is not None
        assert api.response is not None
        assert api.response.max_body_bytes == "${stage.items.0.text}"

    def test_strict_rejects_placeholder(self) -> None:
        with pytest.raises(ValidationError):
            _strict(api={"retries": "${stage.items.0.text}"})


class TestTemplateBounds:
    def test_retries_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _template(api={"retries": _MAX_RETRIES + 1})

    def test_retries_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _template(api={"retries": -1})

    def test_concurrency_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _template(api={"concurrency": _MAX_CONCURRENCY + 1})

    def test_concurrency_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _template(api={"concurrency": 0})

    def test_timeout_nonpositive_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _template(api={"timeout_sec": 0})

    def test_max_body_bytes_nonpositive_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _template(api={"response": {"max_body_bytes": 0}})

    def test_retries_placeholder_accepted(self) -> None:
        api = _template(api={"retries": "${stage.items.0.text}"}).api
        assert api is not None
        assert api.retries == "${stage.items.0.text}"


class TestJsonAlias:
    def test_json_key_validates(self) -> None:
        api = _api(json={"model": "gpt"})
        assert api.json_body == {"model": "gpt"}

    def test_dumps_back_as_json_key(self) -> None:
        api = _api(json={"model": "gpt"})
        dumped = api.model_dump(by_alias=True)
        assert "json" in dumped
        assert dumped["json"] == {"model": "gpt"}

    def test_round_trips_to_equal_model(self) -> None:
        api = _api(json={"model": "gpt"})
        dumped = api.model_dump(by_alias=True)
        assert ApiConfig.model_validate(dumped) == api

    def test_template_json_key(self) -> None:
        api = _template(api={"json": {"model": "gpt"}}).api
        assert api is not None
        assert api.json_body == {"model": "gpt"}


class TestTemplateInstanceConversion:
    def test_template_instance_with_json_keeps_value(self) -> None:
        template = _template(api={"json": {"model": "gpt"}}).api
        assert template is not None
        strict = ApiConfig.model_validate(template)
        assert strict.json_body == {"model": "gpt"}

    def test_template_instance_without_json_keeps_none(self) -> None:
        template = _template(api={}).api
        assert template is not None
        strict = ApiConfig.model_validate(template)
        assert strict.json_body is None

    def test_envelope_conversion_dumps_json_key(self) -> None:
        template = TaskEnvelopeTemplate.model_validate(
            {
                "apiVersion": "flowmesh/v1",
                "kind": "APITask",
                "spec": {
                    "taskType": "api",
                    "data": {"type": "list", "items": ["hi"]},
                    "api": {"json": {"model": "gpt"}},
                },
            }
        )
        strict = TaskEnvelopeStrict.model_validate(template)
        dumped = strict.model_dump_json(by_alias=True)
        assert '"json":{"model":"gpt"}' in dumped


class TestWarningFreeImport:
    def test_import_raises_no_warning(self) -> None:
        src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        env = {**os.environ, "PYTHONPATH": os.path.abspath(src)}
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::UserWarning",
                "-c",
                "import shared.tasks.specs.misc",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr


class TestRedaction:
    def test_redacts_credential_fields(self) -> None:
        spec = _strict(
            api={
                "headers": {"Authorization": "Bearer H"},
                "params": {"api_key": "P"},
                "body": {"secret": "B"},
                "json": {"token": "J"},
                "data": {"access_token": "D"},
            }
        )
        redacted = spec.redact_credentials()
        dumped = redacted.model_dump(by_alias=True)["api"]
        for field in ("headers", "params", "body", "json", "data"):
            assert list(dumped[field].values()) == [REDACTED], field
        assert redacted.has_redacted_credentials()

    def test_in_memory_keeps_real_credential(self) -> None:
        spec = _strict(api={"headers": {"Authorization": "Bearer SECRET"}})
        assert spec.api is not None
        assert spec.api.headers is not None
        assert spec.api.headers["Authorization"] == "Bearer SECRET"
        assert not spec.has_redacted_credentials()

    def test_template_redacts(self) -> None:
        spec = _template(api={"headers": {"Authorization": "Bearer SECRET"}})
        redacted = spec.redact_credentials()
        assert redacted.model_dump()["api"]["headers"]["Authorization"] == REDACTED


class TestDefaults:
    def test_defaults(self) -> None:
        api = ApiConfig.model_validate({})
        assert api.method == "POST"
        assert api.timeout_sec == 60
        assert api.verify_tls is True
        assert api.follow_redirects is True
        assert api.retries == 0
        assert api.concurrency == _MAX_CONCURRENCY
        assert api.response is None

    def test_template_defaults(self) -> None:
        api = ApiConfigTemplate.model_validate({})
        assert api.method == "POST"
        assert api.timeout_sec == 60.0
        assert api.retries == 0
        assert api.concurrency == _MAX_CONCURRENCY
