"""Tests for the dispatcher's redacted-credential detection.

The dispatcher refuses to dispatch a task whose credential was not
retained across a server restart (it was redacted to ``[REDACTED]`` at
persist time). Detection is implemented by each credential-bearing spec.
"""

import pytest

from server.dispatcher import Dispatcher
from shared.tasks import TaskEnvelopeStrict, TaskSpecStrict
from shared.tasks.specs import ApiSpecStrict
from shared.utils.redact import REDACTED, contains_redacted

from .helpers import make_capturing_dispatcher


def _strict_api_task(api: dict) -> TaskEnvelopeStrict:
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": "api", "api": api},
        }
    )


def _strict_spec(api: dict) -> ApiSpecStrict:
    spec = _strict_api_task(api).spec
    assert isinstance(spec, ApiSpecStrict)
    return spec


def _strict_spec_for(task_type: str, **fields: object) -> TaskSpecStrict:
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": task_type, **fields},
        }
    ).spec


def _dispatcher() -> Dispatcher:
    return make_capturing_dispatcher()


class TestContainsRedacted:
    def test_plain_placeholder(self) -> None:
        assert contains_redacted(REDACTED)

    def test_nested_in_dict(self) -> None:
        assert contains_redacted({"auth": {"token": REDACTED}})

    def test_nested_in_list(self) -> None:
        assert contains_redacted([{"token": REDACTED}])

    def test_innocent_value(self) -> None:
        assert not contains_redacted("Bearer SECRET")

    def test_innocent_nested(self) -> None:
        assert not contains_redacted({"model": "gpt", "monkey": "x"})


class TestHasRedactedCredential:
    def test_headers_redacted_detected(self) -> None:
        spec = _strict_spec({"headers": {"Authorization": REDACTED}})
        assert _dispatcher()._has_redacted_credential(spec)

    def test_params_redacted_detected(self) -> None:
        spec = _strict_spec({"params": {"api_key": REDACTED}})
        assert _dispatcher()._has_redacted_credential(spec)

    def test_body_redacted_detected(self) -> None:
        spec = _strict_spec({"body": {"secret": REDACTED}})
        assert _dispatcher()._has_redacted_credential(spec)

    def test_json_redacted_detected(self) -> None:
        spec = _strict_spec({"json": {"token": REDACTED}})
        assert _dispatcher()._has_redacted_credential(spec)

    def test_data_redacted_detected(self) -> None:
        spec = _strict_spec({"data": {"access_token": REDACTED}})
        assert _dispatcher()._has_redacted_credential(spec)

    def test_nested_redacted_detected(self) -> None:
        spec = _strict_spec({"json": {"auth": {"token": REDACTED}}})
        assert _dispatcher()._has_redacted_credential(spec)

    def test_redacted_in_list_detected(self) -> None:
        spec = _strict_spec({"json": [{"token": REDACTED}]})
        assert _dispatcher()._has_redacted_credential(spec)

    @pytest.mark.parametrize(
        ("task_type", "fields"),
        [
            ("data_retrieval", {"data": {"lumid_data_token": REDACTED}}),
            ("rag", {"qdrant": {"api_key": REDACTED}}),
            (
                "serve",
                {
                    "apiKey": REDACTED,
                    "model": {"source": {"identifier": "model"}},
                    "resources": {"hardware": {"gpu": {"count": 1}}},
                },
            ),
            ("ssh", {"authorizedKeys": [REDACTED]}),
        ],
    )
    def test_non_api_task_redacted_credentials_detected(
        self, task_type: str, fields: dict[str, object]
    ) -> None:
        spec = _strict_spec_for(task_type, **fields)
        assert _dispatcher()._has_redacted_credential(spec)

    def test_no_credential_not_detected(self) -> None:
        spec = _strict_spec({"json": {"model": "gpt"}})
        assert not _dispatcher()._has_redacted_credential(spec)

    def test_real_credential_not_detected(self) -> None:
        spec = _strict_spec({"headers": {"Authorization": "Bearer SECRET"}})
        assert not _dispatcher()._has_redacted_credential(spec)

    def test_non_api_spec_not_detected(self) -> None:
        spec = TaskEnvelopeStrict.model_validate(
            {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "metadata": {"name": "t"},
                "spec": {"taskType": "echo", "data": {"token": REDACTED}},
            }
        ).spec
        assert not _dispatcher()._has_redacted_credential(spec)
