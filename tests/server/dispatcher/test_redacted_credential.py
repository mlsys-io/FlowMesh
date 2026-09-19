"""Tests for task-spec redacted-credential detection.

The dispatcher refuses to dispatch a task whose credential was not
retained across a server restart (it was redacted to ``[REDACTED]`` at
persist time). Detection is implemented by each credential-bearing spec.
"""

import pytest

from shared.tasks import TaskEnvelopeStrict, TaskSpecStrict
from shared.utils.redact import REDACTED, contains_redacted


def _strict_spec_for(task_type: str, **fields: object) -> TaskSpecStrict:
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": task_type, **fields},
        }
    ).spec


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


class TestTaskSpecCredentialDetection:
    @pytest.mark.parametrize(
        ("task_type", "fields"),
        [
            ("api", {"api": {"headers": {"Authorization": REDACTED}}}),
            ("api", {"api": {"params": {"api_key": REDACTED}}}),
            ("api", {"api": {"body": {"secret": REDACTED}}}),
            ("api", {"api": {"json": {"token": REDACTED}}}),
            ("api", {"api": {"data": {"access_token": REDACTED}}}),
            ("api", {"api": {"json": {"auth": {"token": REDACTED}}}}),
            ("api", {"api": {"json": [{"token": REDACTED}]}}),
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
            (
                "echo",
                {
                    "output": {
                        "destination": {
                            "type": "http",
                            "headers": {"Authorization": REDACTED},
                        }
                    }
                },
            ),
        ],
        ids=[
            "api_headers",
            "api_params",
            "api_body",
            "api_json",
            "api_data",
            "api_nested",
            "api_list",
            "data_retrieval",
            "rag",
            "serve",
            "ssh",
            "output_headers",
        ],
    )
    def test_redacted_credential_is_detected(
        self, task_type: str, fields: dict[str, object]
    ) -> None:
        spec = _strict_spec_for(task_type, **fields)
        assert spec.has_redacted_credentials()

    @pytest.mark.parametrize(
        ("task_type", "fields"),
        [
            ("api", {"api": {"json": {"model": "gpt"}}}),
            ("api", {"api": {"headers": {"Authorization": "Bearer SECRET"}}}),
            ("echo", {"data": {"token": REDACTED}}),
        ],
        ids=["no_credential", "real_credential", "generic_task"],
    )
    def test_non_redacted_credentials_are_not_detected(
        self, task_type: str, fields: dict[str, object]
    ) -> None:
        spec = _strict_spec_for(task_type, **fields)
        assert not spec.has_redacted_credentials()
