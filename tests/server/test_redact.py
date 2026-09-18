"""Tests for API credential redaction at serialization time."""

import json
from unittest import mock

from server.task.models import TaskRecord
from shared.tasks import TaskEnvelopeTemplate
from shared.tasks.specs import ApiSpecTemplate
from shared.utils.redact import REDACTED, is_sensitive_key, redact_api, redact_raw_yaml


def _api_task(api: dict) -> TaskEnvelopeTemplate:
    return TaskEnvelopeTemplate.model_validate(
        {
            "apiVersion": "mloc/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": "api", "api": api},
        }
    )


def _record(api: dict, source: str = "") -> TaskRecord:
    return TaskRecord(
        task_id="tsk-1",
        workflow_id="wfl-1",
        owner_id="owner",
        source=source,
        task=_api_task(api),
    )


class TestSensitiveKey:
    def test_sensitive_keys_match(self) -> None:
        for key in (
            "Authorization",
            "authorization",
            "token",
            "api-key",
            "api_key",
            "apikey",
            "secret",
            "access_token",
            "bearer",
            "x-api-key",
            "my_token",
            "my_key",
        ):
            assert is_sensitive_key(key), key

    def test_innocent_keys_do_not_match(self) -> None:
        for key in ("monkey", "turkey", "keyword", "keys", "model", "messages"):
            assert not is_sensitive_key(key), key


class TestRedactApi:
    def test_headers_redacted(self) -> None:
        out = redact_api({"headers": {"Authorization": "Bearer SECRET"}})
        assert out is not None
        assert out["headers"]["Authorization"] == REDACTED

    def test_nested_in_dict_redacted(self) -> None:
        out = redact_api({"json": {"auth": {"token": "SECRET-NESTED"}}})
        assert out is not None
        assert out["json"]["auth"]["token"] == REDACTED

    def test_nested_in_list_redacted(self) -> None:
        out = redact_api({"json": [{"token": "SECRET"}]})
        assert out is not None
        assert out["json"][0]["token"] == REDACTED

    def test_all_five_locations_redacted(self) -> None:
        api = {
            "headers": {"Authorization": "Bearer H"},
            "params": {"api_key": "P"},
            "body": {"secret": "B"},
            "json": {"token": "J"},
            "data": {"access_token": "D"},
        }
        out = redact_api(api)
        assert out is not None
        for field in ("headers", "params", "body", "json", "data"):
            assert list(out[field].values()) == [REDACTED], field

    def test_innocent_values_preserved(self) -> None:
        out = redact_api({"json": {"model": "gpt", "monkey": "x"}})
        assert out is not None
        assert out["json"]["model"] == "gpt"
        assert out["json"]["monkey"] == "x"

    def test_original_not_mutated(self) -> None:
        api = {"headers": {"Authorization": "Bearer SECRET"}}
        redact_api(api)
        assert api["headers"]["Authorization"] == "Bearer SECRET"


class TestRedactRawYaml:
    def test_plain_scalar_redacted(self) -> None:
        out = redact_raw_yaml("Authorization: Bearer SECRET\n")
        assert "SECRET" not in out
        assert REDACTED in out

    def test_block_scalar_redacted(self) -> None:
        out = redact_raw_yaml("Authorization: |\n  Bearer SECRET\n")
        assert "SECRET" not in out
        assert REDACTED in out

    def test_folded_scalar_redacted(self) -> None:
        out = redact_raw_yaml("Authorization: >-\n  Bearer SECRET\n")
        assert "SECRET" not in out
        assert REDACTED in out

    def test_nested_yaml_redacted(self) -> None:
        out = redact_raw_yaml("api:\n  json:\n    token: SECRET\n")
        assert "SECRET" not in out

    def test_innocent_yaml_preserved(self) -> None:
        out = redact_raw_yaml("model: gpt-4o\nmonkey: x\n")
        assert "gpt-4o" in out
        assert "monkey: x" in out

    def test_unparseable_yaml_fails_closed(self) -> None:
        assert redact_raw_yaml("a: [unclosed") == REDACTED


class TestTaskRecordSerializer:
    def test_dump_redacts_all_five_locations(self) -> None:
        api = {
            "headers": {"Authorization": "Bearer H"},
            "params": {"api_key": "P"},
            "body": {"secret": "B"},
            "json": {"token": "J"},
            "data": {"access_token": "D"},
        }
        rec = _record(api)
        dumped = rec.model_dump()
        dumped_api = dumped["task"]["spec"]["api"]
        for field in ("headers", "params", "body", "json", "data"):
            assert list(dumped_api[field].values()) == [REDACTED], field

    def test_dump_json_redacts(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        dumped = json.loads(rec.model_dump_json())
        assert dumped["task"]["spec"]["api"]["headers"]["Authorization"] == REDACTED

    def test_source_redacted_in_dump(self) -> None:
        rec = _record(
            {"headers": {"Authorization": "Bearer SECRET"}},
            source="api:\n  headers:\n    Authorization: Bearer SECRET\n",
        )
        dumped = rec.model_dump()
        assert "SECRET" not in dumped["source"]

    def test_in_memory_keeps_real_credential(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        assert isinstance(rec.task.spec, ApiSpecTemplate)
        assert rec.task.spec.api is not None
        assert rec.task.spec.api["headers"]["Authorization"] == "Bearer SECRET"

    def test_no_credential_unchanged(self) -> None:
        api = {"url": "http://x", "json": {"model": "gpt"}}
        rec = _record(api)
        dumped = rec.model_dump()
        assert dumped["task"]["spec"]["api"] == api

    def test_redaction_cached_across_dumps(self) -> None:
        rec = _record(
            {"headers": {"Authorization": "Bearer SECRET"}},
            source="api:\n  headers:\n    Authorization: Bearer SECRET\n",
        )
        with mock.patch(
            "server.task.models.redact_raw_yaml",
            wraps=redact_raw_yaml,
        ) as spy:
            for _ in range(3):
                dumped = rec.model_dump()
                assert dumped["source"] != "SECRET"
                assert (
                    dumped["task"]["spec"]["api"]["headers"]["Authorization"]
                    == REDACTED
                )
        assert spy.call_count == 1


class TestSourceFieldAlias:
    def test_old_key_populates_source(self) -> None:
        rec = TaskRecord.model_validate(
            {
                "task_id": "tsk-1",
                "workflow_id": "wfl-1",
                "owner_id": "owner",
                "raw_yaml": "model: gpt-4o\n",
                "task": _api_task({"url": "http://x"}),
            }
        )
        assert rec.source == "model: gpt-4o\n"

    def test_new_key_populates_source(self) -> None:
        rec = TaskRecord.model_validate(
            {
                "task_id": "tsk-1",
                "workflow_id": "wfl-1",
                "owner_id": "owner",
                "source": "model: gpt-4o\n",
                "task": _api_task({"url": "http://x"}),
            }
        )
        assert rec.source == "model: gpt-4o\n"

    def test_constructed_with_source_kwarg(self) -> None:
        rec = TaskRecord(
            task_id="tsk-1",
            workflow_id="wfl-1",
            owner_id="owner",
            source="model: gpt-4o\n",
            task=_api_task({"url": "http://x"}),
        )
        assert rec.source == "model: gpt-4o\n"
