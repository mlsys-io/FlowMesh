"""Tests for task credential redaction at serialization time."""

import json
from unittest import mock

import pytest

from server.task.models import TaskRecord
from shared.tasks import TaskEnvelopeTemplate
from shared.tasks.specs import (
    ApiSpecTemplate,
    RagSpecTemplate,
    SFTSpecTemplate,
    TaskSpecStrictBase,
    TaskSpecTemplateBase,
)
from shared.utils.redact import (
    REDACTED,
    is_credential_key,
    redact_credential,
    redact_credential_fields,
    redact_raw_yaml,
)


def _api_task(api: dict) -> TaskEnvelopeTemplate:
    return TaskEnvelopeTemplate.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": "api", "api": api},
        }
    )


def _record(api: dict, source: str = "") -> TaskRecord:
    return _record_for_task(_api_task(api), source=source)


def _record_for_task(task: TaskEnvelopeTemplate, source: str = "") -> TaskRecord:
    return TaskRecord(
        task_id="tsk-1",
        workflow_id="wfl-1",
        owner_id="owner",
        source=source,
        task=task,
    )


def _task(task_type: str, **fields: object) -> TaskEnvelopeTemplate:
    return TaskEnvelopeTemplate.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "t"},
            "spec": {"taskType": task_type, **fields},
        }
    )


class TestCredentialKey:
    def test_credential_keys_match(self) -> None:
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
            "authorizedKeys",
            "connection_string",
            "cert_data",
            "AWS_ACCESS_KEY_ID",
            "client_secret",
            "password",
        ):
            assert is_credential_key(key), key

    def test_non_credential_keys_do_not_match(self) -> None:
        for key in ("monkey", "turkey", "keyword", "keys", "model", "messages"):
            assert not is_credential_key(key), key


class TestRedactTask:
    @pytest.mark.parametrize(
        ("task_type", "fields", "path"),
        [
            (
                "api",
                {"api": {"headers": {"Authorization": "Bearer api-secret"}}},
                ("api", "headers", "Authorization"),
            ),
            (
                "data_retrieval",
                {"data": {"lumid_data_token": "lumid-secret"}},
                ("data", "lumid_data_token"),
            ),
            (
                "rag",
                {"qdrant": {"api_key": "qdrant-secret"}},
                ("qdrant", "api_key"),
            ),
            (
                "serve",
                {
                    "apiKey": "serve-secret",
                    "model": {"source": {"identifier": "model"}},
                    "resources": {"hardware": {"gpu": {"count": 1}}},
                },
                ("apiKey",),
            ),
            (
                "data_profiling",
                {"data": {"connection_string": "postgres://u:secret@db/app"}},
                ("data", "connection_string"),
            ),
            (
                "inference",
                {"data": {"connection_string": "s3://u:secret@host/bucket"}},
                ("data", "connection_string"),
            ),
            (
                "embedding",
                {"data": {"connection_string": "s3://u:secret@host/bucket"}},
                ("data", "connection_string"),
            ),
            (
                "sft",
                {"data": {"connection_string": "s3://u:secret@host/bucket"}},
                ("data", "connection_string"),
            ),
            (
                "sft",
                {"checkpoint": {"load": {"headers": {"Authorization": "Bearer x"}}}},
                ("checkpoint", "load", "headers", "Authorization"),
            ),
        ],
    )
    def test_task_record_redacts_credentials(
        self, task_type: str, fields: dict[str, object], path: tuple[str, ...]
    ) -> None:
        record = _record_for_task(_task(task_type, **fields))
        dumped = record.model_dump()
        value: object = dumped["task"]["spec"]
        for key in path:
            assert isinstance(value, dict)
            value = value[key]
        assert value == REDACTED

    def test_nested_shared_credentials_redacted(self) -> None:
        value = {
            "authorizedKeys": ["ssh-secret"],
            "connection_string": "postgres://user:secret@db/app",
            "cert_data": "certificate-secret",
            "env": {"AWS_ACCESS_KEY_ID": "access-secret"},
            "model": {"adapters": [{"headers": {"Authorization": "header-secret"}}]},
        }
        redacted = redact_credential_fields(value)
        assert redacted == {
            "authorizedKeys": [REDACTED],
            "connection_string": REDACTED,
            "cert_data": REDACTED,
            "env": {"AWS_ACCESS_KEY_ID": REDACTED},
            "model": {"adapters": [{"headers": {"Authorization": REDACTED}}]},
        }

    def test_in_memory_task_keeps_non_api_credentials(self) -> None:
        task = _task(
            "rag",
            qdrant={"api_key": "qdrant-secret"},
        )
        record = _record_for_task(task)
        assert isinstance(record.task.spec, RagSpecTemplate)
        assert record.task.spec.qdrant == {"api_key": "qdrant-secret"}
        assert record.model_dump()["task"]["spec"]["qdrant"]["api_key"] == REDACTED

    def test_in_memory_training_spec_keeps_credentials(self) -> None:
        task = _task(
            "sft",
            checkpoint={"load": {"headers": {"Authorization": "Bearer keep"}}},
        )
        record = _record_for_task(task)
        assert isinstance(record.task.spec, SFTSpecTemplate)
        assert record.task.spec.checkpoint is not None
        assert (
            record.task.spec.checkpoint["load"]["headers"]["Authorization"]
            == "Bearer keep"
        )
        dumped = record.model_dump()["task"]["spec"]
        assert dumped["checkpoint"]["load"]["headers"]["Authorization"] == REDACTED

    def test_ssh_credentials_are_redacted_with_valid_shape(self) -> None:
        record = _record_for_task(
            _task(
                "ssh",
                authorizedKeys=["ssh-secret"],
                env={"SERVICE_TOKEN": "env-secret"},
            )
        )
        dumped = record.model_dump()["task"]["spec"]
        assert dumped["authorizedKeys"] == [REDACTED]
        assert dumped["env"] == {"SERVICE_TOKEN": REDACTED}

    def test_base_spec_redaction_is_a_no_op(self) -> None:
        record = _record_for_task(_task("echo", data={"token": "echo-data"}))
        assert record.model_dump()["task"]["spec"]["data"]["token"] == "echo-data"

    def test_output_destination_headers_redacted(self) -> None:
        record = _record_for_task(
            _task(
                "echo",
                output={
                    "destination": {
                        "type": "http",
                        "url": "http://x",
                        "headers": {
                            "Authorization": "Bearer out-secret",
                            "Content-Type": "application/json",
                        },
                    }
                },
            )
        )
        spec = record.model_dump()["task"]["spec"]
        headers = spec["output"]["destination"]["headers"]
        assert headers["Authorization"] == REDACTED
        assert headers["Content-Type"] == "application/json"

    def test_output_headers_redacted_alongside_spec_fields(self) -> None:
        record = _record_for_task(
            _task(
                "api",
                api={"headers": {"api_key": "api-secret"}},
                output={
                    "destination": {
                        "type": "http",
                        "headers": {"Authorization": "Bearer out-secret"},
                    }
                },
            )
        )
        spec = record.model_dump()["task"]["spec"]
        assert spec["api"]["headers"]["api_key"] == REDACTED
        assert spec["output"]["destination"]["headers"]["Authorization"] == REDACTED


class TestRedactCredential:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, None), ("secret", REDACTED), (["secret"], [REDACTED])],
    )
    def test_redacts_scalar_and_list_credentials(
        self, value: object, expected: object
    ) -> None:
        assert redact_credential(value) == expected


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

    def test_dump_excluding_task_does_not_raise(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        dumped = rec.model_dump(exclude={"task"})
        assert "task" not in dumped
        assert dumped["source"] == ""

    def test_dump_excluding_source_does_not_readd_it(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        dumped = rec.model_dump(exclude={"source"})
        assert "source" not in dumped
        assert dumped["task"]["spec"]["api"]["headers"]["Authorization"] == REDACTED

    def test_dump_honors_nested_exclude_within_spec(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        dumped = rec.model_dump(exclude={"task": {"spec": {"api"}}})["task"]["spec"]
        assert "api" not in dumped

    def test_nested_exclude_keeps_other_fields_redacted(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        dumped = rec.model_dump(exclude={"task": {"spec": {"output"}}})["task"]["spec"]
        assert dumped["api"]["headers"]["Authorization"] == REDACTED

    def test_nested_exclude_true_drops_whole_spec_field(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        assert "spec" not in rec.model_dump(exclude={"task": {"spec": True}})["task"]

    def test_spec_dump_honors_by_alias(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        spec = rec.model_dump(by_alias=True)["task"]["spec"]
        assert "_upstreamResults" in spec
        assert spec["api"]["headers"]["Authorization"] == REDACTED

    def test_spec_dump_honors_exclude_none(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        spec = rec.model_dump(exclude_none=True)["task"]["spec"]
        assert "upstreamResults" not in spec
        assert spec["api"]["headers"]["Authorization"] == REDACTED

    def test_spec_dump_honors_exclude_unset(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        spec = rec.model_dump(exclude_unset=True)["task"]["spec"]
        assert "resources" not in spec  # never set on this task
        assert spec["api"]["headers"]["Authorization"] == REDACTED

        # Model-based specs redact via model_copy(update=...) on optional fields;
        # that must not mark unset ones (data/inference/checkpoint, nested model
        # config) as set, or they would surface here.
        rec = _record_for_task(
            _task(
                "inference",
                model={"source": {"identifier": "llama"}},
                data={"lumid_data_token": "SECRET"},
            )
        )
        spec = rec.model_dump(exclude_unset=True)["task"]["spec"]
        assert set(spec) == {"taskType", "model", "data"}
        assert spec["data"]["lumid_data_token"] == REDACTED  # set field still redacted
        assert set(spec["model"]) == {"source"}  # nested config not spuriously set

    def test_spec_dump_honors_exclude_defaults(self) -> None:
        rec = _record({"headers": {"Authorization": "Bearer SECRET"}})
        spec = rec.model_dump(exclude_defaults=True)["task"]["spec"]
        assert "resources" not in spec  # left at its default
        assert spec["api"]["headers"]["Authorization"] == REDACTED

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


_SpecClass = type[TaskSpecStrictBase] | type[TaskSpecTemplateBase]


def _concrete_spec_classes() -> list[_SpecClass]:
    """Every leaf spec class (identified by a concrete ``taskType`` field)."""
    found: list[_SpecClass] = []
    stack: list[type] = []
    stack.extend(TaskSpecStrictBase.__subclasses__())
    stack.extend(TaskSpecTemplateBase.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if (
            issubclass(cls, (TaskSpecStrictBase, TaskSpecTemplateBase))
            and "taskType" in cls.model_fields
        ):
            found.append(cls)
    return found


_SPEC_CLASSES = _concrete_spec_classes()


class TestSpecRedactionChainsToBase:
    """Redaction of shared fields (e.g. the output destination) lives on the
    base spec, so every override must chain ``super()``; a missing call would
    silently drop that coverage for a whole family of task types."""

    @pytest.mark.parametrize("spec_cls", _SPEC_CLASSES, ids=lambda cls: cls.__name__)
    def test_redact_credentials_chains_to_base(self, spec_cls: _SpecClass) -> None:
        base = (
            TaskSpecStrictBase
            if issubclass(spec_cls, TaskSpecStrictBase)
            else TaskSpecTemplateBase
        )
        with mock.patch.object(
            base,
            "redact_credentials",
            autospec=True,
            wraps=base.redact_credentials,
        ) as spy:
            spec_cls.model_construct().redact_credentials()
        spy.assert_called_once()

    @pytest.mark.parametrize("spec_cls", _SPEC_CLASSES, ids=lambda cls: cls.__name__)
    def test_has_redacted_credentials_chains_to_base(
        self, spec_cls: _SpecClass
    ) -> None:
        base = (
            TaskSpecStrictBase
            if issubclass(spec_cls, TaskSpecStrictBase)
            else TaskSpecTemplateBase
        )
        with mock.patch.object(
            base,
            "has_redacted_credentials",
            autospec=True,
            wraps=base.has_redacted_credentials,
        ) as spy:
            spec_cls.model_construct().has_redacted_credentials()
        spy.assert_called_once()
