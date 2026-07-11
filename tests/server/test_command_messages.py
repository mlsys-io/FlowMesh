"""Tests for shared command/dispatch message schemas."""

from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from shared.schemas.command import (
    CommandMessage,
    CommandResponse,
    CommandType,
    DispatchMessage,
    InterruptMessage,
    StopMessage,
    TaskMessage,
)


class TestCommandMessage:
    def test_required_fields(self) -> None:
        msg = CommandMessage(command=CommandType.START_WORKER)
        assert msg.command == CommandType.START_WORKER
        assert msg.command_id  # auto-generated UUID hex
        assert msg.payload is None

    def test_with_payload(self) -> None:
        msg = CommandMessage(
            command=CommandType.STOP_WORKER,
            payload={"worker_name": "w-1"},
        )
        assert msg.payload == {"worker_name": "w-1"}

    def test_command_id_is_unique(self) -> None:
        m1 = CommandMessage(command=CommandType.GET_WORKERS)
        m2 = CommandMessage(command=CommandType.GET_WORKERS)
        assert m1.command_id != m2.command_id

    def test_all_command_types(self) -> None:
        expected = {
            "START_WORKER",
            "CREATE_WORKER",
            "CREATE_WORKER_ON_NODE",
            "GET_WORKERS",
            "STOP_WORKER",
            "DESTROY_WORKER",
            "DESTROY_WORKERS",
            "START_RELAY",
        }
        assert {t.value for t in CommandType} == expected


class TestCommandResponse:
    def test_ok_factory(self) -> None:
        cmd = CommandMessage(command=CommandType.GET_WORKERS)
        resp = CommandResponse.ok(cmd, data={"workers": []})
        assert resp.success is True
        assert resp.command_id == cmd.command_id
        assert resp.data == {"workers": []}

    def test_error_factory(self) -> None:
        cmd = CommandMessage(command=CommandType.START_WORKER)
        resp = CommandResponse.error(cmd, message="worker not found")
        assert resp.success is False
        assert resp.message == "worker not found"
        assert resp.command_id == cmd.command_id


class TestDispatchMessages:
    _adapter: ClassVar[TypeAdapter[DispatchMessage]] = TypeAdapter(DispatchMessage)

    def test_task_message(self) -> None:
        msg = TaskMessage(worker_id="w-1", payload={"task_id": "t-1"})
        assert msg.kind == "task"
        assert msg.worker_id == "w-1"
        dumped = msg.model_dump()
        assert dumped["kind"] == "task"

    def test_interrupt_message(self) -> None:
        msg = InterruptMessage(task_id="t-1", worker_id="w-1")
        assert msg.kind == "interrupt"
        assert msg.reason == "cancelled"

    def test_stop_message(self) -> None:
        msg = StopMessage(task_id="t-1", worker_id="w-1")
        assert msg.kind == "stop"
        assert msg.reason == "stopped"

    def test_custom_reason(self) -> None:
        msg = InterruptMessage(task_id="t-1", worker_id="w-1", reason="timeout")
        assert msg.reason == "timeout"

    def test_roundtrip_serialization(self) -> None:
        msg = TaskMessage(
            worker_id="w-1",
            payload={"task_id": "t-1", "spec": {"taskType": "echo"}},
        )
        data = msg.model_dump()
        restored = TaskMessage.model_validate(data)
        assert restored.worker_id == msg.worker_id
        assert restored.payload == msg.payload

    def test_dispatch_union_parses_task_variant(self) -> None:
        parsed = self._adapter.validate_python(
            {"kind": "task", "worker_id": "w-1", "payload": {"task_id": "t-1"}}
        )
        assert isinstance(parsed, TaskMessage)
        assert parsed.worker_id == "w-1"

    def test_dispatch_union_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python(
                {"kind": "bogus", "worker_id": "w-1", "payload": {}}
            )
