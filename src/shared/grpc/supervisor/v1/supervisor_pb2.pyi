from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import message as _message
from google.protobuf import struct_pb2 as _struct_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class RegisterRequest(_message.Message):
    __slots__ = ("meta",)
    META_FIELD_NUMBER: _ClassVar[int]
    meta: _struct_pb2.Struct

    def __init__(
        self, meta: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("worker_id",)
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str

    def __init__(self, worker_id: _Optional[str] = ...) -> None: ...

class InterruptMessage(_message.Message):
    __slots__ = ("task_id", "reason")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    reason: str

    def __init__(
        self, task_id: _Optional[str] = ..., reason: _Optional[str] = ...
    ) -> None: ...

class StopMessage(_message.Message):
    __slots__ = ("task_id", "reason")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    reason: str

    def __init__(
        self, task_id: _Optional[str] = ..., reason: _Optional[str] = ...
    ) -> None: ...

class TaskMessage(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct

    def __init__(
        self, payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...

class RelayRequest(_message.Message):
    __slots__ = ("relay_token", "endpoint_id")
    RELAY_TOKEN_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    relay_token: str
    endpoint_id: str

    def __init__(
        self, relay_token: _Optional[str] = ..., endpoint_id: _Optional[str] = ...
    ) -> None: ...

class DispatchMessage(_message.Message):
    __slots__ = ("task", "interrupt", "stop", "relay")
    TASK_FIELD_NUMBER: _ClassVar[int]
    INTERRUPT_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    RELAY_FIELD_NUMBER: _ClassVar[int]
    task: TaskMessage
    interrupt: InterruptMessage
    stop: StopMessage
    relay: RelayRequest

    def __init__(
        self,
        task: _Optional[_Union[TaskMessage, _Mapping]] = ...,
        interrupt: _Optional[_Union[InterruptMessage, _Mapping]] = ...,
        stop: _Optional[_Union[StopMessage, _Mapping]] = ...,
        relay: _Optional[_Union[RelayRequest, _Mapping]] = ...,
    ) -> None: ...

class RelayOpen(_message.Message):
    __slots__ = ("relay_token", "endpoint_id")
    RELAY_TOKEN_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    relay_token: str
    endpoint_id: str

    def __init__(
        self, relay_token: _Optional[str] = ..., endpoint_id: _Optional[str] = ...
    ) -> None: ...

class RelayFrame(_message.Message):
    __slots__ = ("open", "data", "eof")
    OPEN_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    EOF_FIELD_NUMBER: _ClassVar[int]
    open: RelayOpen
    data: bytes
    eof: bool

    def __init__(
        self,
        open: _Optional[_Union[RelayOpen, _Mapping]] = ...,
        data: _Optional[bytes] = ...,
        eof: _Optional[bool] = ...,
    ) -> None: ...

class EventMessage(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct

    def __init__(
        self, payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...

class LogMessage(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct

    def __init__(
        self, payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...
