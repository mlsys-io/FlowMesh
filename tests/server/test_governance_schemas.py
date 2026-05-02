from server.governance import Span
from shared.schemas.spans import SpanType


def test_span_otel_round_trip() -> None:
    raw = {
        "name": "model load",
        "context": {
            "trace_id": "0xfbad6be5c4434181a2d394eac830dea1",
            "span_id": "0xa3f1e9d2c5b40678",
        },
        "parent_id": "0x1b2c3d4e5f6a7b8c",
        "start_time": "2026-04-30T14:00:01.000000Z",
        "end_time": "2026-04-30T14:00:55.000000Z",
        "status": {"status_code": "OK"},
        "attributes": {
            "data_id": "tsk-1",
            "batch_id": "tsk-1",
            "flowmesh.type": "compute",
        },
    }
    span = Span.parse_otel_json(raw)
    assert span.name == "model load"
    assert span.context.trace_id == "fbad6be5c4434181a2d394eac830dea1"
    assert span.context.span_id == "a3f1e9d2c5b40678"
    assert span.parent_id == "1b2c3d4e5f6a7b8c"
    assert span.attributes.data_id == "tsk-1"
    assert span.attributes.batch_id == "tsk-1"
    assert span.attributes.flowmesh_type == SpanType.COMPUTE
    assert span.duration_seconds == 54.0


def test_span_marker_zero_duration() -> None:
    raw = {
        "name": "dump to storage",
        "context": {"trace_id": "0x" + "a" * 32, "span_id": "0x" + "b" * 16},
        "parent_id": "0x" + "c" * 16,
        "start_time": "2026-04-30T14:00:01.500000Z",
        "end_time": "2026-04-30T14:00:01.500000Z",
        "status": {"status_code": "OK"},
        "attributes": {"data_id": "tsk-2", "flowmesh.type": "marker"},
    }
    span = Span.parse_otel_json(raw)
    assert span.duration_seconds == 0.0
    assert span.attributes.flowmesh_type == SpanType.MARKER
