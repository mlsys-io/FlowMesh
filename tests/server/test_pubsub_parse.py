"""Decoding of redis-py pub/sub frames shared by the blocking and polling readers."""

import json

from server.clients.redis import parse_pubsub_message


def test_parses_json_object_message() -> None:
    msg = {"type": "message", "data": json.dumps({"kind": "task", "id": 1})}
    assert parse_pubsub_message(msg) == {"kind": "task", "id": 1}


def test_decodes_bytes_payload() -> None:
    msg = {"type": "message", "data": b'{"kind": "stop"}'}
    assert parse_pubsub_message(msg) == {"kind": "stop"}


def test_none_frame_returns_none() -> None:
    # get_message(timeout=...) yields None when nothing arrived
    assert parse_pubsub_message(None) is None


def test_control_frame_returns_none() -> None:
    assert parse_pubsub_message({"type": "subscribe", "data": 1}) is None


def test_missing_data_returns_none() -> None:
    assert parse_pubsub_message({"type": "message", "data": None}) is None


def test_malformed_json_returns_none() -> None:
    assert parse_pubsub_message({"type": "message", "data": "{not json"}) is None
