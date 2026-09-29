"""Tests for n8n workflow translation."""

import base64
import json

import pytest

from server.task.n8n_parser import _decode_secret_part, translate_n8n_workflow
from server.task.parser import parse_workflow
from shared.tasks.specs import ApiSpecTemplate


class TestTranslateN8nWorkflow:
    def test_simple_openai_node(self) -> None:
        """A single OpenAI chat node should produce an API task with correct fields."""
        nodes = [
            {
                "name": "Chat",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {
                        "values": [{"content": "Hello, world!"}],
                    },
                },
            }
        ]
        result = translate_n8n_workflow({"nodes": nodes, "connections": {}})

        assert result["kind"] == "APITask"
        assert result["apiVersion"] == "flowmesh/v1"
        assert "spec" in result

        spec = result["spec"]
        assert spec["taskType"] == "api"
        assert "api" in spec
        api = spec["api"]
        assert api["method"] == "POST"
        assert api["body"]["model"] == "gpt-4"

        assert spec["data"]["type"] == "list"
        assert spec["data"]["items"] == ["Hello, world!"]
        assert api["body"]["messages"][0]["content"] == "{{prompt}}"

    def test_no_task_nodes_raises_value_error(self) -> None:
        """Workflow with no recognized task nodes should raise ValueError."""
        with pytest.raises(ValueError, match="No task nodes found"):
            translate_n8n_workflow({"nodes": [], "connections": {}})

    def test_invalid_json_via_parse_workflow(self) -> None:
        """Non-JSON input to n8n format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_workflow("not json at all {{{", format="n8n")

    def test_api_dependency_resolves_first_row_text(self) -> None:
        """A dependent API node reads the upstream API stage's first-row text."""
        nodes = [
            {
                "name": "Upstream",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {"values": [{"content": "First answer"}]},
                },
            },
            {
                "name": "Downstream",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {"values": [{"content": "Simplify this"}]},
                },
            },
        ]
        connections = {
            "Upstream": {"ai_languageModel": [[{"node": "Downstream"}]]},
        }
        result = translate_n8n_workflow({"nodes": nodes, "connections": connections})

        graph = result["spec"]["graph"]["nodes"]
        assert [n["name"] for n in graph] == ["Upstream", "Downstream"]
        downstream = next(n for n in graph if n["name"] == "Downstream")
        assert downstream["dependsOn"] == ["Upstream"]
        assert downstream["spec"]["data"]["items"] == [
            "The previous stage's response is as follows. Simplify this\n"
            "${Upstream.items.0.text}"
        ]

    def test_openai_credential_parses_with_header_and_no_key(self) -> None:
        """An OpenAI credential yields an Authorization header and no ``key``
        field, so the workflow validates through the submitted path."""
        nodes = [
            {
                "name": "Chat",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {
                    "modelId": {"value": "gpt-4"},
                    "responses": {
                        "values": [{"content": "Hello, world!"}],
                    },
                },
                "credentials": {
                    "openAiApi": {
                        "data": {"apiKey": "sk-secret"},
                    }
                },
            }
        ]
        result = translate_n8n_workflow({"nodes": nodes, "connections": {}})
        api = result["spec"]["api"]
        assert api["headers"]["Authorization"] == "Bearer sk-secret"
        assert "key" not in api

        parsed = parse_workflow(
            json.dumps({"nodes": nodes, "connections": {}}), format="n8n"
        )
        spec = parsed.tasks[0].task.spec
        assert isinstance(spec, ApiSpecTemplate)
        assert spec.api is not None
        assert spec.api.headers is not None
        assert spec.api.headers["Authorization"] == "Bearer sk-secret"


class TestDecodeSecretPart:
    def test_hex_decode(self) -> None:
        data = b"hello"
        encoded = data.hex()
        assert _decode_secret_part(encoded) == data

    def test_base64_decode(self) -> None:
        data = b"hello world"
        encoded = base64.b64encode(data).decode()
        assert _decode_secret_part(encoded) == data

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(Exception):
            _decode_secret_part("!!!not-valid-hex-or-base64!!!")
