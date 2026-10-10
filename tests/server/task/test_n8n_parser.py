"""Tests for n8n workflow translation."""

import base64
import json

import pytest

from server.task.n8n_parser import _decode_secret_part, translate_n8n_workflow
from server.task.parser import parse_workflow
from shared.tasks.specs import (
    ApiSpecTemplate,
    OmniText2VideoSpecTemplate,
    PythonSpecTemplate,
)
from shared.tasks.task_type import TaskType


def set_node(name: str, value: dict) -> dict:
    return {
        "name": name,
        "type": "n8n-nodes-base.set",
        "parameters": {"mode": "raw", "jsonOutput": json.dumps(value)},
    }


def model_node(name: str, model: str) -> dict:
    return {
        "name": name,
        "type": "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference",
        "parameters": {"model": model},
    }


def chain_node(name: str, task_type: str, text: str = "") -> dict:
    return {
        "name": name,
        "type": "@n8n/n8n-nodes-langchain.chainLlm",
        "parameters": {"promptType": "define", "text": text},
        "notes": json.dumps({"taskType": task_type}),
    }


def code_node(name: str, code: str, language: str = "pythonNative") -> dict:
    return {
        "name": name,
        "type": "n8n-nodes-base.code",
        "parameters": {"language": language, "pythonCode": code, "jsCode": code},
    }


def main(*targets: str) -> dict:
    return {"main": [[{"node": t, "type": "main", "index": 0} for t in targets]]}


def lm(target: str) -> dict:
    return {
        "ai_languageModel": [[{"node": target, "type": "ai_languageModel", "index": 0}]]
    }


GPU = {"hardware": {"gpu": {"type": "any", "count": 1}}}


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

    def test_llm_to_video_chain(self) -> None:
        """An LLM chain feeding a video chain through a Format node becomes a
        two-stage graph; the ``Omni`` Set node lands in ``spec.omni``."""

        hardware = GPU
        nodes = [
            set_node("Input Idea", {"type": "list", "items": ["a fox in snow"]}),
            set_node("Resource Spec W", hardware),
            model_node("Model Writer", "Qwen/Qwen2.5-7B-Instruct"),
            chain_node("Write Prompt", "inference", "Expand into a video prompt."),
            set_node(
                "Format Video",
                {
                    "type": "graph_template",
                    "template": {
                        "name": "video_prompt",
                        "text": "{col0_value}",
                        "columns": [
                            {
                                "label": "PROMPT",
                                "node": "Write Prompt",
                                "path": "items[0].output",
                            }
                        ],
                    },
                },
            ),
            set_node("Resource Spec V", hardware),
            set_node("Omni Video", {"num_frames": 81, "enable_cpu_offload": True}),
            model_node("Model Video", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            chain_node("Render Video", "omni_text2video"),
        ]

        connections = {
            "Input Idea": main("Write Prompt"),
            "Resource Spec W": main("Write Prompt"),
            "Model Writer": lm("Write Prompt"),
            "Write Prompt": main("Format Video"),
            "Format Video": main("Render Video"),
            "Resource Spec V": main("Render Video"),
            "Omni Video": main("Render Video"),
            "Model Video": lm("Render Video"),
        }
        payload = {"nodes": nodes, "connections": connections}
        result = translate_n8n_workflow(payload)

        graph = {n["name"]: n for n in result["spec"]["graph"]["nodes"]}
        video = graph["Render Video"]
        assert video["dependsOn"] == ["Write Prompt"]
        assert video["spec"]["taskType"] == "omni_text2video"
        assert video["spec"]["omni"] == {"num_frames": 81, "enable_cpu_offload": True}
        assert video["spec"]["data"]["type"] == "graph_template"
        assert video["spec"]["output"]["artifacts"] == [
            "results.json",
            "logs",
            "artifacts/",
        ]

        parsed = parse_workflow(json.dumps(payload), format="n8n")
        specs = {t.task.spec.taskType: t.task.spec for t in parsed.tasks}
        assert isinstance(specs[TaskType.OMNI_TEXT2VIDEO], OmniText2VideoSpecTemplate)

    def test_python_code_node_becomes_python_stage(self) -> None:
        """A Python Code node becomes a ``python`` stage that depends on the
        task nodes feeding it; a ``Python`` Set node adds its stage fields."""
        compose = "def main(inputs):\n    return {'ok': True}\n"
        nodes = [
            set_node("Input Idea", {"type": "list", "items": ["a fox in snow"]}),
            set_node("Resource Spec W", GPU),
            model_node("Model Writer", "Qwen/Qwen2.5-7B-Instruct"),
            chain_node("Write Narration", "inference", "Write a voice-over."),
            set_node(
                "Format Speech",
                {
                    "type": "graph_template",
                    "template": {
                        "name": "voice_over",
                        "text": "{col0_value}",
                        "columns": [
                            {"node": "Write Narration", "path": "items[0].output"}
                        ],
                    },
                },
            ),
            set_node("Resource Spec S", GPU),
            set_node("Omni Speech", {"speaker": "Vivian", "language": "English"}),
            model_node("Model Speech", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            chain_node("Speak", "omni_text2speech"),
            set_node(
                "Python Env",
                {
                    "requirements": ["imageio-ffmpeg"],
                    "network": "bridge",
                    "timeoutSeconds": 900,
                },
            ),
            code_node("Compose", compose),
        ]
        connections = {
            "Input Idea": main("Write Narration"),
            "Resource Spec W": main("Write Narration"),
            "Model Writer": lm("Write Narration"),
            "Write Narration": main("Format Speech", "Compose"),
            "Format Speech": main("Speak"),
            "Resource Spec S": main("Speak"),
            "Omni Speech": main("Speak"),
            "Model Speech": lm("Speak"),
            "Speak": main("Compose"),
            "Python Env": main("Compose"),
        }
        payload = {"nodes": nodes, "connections": connections}
        result = translate_n8n_workflow(payload)

        graph = {n["name"]: n for n in result["spec"]["graph"]["nodes"]}
        assert sorted(graph) == ["Compose", "Speak", "Write Narration"]
        assert graph["Speak"]["spec"]["output"]["artifacts"] == [
            "results.json",
            "logs",
            "artifacts/",
        ]
        compose_node = graph["Compose"]
        assert sorted(compose_node["dependsOn"]) == ["Speak", "Write Narration"]
        assert compose_node["spec"] == {
            "taskType": "python",
            "code": compose,
            "entrypoint": "main",
            "requirements": ["imageio-ffmpeg"],
            "network": "bridge",
            "timeoutSeconds": 900,
        }

        parsed = parse_workflow(json.dumps(payload), format="n8n")
        specs = {t.task.spec.taskType: t.task.spec for t in parsed.tasks}
        python_spec = specs[TaskType.PYTHON]
        assert isinstance(python_spec, PythonSpecTemplate)
        assert python_spec.network == "bridge"

    def test_javascript_code_node_is_not_translated(self) -> None:
        nodes = [
            set_node("Input Idea", {"type": "list", "items": ["hi"]}),
            model_node("Model Writer", "Qwen/Qwen2.5-7B-Instruct"),
            chain_node("Write", "inference"),
            code_node("Shape", "return items;", language="javaScript"),
        ]
        connections = {
            "Input Idea": main("Write"),
            "Model Writer": lm("Write"),
            "Write": main("Shape"),
        }
        result = translate_n8n_workflow({"nodes": nodes, "connections": connections})

        assert result["spec"]["taskType"] == "inference"

    def test_python_code_node_without_code_raises(self) -> None:
        nodes = [code_node("Compose", "  ")]
        with pytest.raises(ValueError, match="has no Python code"):
            translate_n8n_workflow({"nodes": nodes, "connections": {}})


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
