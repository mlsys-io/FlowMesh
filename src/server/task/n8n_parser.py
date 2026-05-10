import base64
import binascii
import json
import os
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_SET_NODE_TYPE = "n8n-nodes-base.set"
_CHAIN_NODE_TYPE = "@n8n/n8n-nodes-langchain.chainLlm"
_HF_MODEL_NODE_TYPE = "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference"
_OPENAI_CHAT_MODEL_NODE_TYPE = "@n8n/n8n-nodes-langchain.lmChatOpenAi"
_OPENAI_CHAT_NODE_TYPE = "@n8n/n8n-nodes-langchain.openAi"
_N8N_CREDENTIAL_AES_PASSWORD = os.environ.get("N8N_CREDENTIAL_AES_PASSWORD", "").strip()

N8N_NODE_KEY_SCHEMA = {
    "name": str,
    "type": str,
    "parameters": dict,
}


def translate_n8n_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    nodes, connections = _parse_and_validate_node(payload)

    edges = _collect_edges(connections)
    incoming, _ = _index_edges(edges)

    chain_nodes = [node for node in nodes if node["type"] == _CHAIN_NODE_TYPE]
    openai_nodes = [node for node in nodes if node["type"] == _OPENAI_CHAT_NODE_TYPE]

    hf_model_nodes = {
        node["name"]: node for node in nodes if node["type"] == _HF_MODEL_NODE_TYPE
    }
    openai_model_nodes = {
        node["name"]: node
        for node in nodes
        if node["type"] == _OPENAI_CHAT_MODEL_NODE_TYPE
    }

    chain_model_types: dict[str, str] = {}
    for chain in chain_nodes:
        chain_model_types[chain["name"]] = _resolve_chain_model_type(
            chain, incoming, hf_model_nodes, openai_model_nodes
        )

    inference_chain_nodes = [
        node for node in chain_nodes if chain_model_types.get(node["name"]) == "hf"
    ]
    api_chain_nodes = [
        node for node in chain_nodes if chain_model_types.get(node["name"]) == "api"
    ]

    task_nodes = inference_chain_nodes + api_chain_nodes + openai_nodes
    if not task_nodes:
        raise ValueError("No task nodes found in n8n workflow")

    raw_set_nodes = {
        node["name"]: node for node in nodes if node["type"] == _SET_NODE_TYPE
    }

    node_task_types: dict[str, str] = {}
    for chain in inference_chain_nodes:
        task_type = json.loads(chain["notes"])["taskType"]
        if not task_type:
            raise ValueError(f"Chain node '{chain['name']}' missing taskType")
        node_task_types[chain["name"]] = task_type
    for chain in api_chain_nodes:
        node_task_types[chain["name"]] = "api"
    for node in openai_nodes:
        node_task_types[node["name"]] = "api"

    node_specs: dict[str, dict[str, Any]] = {}
    for chain in inference_chain_nodes:
        name = chain["name"]
        spec: dict[str, Any] = {}
        model: dict[str, Any]
        task_type = node_task_types[name]
        spec["taskType"] = task_type

        for source_name, _ in incoming[name]:
            if source := raw_set_nodes.get(source_name):
                fragment = json.loads(source["parameters"]["jsonOutput"])
                if (
                    source_name.startswith("Data")
                    or source_name.startswith("Input")
                    or source_name.startswith("Format")
                ):
                    assert (
                        "data" not in spec
                    ), "Multiple data/input source nodes are not supported"
                    spec["data"] = fragment
                elif source_name.startswith("Resource"):
                    spec["resources"] = fragment
                elif source_name.startswith("Training"):
                    spec["training"] = fragment
                elif source_name.startswith("Runtime"):
                    model = spec.setdefault("model", {})
                    model.update({"vllm": fragment})
                elif source_name.startswith("Checkpoint"):
                    assert (
                        len(incoming[source_name]) == 1
                    ), "Checkpoint node should have exactly one input"
                    parent_name, _ = incoming[source_name][0]
                    assert parent_name in [
                        node["name"] for node in chain_nodes
                    ], "Checkpoint source's parent must be a chain node"
                    spec["checkpoint"] = {
                        "load": {
                            "type": "http",
                            "url": f"${{{parent_name}.final_model_archive}}",
                        }
                    }
                else:
                    raise ValueError(
                        f"Unrecognized fragment source node '{source_name}'"
                    )
            elif source := hf_model_nodes.get(source_name):
                model = spec.setdefault("model", {})
                model.update(
                    {
                        "source": {
                            "type": "huggingface",
                            "identifier": source["parameters"]["model"],
                        }
                    }
                )
                if inference_opts := _parse_inference_options(
                    source["parameters"].get("options") or {}
                ):
                    spec["inference"] = inference_opts

        if prompt_text := _extract_prompt_text(chain):
            if "inference" not in spec:
                spec["inference"] = {}
            spec["inference"]["system_prompt"] = prompt_text

        spec["output"] = _default_output_spec(
            spec["taskType"],
        )
        node_specs[name] = spec

    for node in api_chain_nodes + openai_nodes:
        name = node["name"]
        spec = {
            "taskType": "api",
            "api": _build_api_node_spec(
                node,
                openai_model_nodes,
                incoming,
                [n["name"] for n in task_nodes],
                node_task_types,
            ),
            "output": _default_api_output_spec(),
        }
        node_specs[name] = spec

    workflow_spec: dict[str, Any] = {}
    if len(node_specs) == 1:
        workflow_spec = next(iter(node_specs.values()))
    else:
        graph_nodes = []
        task_node_names = [node["name"] for node in nodes if node["name"] in node_specs]
        for name in task_node_names:
            spec = node_specs[name]
            entry = {"name": name, "spec": spec}
            if depends_on := _node_dependencies(name, incoming, task_node_names):
                entry["dependsOn"] = depends_on
            graph_nodes.append(entry)

        workflow_spec["graph"] = {"nodes": graph_nodes}

    workflow_name: str = payload.get("name", "n8n-workflow")
    has_api = "api" in node_task_types.values()
    has_inference = any(value != "api" for value in node_task_types.values() if value)
    if has_api and has_inference:
        kind = "HybridTask"
    else:
        kind = "APITask" if has_api else "InferenceTask"
    return {
        "apiVersion": "flowmesh/v1",
        "kind": kind,
        "metadata": {
            "name": workflow_name,
            "owner": "n8n",
        },
        "spec": workflow_spec,
    }


def _resolve_chain_model_type(
    node: dict[str, Any],
    incoming: dict[str, list[tuple[str, str]]],
    hf_model_nodes: dict[str, dict[str, Any]],
    openai_model_nodes: dict[str, dict[str, Any]],
) -> str:
    sources: list[str] = []
    for source_name, conn_type in incoming.get(node["name"], []):
        if conn_type != "ai_languageModel":
            continue
        if source_name in hf_model_nodes:
            sources.append("hf")
        elif source_name in openai_model_nodes:
            sources.append("api")
        else:
            raise ValueError(
                f"Chain node '{node['name']}' has unknown model source '{source_name}'"
            )
    if not sources:
        raise ValueError(f"Chain node '{node['name']}' missing model connection")
    if len(sources) != 1:
        raise ValueError(
            f"Chain node '{node['name']}' has multiple model connections: {sources}"
        )
    return sources[0]


def _build_api_node_spec(
    node: dict[str, Any],
    model_nodes: dict[str, dict[str, Any]],
    incoming: dict[str, list[tuple[str, str]]],
    task_node_names: list[str],
    node_task_types: dict[str, str],
) -> dict[str, Any]:
    model_id = _resolve_api_model_id(node, model_nodes, incoming)
    prompt_text = _extract_api_prompt_text(node)
    credential_data = _resolve_api_credentials(node, model_nodes, incoming)
    deps = _node_dependencies(node["name"], incoming, task_node_names)
    if len(deps) > 1:
        raise ValueError(f"API node '{node['name']}' has multiple dependencies: {deps}")
    if deps:
        dep_name = deps[0]
        dep_type = node_task_types.get(dep_name)
        if not dep_type:
            raise ValueError(f"Missing taskType for dependency '{dep_name}'")
        placeholder = _dependency_placeholder(dep_name, dep_type)
        prompt_text = _inject_dependency_prompt(prompt_text, placeholder)

    headers = {
        "Content-Type": "application/json",
    }
    spec = {
        "method": "POST",
        "headers": headers,
        "body": {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
        },
        "response": {
            "parse_json": True,
            "return_body": True,
            "raise_for_status": True,
        },
    }
    if api_key := credential_data.get("api_key"):
        spec["key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    if api_url := credential_data.get("url"):
        spec["url"] = api_url
    return spec


def _resolve_api_model_id(
    node: dict[str, Any],
    model_nodes: dict[str, dict[str, Any]],
    incoming: dict[str, list[tuple[str, str]]],
) -> str:
    if node["type"] == _CHAIN_NODE_TYPE:
        for source_name, conn_type in incoming.get(node["name"], []):
            if conn_type != "ai_languageModel":
                continue
            model_node = model_nodes.get(source_name)
            if model_node:
                model_id = _extract_model_value(
                    model_node.get("parameters", {}).get("model")
                )
                if model_id:
                    return model_id
        raise ValueError(
            f"Chain node '{node['name']}' is missing an OpenAI model connection"
        )
    if node["type"] == _OPENAI_CHAT_NODE_TYPE:
        model_id = _extract_model_value(node.get("parameters", {}).get("modelId"))
        if model_id:
            return model_id
        raise ValueError(f"OpenAI node '{node['name']}' is missing modelId")
    raise ValueError(f"Unsupported API node type '{node['type']}'")


def _resolve_api_credentials(
    node: dict[str, Any],
    model_nodes: dict[str, dict[str, Any]],
    incoming: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    if node["type"] == _CHAIN_NODE_TYPE:
        for source_name, conn_type in incoming.get(node["name"], []):
            if conn_type != "ai_languageModel":
                continue
            if model_node := model_nodes.get(source_name):
                return _extract_openai_credentials(model_node)
        return {}
    if node["type"] == _OPENAI_CHAT_NODE_TYPE:
        return _extract_openai_credentials(node)
    return {}


def _extract_openai_credentials(node: dict[str, Any]) -> dict[str, str]:
    credentials = node.get("credentials")
    if not isinstance(credentials, dict):
        return {}
    openai_api = credentials.get("openAiApi")
    if not isinstance(openai_api, dict):
        return {}

    raw_data = openai_api.get("data")
    data = raw_data if isinstance(raw_data, dict) else openai_api

    result: dict[str, str] = {}
    raw_url = data.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        if _is_truthy_flag(data.get("url_encrypted")):
            raw_url = _decrypt_credential_value(raw_url)
        normalized_url = _normalize_api_url(raw_url)
        if normalized_url:
            result["url"] = normalized_url

    raw_key = data.get("apiKey")
    if isinstance(raw_key, str) and raw_key.strip():
        if _is_truthy_flag(data.get("apiKey_encrypted")):
            raw_key = _decrypt_credential_value(raw_key)
        if raw_key:
            result["api_key"] = raw_key

    return result


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalize_api_url(url: str) -> str:
    base = str(url).strip()
    if not base:
        return ""
    no_trailing = base.rstrip("/")
    lowered = no_trailing.lower()
    if lowered.endswith("/chat/completions"):
        return no_trailing
    if lowered.endswith("/v1"):
        return no_trailing + "/chat/completions"
    return no_trailing + "/v1/chat/completions"


def _decrypt_credential_value(value: str) -> str:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Encrypted credential must be nonce:tag:ciphertext")

    nonce = _decode_secret_part(parts[0])
    tag = _decode_secret_part(parts[1])
    ciphertext = _decode_secret_part(parts[2])
    aesgcm = AESGCM(_N8N_CREDENTIAL_AES_PASSWORD.encode("utf-8"))
    plaintext = aesgcm.decrypt(nonce, ciphertext + tag, None)
    return plaintext.decode("utf-8")


def _decode_secret_part(value: str) -> bytes:
    token = value.strip()
    if not token:
        raise ValueError("Encrypted credential segment cannot be empty")
    try:
        return bytes.fromhex(token)
    except ValueError:
        try:
            return base64.b64decode(token, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Encrypted credential segment is not hex/base64") from exc


def _extract_api_prompt_text(node: dict[str, Any]) -> str:
    if node["type"] == _CHAIN_NODE_TYPE:
        text = _extract_prompt_text(node)
        if not text:
            raise ValueError(f"Chain node '{node['name']}' is missing prompt text")
        return text
    if node["type"] == _OPENAI_CHAT_NODE_TYPE:
        params = node.get("parameters") or {}
        responses = params.get("responses") or {}
        values = responses.get("values")
        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, str):
                    return content
        raise ValueError(f"OpenAI node '{node['name']}' is missing prompt content")
    raise ValueError(f"Unsupported API node type '{node['type']}'")


def _inject_dependency_prompt(prompt_text: str, placeholder: str) -> str:
    prefix = "The previous stage's response is as follows."
    prompt = f"{prefix} {prompt_text}" if prompt_text else prefix
    return f"{prompt}\n{placeholder}"


def _dependency_placeholder(dep_name: str, dep_task_type: str) -> str:
    if dep_task_type == "api":
        return f"${{{dep_name}.text}}"
    if dep_task_type == "inference":
        return f"${{{dep_name}.items.0.output}}"
    raise ValueError(
        f"API nodes cannot depend on taskType '{dep_task_type}' (node '{dep_name}')"
    )


def _extract_model_value(value: Any) -> str | None:
    if isinstance(value, dict):
        if "value" in value:
            return str(value["value"])
        if "cachedResultName" in value:
            return str(value["cachedResultName"])
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_and_validate_node(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[list[dict[str, Any]]]]]]:
    nodes: list[dict[str, Any]] | None = payload.get("nodes")
    connections: dict[str, dict[str, list[list[dict[str, Any]]]]] | None = payload.get(
        "connections"
    )
    if not isinstance(nodes, list) or any(not isinstance(n, dict) for n in nodes):
        raise ValueError("n8n workflow nodes must be a list of dicts")
    if not isinstance(connections, dict):
        raise ValueError("n8n workflow connections must be an object")

    for node_key, dtype in N8N_NODE_KEY_SCHEMA.items():
        if any(
            node_key not in node or not isinstance(node[node_key], dtype)
            for node in nodes
        ):
            raise ValueError(
                f"n8n workflow nodes must have a '{node_key}' field of type {dtype}"
            )

    for node in nodes:
        if node["type"] == _SET_NODE_TYPE:
            params = node["parameters"]
            if (
                not isinstance(params, dict)
                or "mode" not in params
                or params["mode"] != "raw"
            ):
                raise ValueError(
                    f"n8n set node '{node['name']}' must have "
                    "parameters.mode set to 'raw'"
                )

    node_names: set[str] = {node["name"] for node in nodes}
    for source_name, conn_types in connections.items():
        assert source_name in node_names
        for conn_type, outputs in conn_types.items():
            assert isinstance(conn_type, str) and isinstance(outputs, list)
            for output in outputs:
                assert isinstance(output, list)
                for conn in output:
                    assert (
                        isinstance(conn, dict)
                        and "node" in conn
                        and isinstance(conn["node"], str)
                    )

    return nodes, connections


def _collect_edges(
    connections: dict[str, dict[str, list[list[dict[str, Any]]]]],
) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for source_name, conn_types in connections.items():
        for conn_type, outputs in conn_types.items():
            for output in outputs:
                for conn in output:
                    edges.append((source_name, conn["node"], conn_type))
    return edges


def _index_edges(
    edges: list[tuple[str, str, str]],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[str]]]:
    incoming: dict[str, list[tuple[str, str]]] = {}
    outgoing: dict[str, list[str]] = {}
    for source, dest, conn_type in edges:
        incoming.setdefault(dest, []).append((source, conn_type))
        outgoing.setdefault(source, []).append(dest)
    return incoming, outgoing


def _parse_inference_options(options: dict[str, int | float]) -> dict[str, Any]:
    mapping = {
        "maxTokens": "max_tokens",
        "temperature": "temperature",
        "topK": "top_k",
        "topP": "top_p",
    }
    result: dict[str, Any] = {}
    for key, target in mapping.items():
        if key in options:
            result[target] = options[key]
    return result


def _extract_prompt_text(node: dict[str, Any]) -> str | None:
    params = node.get("parameters") or {}
    if not isinstance(params, dict):
        return None
    text = params.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _node_dependencies(
    name: str,
    incoming: dict[str, list[tuple[str, str]]],
    node_names: list[str],
) -> list[str]:
    parents = []
    for source_name, _ in incoming.get(name, []):
        if source_name in node_names:
            parents.append(source_name)
        else:
            parents.extend(_node_dependencies(source_name, incoming, node_names))
    return parents


def _default_api_output_spec() -> dict[str, Any]:
    return {
        "destination": {"type": "http"},
        "artifacts": ["results.json"],
    }


def _default_output_spec(task_type: str) -> dict[str, Any]:
    artifacts = _default_artifacts(task_type)
    return {
        "destination": {
            "type": "http",
            "timeoutSec": 30,
        },
        "artifacts": artifacts,
    }


def _default_artifacts(task_type: str) -> list[str]:
    if task_type == "inference":
        return ["results.json", "logs"]
    elif task_type == "sft":
        return ["results.json", "logs", "final_model"]
    else:
        raise ValueError(
            f"Cannot determine default artifacts for unknown taskType '{task_type}'"
        )
