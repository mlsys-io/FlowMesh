import importlib

_IMPORT_ERRORS: dict[str, str] = {}


def _safe_import(name: str, module: str) -> type | None:
    try:
        pkg = importlib.import_module(module, package=__package__)
        return getattr(pkg, name)
    except Exception as exc:
        _IMPORT_ERRORS[name] = str(exc)
        return None


VLLMExecutor = _safe_import("VLLMExecutor", ".vllm_executor")
VLLMLoRAExecutor = _safe_import("VLLMLoRAExecutor", ".vllm_lora_executor")
PPOExecutor = _safe_import("PPOExecutor", ".ppo_executor")
DPOExecutor = _safe_import("DPOExecutor", ".dpo_executor")
SFTExecutor = _safe_import("SFTExecutor", ".sft_executor")
LoRASFTExecutor = _safe_import("LoRASFTExecutor", ".lora_sft_executor")
ImageClassificationExecutor = _safe_import(
    "ImageClassificationExecutor", ".image_classification_executor"
)
HFTransformersExecutor = _safe_import(
    "HFTransformersExecutor", ".transformers_executor"
)
RAGExecutor = _safe_import("RAGExecutor", ".rag_executor")
AgentExecutor = _safe_import("AgentExecutor", ".agent_executor")
EchoExecutor = _safe_import("EchoExecutor", ".echo_executor")
DataProfilingExecutor = _safe_import(
    "DataProfilingExecutor", ".data_profiling_executor"
)
DataRetrievalExecutor = _safe_import(
    "DataRetrievalExecutor", ".data_retrieval_executor"
)
DiffusersExecutor = _safe_import("DiffusersExecutor", ".diffusers_executor")
APIExecutor = _safe_import("APIExecutor", ".api_executor")
SSHExecutor = _safe_import("SSHExecutor", ".ssh_executor")
OmniText2ImageExecutor = _safe_import(
    "OmniText2ImageExecutor", ".omni_text2image_executor"
)
OmniText2SpeechExecutor = _safe_import(
    "OmniText2SpeechExecutor", ".omni_text2speech_executor"
)
OmniText2AudioExecutor = _safe_import(
    "OmniText2AudioExecutor", ".omni_text2audio_executor"
)
OmniText2GeneralExecutor = _safe_import(
    "OmniText2GeneralExecutor", ".omni_text2general_executor"
)

EXECUTOR_REGISTRY: dict[str, type | None] = {
    "vllm": VLLMExecutor,
    "vllm_lora": VLLMLoRAExecutor,
    "ppo": PPOExecutor,
    "dpo": DPOExecutor,
    "sft": SFTExecutor,
    "lora_sft": LoRASFTExecutor,
    "image_classification": ImageClassificationExecutor,
    "default": HFTransformersExecutor,
    "rag": RAGExecutor,
    "agent": AgentExecutor,
    "echo": EchoExecutor,
    "data_profiling": DataProfilingExecutor,
    "data_retrieval": DataRetrievalExecutor,
    "diffusers": DiffusersExecutor,
    "api": APIExecutor,
    "ssh": SSHExecutor,
    "omni_text2image": OmniText2ImageExecutor,
    "omni_text2speech": OmniText2SpeechExecutor,
    "omni_text2audio": OmniText2AudioExecutor,
    "omni_text2general": OmniText2GeneralExecutor,
}

EXECUTOR_CLASS_NAMES: dict[str, str] = {
    "vllm": "VLLMExecutor",
    "vllm_lora": "VLLMLoRAExecutor",
    "ppo": "PPOExecutor",
    "dpo": "DPOExecutor",
    "sft": "SFTExecutor",
    "lora_sft": "LoRASFTExecutor",
    "image_classification": "ImageClassificationExecutor",
    "default": "HFTransformersExecutor",
    "rag": "RAGExecutor",
    "agent": "AgentExecutor",
    "echo": "EchoExecutor",
    "data_profiling": "DataProfilingExecutor",
    "data_retrieval": "DataRetrievalExecutor",
    "diffusers": "DiffusersExecutor",
    "api": "APIExecutor",
    "ssh": "SSHExecutor",
    "omni_text2image": "OmniText2ImageExecutor",
    "omni_text2speech": "OmniText2SpeechExecutor",
    "omni_text2audio": "OmniText2AudioExecutor",
    "omni_text2general": "OmniText2GeneralExecutor",
}

IMPORT_ERRORS: dict[str, str] = dict(_IMPORT_ERRORS)

__all__ = [
    name
    for name, cls in {
        "VLLMExecutor": VLLMExecutor,
        "VLLMLoRAExecutor": VLLMLoRAExecutor,
        "PPOExecutor": PPOExecutor,
        "DPOExecutor": DPOExecutor,
        "SFTExecutor": SFTExecutor,
        "LoRASFTExecutor": LoRASFTExecutor,
        "ImageClassificationExecutor": ImageClassificationExecutor,
        "HFTransformersExecutor": HFTransformersExecutor,
        "RAGExecutor": RAGExecutor,
        "AgentExecutor": AgentExecutor,
        "EchoExecutor": EchoExecutor,
        "DataProfilingExecutor": DataProfilingExecutor,
        "DataRetrievalExecutor": DataRetrievalExecutor,
        "DiffusersExecutor": DiffusersExecutor,
        "APIExecutor": APIExecutor,
        "SSHExecutor": SSHExecutor,
        "OmniText2ImageExecutor": OmniText2ImageExecutor,
        "OmniText2SpeechExecutor": OmniText2SpeechExecutor,
        "OmniText2AudioExecutor": OmniText2AudioExecutor,
        "OmniText2GeneralExecutor": OmniText2GeneralExecutor,
    }.items()
    if cls is not None
] + ["EXECUTOR_REGISTRY", "IMPORT_ERRORS", "EXECUTOR_CLASS_NAMES"]
