import importlib

from .base_executor import Executor

_IMPORT_ERRORS: dict[str, str] = {}


def _import_executor(name: str, module: str) -> type[Executor] | None:
    try:
        pkg = importlib.import_module(module, package=__package__)
        if issubclass(cls := getattr(pkg, name), Executor):
            return cls
        error = f"{name} is not a subclass of Executor"
    except Exception as exc:
        error = str(exc)
    _IMPORT_ERRORS[name] = error
    return None


VLLMExecutor = _import_executor("VLLMExecutor", ".vllm_executor")
VLLMLoRAExecutor = _import_executor("VLLMLoRAExecutor", ".vllm_lora_executor")
PPOExecutor = _import_executor("PPOExecutor", ".ppo_executor")
DPOExecutor = _import_executor("DPOExecutor", ".dpo_executor")
SFTExecutor = _import_executor("SFTExecutor", ".sft_executor")
LoRASFTExecutor = _import_executor("LoRASFTExecutor", ".lora_sft_executor")
ImageClassificationTrainingExecutor = _import_executor(
    "ImageClassificationTrainingExecutor", ".image_classification_executor"
)
HFTransformersExecutor = _import_executor(
    "HFTransformersExecutor", ".transformers_executor"
)
RAGExecutor = _import_executor("RAGExecutor", ".rag_executor")
AgentExecutor = _import_executor("AgentExecutor", ".agent_executor")
EchoExecutor = _import_executor("EchoExecutor", ".echo_executor")
DataProfilingExecutor = _import_executor(
    "DataProfilingExecutor", ".data_profiling_executor"
)
DataRetrievalExecutor = _import_executor(
    "DataRetrievalExecutor", ".data_retrieval_executor"
)
DiffusersExecutor = _import_executor("DiffusersExecutor", ".diffusers_executor")
APIExecutor = _import_executor("APIExecutor", ".api_executor")
SSHExecutor = _import_executor("SSHExecutor", ".ssh_executor")
OmniText2ImageExecutor = _import_executor(
    "OmniText2ImageExecutor", ".omni_text2image_executor"
)
OmniText2SpeechExecutor = _import_executor(
    "OmniText2SpeechExecutor", ".omni_text2speech_executor"
)
OmniText2AudioExecutor = _import_executor(
    "OmniText2AudioExecutor", ".omni_text2audio_executor"
)
OmniText2GeneralExecutor = _import_executor(
    "OmniText2GeneralExecutor", ".omni_text2general_executor"
)

EXECUTOR_REGISTRY: dict[str, type[Executor] | None] = {
    "vllm": VLLMExecutor,
    "vllm_lora": VLLMLoRAExecutor,
    "ppo": PPOExecutor,
    "dpo": DPOExecutor,
    "sft": SFTExecutor,
    "lora_sft": LoRASFTExecutor,
    "image_classification_training": ImageClassificationTrainingExecutor,
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
    "image_classification_training": "ImageClassificationTrainingExecutor",
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

IMPORT_ERRORS: dict[str, str] = _IMPORT_ERRORS

__all__ = [
    name
    for name, cls in {
        "VLLMExecutor": VLLMExecutor,
        "VLLMLoRAExecutor": VLLMLoRAExecutor,
        "PPOExecutor": PPOExecutor,
        "DPOExecutor": DPOExecutor,
        "SFTExecutor": SFTExecutor,
        "LoRASFTExecutor": LoRASFTExecutor,
        "ImageClassificationTrainingExecutor": ImageClassificationTrainingExecutor,
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
