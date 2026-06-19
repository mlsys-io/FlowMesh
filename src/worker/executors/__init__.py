import importlib
from collections.abc import Iterator, Mapping

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
_OMNI_SPECS: dict[str, tuple[str, str]] = {
    "omni_text2image": ("OmniText2ImageExecutor", ".omni_text2image_executor"),
    "omni_text2speech": ("OmniText2SpeechExecutor", ".omni_text2speech_executor"),
    "omni_text2audio": ("OmniText2AudioExecutor", ".omni_text2audio_executor"),
    "omni_text2general": ("OmniText2GeneralExecutor", ".omni_text2general_executor"),
}


class _LazyRegistry(Mapping[str, type[Executor] | None]):
    """Mapping that defers omni executor imports to first key lookup.

    Backed by a plain dict; omni entries are resolved (their module imported)
    on the first __getitem__ / get() access via the Mapping mixin.
    """

    def __init__(
        self,
        eager: dict[str, type[Executor] | None],
        lazy: dict[str, tuple[str, str]],
    ) -> None:
        self._data: dict[str, type[Executor] | None] = dict(eager)
        self._lazy: dict[str, tuple[str, str]] = dict(lazy)
        for key in lazy:
            self._data[key] = None

    def _resolve(self, key: str) -> type[Executor] | None:
        cls_name, module = self._lazy.pop(key)
        cls = _import_executor(cls_name, module)
        self._data[key] = cls
        return cls

    def __getitem__(self, key: str) -> type[Executor] | None:
        if key in self._lazy:
            return self._resolve(key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


EXECUTOR_REGISTRY: Mapping[str, type[Executor] | None] = _LazyRegistry(
    {
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
    },
    _OMNI_SPECS,
)

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

# Live reference so errors from lazy imports appear immediately
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
    }.items()
    if cls is not None
] + ["EXECUTOR_REGISTRY", "IMPORT_ERRORS", "EXECUTOR_CLASS_NAMES"]
