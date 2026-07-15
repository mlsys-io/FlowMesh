import importlib
from collections.abc import Iterator, Mapping
from typing import overload

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


EXECUTOR_MODULES: dict[str, tuple[str, str]] = {
    "vllm": ("VLLMExecutor", ".vllm_executor"),
    "vllm_lora": ("VLLMLoRAExecutor", ".vllm_lora_executor"),
    "vllm_embedding": ("VLLMEmbeddingExecutor", ".vllm_embedding_executor"),
    "vllm_serve": ("VLLMServeExecutor", ".vllm_serve_executor"),
    "ppo": ("PPOExecutor", ".ppo_executor"),
    "dpo": ("DPOExecutor", ".dpo_executor"),
    "sft": ("SFTExecutor", ".sft_executor"),
    "lora_sft": ("LoRASFTExecutor", ".lora_sft_executor"),
    "image_classification_training": (
        "ImageClassificationTrainingExecutor",
        ".image_classification_executor",
    ),
    "default": ("HFTransformersExecutor", ".transformers_executor"),
    "rag": ("RAGExecutor", ".rag_executor"),
    "agent": ("AgentExecutor", ".agent_executor"),
    "echo": ("EchoExecutor", ".echo_executor"),
    "data_profiling": ("DataProfilingExecutor", ".data_profiling_executor"),
    "data_retrieval": ("DataRetrievalExecutor", ".data_retrieval_executor"),
    "diffusers": ("DiffusersExecutor", ".diffusers_executor"),
    "api": ("APIExecutor", ".api_executor"),
    "ssh": ("SSHExecutor", ".ssh_executor"),
    "omni_text2image": ("OmniText2ImageExecutor", ".omni_text2image_executor"),
    "omni_text2speech": ("OmniText2SpeechExecutor", ".omni_text2speech_executor"),
    "omni_text2audio": ("OmniText2AudioExecutor", ".omni_text2audio_executor"),
    "omni_text2general": (
        "OmniText2GeneralExecutor",
        ".omni_text2general_executor",
    ),
}


class ExecutorRegistry(Mapping[str, type[Executor] | None]):
    def __init__(self) -> None:
        self._executors: dict[str, type[Executor] | None] = {}

    def __getitem__(self, key: str) -> type[Executor] | None:
        if key in self._executors:
            return self._executors[key]
        if key not in EXECUTOR_MODULES:
            raise KeyError(f"Executor {key!r} not found in registry")
        name, module = EXECUTOR_MODULES[key]
        executor = _import_executor(name, module)
        self._executors[key] = executor
        return executor

    def __iter__(self) -> Iterator[str]:
        return iter(EXECUTOR_MODULES)

    def __len__(self) -> int:
        return len(EXECUTOR_MODULES)


EXECUTOR_REGISTRY = ExecutorRegistry()


@overload
def get_executor_class_name[T](key: str, default: T) -> str | T: ...
@overload
def get_executor_class_name(key: str, default: None = None) -> str | None: ...
def get_executor_class_name[T](key: str, default: T | None = None) -> str | T | None:
    return mod[0] if (mod := EXECUTOR_MODULES.get(key)) else default


IMPORT_ERRORS: dict[str, str] = _IMPORT_ERRORS
