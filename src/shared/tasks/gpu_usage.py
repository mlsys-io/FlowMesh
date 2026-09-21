"""Whether running a task spec will allocate GPU memory.

The server uses this to keep GPU work off a worker whose device is held by
another tenant, and the worker uses it both to refuse such a task and to know
whether its own warm executor is holding VRAM. Both sides must agree, so the
answer lives here rather than being derived twice.

Unknowable values answer ``True``: refusing a task that would have fit costs a
reschedule, while admitting one onto a held card costs the task.
"""

from shared.tasks.components.model import ModelConfig, ModelConfigTemplate
from shared.tasks.envelope import TaskSpecStrict, TaskSpecTemplate
from shared.tasks.specs import (
    EmbeddingSpecStrict,
    EmbeddingSpecTemplate,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    SSHSpecStrict,
    SSHSpecTemplate,
)
from shared.tasks.specs.inference import InferenceBackend
from shared.tasks.task_type import TaskType

_ALWAYS_GPU: frozenset[TaskType] = frozenset(
    {
        TaskType.DIFFUSION,
        TaskType.SFT,
        TaskType.LORA_SFT,
        TaskType.PPO,
        TaskType.DPO,
        TaskType.IMAGE_CLASSIFICATION_TRAINING,
        TaskType.SERVE,
        TaskType.OMNI_TEXT2IMAGE,
        TaskType.OMNI_TEXT2SPEECH,
        TaskType.OMNI_TEXT2AUDIO,
        TaskType.OMNI_TEXT2GENERAL,
    }
)

_NEVER_GPU: frozenset[TaskType] = frozenset(
    {
        TaskType.API,
        TaskType.ECHO,
        TaskType.AGENT,
        TaskType.DATA_PROFILING,
        TaskType.DATA_RETRIEVAL,
        TaskType.RAG,
    }
)


def task_uses_gpu(spec: TaskSpecStrict | TaskSpecTemplate) -> bool:
    """Whether executing ``spec`` will allocate GPU memory."""
    task_type = spec.taskType
    if task_type in _ALWAYS_GPU:
        return True
    if task_type in _NEVER_GPU:
        return False
    if isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate)):
        return _inference_uses_gpu(spec)
    if isinstance(spec, (EmbeddingSpecStrict, EmbeddingSpecTemplate)):
        return _embedding_uses_gpu(spec)
    if isinstance(spec, (SSHSpecStrict, SSHSpecTemplate)):
        return _ssh_requests_gpu(spec)
    return True


def _inference_uses_gpu(spec: InferenceSpecStrict | InferenceSpecTemplate) -> bool:
    # Identity, not truthiness: on a template this field may hold an unresolved
    # placeholder string, which must not read as "pinned to CPU".
    if spec.enforce_cpu is True:
        return False
    if spec.backend() is InferenceBackend.TRANSFORMERS:
        return _transformers_uses_gpu(spec.model)
    # VLLM always, and AUTO because the runner prefers vLLM for it.
    return True


def _embedding_uses_gpu(spec: EmbeddingSpecStrict | EmbeddingSpecTemplate) -> bool:
    model = spec.model
    if model is not None and model.vllm is not None:
        return True
    return _transformers_uses_gpu(model)


def _transformers_uses_gpu(model: ModelConfig | ModelConfigTemplate | None) -> bool:
    """Mirror of ``HFTransformersExecutor._pick_device``: only an explicit
    ``cpu`` keeps the model off the GPU; everything else prefers CUDA."""
    config = model.transformers if model is not None else None
    if not config:
        return True
    return config.get("device_map") != "cpu"


def _ssh_requests_gpu(spec: SSHSpecStrict | SSHSpecTemplate) -> bool:
    """An SSH session is GPU-using exactly when it asks for devices.

    A bare ``type`` or ``memory`` without ``count`` still resolves to one device in
    the session config, so any ``gpu`` block counts -- except an explicit
    ``count: 0``, which asks for none.
    """
    resources = spec.resources
    hardware = resources.hardware if resources is not None else None
    gpu = hardware.gpu if hardware is not None else None
    return gpu is not None and gpu.count != 0
