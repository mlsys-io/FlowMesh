from typing import Literal

from ..task_type import TaskType
from .common import ModelInferSpecStrict, ModelInferSpecTemplate


class DiffusionSpecStrict(ModelInferSpecStrict):
    taskType: Literal[TaskType.DIFFUSION]

    def uses_gpu(self) -> bool:
        return True


class DiffusionSpecTemplate(ModelInferSpecTemplate):
    taskType: Literal[TaskType.DIFFUSION]

    def uses_gpu(self) -> bool:
        return True
