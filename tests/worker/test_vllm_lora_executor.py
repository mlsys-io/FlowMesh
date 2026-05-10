import pytest
from pydantic import ValidationError

from shared.tasks import TaskType
from shared.tasks.specs import InferenceSpecStrict
from worker.executors.vllm_lora_executor import VLLMLoRAExecutor

from .factories import make_worker_config


def test_lora_adapter_fields_are_typed_and_extracted() -> None:
    spec = InferenceSpecStrict.model_validate(
        {
            "taskType": TaskType.INFERENCE.value,
            "model": {
                "adapters": [
                    {
                        "type": "lora",
                        "path": "/tmp/final_lora.tar.gz",
                        "name": "math",
                        "apply": "merge",
                        "id": 7,
                        "archive_format": "tar",
                        "headers": {"Authorization": "Bearer token"},
                        "task_id": "tsk-train",
                    }
                ]
            },
        }
    )

    adapter = spec.adapters[0]  # type: ignore[index]
    assert adapter.apply == "merge"
    assert adapter.id == 7
    assert adapter.archive_format == "tar"
    assert adapter.headers == {"Authorization": "Bearer token"}
    assert adapter.task_id == "tsk-train"

    executor = VLLMLoRAExecutor(make_worker_config())
    extracted = executor._extract_adapter_specs(spec)  # noqa: SLF001

    assert len(extracted) == 1
    assert extracted[0].name == "math"
    assert extracted[0].id == 7
    assert extracted[0].apply == "merge"
    assert extracted[0].headers == {"Authorization": "Bearer token"}


def test_lora_adapter_kwargs_wrapper_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InferenceSpecStrict.model_validate(
            {
                "taskType": TaskType.INFERENCE.value,
                "model": {
                    "adapters": [
                        {
                            "type": "lora",
                            "path": "/tmp/final_lora.tar.gz",
                            "kwargs": {"apply": "merge"},
                        }
                    ]
                },
            }
        )
