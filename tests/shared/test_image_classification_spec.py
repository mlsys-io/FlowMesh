"""Tests for the image_classification_training task spec."""

import pytest
from pydantic import ValidationError

from shared.tasks.envelope import TaskEnvelopeStrict
from shared.tasks.specs import ImageClassificationTrainingSpecStrict
from shared.tasks.task_type import TaskType

_BASE = {
    "taskType": "image_classification_training",
    "model": {
        "source": {"type": "huggingface", "identifier": "google/vit-base-patch16-224"}
    },
    "data": {"dataset_name": "cifar10"},
}


def test_envelope_resolves_image_classification_training() -> None:
    env = TaskEnvelopeStrict.model_validate(
        {"apiVersion": "flowmesh/v1", "kind": "ImageClassificationTask", "spec": _BASE}
    )
    assert isinstance(env.spec, ImageClassificationTrainingSpecStrict)
    assert env.spec.taskType == TaskType.IMAGE_CLASSIFICATION_TRAINING


def test_model_identifier_is_required() -> None:
    spec_without_model = {k: v for k, v in _BASE.items() if k != "model"}
    with pytest.raises(ValidationError, match="model.source.identifier"):
        ImageClassificationTrainingSpecStrict.model_validate(spec_without_model)
