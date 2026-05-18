# ruff: noqa: E402
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip(
    "torch", reason="torch not installed (needs --extra inference)"
)

from shared.tasks.components import (
    JsonlExportSpec,
    PostprocessSpec,
)
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG, make_worker_task_message
from worker.executors.transformers_executor import HFTransformersExecutor


def _mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.chat_template = "<dummy chat template>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    def _apply(messages, **kwargs):
        content = messages[-1]["content"]
        if isinstance(content, list):
            content = content[-1]["text"]
        return f"<chat>{content}</chat>"

    def _encode(prompts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {
            "input_ids": torch.tensor([[10, 11], [12, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }

    def _decode(tokens, skip_special_tokens=True):
        values = [int(v) for v in tokens]
        if values == [20, 2]:
            return "first answer"
        if values == [21, 22]:
            return "second answer"
        raise AssertionError(f"Unexpected tokens: {values}")

    tokenizer.apply_chat_template.side_effect = _apply
    tokenizer.side_effect = _encode
    tokenizer.decode.side_effect = _decode
    return tokenizer


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = torch.tensor(
        [
            [10, 11, 20, 2],
            [12, 0, 21, 22],
        ]
    )
    return model


def test_transformers_executor_supports_chat_prompts_and_jsonl_export(
    tmp_path: Path,
) -> None:
    spec = InferenceSpecStrict(
        taskType=TaskType.INFERENCE,
        model=ModelConfig(source=ModelSource(identifier="org/model")),
        data={
            "type": "list",
            "items": [
                [{"role": "user", "content": "hello"}],
                [{"role": "user", "content": "world"}],
            ],
            "metadata": [
                {"row_id": "a"},
                {"row_id": "b"},
            ],
        },
        inference={
            "chat_template_kwargs": {"enable_thinking": False},
            "stop": ["unused stop marker"],
        },
        postprocess=PostprocessSpec(
            jsonl_export=JsonlExportSpec(
                path="rows.jsonl",
                fields={
                    "row_id": "metadata.row_id",
                    "prompt": "prompt",
                    "messages": {"from": "metadata", "key": "prompt"},
                    "answer": "output",
                },
                required_fields=["row_id", "answer"],
            )
        ),
    )
    task = make_worker_task_message(
        spec, task_type=TaskType.INFERENCE, task_id="tsk-chat"
    )

    executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
    executor._tok = _mock_tokenizer()
    executor._model = _mock_model()
    executor._device = "cpu"
    executor._mode = "text-generation"
    executor._model_name = "org/model"

    with patch.object(executor, "_ensure_model") as mock_ensure_model:
        result = executor.run(task, tmp_path)
    mock_ensure_model.assert_called_once_with(spec)

    assert [item["output"] for item in result.items] == [
        "first answer",
        "second answer",
    ]
    assert [item["prompt"] for item in result.items] == [
        "<chat>hello</chat>",
        "<chat>world</chat>",
    ]
    assert [item["metadata"]["row_id"] for item in result.items] == ["a", "b"]
    assert result.usage is not None
    assert result.usage["num_requests"] == 2
    assert "latency_sec" in result.usage

    exported = (tmp_path / "artifacts" / "rows.jsonl").read_text(encoding="utf-8")
    assert '"row_id": "a"' in exported
    assert '"answer": "first answer"' in exported

    rows = [json.loads(line) for line in exported.splitlines() if line.strip()]
    assert len(rows) == 2
    # JSONL fields preserve native JSON types — `messages` (a list of dicts)
    # comes through directly, no inner decoding required.
    assert rows[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert rows[1]["messages"] == [{"role": "user", "content": "world"}]
