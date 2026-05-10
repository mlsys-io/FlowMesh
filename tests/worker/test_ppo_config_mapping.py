"""Unit tests for PPO template-to-PPOConfig field mapping."""

from pathlib import Path
from typing import Any

import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.ppo_executor import PPOExecutor, _resolve_report_to
from worker.executors.utils.huggingface import build_hf_load_kwargs


class TestBuildHfLoadKwargsPaddingSide:
    """``training.padding_side`` is wired into the tokenizer kwargs."""

    def test_padding_side_left_threads_through(self) -> None:
        tok_kwargs, _ = build_hf_load_kwargs(
            revision=None,
            trust_remote_code=False,
            training_cfg={"padding_side": "left"},
            torch_dtype=None,
        )
        assert tok_kwargs["padding_side"] == "left"

    def test_padding_side_omitted_when_absent(self) -> None:
        tok_kwargs, _ = build_hf_load_kwargs(
            revision=None,
            trust_remote_code=False,
            training_cfg={},
            torch_dtype=None,
        )
        assert "padding_side" not in tok_kwargs


class TestResolveReportTo:
    def test_none_disables_logging(self) -> None:
        assert _resolve_report_to(None) == "none"

    def test_empty_string_disables_logging(self) -> None:
        assert _resolve_report_to("") == "none"
        assert _resolve_report_to("   ") == "none"

    def test_string_wraps_in_list(self) -> None:
        assert _resolve_report_to("wandb") == ["wandb"]
        assert _resolve_report_to(" tensorboard ") == ["tensorboard"]

    def test_list_passes_through(self) -> None:
        assert _resolve_report_to(["wandb", "tensorboard"]) == [
            "wandb",
            "tensorboard",
        ]

    def test_empty_list_disables_logging(self) -> None:
        assert _resolve_report_to([]) == "none"
        assert _resolve_report_to([" ", ""]) == "none"

    @pytest.mark.parametrize("bad", [42, 3.14, True, {"x": 1}])
    def test_invalid_type_raises(self, bad: Any) -> None:
        with pytest.raises(ExecutionError):
            _resolve_report_to(bad)


def _build_config(executor: PPOExecutor, training_cfg: dict[str, Any]) -> Any:
    return executor._build_ppo_config(
        training_config=training_cfg,
        response_cfg={},
        checkpoint_dir=Path("/tmp/ppo-cfg-test"),
        per_device_batch=None,
        grad_acc_steps=None,
        num_mini_batches=None,
        dataset_size=8,
    )


class TestPPOConfigMapping:
    @pytest.fixture
    def executor(self) -> PPOExecutor:
        return PPOExecutor.__new__(PPOExecutor)

    def test_log_with_null_disables_logging(self, executor: PPOExecutor) -> None:
        cfg = _build_config(executor, {"log_with": None})
        # HF TrainingArguments normalizes "none" to an empty integration list.
        assert cfg.report_to == []

    def test_log_with_string_maps_to_report_to_list(
        self, executor: PPOExecutor
    ) -> None:
        cfg = _build_config(executor, {"log_with": "wandb"})
        assert cfg.report_to == ["wandb"]

    def test_tracker_project_name_maps_to_project(self, executor: PPOExecutor) -> None:
        cfg = _build_config(
            executor, {"tracker_project_name": "tinyllama-ppo-training"}
        )
        assert cfg.project == "tinyllama-ppo-training"

    def test_absent_keys_keep_ppoconfig_defaults(self, executor: PPOExecutor) -> None:
        cfg_default = _build_config(executor, {})
        cfg_explicit = _build_config(executor, {"log_with": "wandb"})
        # Default project is the HuggingFace TrainingArguments default.
        assert cfg_default.project == "huggingface"
        # Explicit log_with applies; default leaves report_to unchanged from HF.
        assert cfg_default.report_to != cfg_explicit.report_to
