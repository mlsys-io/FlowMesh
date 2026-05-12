"""Unit tests for PPO KL-based early stopping."""

from unittest.mock import patch

import pytest

pytest.importorskip("trl", reason="trl not installed (needs --extra training)")

from worker.executors.ppo_executor import _EarlyStopPPOTrainer, _EarlyStopSignal


def _make_trainer(target_kl: float | None) -> _EarlyStopPPOTrainer:
    """Build a bare ``_EarlyStopPPOTrainer`` without running ``__init__``.

    ``_EarlyStopPPOTrainer.log`` defers to ``super().log`` (TRL/HF Trainer); the
    tests patch that with a no-op so they only exercise our override.
    """
    trainer = _EarlyStopPPOTrainer.__new__(_EarlyStopPPOTrainer)
    trainer.target_kl = target_kl
    return trainer


@patch("trl.trainer.ppo_trainer.PPOTrainer.log", autospec=True)
def test_kl_above_target_raises(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    with pytest.raises(_EarlyStopSignal):
        trainer.log({"objective/kl": 0.2})


@patch("trl.trainer.ppo_trainer.PPOTrainer.log", autospec=True)
def test_kl_below_target_does_not_raise(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    trainer.log({"objective/kl": 0.05})


@patch("trl.trainer.ppo_trainer.PPOTrainer.log", autospec=True)
def test_kl_equal_to_target_does_not_raise(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    trainer.log({"objective/kl": 0.1})


@patch("trl.trainer.ppo_trainer.PPOTrainer.log", autospec=True)
def test_missing_kl_key_is_ignored(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    trainer.log({"loss": 1.0})


@patch("trl.trainer.ppo_trainer.PPOTrainer.log", autospec=True)
def test_threshold_unset_is_no_op(_super_log) -> None:
    """When ``target_kl is None`` the override is a pure pass-through."""
    trainer = _make_trainer(target_kl=None)
    trainer.log({"objective/kl": 9.99})
