"""Unit tests for PPO KL-based early stopping."""

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("trl", reason="trl not installed (needs --extra training)")

from worker.executors.base_executor import ExecutionError
from worker.executors.ppo_executor import (
    PPOExecutor,
    _EarlyStopPPOTrainer,
    _EarlyStopSignal,
)


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


# ---------------------------------------------------------------------------
# _install_kl_early_stopping activation rules
# ---------------------------------------------------------------------------


def _install(training_config: dict[str, Any]) -> _EarlyStopPPOTrainer:
    """Run the installer against a bare trainer and return it."""
    executor = PPOExecutor.__new__(PPOExecutor)
    trainer = _make_trainer(target_kl=None)
    executor._install_kl_early_stopping(trainer, training_config)
    return trainer


def test_install_no_op_when_flag_missing() -> None:
    trainer = _install({"target_kl": 0.1})
    assert trainer.target_kl is None


@pytest.mark.parametrize("flag", [False, "false", "False", 0, "no", "off"])
def test_install_no_op_when_flag_disabled(flag: Any) -> None:
    trainer = _install({"early_stopping": flag, "target_kl": 0.1})
    assert trainer.target_kl is None


@pytest.mark.parametrize("flag", [True, "true", "True", 1, "yes", "on"])
def test_install_arms_when_flag_enabled(flag: Any) -> None:
    trainer = _install({"early_stopping": flag, "target_kl": 0.1})
    assert trainer.target_kl == pytest.approx(0.1)


@pytest.mark.parametrize("bad_target", [None, 0, -0.5])
def test_install_rejects_enabled_without_positive_target(bad_target: Any) -> None:
    cfg: dict[str, Any] = {"early_stopping": True}
    if bad_target is not None:
        cfg["target_kl"] = bad_target
    with pytest.raises(ExecutionError):
        _install(cfg)
