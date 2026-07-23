# ruff: noqa: E402
import pytest

torch = pytest.importorskip(
    "torch", reason="torch not installed (needs --extra inference)"
)

from worker.executors.transformers_executor import _select_vision_features


def _hidden_states(num_layers: int = 3):
    return tuple(
        torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) + layer * 100
        for layer in range(num_layers)
    )


def test_int_layer_default_crops_cls_token() -> None:
    hidden_states = _hidden_states()
    selected = _select_vision_features(hidden_states, -2, "default")
    assert selected.shape == (2, 3, 3)
    torch.testing.assert_close(selected, hidden_states[-2][:, 1:])


def test_int_layer_full_keeps_cls_token() -> None:
    hidden_states = _hidden_states()
    selected = _select_vision_features(hidden_states, -2, "full")
    assert selected.shape == (2, 4, 3)
    torch.testing.assert_close(selected, hidden_states[-2])


def test_list_layer_default_crops_then_concatenates_on_feature_dim() -> None:
    hidden_states = _hidden_states()
    selected = _select_vision_features(hidden_states, [-2, -1], "default")
    assert selected.shape == (2, 3, 6)
    expected = torch.cat([hidden_states[-2][:, 1:], hidden_states[-1][:, 1:]], dim=-1)
    torch.testing.assert_close(selected, expected)


def test_list_layer_full_concatenates_without_cropping() -> None:
    hidden_states = _hidden_states()
    selected = _select_vision_features(hidden_states, [0, 2], "full")
    assert selected.shape == (2, 4, 6)
    expected = torch.cat([hidden_states[0], hidden_states[2]], dim=-1)
    torch.testing.assert_close(selected, expected)
