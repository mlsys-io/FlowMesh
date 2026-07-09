"""Tests for the serve spec dispatch validation."""

from typing import Any

import pytest

from shared.tasks.specs import ServeSpecStrict, ServeSpecTemplate


def _strict(**fields: Any) -> ServeSpecStrict:
    return ServeSpecStrict.model_validate({"taskType": "serve", **fields})


def _template(**fields: Any) -> ServeSpecTemplate:
    return ServeSpecTemplate.model_validate({"taskType": "serve", **fields})


class TestValidateDispatchable:
    def test_without_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _strict(
                model={"source": {"identifier": "Qwen/Qwen3-7B"}}
            ).validate_dispatchable()

    def test_zero_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _strict(
                model={"source": {"identifier": "Qwen/Qwen3-7B"}},
                resources={"hardware": {"gpu": {"count": 0}}},
            ).validate_dispatchable()

    def test_with_gpu_ok(self) -> None:
        _strict(
            model={"source": {"identifier": "Qwen/Qwen3-7B"}},
            resources={"hardware": {"gpu": {"count": 1}}},
        ).validate_dispatchable()

    def test_template_without_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _template(
                model={"source": {"identifier": "Qwen/Qwen3-7B"}}
            ).validate_dispatchable()

    def test_template_with_gpu_ok(self) -> None:
        _template(
            model={"source": {"identifier": "Qwen/Qwen3-7B"}},
            resources={"hardware": {"gpu": {"count": 2}}},
        ).validate_dispatchable()
