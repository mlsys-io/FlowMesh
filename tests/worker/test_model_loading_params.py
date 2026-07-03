# # ruff: noqa: E402
"""Tests for model loading parameters (revision, trust_remote_code, quantization).

Instantiates real executor objects and calls their model-loading methods with
mocked ``from_pretrained`` to verify the correct kwargs are forwarded.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip(
    "torch", reason="torch not installed (needs --extra inference)"
)

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import (
    DiffusionSpecStrict,
    InferenceSpecStrict,
)
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors.diffusers_executor import DiffusersExecutor
from worker.executors.transformers_executor import HFTransformersExecutor
from worker.executors.vllm_executor import VLLMExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model() -> MagicMock:
    """Return a minimal mock that passes model.eval() / model.to() etc."""
    m = MagicMock()
    m.config = MagicMock()
    m.config.use_cache = True
    m.eval.return_value = m
    m.to.return_value = m
    return m


def _mock_tokenizer() -> MagicMock:
    t = MagicMock()
    t.pad_token = None
    t.eos_token = "<eos>"
    t.pad_token_id = None
    t.padding_side = "right"
    return t


# ---------------------------------------------------------------------------
# ModelSpec property tests
# ---------------------------------------------------------------------------


class TestModelSpecProperties:
    """Test convenience properties on ModelSpecStrict."""

    def _make_spec(
        self,
        identifier: str = "org/model",
        revision: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> Any:
        source = ModelSource(
            identifier=identifier,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        return InferenceSpecStrict(
            taskType=TaskType.INFERENCE, model=ModelConfig(source=source)
        )

    def test_model_name(self) -> None:
        assert self._make_spec(identifier="org/model").model_name == "org/model"

    def test_model_revision(self) -> None:
        assert self._make_spec(revision="abc123").model_revision == "abc123"

    def test_model_revision_none(self) -> None:
        assert self._make_spec().model_revision is None

    def test_trust_remote_code_true(self) -> None:
        assert self._make_spec(trust_remote_code=True).model_trust_remote_code is True

    def test_trust_remote_code_default_false(self) -> None:
        assert self._make_spec().model_trust_remote_code is False

    def test_no_model_at_all(self) -> None:
        spec = InferenceSpecStrict(taskType=TaskType.INFERENCE)
        assert spec.model_trust_remote_code is False
        assert spec.model_revision is None
        assert spec.model_name is None


# ---------------------------------------------------------------------------
# Transformers executor
# ---------------------------------------------------------------------------


class TestTransformersExecutor:
    """Verify _ensure_model passes revision / trust_remote_code."""

    @patch("worker.executors.transformers_executor.AutoModelForCausalLM")
    @patch("worker.executors.transformers_executor.AutoTokenizer")
    def test_revision_and_trust_remote_code(
        self, mock_tok_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            model=ModelConfig(
                source=ModelSource(
                    identifier="org/model",
                    revision="v2.0",
                    trust_remote_code=True,
                ),
            ),
        )
        executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
        executor._ensure_model(spec)

        tok_call = mock_tok_cls.from_pretrained.call_args
        assert tok_call.kwargs["revision"] == "v2.0"
        assert tok_call.kwargs["trust_remote_code"] is True

        model_call = mock_model_cls.from_pretrained.call_args
        assert model_call.kwargs["revision"] == "v2.0"
        assert model_call.kwargs["trust_remote_code"] is True

    @patch("worker.executors.transformers_executor.AutoModelForCausalLM")
    @patch("worker.executors.transformers_executor.AutoTokenizer")
    def test_no_revision_omits_kwarg(
        self, mock_tok_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        spec = InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            model=ModelConfig(
                source=ModelSource(identifier="org/model"),
            ),
        )
        executor = HFTransformersExecutor(DEFAULT_WORKER_CONFIG)
        executor._ensure_model(spec)

        model_call = mock_model_cls.from_pretrained.call_args
        assert "revision" not in model_call.kwargs


# ---------------------------------------------------------------------------
# Diffusers executor
# ---------------------------------------------------------------------------


class TestDiffusersExecutor:
    """Verify _ensure_pipeline passes revision / trust_remote_code."""

    @patch("worker.executors.diffusers_executor.AutoPipelineForText2Image")
    def test_revision_and_trust(self, mock_pipe_cls: MagicMock) -> None:
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe_cls.from_pretrained.return_value = mock_pipe

        spec = DiffusionSpecStrict(
            taskType=TaskType.DIFFUSION,
            model=ModelConfig(
                source=ModelSource(
                    identifier="stabilityai/sdxl",
                    revision="fp16",
                    trust_remote_code=True,
                ),
                diffusers={},
            ),
        )
        executor = DiffusersExecutor(DEFAULT_WORKER_CONFIG)
        executor._ensure_pipeline(spec)

        call_kwargs = mock_pipe_cls.from_pretrained.call_args.kwargs
        assert call_kwargs["revision"] == "fp16"
        assert call_kwargs["trust_remote_code"] is True

    @patch("worker.executors.diffusers_executor.AutoPipelineForText2Image")
    def test_no_revision(self, mock_pipe_cls: MagicMock) -> None:
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe_cls.from_pretrained.return_value = mock_pipe

        spec = DiffusionSpecStrict(
            taskType=TaskType.DIFFUSION,
            model=ModelConfig(
                source=ModelSource(identifier="stabilityai/sdxl"),
                diffusers={},
            ),
        )
        executor = DiffusersExecutor(DEFAULT_WORKER_CONFIG)
        executor._ensure_pipeline(spec)

        call_kwargs = mock_pipe_cls.from_pretrained.call_args.kwargs
        assert "revision" not in call_kwargs
        assert "trust_remote_code" not in call_kwargs


# ---------------------------------------------------------------------------
# vLLM engine reuse
# ---------------------------------------------------------------------------


class TestVLLMEngineReuse:
    """The engine-reuse key must include the model identifier/revision so a
    different model forces a reload rather than silently reusing the wrong one.
    """

    def test_inference_reloads_engine_on_model_change(self) -> None:
        pytest.importorskip("vllm")
        executor = VLLMExecutor(DEFAULT_WORKER_CONFIG)
        built: list[str] = []
        shutdowns: list[int] = []

        def fake_llm(**kwargs: Any) -> Any:
            built.append(kwargs["model"])
            return SimpleNamespace()

        def fake_shutdown() -> None:
            shutdowns.append(1)
            executor._llm = None

        def _spec(identifier: str) -> InferenceSpecStrict:
            return InferenceSpecStrict(
                taskType=TaskType.INFERENCE,
                model=ModelConfig(source=ModelSource(identifier=identifier), vllm={}),
            )

        with (
            patch("worker.executors.vllm_executor.LLM", side_effect=fake_llm),
            patch.object(executor, "_shutdown_llm", side_effect=fake_shutdown),
        ):
            executor._ensure_llm(_spec("org/gen-a"), ["t1"])
            # Same identifier + config: the loaded engine is reused, no reload.
            executor._ensure_llm(_spec("org/gen-a"), ["t2"])
            assert built == ["org/gen-a"]
            assert shutdowns == []
            # Different identifier (same vllm/checkpoint): reuse key differs → reload.
            executor._ensure_llm(_spec("org/gen-b"), ["t3"])
            assert shutdowns == [1]
            assert built == ["org/gen-a", "org/gen-b"]


# ---------------------------------------------------------------------------
# vLLM sampling params
# ---------------------------------------------------------------------------


class TestVLLMSamplingParams:
    """Verify _build_sampling_params forwards new fields correctly."""

    def _build(self, inference_cfg: dict[str, Any]) -> Any:
        pytest.importorskip("vllm")
        from worker.executors.vllm_executor import VLLMExecutor

        executor = VLLMExecutor(DEFAULT_WORKER_CONFIG)
        return executor._build_sampling_params(inference_cfg)

    def test_n_forwarded_when_set(self) -> None:
        params = self._build({"n": 4})
        assert params.n == 4

    def test_n_defaults_to_one_when_unset(self) -> None:
        params = self._build({})
        assert params.n == 1

    def test_new_sampling_fields_forwarded(self) -> None:
        params = self._build(
            {
                "min_p": 0.05,
                "min_tokens": 10,
                "repetition_penalty": 1.1,
                "skip_special_tokens": False,
                "seed": 42,
            }
        )
        assert params.min_p == 0.05
        assert params.min_tokens == 10
        assert params.repetition_penalty == 1.1
        assert params.skip_special_tokens is False
        assert params.seed == 42

    def test_stop_token_ids_and_bad_words(self) -> None:
        params = self._build({"stop_token_ids": [2150], "bad_words": ["foo"]})
        assert 2150 in params.stop_token_ids
        assert params.bad_words == ["foo"]


# ---------------------------------------------------------------------------
# huggingface: shared helpers used by all training executors
# ---------------------------------------------------------------------------


class TestPickTorchDtype:
    """Verify ``huggingface.pick_torch_dtype`` honours ``bf16``/``fp16`` flags."""

    def _call(self, training_cfg: dict[str, Any], *, cuda_available: bool) -> Any:
        from worker.executors.utils import huggingface

        with patch.object(torch.cuda, "is_available", return_value=cuda_available):
            return huggingface.pick_torch_dtype(training_cfg)

    def test_bf16_returns_bfloat16(self) -> None:
        assert self._call({"bf16": True}, cuda_available=True) is torch.bfloat16

    def test_fp16_returns_float16(self) -> None:
        assert self._call({"fp16": True}, cuda_available=True) is torch.float16

    def test_bf16_preferred_over_fp16(self) -> None:
        dtype = self._call({"bf16": True, "fp16": True}, cuda_available=True)
        assert dtype is torch.bfloat16

    def test_no_flags_returns_none(self) -> None:
        assert self._call({}, cuda_available=True) is None

    def test_bf16_without_cuda_returns_none(self) -> None:
        assert self._call({"bf16": True}, cuda_available=False) is None


class TestBuildHFLoadKwargs:
    """Verify ``huggingface.build_hf_load_kwargs`` assembles from_pretrained kwargs."""

    def _build(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        from worker.executors.utils import huggingface

        defaults: dict[str, Any] = {
            "revision": None,
            "trust_remote_code": False,
            "training_cfg": {},
            "torch_dtype": None,
        }
        defaults.update(kwargs)
        return huggingface.build_hf_load_kwargs(**defaults)

    def test_revision_forwarded_to_both(self) -> None:
        tok_kwargs, model_kwargs = self._build(revision="abc123")
        assert tok_kwargs == {"revision": "abc123"}
        assert model_kwargs == {"revision": "abc123"}

    def test_trust_remote_code_forwarded_to_both(self) -> None:
        tok_kwargs, model_kwargs = self._build(trust_remote_code=True)
        assert tok_kwargs == {"trust_remote_code": True}
        assert model_kwargs == {"trust_remote_code": True}

    def test_no_revision_omits_kwarg(self) -> None:
        tok_kwargs, model_kwargs = self._build()
        assert "revision" not in tok_kwargs
        assert "revision" not in model_kwargs

    def test_trust_remote_code_false_omits_kwarg(self) -> None:
        tok_kwargs, model_kwargs = self._build(trust_remote_code=False)
        assert "trust_remote_code" not in tok_kwargs
        assert "trust_remote_code" not in model_kwargs

    def test_dtype_forwarded_to_model_only(self) -> None:
        tok_kwargs, model_kwargs = self._build(torch_dtype=torch.bfloat16)
        assert "dtype" not in tok_kwargs
        assert model_kwargs["dtype"] is torch.bfloat16

    def test_none_dtype_omits_kwarg(self) -> None:
        _, model_kwargs = self._build(torch_dtype=None)
        assert "dtype" not in model_kwargs

    def test_4bit_quantization_config(self) -> None:
        pytest.importorskip("bitsandbytes")
        _, model_kwargs = self._build(
            training_cfg={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": False,
            },
            torch_dtype=torch.bfloat16,
        )
        qc = model_kwargs["quantization_config"]
        assert qc.load_in_4bit is True
        assert qc.bnb_4bit_quant_type == "nf4"
        assert qc.bnb_4bit_compute_dtype is torch.bfloat16
        assert qc.bnb_4bit_use_double_quant is False

    def test_8bit_quantization_path(self) -> None:
        _, model_kwargs = self._build(training_cfg={"load_in_8bit": True})
        assert model_kwargs.get("load_in_8bit") is True
        assert "quantization_config" not in model_kwargs

    def test_4bit_takes_precedence_over_8bit(self) -> None:
        pytest.importorskip("bitsandbytes")
        _, model_kwargs = self._build(
            training_cfg={"load_in_4bit": True, "load_in_8bit": True},
        )
        assert "quantization_config" in model_kwargs
        assert "load_in_8bit" not in model_kwargs

    def test_bnb_compute_dtype_uses_torch_dtype_when_set(self) -> None:
        pytest.importorskip("bitsandbytes")
        _, model_kwargs = self._build(
            training_cfg={"load_in_4bit": True}, torch_dtype=torch.float16
        )
        assert model_kwargs["quantization_config"].bnb_4bit_compute_dtype is (
            torch.float16
        )

    def test_bnb_compute_dtype_falls_back_to_bfloat16_when_supported(self) -> None:
        pytest.importorskip("bitsandbytes")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "is_bf16_supported", return_value=True),
        ):
            _, model_kwargs = self._build(
                training_cfg={"load_in_4bit": True}, torch_dtype=None
            )
        assert model_kwargs["quantization_config"].bnb_4bit_compute_dtype is (
            torch.bfloat16
        )

    def test_bnb_compute_dtype_falls_back_to_float16_when_bf16_unsupported(
        self,
    ) -> None:
        pytest.importorskip("bitsandbytes")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "is_bf16_supported", return_value=False),
        ):
            _, model_kwargs = self._build(
                training_cfg={"load_in_4bit": True}, torch_dtype=None
            )
        assert model_kwargs["quantization_config"].bnb_4bit_compute_dtype is (
            torch.float16
        )
