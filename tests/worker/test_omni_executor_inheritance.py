"""Compile-time guard that omni executors sit on the mixin chain.

The mixin chain is the same one the inference and training executors use; the
omni family was the last to predate it. This test fails loudly if a future
refactor regresses the base class.
"""

import pytest

pytest.importorskip("vllm_omni", reason="vllm-omni not installed")

from worker.executors.mixins.data import DataMixin
from worker.executors.mixins.governance import GovernanceMixin
from worker.executors.mixins.inference import InferenceMixin
from worker.executors.omni_executor_base import OmniExecutorBase
from worker.executors.omni_text2audio_executor import OmniText2AudioExecutor
from worker.executors.omni_text2general_executor import OmniText2GeneralExecutor
from worker.executors.omni_text2image_executor import OmniText2ImageExecutor
from worker.executors.omni_text2speech_executor import OmniText2SpeechExecutor


@pytest.mark.parametrize(
    "cls",
    [
        OmniExecutorBase,
        OmniText2AudioExecutor,
        OmniText2GeneralExecutor,
        OmniText2ImageExecutor,
        OmniText2SpeechExecutor,
    ],
    ids=lambda c: c.__name__,
)
def test_omni_executors_use_mixin_chain(cls: type) -> None:
    assert issubclass(cls, InferenceMixin)
    assert issubclass(cls, DataMixin)
    assert issubclass(cls, GovernanceMixin)
