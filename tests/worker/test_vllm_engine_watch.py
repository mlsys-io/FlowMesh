"""Tests for waking calls blocked on a dead vLLM engine core, and the executor's
reload of a dead engine. The engine client is faked, so no GPU is needed."""

import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("vllm.v1.engine.core_client", reason="vllm not installed")

from vllm.v1.engine.core_client import SyncMPClient  # noqa: E402
from vllm.v1.engine.exceptions import EngineDeadError  # noqa: E402

from worker.executors.utils.vllm_engine import EngineCoreWatch  # noqa: E402
from worker.executors.vllm_executor import VLLMExecutor  # noqa: E402

POLL_SEC = 0.01
WAIT_SEC = 5.0


def _client() -> SyncMPClient:
    client = SyncMPClient.__new__(SyncMPClient)
    client.resources = cast(Any, SimpleNamespace(engine_dead=False))
    client.outputs_queue = queue.Queue()
    client.utility_results = {}
    return client


def _llm(engine_core: Any) -> Any:
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=engine_core))


@pytest.fixture
def client() -> SyncMPClient:
    return _client()


@pytest.fixture
def watch(client: SyncMPClient) -> Iterator[EngineCoreWatch]:
    watch = EngineCoreWatch(_llm(client), poll_interval=POLL_SEC)
    yield watch
    watch.stop()


def _kill(client: SyncMPClient) -> None:
    cast(Any, client.resources).engine_dead = True


def test_blocked_get_output_raises_when_engine_core_dies(
    client: SyncMPClient, watch: EngineCoreWatch
) -> None:
    raised: list[BaseException] = []

    def _wait_for_output() -> None:
        try:
            client.get_output()
        except BaseException as exc:
            raised.append(exc)

    waiter = threading.Thread(target=_wait_for_output, daemon=True)
    waiter.start()
    _kill(client)
    waiter.join(timeout=WAIT_SEC)

    assert not waiter.is_alive()
    assert len(raised) == 1 and isinstance(raised[0], EngineDeadError)
    assert watch.dead


def test_pending_utility_call_fails_when_engine_core_dies(
    client: SyncMPClient, watch: EngineCoreWatch
) -> None:
    pending: Future[Any] = Future()
    answered: Future[Any] = Future()
    answered.set_result("ok")
    client.utility_results.update({1: pending, 2: answered})

    _kill(client)

    assert isinstance(pending.exception(timeout=WAIT_SEC), EngineDeadError)
    assert answered.result() == "ok"


def test_live_engine_is_left_alone(
    client: SyncMPClient, watch: EngineCoreWatch
) -> None:
    time.sleep(POLL_SEC * 10)
    assert not watch.dead
    assert client.outputs_queue.empty()


def test_stopped_watch_wakes_nothing(
    client: SyncMPClient, watch: EngineCoreWatch
) -> None:
    watch.stop()
    _kill(client)
    time.sleep(POLL_SEC * 10)
    assert client.outputs_queue.empty()


def test_in_process_engine_core_is_not_watched() -> None:
    watch = EngineCoreWatch(_llm(object()), poll_interval=POLL_SEC)
    assert not watch.dead
    watch.stop()


def _executor() -> VLLMExecutor:
    return VLLMExecutor(cast(Any, SimpleNamespace()), lifecycle=None)


def _init(executor: VLLMExecutor) -> None:
    executor._init_vllm_engine(
        ident="org/model",
        vllm_cfg={},
        checkpoint_cfg={},
        new_inference_spec={"same": True},
        requested_gpu_count=1,
        revision=None,
        extra_llm_kwargs={},
        adjust_tp=lambda size: size,
        task_ids=None,
    )


def _loaded(executor: VLLMExecutor, *, dead: bool) -> None:
    executor._llm = cast(Any, object())
    executor._inference_spec = {"same": True}
    executor._engine_watch = MagicMock(spec=EngineCoreWatch, dead=dead)


def test_live_engine_is_reused() -> None:
    executor = _executor()
    _loaded(executor, dead=False)
    with (
        patch.object(VLLMExecutor, "_shutdown_llm") as shutdown,
        patch("worker.executors.vllm_executor.LLM") as llm_cls,
    ):
        _init(executor)
    shutdown.assert_not_called()
    llm_cls.assert_not_called()


def test_dead_engine_is_shut_down_and_reloaded() -> None:
    executor = _executor()
    _loaded(executor, dead=True)

    def _drop(self: VLLMExecutor) -> None:
        self._llm = None
        self._engine_watch = None

    with (
        patch.object(
            VLLMExecutor, "_shutdown_llm", autospec=True, side_effect=_drop
        ) as shutdown,
        patch.object(
            VLLMExecutor, "_compute_safe_utilization", return_value=(0.9, None)
        ),
        patch(
            "worker.executors.vllm_executor.LLM", return_value=_llm(_client())
        ) as llm_cls,
    ):
        _init(executor)

    shutdown.assert_called_once()
    llm_cls.assert_called_once()
    assert executor._engine_watch is not None and not executor._engine_watch.dead
    executor._engine_watch.stop()


def test_shutdown_stops_the_watch_before_the_client() -> None:
    client = _client()
    executor = _executor()
    with (
        patch.object(
            VLLMExecutor, "_compute_safe_utilization", return_value=(0.9, None)
        ),
        patch("worker.executors.vllm_executor.LLM", return_value=_llm(client)),
    ):
        _init(executor)
    assert isinstance(executor._engine_watch, EngineCoreWatch)

    watches_at_client_shutdown: list[EngineCoreWatch | None] = []

    def _client_shutdown(timeout: float | None = None) -> None:
        # A real client shutdown marks itself dead, which a running watch would
        # take for a crash.
        watches_at_client_shutdown.append(executor._engine_watch)
        _kill(client)

    with patch.object(client, "shutdown", side_effect=_client_shutdown):
        executor._shutdown_llm()

    assert watches_at_client_shutdown == [None]
    assert client.outputs_queue.empty()
