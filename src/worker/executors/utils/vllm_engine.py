"""Liveness watch for the out-of-process vLLM engine core."""

import logging
import queue
import threading
from concurrent.futures import Future, InvalidStateError
from typing import TYPE_CHECKING, Any

try:
    from vllm.v1.engine.core_client import BackgroundResources, SyncMPClient
    from vllm.v1.engine.exceptions import EngineDeadError
except Exception:
    if TYPE_CHECKING:
        from vllm.v1.engine.core_client import BackgroundResources, SyncMPClient
        from vllm.v1.engine.exceptions import EngineDeadError
    else:
        BackgroundResources = None
        SyncMPClient = None
        EngineDeadError = None

if TYPE_CHECKING:
    from vllm import LLM

logger = logging.getLogger(__name__)


class EngineCoreWatch:
    """Wakes callers blocked on a vLLM engine core whose process has died.

    vLLM's own monitor marks the client dead when the engine core exits and
    rejects new requests, but a call already waiting on outputs or a utility
    result is never woken and blocks forever. The watch polls that mark and, once
    it is set, fails those waits with ``EngineDeadError``.

    An engine core running in-process has no separate process to lose, so the
    watch is inert for it. Only the client's queue, pending utility futures and
    resources are held, never the client, so the client is still finalized when
    it is dropped.
    """

    def __init__(self, llm: "LLM", poll_interval: float = 1.0) -> None:
        self._resources: BackgroundResources | None = None
        self._outputs: queue.Queue[Any] | None = None
        self._utility_results: dict[int, Future[Any]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        client = llm.llm_engine.engine_core
        if SyncMPClient is None or not isinstance(client, SyncMPClient):
            return
        self._resources = client.resources
        self._outputs = client.outputs_queue
        self._utility_results = client.utility_results
        self._thread = threading.Thread(
            target=self._watch,
            args=(poll_interval,),
            name="EngineCoreWatch",
            daemon=True,
        )
        self._thread.start()

    @property
    def dead(self) -> bool:
        return self._resources is not None and self._resources.engine_dead

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _watch(self, poll_interval: float) -> None:
        while not self._stop.wait(poll_interval):
            if self.dead:
                logger.warning("vLLM engine core exited; failing calls waiting on it")
                self._wake()
                return

    def _wake(self) -> None:
        assert self._outputs is not None
        self._outputs.put_nowait(EngineDeadError())
        for future in list(self._utility_results.values()):
            try:
                future.set_exception(EngineDeadError())
            except InvalidStateError:
                pass
