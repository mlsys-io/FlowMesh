"""Per-run cancel and graceful-stop signals handed to ``Executor.run``."""

import itertools
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

type Callback = Callable[[], None]


class RunControl:
    """Cancel and graceful-stop signals for a single executor run.

    Both signals are latched and one-way. Every method is thread-safe. Only the
    runner raises signals; executors read them, wait on them, or register callbacks.
    """

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._stop = threading.Event()
        self._cancel_callbacks: dict[int, Callback] = {}
        self._stop_callbacks: dict[int, Callback] = {}
        self._tokens = itertools.count()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def wait_for_cancel(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds; return whether cancel was requested."""
        return self._cancel.wait(timeout)

    def raise_if_cancelled(self, message: str) -> None:
        """Raise ``TaskCancelledError(message)`` if cancel was requested."""
        # Imported here: base_executor imports this module for the run() signature.
        from .base_executor import TaskCancelledError

        if self._cancel.is_set():
            raise TaskCancelledError(message)

    def on_cancel(self, fn: Callback) -> Callback:
        """Run ``fn`` once when cancel is requested; return its unregister function.

        ``fn`` runs immediately on the calling thread if cancel was already requested.
        """
        return self._register(self._cancel, self._cancel_callbacks, fn)

    def on_stop(self, fn: Callback) -> Callback:
        """Run ``fn`` once when stop is requested; return its unregister function.

        ``fn`` runs immediately on the calling thread if stop was already requested.
        """
        return self._register(self._stop, self._stop_callbacks, fn)

    def request_cancel(self) -> None:
        self._raise(self._cancel, self._cancel_callbacks)

    def request_stop(self) -> None:
        self._raise(self._stop, self._stop_callbacks)

    def _register(
        self, signal: threading.Event, callbacks: dict[int, Callback], fn: Callback
    ) -> Callback:
        with self._lock:
            if not signal.is_set():
                token = next(self._tokens)
                callbacks[token] = fn

                def unregister() -> None:
                    with self._lock:
                        callbacks.pop(token, None)

                return unregister
        self._invoke(fn)
        return lambda: None

    def _raise(self, signal: threading.Event, callbacks: dict[int, Callback]) -> None:
        with self._lock:
            if signal.is_set():
                return
            signal.set()
            pending = list(callbacks.values())
            callbacks.clear()
        for fn in pending:
            self._invoke(fn)

    def _invoke(self, fn: Callback) -> None:
        try:
            fn()
        except Exception:
            logger.warning(
                "Run control callback for task %s raised",
                self._task_id,
                exc_info=True,
            )
