import logging
import time
from abc import ABC, abstractmethod
from threading import Event, Lock
from typing import Any

from redis.client import PubSub

from ...clients.redis import REDIS_CONN_ERRORS, SyncRedisClient, parse_pubsub_message

_POLL_TIMEOUT_SEC = 0.25
_RECONNECT_BACKOFF_SEC = 1.0


class RebindableReader(ABC):
    """Reader for a per-node control-plane pub/sub channel that can be rebound to a new
    node id and recover a dropped connection.

    A redis-py ``PubSub`` is single-thread-only, so both mutations happen on the thread
    that owns it: ``rebind``, called from another thread, only records the target id,
    and the reader applies the actual ``subscribe``/``unsubscribe`` between
    ``get_message`` polls via ``_apply_pending_rebind``. On a dropped connection, the
    reader re-subscribes with backoff, which guarantees a pending rebind eventually
    lands. Subclasses supply the channel name and per-message handling.
    """

    _label = "reader"

    def __init__(
        self, redis: SyncRedisClient, node_id: str, logger: logging.Logger
    ) -> None:
        self.logger = logger
        self._redis = redis
        # node_id is mutated only by the reader thread once running.
        self._node_id = node_id
        self._pubsub: PubSub | None = None
        self._running = False

        self._rebind_lock = Lock()
        self._pending_node_id: str | None = None
        self._rebind_applied = Event()

    @abstractmethod
    def _channel(self, node_id: str) -> str:
        """Return the pub/sub channel this reader subscribes to for a node id."""

    @abstractmethod
    def _handle_message(self, data: Any) -> None:
        """Process one decoded pub/sub payload."""

    def rebind(self, node_id: str) -> None:
        """Request moving the subscription to a new node id.

        Records the target under a lock; the reader thread applies the actual
        ``subscribe``/``unsubscribe`` between polls. ``wait_rebound`` blocks until
        the switch has taken effect.
        """
        if node_id == self._node_id:
            self._rebind_applied.set()
            return
        self._rebind_applied.clear()
        with self._rebind_lock:
            self._pending_node_id = node_id

    def wait_rebound(self, timeout: float) -> bool:
        return self._rebind_applied.wait(timeout)

    def _subscribe(self) -> None:
        """Open the initial subscription to the current node's channel."""
        self._pubsub = self._redis.subscribe_control(self._channel(self._node_id))

    def _apply_pending_rebind(self, current_id: str) -> str:
        pubsub = self._pubsub
        if pubsub is None:
            raise RuntimeError(f"{self._label} pubsub not initialized")
        with self._rebind_lock:
            pending = self._pending_node_id
            self._pending_node_id = None
        if pending is None or pending == current_id:
            return current_id
        try:
            pubsub.subscribe(self._channel(pending))
            pubsub.unsubscribe(self._channel(current_id))
        except REDIS_CONN_ERRORS:
            # The connection dropped mid-switch; re-arm the target (unless a newer
            # rebind already superseded it) so the reader re-applies it after
            # reconnecting, rather than silently dropping the move.
            with self._rebind_lock:
                if self._pending_node_id is None:
                    self._pending_node_id = pending
            raise
        self._node_id = pending
        self._rebind_applied.set()
        self.logger.info(
            "%s rebound from node %s to %s", self._label, current_id, pending
        )
        return pending

    def _resubscribe(self, current_id: str) -> bool:
        """Re-establish the subscription after a dropped connection, retrying
        with backoff until it succeeds or the reader is stopped."""
        if self._pubsub is not None:
            try:
                self._pubsub.close()
            except REDIS_CONN_ERRORS:
                pass
        while self._running:
            # Backoff before every attempt so a connection that accepts SUBSCRIBE
            # but drops on the next read can't drive a tight reconnect loop.
            time.sleep(_RECONNECT_BACKOFF_SEC)
            try:
                self._pubsub = self._redis.subscribe_control(self._channel(current_id))
            except REDIS_CONN_ERRORS as exc:
                self.logger.warning(
                    "%s resubscribe failed (%s); retrying", self._label, exc
                )
                continue
            self.logger.info("%s reconnected on node %s", self._label, current_id)
            return True
        return False

    def _read_loop(self) -> None:
        current_id = self._node_id
        while self._running and (pubsub := self._pubsub):
            try:
                current_id = self._apply_pending_rebind(current_id)
                msg = pubsub.get_message(timeout=_POLL_TIMEOUT_SEC)
                data = parse_pubsub_message(msg)
                if data is not None:
                    try:
                        self._handle_message(data)
                    except Exception as exc:
                        # A malformed frame must not take down the reader; handlers
                        # touch no Redis, so this can't mask a connection drop.
                        self.logger.exception(
                            "%s failed to handle message: %s", self._label, exc
                        )
            except REDIS_CONN_ERRORS as exc:
                if not self._running:
                    break
                self.logger.warning(
                    "%s connection lost (%s); reconnecting", self._label, exc
                )
                self._resubscribe(current_id)
            except Exception as exc:
                if self._running:
                    self.logger.exception("%s loop error: %s", self._label, exc)
                break
