import json
import logging
import threading
from collections.abc import Callable

import httpx
from lumid_hooks import PrincipalContext

from shared.schemas.event import NodeEvent, serialize_event
from shared.schemas.node import NodeInfo
from shared.utils.http import auth_headers
from shared.utils.time import now_iso

from ...clients.redis import NODE_EVENT_CHANNEL, SyncRedisClient
from ...config import NodeRole
from ...registries.node import NodeRegistry


class Lifecycle:
    """Manages node registration, heartbeats, and unregistration."""

    def __init__(
        self,
        redis: SyncRedisClient,
        node_registry: NodeRegistry,
        node_info: NodeInfo,
        role: NodeRole,
        base_url: str,
        hb_sec: int,
        hb_ttl_sec: int,
        logger: logging.Logger,
        system_principal: PrincipalContext,
        current_gpu_count_getter: Callable[[], int] | None = None,
        on_reregister: Callable[[str], None] | None = None,
    ) -> None:
        self._redis = redis
        self._node_registry = node_registry
        self._node_info = node_info
        self._role = role
        self._base_url = base_url
        self.hb_sec = hb_sec
        self.hb_ttl_sec = hb_ttl_sec
        self.logger = logger
        self._system_principal = system_principal
        self._current_gpu_count_getter = current_gpu_count_getter
        self._on_reregister = on_reregister

        self._node_id: str | None = None
        self._stop_event = threading.Event()
        self._stop_event.set()  # Initially stopped
        self._hb_thread: threading.Thread | None = None
        self._hb_lock = threading.Lock()
        self._unregister_published: bool = False

    @property
    def node_id(self) -> str:
        """Return the assigned node_id. Only valid after ``start()``."""
        if self._node_id is None:
            raise RuntimeError("Lifecycle not started; node_id not yet assigned")
        return self._node_id

    def set_reregister_callback(self, callback: Callable[[str], None]) -> None:
        """Register a hook invoked with the new node id after a re-register.

        Lets components that captured the original node id (dispatch/command
        subscriptions, worker homing) rebind once the node re-registers under a
        new id, so dispatch is restored — not just registration.
        """
        self._on_reregister = callback

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def _register(self) -> str:
        """Register with the root and return the assigned node_id."""
        if self._role is NodeRole.ROOT:
            return self._register_direct()
        return self._register_http()

    def _register_direct(self) -> str:
        """Root node: register directly via NodeRegistry (Redis)."""
        node_id = self._node_registry.register_node(self._node_info)
        self.logger.info("Node registered (direct): %s", node_id)
        return node_id

    def _register_http(self) -> str:
        """Worker node: register via HTTP on the root node."""
        url = f"{self._base_url.rstrip('/')}/api/v1/nodes/register"
        self.logger.info(
            "Registering node %s with root at %s", self._node_info.alias, url
        )
        payload = self._node_info.model_dump()
        resp = httpx.post(url, json=payload, headers=auth_headers(), timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        node_id = data.get("node_id")
        if not node_id:
            raise RuntimeError("Node registration failed: missing node_id")
        self.logger.info("Node registered (HTTP): %s", node_id)
        return node_id

    # ------------------------------------------------------------------ #
    # Event publishing
    # ------------------------------------------------------------------ #

    def _publish_event(self, event_type: str, **extra: object) -> None:
        event = NodeEvent(
            type=event_type,
            ts=now_iso(),
            node_id=self.node_id,
            tags=[],
            payload={},
            actor=self._system_principal.model_dump(),
        )
        payload = serialize_event(event) | extra
        self._redis.publish_telemetry(NODE_EVENT_CHANNEL, json.dumps(payload))

    def _current_gpu_count(self) -> int | None:
        getter = self._current_gpu_count_getter
        if getter is None:
            return None
        try:
            return max(0, int(getter()))
        except Exception as exc:
            self.logger.warning(
                "Failed to determine current GPU count for node %s: %s",
                self._node_id,
                exc,
            )
            return None

    def heartbeat_now(self) -> None:
        with self._hb_lock:
            ts = now_iso()
            self._node_registry.update_node_hb(
                self.node_id,
                ts,
                self.hb_ttl_sec,
                current_gpu_count=self._current_gpu_count(),
            )
            gpu_count = self._current_gpu_count()
            hb_payload: dict[str, object] = {"ttl_sec": self.hb_ttl_sec}
            if gpu_count is not None:
                hb_payload["current_gpu_count"] = gpu_count
            self._publish_event("SV_HEARTBEAT", payload=hb_payload)

    def start(self) -> str:
        if self._hb_thread is not None:
            self.logger.warning("Node lifecycle already started")
            return self.node_id

        self._node_id = self._register()
        self._unregister_published = False
        self._publish_event("SV_REGISTER")
        self.heartbeat_now()

        self._stop_event.clear()
        self._hb_thread = threading.Thread(
            target=self._hb_loop,
            name=f"NodeLifecycle[{self._node_id}]",
            daemon=True,
        )
        self._hb_thread.start()
        return self._node_id

    def _reregister_if_lost(self) -> None:
        """Re-register when the root registry no longer holds this node.

        The node record lives in the shared control-plane Redis. When the root
        is redeployed its registry is rebuilt from scratch, dropping records for
        nodes that registered before the restart. Those nodes keep heartbeating,
        but ``update_node_hb`` early-returns for an unknown node, so they never
        reappear and are orphaned. Detect the missing record and re-register so
        the fleet self-heals within one heartbeat interval, without a restart.
        """
        if self._node_registry.node_exists(self.node_id):
            return
        self.logger.warning(
            "Node %s missing from root registry; re-registering", self._node_id
        )
        self._node_id = self._register()
        self._unregister_published = False
        self._publish_event("SV_REGISTER")
        self.logger.info("Node re-registered as %s", self._node_id)
        if self._on_reregister is not None:
            try:
                self._on_reregister(self._node_id)
            except Exception as exc:
                self.logger.warning(
                    "Re-register callback failed for node %s: %s",
                    self._node_id,
                    exc,
                )

    def _hb_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._reregister_if_lost()
                self.heartbeat_now()
            except Exception as exc:
                self.logger.warning(
                    "Node heartbeat failed for %s: %s", self._node_id, exc
                )
            self._stop_event.wait(self.hb_sec)

    def publish_unregister(self) -> None:
        if self._unregister_published:
            return
        self._publish_event("SV_UNREGISTER")
        self._unregister_published = True

    def stop(self) -> None:
        self._stop_event.set()

        if self._hb_thread is not None:
            self._hb_thread.join()
            self._hb_thread = None

        try:
            self.publish_unregister()
        finally:
            try:
                self._node_registry.unregister_node(self.node_id)
            except Exception as exc:
                self.logger.warning(
                    "Failed to unregister node %s during shutdown: %s",
                    self._node_id,
                    exc,
                )
