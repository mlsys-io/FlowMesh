"""Local ports the supervisor may ask this worker to relay to."""

import logging
import threading

logger = logging.getLogger(__name__)


class EndpointRegistry:
    """The ports a worker will relay to, keyed by the id the supervisor names.

    An executor publishes an endpoint once it is listening and withdraws it once
    it stops. Resolution is the only authority on what a relay may reach: an id
    that was never published, or has been withdrawn, resolves to nothing.
    """

    def __init__(self) -> None:
        self._ports: dict[str, int] = {}
        self._lock = threading.Lock()

    def publish(self, endpoint_id: str, port: int) -> None:
        with self._lock:
            previous = self._ports.get(endpoint_id)
            self._ports[endpoint_id] = port
        if previous is not None and previous != port:
            logger.warning(
                "Endpoint %s republished on port %s, replacing port %s",
                endpoint_id,
                port,
                previous,
            )

    def withdraw(self, endpoint_id: str) -> None:
        with self._lock:
            self._ports.pop(endpoint_id, None)

    def resolve(self, endpoint_id: str) -> int | None:
        with self._lock:
            return self._ports.get(endpoint_id)
