"""Node direct API client."""

import os
from typing import Any

import httpx
from flowmesh.exceptions import FlowMeshConnectionError, FlowMeshError

_DEFAULT_TIMEOUT = 30.0
_WORKER_CREATE_TIMEOUT = 600.0


def _raise_for_status(response: httpx.Response, method: str) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.json()
    except Exception:
        body = response.text
    message = ""
    if isinstance(body, dict):
        message = body.get("detail", "") or body.get("message", "")
    if not message:
        message = str(body)
    raise FlowMeshError(
        f"{method} {response.url} returned {response.status_code}: {message}"
    )


class NodeClient:
    """Client for a node's worker management HTTP API.

    Used by the CLI stack sub-package for worker lifecycle management.

    Args:
        base_url: Node HTTP endpoint (e.g. ``http://localhost:8000``).
        token: Optional bearer token for authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_url = base_url or _default_node_url()
        resolved_token = token or os.getenv("FLOWMESH_API_KEY") or None
        headers: dict[str, str] = {"Accept": "application/json"}
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"
        self._base_url = resolved_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    # -- Workers --------------------------------------------------------- #

    def list_workers(self) -> list[dict[str, Any]]:
        """List all workers managed by this node."""
        return self._request("GET", "/api/v1/stack/workers")

    def create_worker(self, config: str | dict[str, Any]) -> dict[str, Any]:
        """Create a worker from a JSON/YAML config.

        Args:
            config: Worker init config as a JSON string or dict.
        """
        if isinstance(config, dict):
            return self._request(
                "POST",
                "/api/v1/stack/workers",
                json_body=config,
                timeout=_WORKER_CREATE_TIMEOUT,
            )
        return self._request(
            "POST",
            "/api/v1/stack/workers",
            data=config,
            headers={"Content-Type": "application/json"},
            timeout=_WORKER_CREATE_TIMEOUT,
        )

    def start_worker(self, name: str) -> None:
        """Start a stopped worker."""
        self._request("POST", f"/api/v1/stack/workers/{name}/start")

    def stop_worker(self, name: str) -> None:
        """Stop a running worker."""
        self._request("POST", f"/api/v1/stack/workers/{name}/stop")

    def destroy_worker(self, name: str) -> None:
        """Destroy a single worker, removing its container."""
        self._request("DELETE", f"/api/v1/stack/workers/{name}")

    def destroy_all_workers(self, *, ignore_unreachable: bool = False) -> bool:
        """Destroy all workers managed by this node.

        Returns ``True`` on success, ``False`` when ``ignore_unreachable=True``
        and the FlowMesh server was unreachable. Other errors propagate.
        """
        try:
            self._request(
                "DELETE",
                "/api/v1/stack/workers",
                headers={"Content-Type": "application/json"},
            )
        except FlowMeshConnectionError:
            if not ignore_unreachable:
                raise
            return False
        return True

    def drain_workers(self, *, ignore_unreachable: bool = False) -> bool:
        """Drain the node's managed workers ahead of a service restart.

        Destroys every worker the node manages so their in-flight tasks are
        released (``WORKER_UNREGISTER`` → requeue) before the worker-managing
        service is recreated. Returns ``True`` on success, ``False`` when
        ``ignore_unreachable=True`` and the server was unreachable.
        """
        return self.destroy_all_workers(ignore_unreachable=ignore_unreachable)

    def worker_names(self) -> list[str]:
        """Return a list of all worker names."""
        data = self.list_workers()
        names: list[str] = []
        for item in data:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        return names

    # -- Transport ------------------------------------------------------- #

    def _request(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        data: str | bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = path
        kwargs: dict[str, Any] = {}
        if json_body is not None:
            kwargs["json"] = json_body
        if data is not None:
            kwargs["content"] = data
        if headers:
            kwargs["headers"] = headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(
                f"Failed to connect to {self._base_url}{path}: {exc}"
            )
        _raise_for_status(response, method)
        if not response.content:
            return None
        return response.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "NodeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _default_node_url() -> str:
    port = os.getenv("SERVER_HTTP_PORT", os.getenv("SERVER_APP_PORT", "8000"))
    return f"http://localhost:{port}"
