import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx

from shared.schemas.result import APIResult
from shared.tasks.specs import ApiSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.redact import is_credential_key

from .base_executor import ExecutionError, Executor, ExecutorTask, TaskCancelledError

logger = logging.getLogger(__name__)

# Cache key: (base_url, timeout_seconds, verify_tls, follow_redirects)
_ClientKey = tuple[str, float, bool, bool]

# Fixed delay between retry attempts.
_RETRY_BACKOFF_SEC = 1.0


def _is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status is transient and worth retrying."""
    return status_code >= 500 or status_code in (408, 429)


class APIExecutor(Executor):
    """Performs a single HTTP request defined by task YAML.

    Defaults to the Nebula endpoint via ``NEBULA_API_BASE_URL`` and authenticates
    with ``NEBULA_API_TOKEN``. ``spec.api.url`` overrides the endpoint and
    ``spec.api.headers`` may supply a credential header (``Authorization``,
    ``X-API-Key``, etc.) directly. A custom ``spec.api.url`` requires its own
    credential: the Nebula token is never sent to an endpoint the caller chose.
    """

    name = "api"
    supported_task_types = frozenset({TaskType.API})

    # ---- Class-level connection pool (shared across all instances) ----
    _clients: ClassVar[dict[_ClientKey, httpx.Client]] = {}
    _clients_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_event = threading.Event()

    def cancel(self, task_id: str) -> None:
        """Signal the executor to abort the current request and any retries."""
        self._cancel_event.set()

    @classmethod
    def _base_url(cls, url: str) -> str:
        """Extract scheme + host + port from a URL for pool keying."""
        parsed = httpx.URL(url)
        # httpx.URL exposes .scheme, .host, .port; rebuild origin string
        port = parsed.port
        if port is None:
            return f"{parsed.scheme}://{parsed.host}"
        return f"{parsed.scheme}://{parsed.host}:{port}"

    @classmethod
    def _get_client(
        cls,
        base_url: str,
        timeout: httpx.Timeout,
        verify_tls: bool,
        follow_redirects: bool,
    ) -> httpx.Client:
        """Return a cached client or create a new one for the given parameters."""
        timeout_sec = timeout.connect  # all four fields are set to same value
        if timeout_sec is None:
            timeout_sec = 0.0
        key: _ClientKey = (base_url, float(timeout_sec), verify_tls, follow_redirects)
        with cls._clients_lock:
            client = cls._clients.get(key)
            if client is not None and not client.is_closed:
                return client
            # Create a new client for this combination
            client = httpx.Client(
                timeout=timeout,
                verify=verify_tls,
                follow_redirects=follow_redirects,
            )
            cls._clients[key] = client
            logger.debug(
                "Created new HTTP client for %s (verify=%s, timeout=%s)",
                base_url,
                verify_tls,
                timeout_sec,
            )
            return client

    def _request_with_retries(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        headers: dict[str, Any],
        params: dict[str, Any] | None,
        request_kwargs: dict[str, Any],
        retries: int,
    ) -> httpx.Response:
        """Issue the request, retrying transient failures up to ``retries`` times.

        A retryable failure is a connection error or a transient HTTP status
        (5xx, 408, 429). Non-retryable failures and a cancelled task stop the
        loop immediately. The final attempt's failure propagates to the caller.
        """
        attempt = 0
        while True:
            if self._cancel_event.is_set():
                raise TaskCancelledError("API request cancelled")
            try:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    **request_kwargs,
                )
            except httpx.RequestError:
                if attempt < retries:
                    attempt += 1
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
                raise
            if resp.is_error and _is_retryable_status(resp.status_code):
                if attempt < retries:
                    attempt += 1
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
            return resp

    @classmethod
    def close_all_clients(cls) -> None:
        """Close and discard all cached HTTP clients."""
        with cls._clients_lock:
            for key, client in cls._clients.items():
                try:
                    client.close()
                except Exception:
                    logger.debug("Error closing HTTP client for %s", key[0])
            cls._clients.clear()
            logger.debug("All cached HTTP clients closed")

    def cleanup_after_run(self) -> None:
        """Close the connection pool when the runner deactivates this executor."""
        self.close_all_clients()

    def run(self, task: ExecutorTask, out_dir: Path) -> APIResult:
        spec = self.require_spec(task, ApiSpecStrict)
        api_cfg = spec.api or {}
        if not isinstance(api_cfg, dict):
            raise ExecutionError("spec.api must be a mapping")

        url = api_cfg.get("url")
        method = str(api_cfg.get("method", "POST")).upper()
        headers = api_cfg.get("headers", {})
        if not isinstance(headers, dict):
            raise ExecutionError("spec.api.headers must be a mapping")

        if url is None:
            url = os.getenv("NEBULA_API_BASE_URL")
            if not url:
                raise ExecutionError("spec.api.url or NEBULA_API_BASE_URL is required")
            url = url.rstrip("/") + "/v1/chat/completions"

            if not any(is_credential_key(k) for k in headers):
                token = os.getenv("NEBULA_API_TOKEN")
                if not token:
                    raise ExecutionError(
                        "no credential configured: set an Authorization header or "
                        "NEBULA_API_TOKEN"
                    )
                headers["Authorization"] = f"Bearer {token}"

        params = api_cfg.get("params")
        if params is not None and not isinstance(params, dict):
            raise ExecutionError("spec.api.params must be a mapping")

        timeout_sec = api_cfg.get("timeout_sec", 60)
        if not isinstance(timeout_sec, (int, float)):
            raise ExecutionError("spec.api.timeout_sec must be a number")
        timeout = httpx.Timeout(timeout_sec)

        verify_tls = api_cfg.get("verify_tls", True)
        follow_redirects = api_cfg.get("follow_redirects", True)

        body = api_cfg.get("body")
        json_payload = api_cfg.get("json")
        data_payload = api_cfg.get("data")

        if json_payload is not None and body is not None:
            raise ExecutionError(
                "spec.api.json and spec.api.body are mutually exclusive"
            )

        request_kwargs: dict[str, Any] = {}
        if json_payload is not None:
            request_kwargs["json"] = json_payload
        elif body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = body
        elif data_payload is not None:
            request_kwargs["data"] = data_payload

        response_cfg = api_cfg.get("response") or {}
        if response_cfg and not isinstance(response_cfg, dict):
            raise ExecutionError("spec.api.response must be a mapping")

        include_headers = bool(response_cfg.get("include_headers", False))
        # return_body is a JSON backdoor: keep raw text when JSON isn't usable.
        return_body = bool(response_cfg.get("return_body", True))
        parse_json = bool(response_cfg.get("parse_json", True))
        raise_for_status = bool(response_cfg.get("raise_for_status", True))
        max_body_bytes = int(response_cfg.get("max_body_bytes", 200000))

        retries = api_cfg.get("retries", 0)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ExecutionError("spec.api.retries must be a non-negative integer")

        try:
            base = self._base_url(str(url))
            client = self._get_client(base, timeout, verify_tls, follow_redirects)
            resp = self._request_with_retries(
                client,
                method,
                str(url),
                headers,
                params,
                request_kwargs,
                retries,
            )
        except httpx.RequestError as exc:
            raise ExecutionError(f"API request failed: {exc}", retryable=True) from exc

        body_bytes = resp.content
        truncated = False
        if max_body_bytes is not None and len(body_bytes) > max_body_bytes:
            body_bytes = body_bytes[:max_body_bytes]
            truncated = True

        result = APIResult(
            ok=resp.is_success,
            executor=self.name,
            method=method,
            url=str(resp.url),
            status_code=resp.status_code,
            truncated=truncated,
        )

        if include_headers:
            result.headers = dict(resp.headers)

        body_text: str | None = None
        if return_body:
            encoding = resp.encoding or "utf-8"
            body_text = body_bytes.decode(encoding, errors="replace")

        if parse_json:
            result.response_json = resp.json()
            if not isinstance(result.response_json, dict):
                raise ExecutionError("Response is not a valid JSON mapping")
            usage = result.response_json.get("usage")
            if not isinstance(usage, dict):
                raise ExecutionError(
                    "spec.api.response.parse_json is true but response JSON "
                    f"does not contain usage info: {result.response_json}"
                )
            result.usage = usage
            try:
                result.text = result.response_json["choices"][0]["message"]["content"]
            except Exception as exc:
                raise ExecutionError(
                    "spec.api.response.parse_json is true but response JSON "
                    f"does not contain message.content: {result.response_json}"
                ) from exc
        elif return_body:
            result.text = body_text

        if raise_for_status and resp.is_error:
            message = f"API request returned status {resp.status_code}"
            if body_text:
                message = f"{message}: {body_text[:200]}"
            retryable = _is_retryable_status(resp.status_code)
            raise ExecutionError(message, retryable=retryable)

        return result
