import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar

import httpx

from shared.schemas.result import APIItem, APIResult
from shared.tasks.specs import ApiSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils.redact import is_credential_key

from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    TaskCancelledError,
)
from .mixins.data import DataMixin

logger = logging.getLogger(__name__)

_ClientKey = tuple[str, float, bool, bool, int]

# Worker-side per-row slot in a batched request body. Server-side stage
# references are ${...}; this is a worker-side token, hence {{...}}.
_PROMPT_PLACEHOLDER = "{{prompt}}"

# Fixed delay between retry attempts.
_RETRY_BACKOFF_SEC = 1.0

_MAX_CONCURRENCY = 8


def _is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status is transient and worth retrying."""
    return status_code >= 500 or status_code in (408, 429)


class APIExecutor(DataMixin, Executor):
    """Performs HTTP requests defined by task YAML.

    ``spec.data`` is required and yields one row per request: each row's prompt
    is substituted for the ``{{prompt}}`` placeholder in the request body and
    one request is issued per row, returned row-aligned in ``APIResult.items``.
    A single request is a one-row ``spec.data``.

    Defaults to the Nebula endpoint via ``NEBULA_API_BASE_URL`` and authenticates
    with ``NEBULA_API_TOKEN``. ``spec.api.url`` overrides the endpoint and
    ``spec.api.headers`` may supply a credential header (``Authorization``,
    ``X-API-Key``, etc.) directly. A custom ``spec.api.url`` may be
    unauthenticated; the Nebula token is never sent to an endpoint the caller
    chose.

    Parallel row requests and the HTTP connection pool are capped at
    ``_MAX_CONCURRENCY``; the pool is sized to the effective concurrency so
    parallel requests never queue on connections.
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
        concurrency: int,
    ) -> httpx.Client:
        """Return a cached client or create a new one for the given parameters.

        The cache key is ``(base_url, timeout_sec, verify_tls, follow_redirects,
        concurrency)``; the pool is sized to ``concurrency`` so parallel row
        requests never queue on connections."""
        timeout_sec = timeout.connect  # all four fields are set to same value
        if timeout_sec is None:
            timeout_sec = 0.0
        key: _ClientKey = (
            base_url,
            float(timeout_sec),
            verify_tls,
            follow_redirects,
            concurrency,
        )
        with cls._clients_lock:
            client = cls._clients.get(key)
            if client is not None and not client.is_closed:
                return client
            client = httpx.Client(
                timeout=timeout,
                verify=verify_tls,
                follow_redirects=follow_redirects,
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
            )
            cls._clients[key] = client
            logger.debug(
                "Created new HTTP client for %s (verify=%s, timeout=%s, "
                "concurrency=%s)",
                base_url,
                verify_tls,
                timeout_sec,
                concurrency,
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

    @staticmethod
    def _prompt_to_str(prompt: Any) -> str:
        """Render a row's prompt as a string for body substitution."""
        if isinstance(prompt, str):
            return prompt
        return json.dumps(prompt)

    @classmethod
    def _substitute_prompt(cls, value: Any, prompt: str) -> Any:
        """Replace ``{{prompt}}`` in the request body with a row's prompt."""
        if isinstance(value, str):
            if value == _PROMPT_PLACEHOLDER:
                return prompt
            return value.replace(_PROMPT_PLACEHOLDER, prompt)
        if isinstance(value, dict):
            return {k: cls._substitute_prompt(v, prompt) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._substitute_prompt(v, prompt) for v in value]
        return value

    def _build_request_kwargs(
        self, api_cfg: dict[str, Any], prompt: str | None
    ) -> dict[str, Any]:
        """Build httpx request kwargs from ``spec.api``, substituting the row
        prompt when batching."""
        json_payload = api_cfg.get("json")
        body = api_cfg.get("body")
        data_payload = api_cfg.get("data")

        if json_payload is not None and body is not None:
            raise ExecutionError(
                "spec.api.json and spec.api.body are mutually exclusive"
            )

        request_kwargs: dict[str, Any] = {}
        if json_payload is not None:
            request_kwargs["json"] = (
                self._substitute_prompt(json_payload, prompt)
                if prompt is not None
                else json_payload
            )
        elif body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = (
                    self._substitute_prompt(body, prompt)
                    if prompt is not None
                    else body
                )
            else:
                request_kwargs["content"] = (
                    self._substitute_prompt(body, prompt)
                    if prompt is not None
                    else body
                )
        elif data_payload is not None:
            request_kwargs["data"] = (
                self._substitute_prompt(data_payload, prompt)
                if prompt is not None
                else data_payload
            )
        return request_kwargs

    def _parse_response(
        self,
        resp: httpx.Response,
        *,
        response_cfg: dict[str, Any],
        max_body_bytes: int,
    ) -> tuple[APIItem, str | None]:
        """Turn one HTTP response into an APIItem, applying response config.

        Returns the item and the raw body text (used for error messages).
        """
        body_bytes = resp.content
        truncated = False
        if max_body_bytes is not None and len(body_bytes) > max_body_bytes:
            body_bytes = body_bytes[:max_body_bytes]
            truncated = True

        item = APIItem(
            index=0,
            url=str(resp.url),
            status_code=resp.status_code,
            truncated=truncated,
        )

        if response_cfg.get("include_headers", False):
            item.headers = dict(resp.headers)

        body_text: str | None = None
        if response_cfg.get("return_body", True):
            encoding = resp.encoding or "utf-8"
            body_text = body_bytes.decode(encoding, errors="replace")

        if response_cfg.get("parse_json", True):
            item.response_json = resp.json()
            if not isinstance(item.response_json, dict):
                raise ExecutionError("Response is not a valid JSON mapping")
            usage = item.response_json.get("usage")
            if isinstance(usage, dict):
                item.usage = usage
            try:
                item.text = item.response_json["choices"][0]["message"]["content"]
            except Exception:
                item.text = None
        elif response_cfg.get("return_body", True):
            item.text = body_text

        return item, body_text

    def run(self, task: ExecutorTask, out_dir: Path) -> APIResult:
        self._cancel_event = threading.Event()
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

        response_cfg = api_cfg.get("response") or {}
        if response_cfg and not isinstance(response_cfg, dict):
            raise ExecutionError("spec.api.response must be a mapping")

        max_body_bytes = int(response_cfg.get("max_body_bytes", 200000))
        raise_for_status = bool(response_cfg.get("raise_for_status", True))

        retries = api_cfg.get("retries", 0)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ExecutionError("spec.api.retries must be a non-negative integer")

        concurrency = int(api_cfg.get("concurrency", _MAX_CONCURRENCY))
        if concurrency < 1:
            raise ExecutionError("spec.api.concurrency must be >= 1")
        concurrency = min(concurrency, _MAX_CONCURRENCY)

        base = self._base_url(str(url))
        client = self._get_client(
            base, timeout, verify_tls, follow_redirects, concurrency
        )

        entry = self._collect_prompts_for_spec(spec, task_id=task.task_id)
        prompts = entry.prompts
        if not prompts:
            raise ExecutionError("spec.data produced no rows")

        request_kwargs = self._build_request_kwargs(api_cfg, None)

        def _issue(idx: int, prompt: Any) -> APIItem:
            if self._cancel_event.is_set():
                raise TaskCancelledError("API task cancelled")
            prompt_str = self._prompt_to_str(prompt)
            kwargs = self._substitute_prompt(request_kwargs, prompt_str)
            try:
                resp = self._request_with_retries(
                    client,
                    method,
                    str(url),
                    headers,
                    params,
                    kwargs,
                    retries,
                )
            except httpx.RequestError as exc:
                raise ExecutionError(
                    f"API request failed (row {idx}): {exc}", retryable=True
                ) from exc

            if raise_for_status and resp.is_error:
                message = f"API request returned status {resp.status_code} (row {idx})"
                body_text = resp.text[:200]
                if body_text:
                    message = f"{message}: {body_text}"
                retryable = resp.status_code >= 500 or resp.status_code in (408, 429)
                raise ExecutionError(message, retryable=retryable)

            item, _ = self._parse_response(
                resp,
                response_cfg=response_cfg,
                max_body_bytes=max_body_bytes,
            )
            item.index = idx
            item.prompt = prompt_str

            return item

        results: dict[int, APIItem] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for idx, prompt in enumerate(prompts):
                if self._cancel_event.is_set():
                    raise TaskCancelledError("API task cancelled")
                futures[pool.submit(_issue, idx, prompt)] = idx
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        items = [results[idx] for idx in range(len(prompts))]

        return APIResult(
            ok=True,
            executor=self.name,
            method=method,
            url=str(url),
            status_code=items[0].status_code,
            truncated=any(item.truncated for item in items),
            items=items,
        )
