import threading
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

from .adapters.base import WorkerAdapter, WorkerTokenType


@dataclass(slots=True)
class _WorkerRegistryState:
    registry: dict[WorkerTokenType, WorkerAdapter] = field(default_factory=dict)
    alias_token_map: dict[str, WorkerTokenType] = field(default_factory=dict)
    token_id_map: dict[WorkerTokenType, str] = field(default_factory=dict)


class WorkerRegistry:
    """Thread-safe map of worker adapters by token and alias, and of each token's
    current worker id.

    ``on_worker_id_released`` is called with every worker id whose token binding is
    replaced or removed, after the registry lock is released.
    """

    def __init__(
        self, on_worker_id_released: Callable[[str], None] | None = None
    ) -> None:
        self._state = _WorkerRegistryState()
        self._lock = threading.Lock()
        self._on_worker_id_released = on_worker_id_released

    def _release(self, worker_ids: Iterable[str | None]) -> None:
        callback = self._on_worker_id_released
        if callback is None:
            return
        for worker_id in worker_ids:
            if worker_id is not None:
                callback(worker_id)

    @contextmanager
    def _get_state(self) -> Generator[_WorkerRegistryState, None, None]:
        with self._lock:
            yield self._state

    def new_token(self) -> WorkerTokenType:
        return uuid.uuid4().hex  # type: ignore

    def add(self, worker: WorkerAdapter) -> None:
        token = worker.token
        alias = worker.alias
        with self._get_state() as state:
            if token in state.registry:
                raise ValueError(f"Worker with token '{token}' already exists")
            if alias in state.alias_token_map:
                raise ValueError(f"Worker with alias '{alias}' already exists")
            state.registry[token] = worker
            state.alias_token_map[alias] = token

    def exists(self, token: WorkerTokenType) -> bool:
        with self._get_state() as state:
            return token in state.registry

    def get(self, token: WorkerTokenType) -> WorkerAdapter:
        with self._get_state() as state:
            return state.registry[token]

    def try_get(self, token: WorkerTokenType) -> WorkerAdapter | None:
        with self._get_state() as state:
            return state.registry.get(token)

    def pop(self, token: WorkerTokenType) -> WorkerAdapter:
        with self._get_state() as state:
            worker = state.registry.pop(token)
            del state.alias_token_map[worker.alias]
            released = state.token_id_map.pop(token, None)
        self._release([released])
        return worker

    def try_pop(self, token: WorkerTokenType) -> WorkerAdapter | None:
        with self._get_state() as state:
            worker = state.registry.pop(token, None)
            if worker is None:
                return None
            del state.alias_token_map[worker.alias]
            released = state.token_id_map.pop(token, None)
        self._release([released])
        return worker

    def clear(self) -> None:
        with self._get_state() as state:
            state.registry.clear()
            state.alias_token_map.clear()
            released = list(state.token_id_map.values())
            state.token_id_map.clear()
        self._release(released)

    def exists_by_alias(self, alias: str) -> bool:
        with self._get_state() as state:
            return alias in state.alias_token_map

    def get_by_alias(self, alias: str) -> WorkerAdapter:
        with self._get_state() as state:
            token = state.alias_token_map[alias]
            return state.registry[token]

    def try_get_by_alias(self, alias: str) -> WorkerAdapter | None:
        with self._get_state() as state:
            token = state.alias_token_map.get(alias)
            if token is None:
                return None
            return state.registry.get(token)

    def pop_by_alias(self, alias: str) -> WorkerAdapter:
        with self._get_state() as state:
            token = state.alias_token_map.pop(alias)
            worker = state.registry.pop(token)
            released = state.token_id_map.pop(token, None)
        self._release([released])
        return worker

    def try_pop_by_alias(self, alias: str) -> WorkerAdapter | None:
        with self._get_state() as state:
            token = state.alias_token_map.pop(alias, None)
            if token is None:
                return None
            released = state.token_id_map.pop(token, None)
            worker = state.registry.pop(token)
        self._release([released])
        return worker

    def all_workers(self) -> list[WorkerAdapter]:
        with self._get_state() as state:
            return list(state.registry.values())

    def set_worker_id(self, token: WorkerTokenType, worker_id: str) -> None:
        with self._get_state() as state:
            previous = state.token_id_map.get(token)
            state.token_id_map[token] = worker_id
        if previous != worker_id:
            self._release([previous])

    def get_worker_id(self, token: WorkerTokenType) -> str | None:
        with self._get_state() as state:
            return state.token_id_map.get(token)
