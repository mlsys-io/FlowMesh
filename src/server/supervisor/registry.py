import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .adapters.base import WorkerAdapter, WorkerTokenType


@dataclass(slots=True)
class _WorkerRegistryState:
    registry: dict[WorkerTokenType, WorkerAdapter] = field(default_factory=dict)
    name_token_map: dict[str, WorkerTokenType] = field(default_factory=dict)
    token_id_map: dict[WorkerTokenType, str] = field(default_factory=dict)


class WorkerRegistry:
    def __init__(self) -> None:
        self._state = _WorkerRegistryState()
        self._lock = threading.Lock()

    @contextmanager
    def _get_state(self) -> Generator[_WorkerRegistryState, None, None]:
        with self._lock:
            yield self._state

    def new_token(self) -> WorkerTokenType:
        return uuid.uuid4().hex  # type: ignore

    def add(self, worker: WorkerAdapter) -> None:
        token = worker.token
        name = worker.name
        with self._get_state() as state:
            if token in state.registry:
                raise ValueError(f"Worker with token '{token}' already exists")
            if name in state.name_token_map:
                raise ValueError(f"Worker with name '{name}' already exists")
            state.registry[token] = worker
            state.name_token_map[name] = token

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
            del state.name_token_map[worker.name]
            state.token_id_map.pop(token, None)
            return worker

    def try_pop(self, token: WorkerTokenType) -> WorkerAdapter | None:
        with self._get_state() as state:
            worker = state.registry.pop(token, None)
            if worker is None:
                return None
            del state.name_token_map[worker.name]
            state.token_id_map.pop(token, None)
            return worker

    def clear(self) -> None:
        with self._get_state() as state:
            state.registry.clear()
            state.name_token_map.clear()
            state.token_id_map.clear()

    def exists_by_name(self, name: str) -> bool:
        with self._get_state() as state:
            return name in state.name_token_map

    def get_by_name(self, name: str) -> WorkerAdapter:
        with self._get_state() as state:
            token = state.name_token_map[name]
            return state.registry[token]

    def try_get_by_name(self, name: str) -> WorkerAdapter | None:
        with self._get_state() as state:
            token = state.name_token_map.get(name)
            if token is None:
                return None
            return state.registry.get(token)

    def pop_by_name(self, name: str) -> WorkerAdapter:
        with self._get_state() as state:
            token = state.name_token_map.pop(name)
            worker = state.registry.pop(token)
            state.token_id_map.pop(token, None)
            return worker

    def try_pop_by_name(self, name: str) -> WorkerAdapter | None:
        with self._get_state() as state:
            token = state.name_token_map.pop(name, None)
            if token is None:
                return None
            state.token_id_map.pop(token, None)
            return state.registry.pop(token)

    def all_workers(self) -> list[WorkerAdapter]:
        with self._get_state() as state:
            return list(state.registry.values())

    def set_worker_id(self, token: WorkerTokenType, worker_id: str) -> None:
        with self._get_state() as state:
            state.token_id_map[token] = worker_id

    def get_worker_id(self, token: WorkerTokenType) -> str | None:
        with self._get_state() as state:
            return state.token_id_map.get(token)
