from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.clients.redis import WORKERS_CORDONED_SET_KEY, WORKERS_SET_KEY
from server.registries.worker import Worker, WorkerRegistry
from server.schemas.worker import WorkerCordon
from shared.schemas.worker import WorkerCapabilities, WorkerStatus
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.task_type import TaskType


def _task() -> TaskEnvelopeStrict:
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "spec": {"taskType": "echo", "resources": None},
        }
    )


def _worker(worker_id: str, alias: str | None, node_alias: str = "node-a") -> Worker:
    return Worker(
        id=worker_id,
        alias=alias,
        namespace="default",
        cluster="default",
        node_id=f"nde-{node_alias}",
        node_alias=node_alias,
        status=WorkerStatus.IDLE,
        capabilities=WorkerCapabilities(
            supported_task_types=frozenset({TaskType.ECHO})
        ),
    )


def _cordon(alias: str, node_alias: str = "node-a") -> WorkerCordon:
    return WorkerCordon(node_alias=node_alias, alias=alias)


class _Registry(WorkerRegistry):
    def __init__(
        self,
        workers: list[Worker],
        cordons: Sequence[WorkerCordon] = (),
        stale: frozenset[str] = frozenset(),
    ) -> None:
        self._workers = {w.id: w for w in workers}
        self._stale = stale
        self.cordoned: set[str] = set()
        rds: Any = MagicMock()
        rds.sync.set_members.side_effect = self._set_members
        rds.asyncio.set_members = AsyncMock(side_effect=self._set_members)
        rds.sync.sadd.side_effect = self._sadd
        rds.sync.srem.side_effect = self._srem
        rds.asyncio.sadd = AsyncMock(side_effect=self._sadd)
        rds.asyncio.srem = AsyncMock(side_effect=self._srem)
        super().__init__(cast(Any, rds))
        for cordon in cordons:
            self.set_cordon(cordon, cordoned=True)

    def _set_members(self, key: str) -> set[str]:
        if key == WORKERS_CORDONED_SET_KEY:
            return set(self.cordoned)
        if key == WORKERS_SET_KEY:
            return set(self._workers)
        return set()

    def _sadd(self, key: str, member: str) -> int:
        added = member not in self.cordoned
        self.cordoned.add(member)
        return int(added)

    def _srem(self, key: str, member: str) -> int:
        removed = member in self.cordoned
        self.cordoned.discard(member)
        return int(removed)

    def get_worker(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    async def get_worker_async(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    def get_worker_ids(self) -> set[str]:
        return set(self._workers)

    async def get_worker_ids_async(self) -> set[str]:
        return set(self._workers)

    def get_workers(self, worker_ids: Sequence[str]) -> list[Worker | None]:
        return [self._workers.get(worker_id) for worker_id in worker_ids]

    async def get_workers_async(self, worker_ids: Sequence[str]) -> list[Worker | None]:
        return [self._workers.get(worker_id) for worker_id in worker_ids]

    def is_worker_stale(self, worker_id: str) -> bool:
        return worker_id in self._stale

    async def is_worker_stale_async(self, worker_id: str) -> bool:
        return worker_id in self._stale

    def get_worker_heartbeat(self, worker_id: str) -> str | None:
        return None


def test_cordoned_worker_is_not_offered_new_work() -> None:
    registry = _Registry(
        [_worker("wkr-1", "alpha"), _worker("wkr-2", "beta")],
        cordons=[_cordon("alpha")],
    )
    assert [w.id for w in registry.idle_satisfying_pool(_task())] == ["wkr-2"]


def test_cordoned_worker_is_excluded_from_the_eligibility_set() -> None:
    registry = _Registry([_worker("wkr-1", "alpha")], cordons=[_cordon("alpha")])
    assert registry.satisfying_workers(_task()) == []


def test_same_alias_on_another_node_is_not_cordoned() -> None:
    registry = _Registry(
        [_worker("wkr-1", "alpha", "node-a"), _worker("wkr-2", "alpha", "node-b")],
        cordons=[_cordon("alpha")],
    )
    assert [w.id for w in registry.idle_satisfying_pool(_task())] == ["wkr-2"]


def test_cordon_matches_a_worker_reregistered_under_a_new_id() -> None:
    registry = _Registry([_worker("wkr-9", "alpha")], cordons=[_cordon("alpha")])
    assert registry.idle_satisfying_pool(_task()) == []


def test_a_worker_with_no_alias_is_not_excluded() -> None:
    registry = _Registry([_worker("wkr-1", None)], cordons=[_cordon("alpha")])
    assert [w.id for w in registry.idle_satisfying_pool(_task())] == ["wkr-1"]


def test_the_cordon_set_is_read_once_per_dispatch() -> None:
    registry = _Registry([_worker(f"wkr-{i}", f"alias-{i}") for i in range(10)])
    registry.idle_satisfying_pool(_task())
    rds = cast(Any, registry._rds)
    cordon_reads = [
        c
        for c in rds.sync.set_members.call_args_list
        if c.args and c.args[0] == WORKERS_CORDONED_SET_KEY
    ]
    assert len(cordon_reads) == 1


def test_list_workers_reports_cordon_state() -> None:
    registry = _Registry(
        [_worker("wkr-1", "alpha"), _worker("wkr-2", "beta")],
        cordons=[_cordon("alpha")],
    )
    by_id = {w.id: w.cordoned for w in registry.list_workers()}
    assert by_id == {"wkr-1": True, "wkr-2": False}


@pytest.mark.asyncio
async def test_set_cordon_reports_whether_it_changed_anything() -> None:
    registry = _Registry([])
    cordon = _cordon("alpha")
    assert await registry.set_cordon_async(cordon, cordoned=True) is True
    assert await registry.set_cordon_async(cordon, cordoned=True) is False
    assert await registry.list_cordons_async() == [cordon]
    assert await registry.set_cordon_async(cordon, cordoned=False) is True
    assert await registry.set_cordon_async(cordon, cordoned=False) is False


def test_list_cordons_round_trips_separator_characters() -> None:
    cordon = _cordon('we"ird,alias', "node/a")
    registry = _Registry([], cordons=[cordon, _cordon("alpha")])
    registry.cordoned.add("not-json")
    assert registry.list_cordons() == [_cordon("alpha"), cordon]


@pytest.mark.asyncio
async def test_is_cordoned_keys_on_node_alias_and_worker_alias() -> None:
    registry = _Registry([], cordons=[_cordon("alpha")])
    assert await registry.is_cordoned_async(_worker("wkr-1", "alpha", "node-a"))
    assert not await registry.is_cordoned_async(_worker("wkr-1", "alpha", "node-b"))
    assert not registry.is_cordoned(_worker("wkr-1", None, "node-a"))


@pytest.mark.asyncio
async def test_live_worker_ids_for_cordon_match_the_key_and_skip_stale() -> None:
    registry = _Registry(
        [
            _worker("wkr-1", "alpha", "node-a"),
            _worker("wkr-2", "alpha", "node-a"),
            _worker("wkr-3", "alpha", "node-b"),
            _worker("wkr-4", "beta", "node-a"),
        ],
        stale=frozenset({"wkr-1"}),
    )
    cordon = _cordon("alpha")
    assert await registry.live_worker_ids_for_cordon_async(cordon) == ["wkr-2"]
    assert registry.live_worker_ids_for_cordon(cordon) == ["wkr-2"]
