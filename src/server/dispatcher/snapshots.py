from collections import OrderedDict
from pathlib import Path

from shared.schemas.result import result_file_path
from shared.utils.result_delivery import (
    SnapshotCoverage,
    SnapshotIdentity,
    artifacts_ready,
    read_receipt,
    snapshot_identity,
    validated_snapshot,
)

DEFAULT_NAME_LIMIT = 200_000


class SnapshotCache:
    """Upstream result snapshots in ``results_dir`` that have been validated.

    A snapshot is trusted while its envelope and receipt files are unchanged, since
    every change to its artifacts replaces or removes the receipt. Entries are
    evicted least recently used once they hold more than ``name_limit`` artifact
    names in total.
    """

    def __init__(self, results_dir: Path, name_limit: int = DEFAULT_NAME_LIMIT):
        self._results_dir = results_dir
        self._name_limit = name_limit
        self._entries: OrderedDict[str, tuple[SnapshotIdentity, SnapshotCoverage]] = (
            OrderedDict()
        )
        self._names = 0

    def artifacts_ready(self, task_id: str, paths: list[str] | None) -> bool:
        """``artifacts_ready`` for ``task_id``, matching files by size, validating
        each snapshot once."""
        base_dir = self._base_dir(task_id)
        coverage = self._lookup(task_id, snapshot_identity(base_dir))
        if coverage is None:
            try:
                snapshot = validated_snapshot(base_dir, task_id, verify_content=False)
            except (OSError, ValueError):
                self._forget(task_id)
                return False
            if snapshot is None:
                return artifacts_ready(base_dir, task_id, paths, verify_content=False)
            receipt, identity = snapshot
            coverage = SnapshotCoverage.of(receipt)
            self._remember(task_id, identity, coverage)
        return coverage.covers(paths)

    def generation(self, task_id: str) -> str | None:
        """The generation of the snapshot ``task_id`` holds; ``None`` without a
        receipt."""
        base_dir = self._base_dir(task_id)
        coverage = self._lookup(task_id, snapshot_identity(base_dir))
        if coverage is not None:
            return coverage.generation
        receipt = read_receipt(base_dir)
        return receipt.generation if receipt is not None else None

    def _base_dir(self, task_id: str) -> Path:
        return result_file_path(self._results_dir, task_id).parent

    def _lookup(
        self, task_id: str, identity: SnapshotIdentity | None
    ) -> SnapshotCoverage | None:
        entry = self._entries.get(task_id)
        if entry is None or identity is None or entry[0] != identity:
            return None
        self._entries.move_to_end(task_id)
        return entry[1]

    def _remember(
        self, task_id: str, identity: SnapshotIdentity, coverage: SnapshotCoverage
    ) -> None:
        self._forget(task_id)
        if len(coverage.names) > self._name_limit:
            return
        self._entries[task_id] = (identity, coverage)
        self._names += len(coverage.names)
        while self._names > self._name_limit:
            _, (_, evicted) = self._entries.popitem(last=False)
            self._names -= len(evicted.names)

    def _forget(self, task_id: str) -> None:
        entry = self._entries.pop(task_id, None)
        if entry is not None:
            self._names -= len(entry[1].names)
