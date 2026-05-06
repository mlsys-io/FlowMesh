"""Drift-hazard test: server's `Worker` must satisfy `flowmesh_hook.WorkerView`.

`SupplierResolver` consumes a structural `WorkerView` Protocol declared in the
hook package; the server's concrete `Worker` Pydantic model must keep
exposing every attribute that `WorkerView` declares. A field rename in
`Worker` would otherwise break plugins silently — this test catches it.
"""

import typing

from flowmesh_hook import WorkerView

from server.registries.worker import Worker


def _make_minimal_worker() -> Worker:
    return Worker(
        id="wkr-test-0001",
        namespace="ns",
        cluster="local",
        node_id="nod-test-0001",
        node_alias="node-0",
    )


def test_worker_satisfies_workerview_protocol() -> None:
    worker = _make_minimal_worker()
    assert isinstance(worker, WorkerView)


def test_worker_exposes_every_workerview_attribute() -> None:
    worker = _make_minimal_worker()
    for attr in typing.get_type_hints(WorkerView):
        assert hasattr(worker, attr), f"Worker is missing WorkerView attr: {attr!r}"
