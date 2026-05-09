"""Supplier-resolver hook (FlowMesh-specific).

Maps an assigned worker to the supplier responsible for the underlying
hardware (e.g. cloud account id, vendor identifier). The runtime calls every
registered resolver at dispatch time and stamps the first non-`None` result
on the task record; downstream `UsageSink`s receive that value.

With no resolvers registered, the supplier id stays at its `""` default.
"""

from typing import Protocol, runtime_checkable

from .worker_view import WorkerView


@runtime_checkable
class SupplierResolver(Protocol):
    name: str

    def resolve(self, worker: WorkerView) -> str | None:
        """Return the supplier id for `worker`, or `None` to defer."""
        ...
