"""Supplier-resolver hook.

Maps an assigned `Worker` to the supplier responsible for the underlying
hardware (e.g. cloud account id, vendor identifier). The runtime calls every
registered resolver at dispatch time and stamps the first non-`None` result
on `TaskRecord.supplier_id`; downstream `UsageSink`s receive that value.

With no resolvers registered, `supplier_id` stays at its `""` default.
"""

from typing import Protocol, runtime_checkable

from ..registries.worker import Worker


@runtime_checkable
class SupplierResolver(Protocol):
    name: str

    def resolve(self, worker: Worker) -> str | None:
        """Return the supplier id for `worker`, or `None` to defer."""
        ...
