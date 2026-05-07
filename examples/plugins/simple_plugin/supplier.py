"""`SupplierResolver` example: return worker namespace as the supplier ID.

The protocol method is **synchronous** and receives no logger — it runs on
the dispatch hot path. Real resolvers stay equally cheap.
"""

from flowmesh_hook import WorkerView


class SimpleSupplierResolver:
    name = "simple_plugin.supplier"

    def resolve(self, worker: WorkerView) -> str | None:
        return worker.namespace
