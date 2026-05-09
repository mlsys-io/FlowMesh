"""FlowMesh's bindings shape — Protocol contract + concrete convenience class.

`HookBindings` is a runtime-checkable Protocol extending
`lumid_hooks.HookBindings` with FlowMesh's `supplier_resolvers` field. The
server gates plugin load against the shared Protocol (so shared-only plugins
pass) and narrows on this Protocol to drain `supplier_resolvers` from
FlowMesh-extended plugins. The field is declared read-only so frozen dataclass
instances satisfy the Protocol under mypy.

`BaseBindings` is a frozen dataclass extending `lumid_hooks.BaseBindings` with
`supplier_resolvers` default-factoried. FlowMesh-only plugins return
`BaseBindings(...)` directly. Cross-host plugins skip both and return their own
dataclass that names `supplier_resolvers` alongside other hosts' fields —
structural typing handles it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from lumid_hooks import BaseBindings as SharedBaseBindings
from lumid_hooks import HookBindings as SharedHookBindings

from .supplier import SupplierResolver


@runtime_checkable
class HookBindings(SharedHookBindings, Protocol):
    @property
    def supplier_resolvers(self) -> Sequence[SupplierResolver]: ...


@dataclass(frozen=True)
class BaseBindings(SharedBaseBindings):
    supplier_resolvers: Sequence[SupplierResolver] = field(default_factory=tuple)
