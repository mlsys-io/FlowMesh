"""Drift-hazard test: FlowMesh's `BaseBindings` must cover every public hook
protocol — both the shared `lumid_hooks` ones and FlowMesh's own.

Adding a new hook `Protocol` to either package without adding a corresponding
field to `BaseBindings` would silently leave the new hook unwired. This test
fails loudly in that case.

`BaseBindings` is the concrete dataclass plugins use as a convenience base; the
sibling `HookBindings` Protocol enumerates the same fields and is what the
server gates against. Iterating `BaseBindings` keeps the assertion in plain
`dataclasses.fields` territory.
"""

import dataclasses
import typing

import flowmesh_hook
import lumid_hooks
from flowmesh_hook import BaseBindings

_HOOK_MODULES = frozenset(
    {
        "lumid_hooks.identity",
        "lumid_hooks.submission",
        "lumid_hooks.usage",
        "lumid_hooks.permissions",
        "lumid_hooks.registrar",
        "flowmesh_hook.supplier",
    }
)


def _public_hook_protocols() -> set[type]:
    """`@runtime_checkable` Protocol classes that represent a hook contract.

    Type-only protocols (e.g. `WorkerView`, `HookBindings`) live alongside
    other types and are excluded; hook protocols live in their own per-hook
    modules.
    """
    found: set[type] = set()
    for module in (lumid_hooks, flowmesh_hook):
        for name in module.__all__:
            obj = getattr(module, name)
            if not isinstance(obj, type):
                continue
            if not getattr(obj, "_is_runtime_protocol", False):
                continue
            if obj.__module__ in _HOOK_MODULES:
                found.add(obj)
    return found


def _bindings_field_protocols() -> set[type]:
    """The Protocol class inside each `Sequence[...]` field annotation."""
    found: set[type] = set()
    hints = typing.get_type_hints(BaseBindings)
    for f in dataclasses.fields(BaseBindings):
        annotation = hints[f.name]
        args = typing.get_args(annotation)
        assert args, f"field {f.name!r} must be parameterized (e.g. Sequence[X])"
        inner = args[0]
        # `UsageSink[Row]` resolves to a generic alias; unwrap to the origin.
        origin = typing.get_origin(inner)
        found.add(origin if origin is not None else inner)
    return found


def test_bindings_cover_every_public_hook_protocol() -> None:
    public = _public_hook_protocols()
    bound = _bindings_field_protocols()
    missing = public - bound
    extra = bound - public
    assert not missing, f"BaseBindings missing field for protocol(s): {missing}"
    assert not extra, f"BaseBindings has field(s) for unknown protocol(s): {extra}"


def test_bindings_fields_default_to_empty_sequence() -> None:
    bindings = BaseBindings()
    for f in dataclasses.fields(BaseBindings):
        value = getattr(bindings, f.name)
        assert len(value) == 0, f"field {f.name!r} default is not empty"


def test_bindings_is_frozen() -> None:
    bindings = BaseBindings()
    try:
        bindings.identity_providers = ()  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("BaseBindings should be frozen")
