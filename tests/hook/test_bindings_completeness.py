"""Drift-hazard test: `HookBindings` must cover every public hook protocol.

Adding a new `Protocol` to `flowmesh_hook` without adding a corresponding
field to `HookBindings` would silently leave the new hook unwired. This test
fails loudly in that case.
"""

import dataclasses
import typing

import flowmesh_hook
from flowmesh_hook import HookBindings

_HOOK_MODULES = frozenset(
    {
        "flowmesh_hook.identity",
        "flowmesh_hook.submission",
        "flowmesh_hook.usage",
        "flowmesh_hook.permissions",
        "flowmesh_hook.supplier",
    }
)


def _public_hook_protocols() -> set[type]:
    """`@runtime_checkable` Protocol classes that represent a hook contract.

    Type-only protocols (e.g. `WorkerView`) live in `flowmesh_hook.types` and
    are excluded; hook protocols live in their own per-hook modules.
    """
    found: set[type] = set()
    for name in flowmesh_hook.__all__:
        obj = getattr(flowmesh_hook, name)
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
    hints = typing.get_type_hints(HookBindings)
    for f in dataclasses.fields(HookBindings):
        annotation = hints[f.name]
        args = typing.get_args(annotation)
        assert args, f"field {f.name!r} must be parameterized (e.g. Sequence[X])"
        inner = args[0]
        found.add(inner)
    return found


def test_bindings_cover_every_public_hook_protocol() -> None:
    public = _public_hook_protocols()
    bound = _bindings_field_protocols()
    missing = public - bound
    extra = bound - public
    assert not missing, f"HookBindings missing field for protocol(s): {missing}"
    assert not extra, f"HookBindings has field(s) for unknown protocol(s): {extra}"


def test_bindings_fields_default_to_empty_sequence() -> None:
    bindings = HookBindings()
    for f in dataclasses.fields(HookBindings):
        value = getattr(bindings, f.name)
        assert len(value) == 0, f"field {f.name!r} default is not empty"


def test_bindings_is_frozen() -> None:
    bindings = HookBindings()
    try:
        bindings.identity_providers = ()  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("HookBindings should be frozen")
