# FlowMesh Hook

Plugin contract surface for the FlowMesh server. Exposes the protocols,
shared types, and `HookBindings` aggregate that third-party plugins
return from their `install()` entry point.

The only runtime dependency is `pydantic`, which `PrincipalContext`
uses as its model base. The package does not import the server or
worker packages, so plugins can `pip install flowmesh[hook]` without
pulling in the heavy core stack.
