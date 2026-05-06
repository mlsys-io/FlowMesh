# FlowMesh Hook

Plugin contract surface for the FlowMesh server. Exposes the protocols,
shared types, and `HookBindings` aggregate that third-party plugins
return from their `install()` entry point.

This package has no runtime dependencies and does not import the server
or worker packages, so plugins can `pip install flowmesh[hook]` without
pulling in the heavy core stack.
