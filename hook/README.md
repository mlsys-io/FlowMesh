# FlowMesh Hook

FlowMesh-specific plugin extension surface. Carries the pieces FlowMesh adds
on top of [`lumid-hooks`](https://github.com/mlsys-io/lumid.hooks):

- `HookBindings` — concrete dataclass with FlowMesh's six fields (the five
  shared from `lumid-hooks` plus `supplier_resolvers`).
- `ResourceType` / `ResourceAction` — FlowMesh resource and action enums.
- `SupplierResolver` / `WorkerView` — supplier attribution at dispatch time.
- `UsageRow` / `FlowMeshUsageSink` — FlowMesh's usage row shape and the
  parametrized sink alias.

Shared protocols (`IdentityProvider`, `SubmissionGuard`, `PermissionChecker`,
`ResourceRegistrar`, `UsageSink`) and shared types (`PrincipalContext`,
`ResourceRef`) live in `lumid-hooks` and should be imported from there.

The package depends only on `lumid-hooks` (transitively `pydantic`); it does
not pull in the server or worker stack, so plugins can `pip install
flowmesh-hook` without the heavy core.
