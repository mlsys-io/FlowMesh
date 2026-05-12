# CLI usage (`flowmesh`)

The CLI entry point is `flowmesh` (provided by `flowmesh-cli`). Run any
subcommand with `-h` / `--help` for the full flag list — this doc covers
the common ones. Stack commands (`flowmesh stack ...`) come from the
separate `flowmesh-cli-stack` package and operate on the local Compose
stack and its containers; everything else works against any reachable
FlowMesh server.

## Top-level groups

```
flowmesh info | health | init | deinit | config
flowmesh workflow {submit, validate, list, info, watch, cancel, logs}
flowmesh task     {list, info, watch, stop, logs}
flowmesh worker   {list, info}
flowmesh node     {list, info, worker {list, start, stop}}
flowmesh ssh      {connect, run, proxy, connections}
flowmesh result   {fetch, download}
flowmesh trace    {fetch, analyze}
flowmesh system   {metrics}
flowmesh stack    {build, push, pull, pullall, up, down, restart, ps, logs}
flowmesh stack bundle {export, init}
flowmesh stack worker {up, start, stop, down, list, pull}
```

## Common workflows

Submit a workflow and watch it:

```bash
flowmesh workflow submit examples/templates/echo_local.yaml
flowmesh workflow watch <wfl-id>              # blocks until DONE / FAILED
flowmesh workflow logs show <wfl-id>          # recent log entries
flowmesh workflow logs stream <wfl-id>        # SSE log stream
flowmesh workflow logs download <wfl-id> -o logs/
```

List, filter, paginate:

```bash
flowmesh workflow list --status DONE
flowmesh task list --workflow-id <wfl-id> --status FAILED
```

Pull results / artifacts:

```bash
flowmesh result fetch <tsk-id>                                  # JSON result
flowmesh result download <tsk-id> --include all -o bundle.tgz   # tar.gz bundle
```

Fetch or analyze workflow traces:

```bash
flowmesh trace fetch spans <wfl-id> -o spans.jsonl
flowmesh trace analyze <wfl-id> --format critical-path
```

`trace fetch` accepts `spans`, `assets`, or `lineage`. `trace analyze
--format` accepts `rich`, `critical-path` (`cp`), `end-to-end` (`e2e`),
`queuing`, `lineage`, or `json`.

## Local stack lifecycle

`.env` controls registry, version tag, and ports — see
`cli/stack/src/flowmesh_cli_stack/assets/.env.example`. Set
`FLOWMESH_VERSION` to a PR-identifying slug (e.g. `myfeature`) so parallel
PRs don't overwrite each other's local images.

When multiple deployments share one host, give each stack its own
`FLOWMESH_STACK_SUFFIX` and distinct `SERVER_HTTP_PORT`,
`SERVER_GRPC_PORT`, `REDIS_CONTROL_PORT`, and `REDIS_TELEMETRY_PORT`.
The suffix isolates Docker object names (including containers, volumes, and networks); the ports isolate host bindings.

```bash
flowmesh stack up                          # Server + Redis + Supervisor (root)
flowmesh stack worker up cpu 1             # 1 CPU worker
flowmesh stack worker up gpu --targets 0   # 1 GPU worker pinned to GPU 0
flowmesh stack down
```

`flowmesh stack up` reads `NODE_ROLE` from the env file (default `root`). On a
root node, both local Redis services are deployed alongside the server. On a
worker node (`NODE_ROLE=worker`), Redis services are skipped — the worker
server connects to the root node's Redis via `REDIS_CONTROL_URL` and
`REDIS_TELEMETRY_URL`, which must be set in the worker's `.env` to reachable
endpoints on the root node.

After changing executor code, rebuild the affected image before bringing
the stack back up — running containers don't pick up source changes:

```bash
flowmesh stack build flowmesh_worker_cpu flowmesh_worker_gpu
```

`flowmesh stack build` loads native images for the local client platform.
Use `flowmesh stack build --no-builder` to skip exporting the standalone
GPU builder image when you only need the runtime images locally.
`flowmesh stack push` publishes multi-platform images (`linux/amd64` and
`linux/arm64`) for the stack bake targets.
Use `flowmesh stack push --no-builder` to skip publishing the standalone
GPU builder image while still building GPU runtime images.
Use `--image-tag <tag>` and `--build-ref <sha>` on either `build` or `push`
to override `FLOWMESH_VERSION` / `FLOWMESH_BUILD_REF` per invocation
without editing the env file; both values flow through to the
`org.opencontainers.image.{version,revision}` labels on the built images.
Values in `.env` always win over shell-set environment variables, so
`--image-tag` / `--build-ref` are the only way to override them without
editing the file.
`flowmesh stack push` also refreshes per-target registry build caches so
subsequent multi-platform pushes can reuse `arm64` and multi-stage layers.
Set `FLOWMESH_CACHE_VERSION` only when you want to intentionally start a
new remote cache lineage.
`flowmesh stack build` runs on the native `docker` driver and reuses
the local layer cache for fast iteration. `flowmesh stack push`
requires a `buildx` builder with the `docker-container` driver so it
can build multi-platform images and share the registry cache across
machines. Pass `--builder <name>` to either command to use a builder
other than the default, and `-f`/`--force` to skip the confirmation
prompt when the active `buildx` builder needs to switch.

To hand off a deployment bundle with bootstrap/config assets:

```bash
flowmesh stack bundle export             # root node (default)
flowmesh stack bundle export worker      # worker node
flowmesh stack bundle export --include-wheels
```

By default, the bundle's `install.sh` installs the published
`flowmesh[cli]` package for the current release. Use `--include-wheels`
when you need the archive to carry locally-built CLI/SDK wheels instead.

Alternatively, use `stack bundle init` to prepare a directory for
deployment. It creates empty `secrets/tls/{server,redis}/` and
`configs/worker_config.yaml`, and writes `.env` from the shipped
example. The normal flow is:

```bash
pip install flowmesh[cli]
flowmesh stack bundle init
# edit .env, configs/worker_config.yaml, drop TLS certs into secrets/tls/{server,redis}/
flowmesh stack pull
flowmesh stack up
```

Existing files are preserved. Use `--dest <path>` to scaffold elsewhere
and `--force` to overwrite `.env` without prompting.

## SSH tasks

```bash
flowmesh ssh connect <tsk-id>          # interactive shell into an SSH task
flowmesh ssh run <tsk-id> -- <cmd>     # one-shot exec
flowmesh ssh connections               # list active proxy/forward connections
```
