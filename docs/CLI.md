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
flowmesh stack worker {up, start, stop, down, list, pull}
```

## Common workflows

Submit a workflow and watch it:

```bash
flowmesh workflow submit templates/echo_local.yaml
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

When multiple people share one host, give each stack its own
`FLOWMESH_STACK_SUFFIX` and distinct `SERVER_HTTP_PORT`,
`SERVER_GRPC_PORT`, `REDIS_CONTROL_PORT`, and `REDIS_TELEMETRY_PORT`.
The suffix isolates Docker object names; the ports isolate host bindings.

```bash
flowmesh stack up                          # Server + Redis + Supervisor
flowmesh stack worker up cpu 1             # 1 CPU worker
flowmesh stack worker up gpu --targets 0   # 1 GPU worker pinned to GPU 0
flowmesh stack down
```

After changing executor code, rebuild the affected image before bringing
the stack back up — running containers don't pick up source changes:

```bash
flowmesh stack build flowmesh_worker_cpu flowmesh_worker_gpu
```

## SSH tasks

```bash
flowmesh ssh connect <tsk-id>          # interactive shell into an SSH task
flowmesh ssh run <tsk-id> -- <cmd>     # one-shot exec
flowmesh ssh connections               # list active proxy/forward connections
```
