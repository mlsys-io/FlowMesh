# AGENTS.md — FlowMesh

Routing doc for coding agents (Claude Code, Codex, Cursor, …) working in
this repository. The full project rules live in dedicated docs; read the
ones relevant to your task before editing.

FlowMesh is a service fabric for running LLM agentic workflows on
distributed GPU workers. The server parses a workflow, turns it into a
DAG of tasks, dispatches each task to a worker, and collects results
and artifacts.

## Where to read what

- **@docs/ARCHITECTURE.md** — topology diagram,
  components (server / supervisor / worker), communication protocols,
  object IDs, task state machine, directory map, runtime behavior
  (task merging, stage stickiness, context reuse, log streams), plugin
  hooks. Read before any cross-component change.
- **@docs/CODE_STYLE.md** — Python rules,
  docstring conventions, bandit security rules and `# nosec` policy,
  pip-audit policy. Read before writing any source code.
- **@CONTRIBUTING.md** — setup, pre-commit hooks,
  testing, dependency-pin workflow, **commit and PR title conventions**,
  DCO sign-off. Read before committing or opening a PR.
- **[`docs/API.md`](docs/API.md)** — common server REST endpoints
  (workflows, tasks, results, workers, nodes, SSH, system) and cursor
  pagination contract. Read before calling the server directly or
  changing a router.
- **[`docs/SDK.md`](docs/SDK.md)** — client usage, common operations, error contract.
- **[`docs/CLI.md`](docs/CLI.md)** — `flowmesh ...` command groups,
  common workflows (submit/watch/logs), local stack lifecycle, SSH tasks.
- **[`docs/EXECUTORS.md`](docs/EXECUTORS.md)** — `taskType → Executor`
  registry table, helper utilities, and the `AgentExecutor` env
  requirements (`UTU_LLM_*`, `SERPER_API_KEY`, `JINA_API_KEY`).
- **[`docs/WORKFLOWS.md`](docs/WORKFLOWS.md)** — workflow YAML format
  hierarchy: single task, multi-stage DAG (`spec.stages`), graph DAG
  (`taskType: graph_template`), and schedule hints (`epoch_groups`,
  `schedule_in_epoch_order`).
- **[`docs/ENV.md`](docs/ENV.md)** — curated server / worker /
  supervisor env var tables (the knobs you actually tune). Full schema
  in `cli/stack/src/flowmesh_cli_stack/env_schema.py`.
- **[`docs/PLUGINS.md`](docs/PLUGINS.md)** — plugin extension contract,
  loader semantics (`FLOWMESH_PLUGINS`), and a worked example.

Concrete examples and runnable workflows live in `templates/`.
When code, APIs, CLI commands, SDK methods, env vars, workflow formats, or
runtime behavior change, update the corresponding docs in the same PR.

## Reminders

The full justification lives in the linked docs; these are the rules
that come up most often, surfaced here so you can't miss them.

- **PR title type**: one of `feat | fix | refactor | chore | test |
  perf | docs`. Anything else fails the title check
  (`scripts/ci/check_pr_title.py`). Use `docs:` for doc-only PRs —
  those skip the code-related CI jobs (lint, tests, security,
  env/requirements sync).
- **Docstrings**: describe what the code *does*, not what it
  *replaced*. No "in-process replacement for X", no "previously did Y".
- **Comments**: default to none. Add a short one only when the *why* is
  non-obvious; let names self-document.
- **Test failures**: never ignore one. CI guards the suite, so a red
  test on `main` is a CI gap, not a permission to skip — fix it as part
  of your PR even when the failure looks unrelated to your change.
  Every PR ships a runnable build.
- **Cluster management goes through SDK / CLI**: reach for `flowmesh ...`
  or the Python SDK. Raw `docker` is a last-resort escape hatch when the
  SDK / CLI doesn't expose what you need.

## Dev workflow

Use `CONTRIBUTING.md` for setup, hooks, tests, dependency pins, and
commit rules. Use `docs/CLI.md` for local stack and workflow commands.
