# Contributing to FlowMesh

Thank you for your interest in FlowMesh! We welcome bug fixes, new features, documentation improvements, and feedback of all kinds.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (required to run the local stack via `flowmesh stack up`)
- NVIDIA Container Toolkit (only if you plan to run GPU workers)

### Setup

```bash
uv sync --extra dev
```

The `dev` extra is required for all contributors (linters, formatters, type
checkers, test runners). Add more extras for the component you're working on:

| Extra | Purpose |
|-------|---------|
| `dev` | Linters, formatters, type checkers, test runners (required for all contributors) |
| `server` | Server dependencies (FastAPI, Redis, gRPC, SQLAlchemy, asyncpg, Docker) |
| `worker-core` | Worker process baseline (gRPC, httpx, docker) |
| `inference` | Inference executors (transformers, diffusers, accelerate, torch) |
| `inference-gpu` | GPU-only inference deps (vLLM, vllm-omni, bitsandbytes, flashinfer) |
| `training` | Training executors (trl, peft) |
| `training-gpu` | GPU-only training deps (deepspeed) |
| `rag` | RAG executor (Qdrant, fastembed) |
| `agent` | Agent executor / Utu framework (OpenAI Agents SDK, Hydra, tools) |
| `analytics` | Data analytics and connectors (pandas, boto3, psycopg) |
| `observability` | Tracing and monitoring (OpenTelemetry, Arize Phoenix) |
| `cli` | CLI package |
| `sdk` | SDK package |

For full development across all components:

```bash
uv sync --all-extras
```

### Install Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to enforce formatting, linting, type checking, and DCO sign-off on every commit.

```bash
uv run pre-commit install --install-hooks -t pre-commit -t prepare-commit-msg -t commit-msg
```

This installs three hook stages:
- **pre-commit** — runs gitleaks, isort, black, ruff, mypy, and codespell on staged files.
- **prepare-commit-msg** — automatically appends a
  [DCO sign-off](#signing-off-commits-dco) line to your commit message.
- **commit-msg** — verifies the sign-off is present (safety net).

## Code Style

| Tool | Purpose | Config |
|------|---------|--------|
| [gitleaks](https://github.com/gitleaks/gitleaks) | Committed-secret detection | - |
| [isort](https://pycqa.github.io/isort/) | Import sorting | `pyproject.toml` `[tool.isort]` |
| [Black](https://black.readthedocs.io/) | Code formatting | `pyproject.toml` `[tool.black]` |
| [Ruff](https://docs.astral.sh/ruff/) | Linting | `pyproject.toml` `[tool.ruff]` |
| [mypy](https://mypy.readthedocs.io/) | Type checking | `pyproject.toml` `[tool.mypy]` |
| [codespell](https://github.com/codespell-project/codespell) | Spell checking | `pyproject.toml` `[tool.codespell]` |
| [bandit](https://bandit.readthedocs.io/) | Python source security audit | `pyproject.toml` `[tool.bandit]` |

You can also run all checks manually:

```bash
uv run pre-commit run --all-files            # Full repo
uv run pre-commit run                        # Staged files only
uv run pre-commit run --files src/server/*.py  # Specific files
```

## Testing

```bash
uv run pytest tests/                   # All tests
uv run pytest tests/server/            # Server-specific tests
uv run pytest tests/test_core_flow.py  # Single file
```

If your change touches shared schemas or proto definitions, verify downstream compatibility across Server and Worker packages.

## Dependency Pins

Dependency versions live in two places with different styles:

- **`pyproject.toml`** — `>=X.Y.Z` lower bounds. Expresses a compatibility floor; lets uv resolve the current acceptable version.
- **`src/worker/requirements/requirements{,.gpu}.txt`** — exact `==X.Y.Z` pins. These feed the worker Docker images (`uv pip install --requirement …`), which need deterministic, reproducible installs.

The requirements files are **auto-generated** from `pyproject.toml` + `uv.lock` by `scripts/dev/sync_requirements.py`. Do not edit them by hand.

When you bump a dependency:

```bash
# 1. Raise the `>=` floor in pyproject.toml (or add / remove a package).
# 2. Re-lock.
uv lock

# 3. Regenerate the worker requirements files.
uv run scripts/dev/sync_requirements.py --write

# 4. Commit all three together: pyproject.toml, uv.lock, requirements*.txt.
```

The `sync-requirements` pre-commit hook (and the `Requirements Sync Check` CI job) enforces this — a PR that edits `pyproject.toml` or `uv.lock` without regenerating the requirements files will fail with a diff pointing at the stale file.

## Signing Off Commits (DCO)

All commits must carry a [Developer Certificate of Origin](https://developercertificate.org/) sign-off line. If you installed the pre-commit hooks as described above, this is handled automatically — the `prepare-commit-msg` hook appends the sign-off to every commit, and the `commit-msg` hook verifies it. CI also checks all PR commits.

**Fixing unsigned commits** (e.g. commits made before installing hooks):

```bash
git rebase --signoff HEAD~N   # N = number of commits to fix
git push --force-with-lease
```

## Project Layout

FlowMesh follows a multi-tier architecture (Server / Worker) with
shared schemas, SDK, and CLI packages.
