# Code style

## Python

- Python 3.12+ (see `[project]` in `pyproject.toml`).
- Top-level imports only; inline imports only to break circular imports.
- Prefer `typing.Any` over `object`; use `object` only when `Any` is
  semantically wrong.
- Prefer `X | Y` and `X | None` over `typing.Union[X, Y]` /
  `typing.Optional[X]`.
- Don't write `from __future__ import annotations` — use `typing.Self` or
  quoted forward references instead.
- No `hasattr` / `getattr` that bypasses type checking. Use `isinstance`
  guards. Acceptable `getattr` uses: dynamic dispatch, providing a
  default, reaching into untyped third-party APIs.
- `# type: ignore[<error-code>]` only after exhausting alternatives.
  Never a bare `# type: ignore`.
- Use `path.as_posix()` when a `Path` is serialized to a string for
  internal use (data structures, APIs, storage, test assertions). Use
  `str(path)` only for user-facing output (log messages, CLI display)
  where an OS-native path is more readable.

## Comments and docstrings

- Default to **no comments**. Comment only when *why* is non-obvious.
  Names self-document.
- Don't reference the current task / fix / caller in comments ("used by
  X", "added for Y", "handles issue #123") — those rot.
- Docstrings describe what the function/module *does*, not what it
  *replaced*. No "in-process replacement for X", no "previously did Y" —
  read the docstring as if seeing the code for the first time.

## Object IDs

3-char prefixes: `wfl-`, `tsk-`, `ssn-`, `scn-`, `cmd-`. Always use
`new_*_id()` helpers in `src/shared/utils/ids.py`. Never use `uuid4()`
or `secrets.token_hex` for IDs.

## Security rules (bandit-enforced)

CI runs `bandit` with no severity / confidence threshold. Every finding
must have a source-level fix, a documented skip in `[tool.bandit]` in
`pyproject.toml`, or a per-line `# nosec BXXX` with a one-line written
rationale at the call site. A bare `# nosec` (no rule code, no reason)
is disallowed.

When writing new code, follow these rules:

- **B113** — every `requests.get/post/...` call passes `timeout=`. No
  implicit defaults; hung connections are a DoS.
- **B202** — `tarfile.extractall(..., filter="data")` (Python 3.12+).
  For zipfile, iterate `infolist()`, validate each member resolves under
  the destination, extract per-member. Never `zipfile.extractall` on
  untrusted archives.
- **B310** — don't use `urllib.request.urlopen`. Use `requests` and
  validate the URL scheme (`http`/`https` only) before fetching.
- **B324** — `hashlib.md5(..., usedforsecurity=False)` is required for
  cache-key / fingerprint use. Never MD5 across a security boundary.
- **B506** — `yaml.safe_load`, never `yaml.load(..., Loader=FullLoader)`.
- **B603** — every `subprocess.run/call/Popen/...` needs a per-line
  `# nosec B603` with a one-line rationale (e.g. `argv list, no
  shell=True, absolute path via shutil.which()`). The B404 import-level
  rule is project-skipped because B602/B607 catch the actually-dangerous
  patterns; B603 is enforced per-site so every shellout is visible at
  the call line.
- **B607** — prefer the vendored SDK (`pynvml`, `docker-py`, `GitPython`)
  over shelling out via `nvidia-smi` / `docker` / `git`. If shelling out
  is unavoidable, the absolute path must be provided.
- **B614** — `torch.load(..., weights_only=True)`. Pickle deserialization
  is RCE waiting to happen.
- **B701** — `Environment(autoescape=select_autoescape())`. The default
  `False` is unsafe even for non-HTML templates.
- **B108** — use `tempfile.gettempdir()` or `tempfile.NamedTemporaryFile`.
  The literal `"/tmp"` in Python source is forbidden; for an
  in-container sentinel, build it from `PurePosixPath` segments.

When a documented `[tool.bandit]` skip stops being true (e.g. a sandbox
stops being a sandbox), remove the skip and fix the call sites — don't
widen the skip list silently.

## Dependency CVE scanning (pip-audit)

CI runs `pip-audit` against each generated requirements file
(`src/server/requirements.txt`,
`src/worker/requirements/requirements.txt`,
`src/worker/requirements/requirements.gpu.txt`). The job lives in
`.github/workflows/security.yml`.

When pip-audit reports a new CVE, the only real fix is to bump the
offending dep in `pyproject.toml`, then `uv lock` and `uv run
scripts/dev/sync_requirements.py --write`. Silencing via `--ignore-vuln`
is a last resort; every silenced advisory needs a written
upgrade-blocker. The currently-ignored advisories and the upgrade
blocker that justifies each are listed below; the same list is encoded
as `--ignore-vuln` flags in `.github/workflows/security.yml`.

The worker GPU `vllm` is pinned to the `+cu129` release wheel (the PyPI
wheel is built for CUDA 13, incompatible with the CUDA 12.9 worker). Its
local version is not on PyPI, so pip-audit skips it — like
`flashinfer-jit-cache` — which is why the GPU run omits `--strict`. vLLM
CVE exposure tracks PyPI `vllm 0.24.0` regardless of the build variant.

| Advisory | Package | Fix version | Why ignored |
|----------|---------|-------------|-------------|
| GHSA-rrmf-rvhw-rf47 | torch | (none) | no fix version published |
| PYSEC-2026-87 | lxml | 6.1.0 | crawl4ai 0.8.6 caps lxml<6 |
| GHSA-w8v5-vhqr-4h9v | diskcache | (none) | upstream unmaintained, no fixed version published |
| PYSEC-2026-597 | nltk | (none) | latest 3.9.4 is last_affected, no fixed release; transitive via crawl4ai and not reachable |
| PYSEC-2026-3447 | setuptools | 83.0.0 | vllm 0.24.0 pins setuptools<81 for py>=3.12, so the fixed >=83.0.0 is unsatisfiable; no reachable fix |

The worker GPU audit ignores `GHSA-rrmf-rvhw-rf47`,
`GHSA-w8v5-vhqr-4h9v`, and `PYSEC-2026-3447`; the worker CPU audit
ignores `GHSA-rrmf-rvhw-rf47`, `PYSEC-2026-87`, `PYSEC-2026-597`, and
`PYSEC-2026-3447`; the server audit ignores nothing.

When a blocker lifts (e.g. crawl4ai unpins lxml, or a fixed torch
release ships), drop the corresponding `--ignore-vuln` flag from the
workflow and the row from this table — don't extend the rationale to
unrelated packages.
