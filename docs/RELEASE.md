# Release

FlowMesh publishes a lightweight `flowmesh` metapackage plus the public SDK,
CLI, stack helper, and hook distributions:

| Distribution | Source |
|--------------|--------|
| `flowmesh` | `pyproject.toml` |
| `flowmesh-sdk` | `sdk/` |
| `flowmesh-sdk-stack` | `sdk/stack/` |
| `flowmesh-cli` | `cli/` |
| `flowmesh-cli-stack` | `cli/stack/` |
| `flowmesh-hook` | `hook/` |

Runtime source under `src/` is not published to PyPI. Server and worker images
continue to copy that source directly and install their generated requirements
files.

## PyPI setup

Use PyPI Trusted Publishing instead of long-lived API tokens. Configure pending
or active trusted publishers for each distribution above on both PyPI and
TestPyPI.

Use this publisher configuration:

| Setting | Value |
|---------|-------|
| Owner | `mlsys-io` |
| Repository | `FlowMesh` |
| Workflow | `release.yml` |
| Environment | `pypi` for PyPI, `testpypi` for TestPyPI |

Create matching GitHub environments named `pypi` and `testpypi`. The `pypi`
environment should require manual approval. The `testpypi` environment can be
left open or use lighter approval rules.

The `pypi` approver should verify the matching TestPyPI run before approving
the production publish job.

## Prepare a release

1. Pick the next synchronized package version, for example `0.1.1`.
2. Update package versions and first-party pins:

   ```bash
   uv run scripts/dev/bump_version.py 0.1.1
   ```

3. Re-lock:

   ```bash
   uv lock
   ```

4. Validate the release metadata:

   ```bash
   uv run scripts/ci/check_release_version.py --tag v0.1.1
   ```

5. Build and smoke-test the distributions:

   ```bash
   uv sync --all-packages --group dev --frozen
   uv build --all-packages --out-dir dist
   uv run scripts/ci/check_package_build.py --dist dist
   ```

6. Run the normal validation suite:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest tests/ --ignore=tests/worker/test_mp_executor_cleanup_gpu.py
   ```

7. Open and merge a release prep PR with the version bump, `uv.lock`, and any
   release notes or docs updates.

## Publish to TestPyPI

After the release prep PR lands, create and push a signed or annotated tag:

```bash
git tag -a v0.1.1 -m "chore: release v0.1.1"
git push origin v0.1.1
```

Run the `Release` workflow manually with:

- `tag`: `v0.1.1`
- `publish_target`: `testpypi`

Or from the CLI:

```bash
gh workflow run release.yml -f tag=v0.1.1 -f publish_target=testpypi
```

Then verify the published artifacts from TestPyPI in a fresh environment:

```bash
python -m venv .venv-testpypi
. .venv-testpypi/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "flowmesh[sdk]"
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "flowmesh[hook]"
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "flowmesh[cli]"
flowmesh --help
```

## Publish to PyPI

Create a GitHub Release from the same `vX.Y.Z` tag. Publishing the release
triggers `.github/workflows/release.yml`, which rebuilds from the tag,
validates versions, smoke-tests the wheels, and publishes the exact uploaded
artifact set to PyPI after the `pypi` environment approval.

Do not move or force-update release tags. The release workflow assumes the tag
already passed PR or main-branch CI, and moving a tag can bypass that validation
history.

After the workflow finishes, verify PyPI installs in a fresh environment:

```bash
python -m venv .venv-pypi
. .venv-pypi/bin/activate
python -m pip install --upgrade pip
python -m pip install "flowmesh[sdk]"
python -m pip install "flowmesh[hook]"
python -m pip install "flowmesh[cli]"
flowmesh --help
```

The release workflow publishes all FlowMesh distributions from the same build.
Do not upload packages manually unless the workflow itself is unavailable and
the release owner has documented the fallback in the release notes.

## If a release goes wrong

PyPI versions are immutable: once `vX.Y.Z` is published you cannot edit,
re-upload, or replace it. Recovery options:

- **Yank** the bad release on PyPI. `pip install` still installs the version
  when it is explicitly pinned, but resolution skips it otherwise. Use yank
  for security or correctness bugs that warrant skipping the version
  entirely.
- **Cut the next patch.** Bump to `vX.Y.(Z+1)`, fix forward, and publish.
  This is the default path for any non-critical bug.
- **`.postN` re-release** of the same source release when the only change is
  packaging metadata (LICENSE, classifiers, README) and no Python code
  changed. Rare.

Do not delete or reuse a published version number under any circumstance.
