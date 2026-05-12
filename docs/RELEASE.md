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

## Push container images

After TestPyPI verification, push the FlowMesh container images from the build
host. The Dockerfiles record the `org.opencontainers.image.{version,revision}`
labels from `--image-tag` and `--build-ref`; `release-images.yml` later asserts
these against the GitHub Release tag and the tagged commit's SHA, so set both
flags explicitly.

```bash
git fetch origin tag v0.1.1
git checkout v0.1.1
flowmesh stack push --image-tag v0.1.1 --build-ref "$(git rev-parse HEAD)"
```

Six manifests are written under `ghcr.io/mlsys-io/`:

- `flowmesh_server:v0.1.1`
- `flowmesh_worker:v0.1.1-cpu`
- `flowmesh_worker:v0.1.1-gpu`
- `flowmesh_worker_builder:v0.1.1-gpu`
- `flowmesh_ssh:v0.1.1-cpu`
- `flowmesh_ssh:v0.1.1-gpu`

Each manifest is a multi-arch OCI index covering `linux/amd64,linux/arm64`.
Verify before continuing:

```bash
docker buildx imagetools inspect ghcr.io/mlsys-io/flowmesh_server:v0.1.1
```

### First-time GHCR setup

The very first `flowmesh stack push` creates each of the six packages under
`ghcr.io/mlsys-io/`. Before `release-images.yml` can verify or retag, two
manual steps are needed per package (one-time):

1. Set visibility to **Public** at
   `github.com/orgs/mlsys-io/packages/container/<name>/settings`
   (Danger Zone → Change visibility). Without this, the verify job's
   anonymous reads fail.
2. **Link to the FlowMesh repository** with role **Write** (same page →
   Manage Actions access → Add repository). Without this, the retag-latest
   job's `GITHUB_TOKEN` cannot write `:latest`.

Also create a GitHub environment named **`ghcr`** at
`github.com/mlsys-io/FlowMesh/settings/environments` with required
reviewers configured — same approval pattern as the `pypi` environment.
The retag-latest job runs only after a reviewer approves.

## Publish to PyPI

Create a GitHub Release from the same `vX.Y.Z` tag. Publishing the release
triggers both `.github/workflows/release.yml` and
`.github/workflows/release-images.yml`. The first rebuilds from the tag,
validates versions, smoke-tests the wheels, and publishes the artifact set to
PyPI after the `pypi` environment approval. The second verifies that the
images you pushed earlier match the tag and commit, then retags them as
`:latest` after the `ghcr` environment approval. Pre-release tags
(`vX.Y.Z-rc1`, `vX.Y.Z.dev1`, etc.) still verify but skip the `:latest`
retag; post-releases (`vX.Y.Z.postN`) move `:latest` forward.

Do not move or force-update release tags. The release workflows assume the
tag already passed PR or main-branch CI, and moving a tag can bypass that
validation history.

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
- **Images failed but PyPI succeeded.** Push from the build host with the
  correct `--image-tag` and `--build-ref`, then re-run `release-images.yml`
  from the Actions UI with the same `tag`. PyPI is unaffected. The retag
  step is idempotent — re-runs replace the digest block in the Release body
  in place.

Do not delete or reuse a published version number under any circumstance.
