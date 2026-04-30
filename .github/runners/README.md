# FlowMesh self-hosted runner image

`Dockerfile` here builds a custom GitHub Actions runner image based on `myoung34/github-runner` with the uv wheel cache pre-populated for the project's `--all-extras` install. CI workflows that run `uv sync --all-extras --frozen` warm-hit this cache instead of redownloading every wheel.

## Build

Build context is the repo root, so run from the top of your FlowMesh checkout:

```bash
git checkout main && git pull
docker build \
  --build-arg UV_VERSION=0.11.8 \
  -t flowmesh-oss-ci-runner:0.11.8-$(date +%Y%m%d) \
  -t flowmesh-oss-ci-runner:latest \
  -f .github/runners/Dockerfile \
  .
```

The two tags give a date-stamped reference for rollback plus a moving `:latest` for the runner systemd units. Build it on a `linux/amd64` host so the cached wheels are platform-correct for the runners (which are also `linux/amd64`).

## Deploy to a runner host

Update the runner systemd unit's `docker run` command to use the new image. Where you currently have:

```
ExecStart=/usr/bin/docker run --rm \
    ... \
    myoung34/github-runner:latest
```

change the image reference:

```
ExecStart=/usr/bin/docker run --rm \
    ... \
    flowmesh-oss-ci-runner:latest
```

Reload + restart the units:

```bash
sudo systemctl daemon-reload
sudo systemctl restart 'flowmesh-oss-ci-cuda-runner@*.service'
sudo systemctl restart 'flowmesh-oss-ci-gpu-runner@*.service'
```

## Refresh cadence

Rebuild the image when:

- **`uv.lock` changed on `main`.** New / updated wheels won't be in the image; CI will fall through to network for those. Not unsafe, just slower until the rebuild.
- **`UV_VERSION` bumped.** uv's cache layout changes across minor versions; a mismatch between the image-time uv and CI-time uv (set by `setup-uv@v7`'s `version:` input) can leave the cache unused. After bumping, update both `--build-arg UV_VERSION=...` here AND every `setup-uv@v7` invocation in `.github/workflows/*.yml` to match.

A stale cache is never unsafe — uv falls through to network for any wheel not in the image. The cost of staleness is install time, not correctness.

## Workflow side

Every `setup-uv@v7` invocation in CI must pin its `version:` to the same `UV_VERSION` the image was built with, and should keep `enable-cache: false` (the runner image already supplies the cache; GHA cache would override it):

```yaml
- uses: astral-sh/setup-uv@94527f2e458b27549849d47d273a16bec83a01e9  # v7
  with:
    version: "0.11.8"
    enable-cache: false
```
