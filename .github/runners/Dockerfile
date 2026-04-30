# syntax=docker/dockerfile:1
#
# FlowMesh self-hosted GitHub Actions runner image.
#
# Bases on myoung34/github-runner with the uv wheel cache pre-populated
# for the project's --all-extras dep set. CI workflows that run
# ``uv sync --all-extras --frozen`` warm-hit this cache instead of
# re-downloading every wheel on each run.
#
# Build context MUST be the repository root (FlowMesh is a private repo;
# we COPY from the build context rather than git-cloning at build time
# to avoid auth at build time):
#
#   cd <FlowMesh checkout>
#   git checkout main && git pull
#   docker build \
#     --build-arg UV_VERSION=0.11.8 \
#     -t flowmesh-oss-ci-runner:0.11.8-$(date +%Y%m%d) \
#     -t flowmesh-oss-ci-runner:latest \
#     -f .github/runners/Dockerfile \
#     .
#
# Refresh cadence:
#   - uv.lock changed on main → rebuild to refresh the cache.
#   - UV_VERSION bumped → rebuild AND update setup-uv's ``version:``
#     input in the workflows to match (see .github/runners/README.md).
#
# A stale cache is slower, not unsafe — uv falls through to network
# for any wheel not already cached in the image.
FROM myoung34/github-runner:latest

# Pinned uv version. Workflows MUST pin setup-uv@v7's ``version:``
# input to the same value or the cache layout written here may not
# match what CI reads back.
ARG UV_VERSION=0.11.8
ENV UV_CACHE_DIR=/root/.cache/uv

# Install pinned uv into a shared location.
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
      | env UV_INSTALL_DIR=/usr/local/bin sh \
 && uv --version

# Pre-populate the uv cache by sync'ing the project's full --all-extras
# dep set against the FlowMesh source in the build context. Only the
# resulting $UV_CACHE_DIR survives in the final image — the source tree
# and the temporary venv are deleted.
COPY . /tmp/flowmesh
RUN cd /tmp/flowmesh \
 && uv sync --all-extras --frozen \
 && cd / \
 && rm -rf /tmp/flowmesh
