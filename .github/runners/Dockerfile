# FlowMesh self-hosted GitHub Actions Runner Docker image
#
# To build it, run the following command from the repo root:
#   docker build -t flowmesh_ci_runner:latest -f .github/runners/Dockerfile .

FROM myoung34/github-runner:latest

LABEL org.opencontainers.image.title="FlowMesh CI Runner" \
      org.opencontainers.image.description="FlowMesh CI Runner" \
      org.opencontainers.image.source="https://github.com/mlsys-io/FlowMesh" \
      org.opencontainers.image.url="https://github.com/mlsys-io/FlowMesh"

ARG UV_VERSION=0.11.8

ENV UV_CACHE_DIR=/root/.cache/uv

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
      | env UV_INSTALL_DIR=/usr/local/bin sh \
 && uv --version
COPY . /tmp/flowmesh
RUN cd /tmp/flowmesh \
 && uv sync --all-extras --frozen \
 && cd / \
 && rm -rf /tmp/flowmesh
