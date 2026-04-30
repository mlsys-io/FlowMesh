# FlowMesh self-hosted GitHub Actions runner image.
#
# To use it, clone the repo, and run the following command from the repo root:
#   docker build -t flowmesh-ci-runner:latest -f .github/runners/Dockerfile .

FROM myoung34/github-runner:latest

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
