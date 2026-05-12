ARG TZ=Asia/Singapore
ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04
ARG TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 8.9 9.0 10.0 12.0'

# Builder stage keeps development-only CUDA bits
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG TZ
ARG CUDA_VERSION
ARG TORCH_CUDA_ARCH_LIST
ENV TZ=${TZ} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_TORCH_BACKEND=cu129 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY="unsafe-best-match" \
    UV_LINK_MODE=copy \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# The CUDA base image ships an NVIDIA apt source, but these Dockerfiles only
# need standard Ubuntu packages. Dropping the external repo avoids transient
# mirror-sync failures during apt metadata refresh.
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates ccache curl git libibverbs-dev libnuma-dev tini tzdata \
      python3 python3-dev python3-pip python3-venv \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Workaround for Triton/PyTorch CUDA compatibility issues
# References: https://github.com/openai/triton/issues/2507
#             https://github.com/pytorch/pytorch/issues/107960
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Create isolated venv and upgrade pip + uv
RUN python3 -m venv /opt/py312 \
 && /opt/py312/bin/pip install --upgrade pip uv
ENV PATH=/opt/py312/bin:$PATH

WORKDIR /opt

# Install GPU-specific dependencies (Heavy, rarely changes)
COPY src/worker/requirements/requirements.gpu.txt /tmp/requirements.gpu.txt
RUN uv pip install --python /opt/py312/bin/python --system --requirement /tmp/requirements.gpu.txt \
 && rm -f /tmp/requirements.gpu.txt \
 && rm -rf /root/.cache/uv /root/.cache/pip /root/.cache/ccache

ARG BUILD_VERSION=dev
ARG BUILD_REF=local
ARG BUILD_CREATED=unknown
LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${BUILD_CREATED}" \
      org.opencontainers.image.revision="${BUILD_REF}"
