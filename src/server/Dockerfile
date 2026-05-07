# FlowMesh Server Docker image
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FlowMesh Server" \
      org.opencontainers.image.description="FlowMesh server (orchestrator + supervisor)" \
      org.opencontainers.image.source="https://github.com/mlsys-io/FlowMesh" \
      org.opencontainers.image.url="https://github.com/mlsys-io/FlowMesh"

ARG TZ=Asia/Singapore
ENV TZ=${TZ} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git lsof net-tools procps redis-server tini tzdata \
      python3 python3-pip python3-venv \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Create isolated venv and install uv
RUN python3 -m venv /opt/py312 \
 && /opt/py312/bin/pip install --upgrade pip uv
ENV PATH=/opt/py312/bin:$PATH

WORKDIR /app

# Copy only dependency files first (leverage Docker layer caching)
COPY src/server/requirements.txt ./src/server/requirements.txt

# Install dependencies before copying source code
RUN uv pip install --python /opt/py312/bin/python --system -r src/server/requirements.txt

COPY src/server ./src/server
COPY src/shared ./src/shared
COPY hook/src ./hook/src

# Ensure packages and plugins, if any, are importable.
ENV PYTHONPATH=/app/src:/app/hook/src:/app/plugins
RUN mkdir -p /app/plugins

# Default server env
ENV REDIS_URL=redis://redis_control:6379/0 \
    REDIS_CONTROL_URL=redis://redis_control:6379/0 \
    REDIS_TELEMETRY_URL=redis://redis_telemetry:6379/0 \
    RESULTS_DIR=/mnt/flowmesh-results \
    WORKER_RESULTS_DIR=flowmesh_results \
    SERVER_METRICS_DIR=/mnt/flowmesh-metrics \
    LOG_DIR=/var/log/flowmesh-server \
    LOG_FILE=/var/log/flowmesh-server/server.log \
    LOG_STREAM_MAXLEN_TASK=50000 \
    LOG_STREAM_MAXLEN_WORKFLOW=200000 \
    LOG_STREAM_TTL_SEC=3600 \
    TASK_LOG_ARCHIVE_FLUSH_INTERVAL_SEC=5 \
    TASK_LOG_ARCHIVE_FLUSH_MAX_ENTRIES=100 \
    LOG_LEVEL=INFO \
    SERVER_APP_PORT=8000 \
    SERVER_GRPC_PORT=50051

# Create directories
RUN mkdir -p /mnt/flowmesh-results /mnt/flowmesh-metrics /var/log/flowmesh-server

EXPOSE 8000 50051

ARG BUILD_VERSION=dev
ARG BUILD_REF=local
ARG BUILD_CREATED=unknown
LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${BUILD_CREATED}" \
      org.opencontainers.image.revision="${BUILD_REF}"

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "server.main"]
