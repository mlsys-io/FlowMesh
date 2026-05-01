# Build CPU image (no vLLM)
docker build -f src/worker/docker/Dockerfile.cpu -t yourrepo/flowmesh_worker:cpu-latest .

# Build CUDA image (installs vLLM + GPU extras)
docker build -f src/worker/docker/Dockerfile.cuda -t yourrepo/flowmesh_worker:cuda-latest .

# Build SSH session image (CPU, for direct/proxy/forward-mode SSH tasks)
docker build -f src/worker/docker/Dockerfile.ssh.cpu -t yourrepo/flowmesh_ssh:latest-cpu .

# Build SSH session image (GPU, for SSH tasks requiring GPU access)
docker build -f src/worker/docker/Dockerfile.ssh.gpu -t yourrepo/flowmesh_ssh:latest-gpu .

# Run (CPU)
docker run --rm \
  -e GUARDIAN_GRPC_TARGET="host.docker.internal:50051" \
  -e WORKER_HB_FILE="/tmp/flowmesh_worker_health/worker.hb" \
  -e RESULTS_DIR=/app/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/results_cpu:/app/results" \
  yourrepo/flowmesh_worker:cpu-latest

# Run (GPU; host must have NVIDIA Container Toolkit)
docker run --rm --gpus all \
  -e GUARDIAN_GRPC_TARGET="host.docker.internal:50051" \
  -e WORKER_HB_FILE="/tmp/flowmesh_worker_health/worker.hb" \
  -e RESULTS_DIR=/app/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/results_gpu:/app/results" \
  yourrepo/flowmesh_worker:cuda-latest

## TLS CA injection

If the guardian uses TLS, pass the internal CA via env:

```
scripts/dev/generate_guardian_tls_certs.sh <guardian-host>
export GUARDIAN_GRPC_TLS_CA_B64="$(base64 -w 0 secrets/tls/guardian/guardian-ca.pem)"
docker run --rm \
  -e GUARDIAN_GRPC_TLS_CA_B64 \
  -e GUARDIAN_GRPC_TARGET="host.docker.internal:50051" \
  yourrepo/flowmesh_worker:cpu-latest
```

## docker-compose with NFS shared results

Edit `worker/docker-compose.yml` and set the following environment variables before
running `docker compose up` (for example via an `.env` file in the same directory):

```
NFS_SERVER=10.0.0.10          # NFS server hostname or IP (defaults to localhost)
NFS_EXPORT_PATH=/srv/flowmesh/results
NFS_VERSION=4                 # optional, defaults to 4
```

The compose file mounts the shared export at `/mnt/flowmesh-results` inside the
worker container and sets `RESULTS_DIR` accordingly so that all task outputs are
stored on the NFS share.
