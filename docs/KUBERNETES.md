# Kubernetes deployment

FlowMesh deploys into an existing Kubernetes cluster with
`flowmesh stack ... --backend k8s`. One namespace holds one FlowMesh node: a
server Deployment with its supervisor, two Redis StatefulSets, and the worker
pods the supervisor creates.

## Prerequisites

- A reachable cluster and `kubectl` on `PATH`.
- A storage class for the Redis and results volumes, or `K8S_STORAGE_CLASS`
  pointing at one.
- For GPU workers: the NVIDIA device plugin, so nodes advertise
  `nvidia.com/gpu`.

## Deploying

```bash
flowmesh stack init                        # scaffold .env
# set STACK_BACKEND=k8s and K8S_NAMESPACE in .env
flowmesh stack up --backend k8s            # apply manifests, wait for rollout
flowmesh stack ps --backend k8s            # stack pods and worker pods
flowmesh stack logs server --backend k8s
flowmesh stack down --backend k8s
```

`STACK_BACKEND=k8s` in the env file makes `--backend` unnecessary on every
call. The same `.env` drives both backends.

`flowmesh stack up` applies the namespace, RBAC, Redis, and server manifests in
that order, then waits on each workload's rollout. `down` drains workers before
deleting the stack; `clean` also removes its persistent volume claims.

`restart` rolls the workloads in place. With `--image-tag` it applies instead,
because a rollout restart does not change the pod template's image.

`build`, `push`, `pull`, and `pullall` act on a local Docker daemon and take no
`--backend`; on Kubernetes each node's kubelet pulls images itself.

## Server topology

The server Deployment runs `replicas: 1` with the `Recreate` strategy. A
worker's token lives in the registry of the supervisor that minted it, so a
second replica behind the Service would reject registrations from workers the
other replica started. Scheduling state is persisted to Redis and rebuilt on
startup, so restarting the pod is safe.

Two Services front the Deployment:

| Service | Purpose |
|---------|---------|
| `flowmesh-server` | REST API on `SERVER_HTTP_PORT`. `K8S_SERVER_SERVICE_TYPE` selects `ClusterIP`, `NodePort`, or `LoadBalancer`. |
| `flowmesh-supervisor` | Headless; worker pods dial it for gRPC on `SERVER_GRPC_PORT`. |

The whole env file is carried into the pod as the `flowmesh-server-env` Secret,
which is the Kubernetes equivalent of compose's `env_file`. Values come from
the env file alone, so nothing else in the operator's shell reaches the
cluster.

## Workers

Worker pods are created by the supervisor through the Kubernetes API, not by a
separate manifest. Point `SERVER_WORKER_CONFIG` at a worker config declaring
the `kubernetes` provider; a starting point ships as
`cli/stack/src/flowmesh_cli_stack/assets/k8s/worker_config.k8s.yaml`.

```yaml
workers:
  - provider: kubernetes
    init_on_start: true
    worker_config:
      worker_type: gpu
      gpu_count: 1
      shm_size: 8Gi
      node_selector:
        nvidia.com/gpu.present: "true"
```

Each worker pod:

- runs with `restartPolicy: Always` and carries the labels
  `flowmesh.io/managed`, `flowmesh.io/node-alias`, and
  `flowmesh.io/worker-name`;
- receives its credentials from a per-worker Secret through `envFrom`, never
  inline in the pod spec;
- requests GPUs as `nvidia.com/gpu` (or `gpu_resource_name`) and leaves device
  assignment to the device plugin;
- has no owner reference to the server pod, so a server rollout does not take
  running workers with it. Pods left by an unclean exit are reaped at
  supervisor startup, because their tokens died with the previous registry.

Set `shm_size` for training and inference workloads; the 64 MiB default
`/dev/shm` is too small for torch dataloaders.

### Configuration reference

`results_pvc`, `results_mount_path`, `hf_cache_pvc`, `cpu_request`,
`cpu_limit`, `memory_request`, `memory_limit`, `tolerations`,
`service_account_name`, `image_pull_secrets`, `runtime_class_name`,
`priority_class_name`, `pod_labels`, and `pod_annotations` map onto their pod
spec equivalents. `pod_overrides` is merged onto the generated manifest for
anything else — mappings merge recursively, sequences and scalars replace.

### Results

Workflows declare where their output goes. Without a `results_pvc` a worker
writes to an ephemeral volume, so a workflow that keeps results either sets
`output.destination`, enables `WORKER_UPLOAD_RESULTS` to upload them to the
server, or names a claim through `results_pvc`.

### SSH tasks

Worker pods have no Docker socket, so the SSH executor does not load and
workers never advertise the `ssh` task type. The dispatcher routes SSH tasks
only to workers that advertise it.

## RBAC

`flowmesh stack up` creates a `flowmesh-server` ServiceAccount and a namespaced
Role granting pods, pod logs, secrets, and events — everything the supervisor
needs to run workers.

Setting `K8S_ENABLE_NODE_RBAC=true` additionally binds a cluster-scoped
ClusterRole granting `get` and `list` on nodes. This is the one cluster-scoped
grant, and it is optional: it lets the supervisor report a worker's hardware
from node allocatable capacity and GPU labels before the worker starts.
Without it that preview is empty, and hardware still arrives from the worker
itself at registration.

## TLS

`SERVER_GRPC_TLS_SECRET` names a Secret mounted at `/etc/ssl/server`, and
`REDIS_TLS_SECRET` one mounted at `/etc/ssl/redis`. The certificate must carry
the supervisor Service DNS name
(`flowmesh-supervisor.<namespace>.svc.cluster.local`) as a SAN, because that is
the name workers dial. The CA reaches workers automatically.

## Joining an external root node

A cluster can also run as a worker node against a root node deployed elsewhere.
Set `NODE_ROLE=worker` and point `REDIS_CONTROL_URL` and `REDIS_TELEMETRY_URL`
at the root node's Redis; the Redis StatefulSets are then not deployed.

## Not yet supported

Workers deployed directly with `kubectl` — a Deployment or DaemonSet that
enrolls itself rather than being created by the supervisor — need an enrollment
flow that mints a token at registration rather than at spawn. Today every
worker's token is minted by the supervisor before it starts the pod.
