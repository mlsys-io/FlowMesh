# Rolling image updates

FlowMesh nodes can be updated to a new image one at a time without tearing the
whole cluster down. The rollout itself is driven externally — by an operator or
a cluster-management tool — and FlowMesh provides the primitives that make each
node safe to recreate and the root node able to resume in-flight work.

## Operator flow

Recreate one node at a time, leaving the others serving. Update the **root node
last**.

```bash
# On each node host, in turn:
flowmesh stack restart server --image-tag <new-version>
```

`flowmesh stack restart server` (see [`CLI.md`](CLI.md)):

1. Drains the node's managed workers, so their in-flight tasks are released and
   requeued onto other nodes.
2. Recreates only the `server` Compose service (`--no-deps --force-recreate`),
   pulling the new image. Redis and any other services keep running.
3. Blocks (`--wait`) until the recreated server passes its healthcheck.

Because the worker-managing process is the `server` service on both root and
worker nodes, the same command works everywhere.

## What happens to in-flight work

**Worker nodes.** Draining a node tears down its workers. Each worker's
departure produces a `WORKER_UNREGISTER` (the supervisor synthesizes one if the
worker did not send it), so the server recovers the worker's `DISPATCHED` tasks
and requeues them onto other eligible nodes. A recreated node's supervisor
re-creates its configured workers, which re-register themselves on startup. No
cordon step is required.

**Root node.** The root holds the dispatcher's scheduling state in memory, so a
naive restart would lose every in-flight workflow. Two mechanisms make a root
restart safe:

- **Durable scheduler state.** Each task's mutable state (status, attempts,
  assigned worker, failed workers, merge linkage), its dependency edges, and its
  epoch index are persisted to Redis on every transition, along with per-workflow
  epoch ordering and frontier. On startup the server rebuilds the full task DAG,
  ready queue, and epoch frontiers from these records
  (`TaskRuntime.rehydrate`).
- **Replayable task events.** Task lifecycle events flow through a durable Redis
  stream consumed from a persisted cursor. Completions that occur while the root
  is down are replayed on startup rather than dropped; event handlers are
  idempotent so replay cannot double-apply. In-flight tasks are left assigned to
  their worker — surviving workers' completions arrive via the stream, and
  workers that genuinely departed are reclaimed by the watchdog.

Rehydration runs inside the ASGI lifespan **before it yields**, so the server
does not accept traffic (and its healthcheck does not pass) until scheduling
state is fully restored. Readiness is therefore implicit — no separate probe is
needed, and `stack restart`'s `--wait` blocks until the node is genuinely ready.

The result is a brief control-plane pause on the root (the server container
recreate plus rehydration) during which workers keep running their tasks; no
workflow is lost.

## Constraints

- **Recreate only the `server` service on the root.** Leave `redis_control` and
  `redis_telemetry` running so durable state and the event stream survive.
  Updating the Redis image is a heavier, control-plane-wide outage and is out of
  scope for a brief-pause rolling update.
- **Co-located root workers are recreated.** Workers running on the root host die
  with the root's supervisor; their tasks requeue and re-run. To avoid this,
  prefer not to run workers on the root node.
