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
  stream consumed from a persisted cursor. The ordering is what makes replay
  safe: a transition is written to durable scheduler state *before* its event is
  emitted, and the consumer advances the cursor only *after* it has handled an
  entry. Delivery is therefore at-least-once — a crash between handling an entry
  and persisting the cursor simply replays that entry on the next startup.
  Handlers are idempotent (a terminal task ignores late dispatch / start /
  update events, and a repeated completion is dropped), so replay cannot
  double-apply. Completions that occur while the root is down are replayed on
  startup rather than dropped. In-flight tasks are left assigned to their worker
  — surviving workers' completions arrive via the stream, and workers that
  genuinely departed are reclaimed by the watchdog.
- **Heartbeat grace for rehydrated work.** Worker heartbeats are dropped while
  the root is down, so a surviving worker briefly looks stale once the root is
  back. The watchdog gives any worker that owns rehydrated in-flight tasks an
  extended grace (`WORKER_REHYDRATION_GRACE_SEC`, default 120s) before it may
  reclaim those tasks, so a worker that is merely catching up is not mistaken
  for a dead one and its tasks are not needlessly requeued.

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
- **The no-worker grace restarts on a root restart.** The window before a task
  that no worker can satisfy is failed (`TASK_NO_WORKER_GRACE_SEC`) is tracked
  with ephemeral scheduler state that is intentionally not persisted, so it
  starts fresh after a restart. This is deliberate: the restart is itself a
  disruption, and a fresh window avoids grace-failing a waiting task the instant
  the control plane comes back.

## State lifetime

Cluster state (workflows, durable scheduler records, the task-event stream)
lives as long as the Redis volumes — **not** the server process. Stopping or
recreating the server never clears it, which is what lets a restart resume
in-flight work; a plain `stack down` / `stack up` likewise resumes the queue.
To reset to a clean slate, remove the Redis volumes with `flowmesh stack clean`
(`down -v`).
