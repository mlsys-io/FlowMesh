# Service restarts

FlowMesh supports fine-grained, in-place restarts: any Compose service — most
importantly the root server — can be recreated without tearing the cluster down
and without losing in-flight work. The same machinery covers every kind of
restart — a crash, a config change, a single-service redeploy, or a rolling
image bump — because the root's scheduling state is durable and rebuilt on
startup and task lifecycle events are replayable. A restarted root resumes its
in-flight workflows instead of dropping them.

## The restart primitive

`flowmesh stack restart [SERVICE ...]` (see [`CLI.md`](CLI.md)) recreates one or
more Compose services in place, leaving the rest of the stack running:

```bash
flowmesh stack restart server                # recreate just the server in place
flowmesh stack restart redis_control server  # recreate several services in one call
flowmesh stack restart                       # whole-stack drain + down + up
```

For each invocation it:

1. Drains the node's managed workers **once** if any named service manages
   workers (the `server` / supervisor), so their in-flight tasks are released
   and requeued onto other nodes.
2. Recreates only the named services (`--no-deps --force-recreate`), optionally
   pulling a new image (`--pull`, on by default). Redis and any unnamed service
   keep running.
3. Blocks (`--wait`) until the recreated services pass their healthchecks.

With no argument it restarts the whole stack (drain + `down` + `up`). Because
the worker-managing process is the `server` service on both root and worker
nodes, the same command works everywhere.

## What survives a restart

**Worker nodes.** Draining a node tears down its workers. Each worker's
departure produces a `WORKER_UNREGISTER` (the supervisor synthesizes one if the
worker did not send it), so the server recovers the worker's `DISPATCHED` tasks
and requeues them onto other eligible nodes. A recreated node's supervisor
re-creates its configured workers, which re-register themselves on startup. No
cordon step is required.

**Root node.** The root holds the dispatcher's scheduling state in memory, so a
naive restart would lose every in-flight workflow. Three mechanisms make a root
restart safe:

- **Durable scheduler state.** Each task's mutable state (status, attempts,
  assigned worker, failed workers, merge linkage), its dependency edges, and its
  epoch index are persisted to Redis on every transition, along with per-workflow
  epoch ordering and frontier. On startup the server rebuilds the full task DAG,
  ready queue, and epoch frontiers from these records (`TaskRuntime.rehydrate`).
  A transition's task records, workflow status-set membership, and schedule
  snapshot are written as a single atomic Redis transaction
  (`WorkflowRegistry.commit_transition`), so a crash mid-persist commits the whole
  transition or none of it. Event-driven transitions are additionally healed by replay;
  the API-driven workflow cancel relies on this atomicity alone.
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

## Use case: rolling image updates

Because each node survives an in-place server restart, a cluster can be moved to
a new image one node at a time without a full teardown. The rollout itself is
driven externally — by an operator or a cluster-management tool — using the same
primitive with an explicit tag:

```bash
# On each node host, in turn — update the root node last:
flowmesh stack restart server --image-tag <new-version>
```

Recreate one node at a time, leaving the others serving, and update the **root
node last** so the control plane is the final hop. Each worker node's in-flight
tasks requeue while it is down and its workers re-register once it is back; the
root's durable state carries its in-flight workflows across its own restart.

## Constraints

- **Recreate only the `server` service on the root.** Leave `redis_control` and
  `redis_telemetry` running so durable state and the event stream survive.
  Updating the Redis image is a heavier, control-plane-wide outage and is out of
  scope for a brief in-place restart.
- **Co-located root workers are recreated.** Workers running on the root host die
  with the root's supervisor; their batch tasks requeue and re-run (cancellable
  in-flight work such as an SSH session is cancelled instead). To avoid this,
  prefer not to run workers on the root node.
- **The no-worker grace restarts on a root restart.** The window before a task
  that no worker can satisfy is failed (`TASK_NO_WORKER_GRACE_SEC`) is tracked
  with ephemeral scheduler state that is intentionally not persisted, so it
  starts fresh after a restart. This is deliberate: the restart is itself a
  disruption, and a fresh window avoids grace-failing a waiting task the instant
  the control plane comes back.

## State lifetime

Cluster state (workflows, durable scheduler records, the task-event stream)
lives in the two Redis instances, which snapshot to disk (`redis-server --save`)
on named Docker volumes — `<slug>_redis_control_data` and
`<slug>_redis_telemetry_data`. The state therefore follows the *volumes*, not
the container or the server process:

- `stack restart` and `stack down` recreate or stop containers **without**
  `-v`, so the volumes and the state persist; Redis saves on the SIGTERM
  from a graceful stop and reloads the snapshot on the next start. This is what
  lets a restart (or a plain `stack down` / `stack up`) resume in-flight work.
- `flowmesh stack clean` is the only command that wipes the state: it runs
  `down -v`, removing the volumes. (`stack purge` only deletes images; it does
  not touch the volumes.)

Persistence is snapshot-based (RDB), not write-synchronous, so a *graceful*
restart preserves everything, but an abrupt loss of a Redis container (kill,
OOM, host crash) can drop up to the last snapshot window — 60s for control,
300s for telemetry.
