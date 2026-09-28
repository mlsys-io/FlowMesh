"""TSQueue must not park executor threads.

Regression: ``get`` ran ``queue.Queue.get`` in the loop's default executor. A
cancelled ``StreamTasks`` (worker disconnect) left that thread blocked forever;
after ``min(32, cpu_count + 4)`` worker reconnects the pool was exhausted and
every later dispatch ``put`` queued behind it, so tasks stayed DISPATCHED on
idle, heartbeating workers.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from server.utils.helpers import TSQueue


def test_cancelled_gets_do_not_starve_later_puts() -> None:
    dead_queues: list[TSQueue[str | None]] = []
    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.new_event_loop()
    # A tiny default executor makes exhaustion immediate if get/put used it.
    loop.set_default_executor(executor)

    async def scenario() -> str | None:
        # Streams that open a get and are then cancelled (worker disconnects).
        for _ in range(10):
            dead: TSQueue[str | None] = TSQueue()
            dead_queues.append(dead)
            waiter = asyncio.ensure_future(dead.get())
            await asyncio.sleep(0.01)
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass

        live: TSQueue[str | None] = TSQueue()
        getter = asyncio.ensure_future(live.get())
        await asyncio.sleep(0)

        # Hand over from another thread, the way TaskListener does.
        threading.Thread(
            target=lambda: asyncio.run_coroutine_threadsafe(live.put("task"), loop),
            daemon=True,
        ).start()
        try:
            return await asyncio.wait_for(getter, timeout=2)
        except TimeoutError:
            return None

    try:
        result = loop.run_until_complete(scenario())
    finally:
        # Release any thread an executor-backed implementation left blocked, so
        # a regression fails this assertion instead of hanging the test run.
        for dead in dead_queues:
            dead._q.put_nowait(None)
        executor.shutdown(wait=True)
        loop.close()
    assert result == "task"


def test_fifo_and_qsize() -> None:
    async def scenario() -> list[int]:
        q: TSQueue[int] = TSQueue()
        for i in range(3):
            await q.put(i)
        assert q.qsize() == 3
        return [await q.get() for _ in range(3)]

    assert asyncio.run(scenario()) == [0, 1, 2]
