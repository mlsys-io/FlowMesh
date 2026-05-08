import asyncio
import multiprocessing as mp
import threading
from collections.abc import Iterable
from multiprocessing.queues import Queue as MPQueue

type TaskIDType = str


class Sentinel:
    pass


class TaskSender[T, R]:
    _SENTINEL = Sentinel()

    def __init__(
        self,
        send_q: MPQueue[tuple[TaskIDType, T | Sentinel]],
        recv_q: MPQueue[tuple[TaskIDType, R | Sentinel]],
    ) -> None:
        self._send_q = send_q
        self._recv_q = recv_q
        self._result_pool: dict[TaskIDType, asyncio.Future[R]] = {}
        self._puller: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        if self._puller is not None and self._puller.is_alive():
            return
        self._loop = asyncio.get_running_loop()
        self._puller = threading.Thread(target=self._pulling_loop, daemon=True)
        self._puller.start()

    def stop(self) -> None:
        if self._puller is not None:
            self._recv_q.put(("", self._SENTINEL))
            self._puller.join(timeout=5)
            self._puller = None

    async def send(
        self, task_id: TaskIDType, task: T, timeout: float | None = None
    ) -> R:
        fut = asyncio.Future[R]()
        self._result_pool[task_id] = fut
        self._send_q.put((task_id, task))
        to_await = fut if timeout is None else asyncio.wait_for(fut, timeout)
        try:
            return await to_await
        finally:
            self._result_pool.pop(task_id, None)

    def _pulling_loop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        while True:
            task_id, result = self._recv_q.get()
            if isinstance(result, Sentinel):
                break
            fut = self._result_pool.pop(task_id, None)
            if fut is None or fut.done():
                continue
            loop.call_soon_threadsafe(fut.set_result, result)


class TaskReceiver[T, R]:
    _SENTINEL = Sentinel()

    def __init__(
        self,
        recv_q: MPQueue[tuple[TaskIDType, T | Sentinel]],
        send_q: MPQueue[tuple[TaskIDType, R | Sentinel]],
    ) -> None:
        self._recv_q = recv_q
        self._send_q = send_q

    def stop(self) -> None:
        self._recv_q.put(("", self._SENTINEL))

    def task_stream(self) -> Iterable[tuple[TaskIDType, T]]:
        while True:
            task_id, task = self._recv_q.get()
            if isinstance(task, Sentinel):
                break
            yield task_id, task

    def send_result(self, task_id: TaskIDType, result: R) -> None:
        self._send_q.put((task_id, result))


def create_task_channel[T, R]() -> tuple[TaskSender[T, R], TaskReceiver[T, R]]:
    send_q: MPQueue[tuple[TaskIDType, T | Sentinel]] = mp.Queue()
    recv_q: MPQueue[tuple[TaskIDType, R | Sentinel]] = mp.Queue()
    sender = TaskSender(send_q, recv_q)
    receiver = TaskReceiver(send_q, recv_q)
    return sender, receiver
