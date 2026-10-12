import threading


class WorkerDeathMarks:
    """Workers the watchdog has declared dead, shared with the scheduler.

    Marks live in memory, so a restarted server starts with none and declares a
    worker dead only after observing its heartbeat expire for the watchdog's grace.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._marks: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def is_dead(self, worker_id: str) -> bool:
        if not self._enabled or not worker_id:
            return False
        with self._lock:
            return worker_id in self._marks

    def mark(self, worker_id: str) -> None:
        if not self._enabled or not worker_id:
            return
        with self._lock:
            self._marks.add(worker_id)

    def clear(self, worker_id: str) -> None:
        if not self._enabled or not worker_id:
            return
        with self._lock:
            self._marks.discard(worker_id)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._marks)
