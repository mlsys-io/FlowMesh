"""WorkerSupervisor keeps the parent-process node_id current after re-register.

The child re-puts its new node id on the handshake queue; the parent watcher
picks it up, updates the cached id, and notifies listeners (which refresh
``app.state.node_id`` and the EventMonitor's own-node in production).
"""

import logging
import queue
from threading import Event, Thread

from server.supervisor.supervisor import WorkerSupervisor

_LOGGER = logging.getLogger("test.node_id_watch")


def _build_supervisor() -> WorkerSupervisor:
    sup = WorkerSupervisor.__new__(WorkerSupervisor)
    sup._logger = _LOGGER
    sup._node_id = "nde-1"
    sup._node_id_queue = queue.Queue()  # type: ignore[assignment]
    sup._node_id_listeners = []
    sup._node_id_stop = False
    sup._node_id_watcher = None
    return sup


def test_watcher_updates_id_and_notifies_listeners() -> None:
    sup = _build_supervisor()
    seen: list[str] = []
    notified = Event()

    def _listener(new_id: str) -> None:
        seen.append(new_id)
        notified.set()

    sup.add_node_id_listener(_listener)

    watcher = Thread(target=sup._watch_node_id, daemon=True)
    watcher.start()
    try:
        sup._node_id_queue.put("nde-2")
        assert notified.wait(2.0)
    finally:
        sup._node_id_stop = True
        watcher.join(timeout=2.0)

    assert seen == ["nde-2"]
    assert sup.node_id == "nde-2"


def test_watcher_survives_a_failing_listener() -> None:
    sup = _build_supervisor()
    good_seen: list[str] = []
    second_ran = Event()

    def _boom(_new_id: str) -> None:
        raise RuntimeError("listener failed")

    def _good(new_id: str) -> None:
        good_seen.append(new_id)
        second_ran.set()

    sup.add_node_id_listener(_boom)
    sup.add_node_id_listener(_good)

    watcher = Thread(target=sup._watch_node_id, daemon=True)
    watcher.start()
    try:
        sup._node_id_queue.put("nde-2")
        assert second_ran.wait(2.0)
    finally:
        sup._node_id_stop = True
        watcher.join(timeout=2.0)

    # a raising listener must not stop later listeners or the watcher
    assert good_seen == ["nde-2"]
    assert sup.node_id == "nde-2"
