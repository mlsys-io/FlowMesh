"""Wire-contract names + type enum for the OTel-shape span rows in ``spans.jsonl``.

Both the worker (producer, ``src/worker/executors/mixins/governance.py``) and
the server analyzer (consumer, ``src/server/governance/analyzer.py``) read
these. Pydantic parsing models for span rows live server-side at
``src/server/governance/spans.py``.
"""

from enum import StrEnum


class SpanType(StrEnum):
    """Producer-side type in ``attributes["flowmesh.type"]``."""

    COMPUTE = "compute"
    NETWORK = "network"
    MARKER = "marker"


# Wire-contract span names. The worker emits these (`_task_span` opens a
# ``"task"`` span; ``_record_output`` opens a ``"dump to storage"`` span);
# the analyzer reads them to identify the root span and the data-ready boundary.
TASK_SPAN_NAME = "task"
READY_SPAN_NAME = "dump to storage"
