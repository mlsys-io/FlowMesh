"""Tests for VLLMServeExecutor."""

import collections
import importlib.metadata
import io
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.serve import ServeSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors.vllm_serve_executor import VLLMServeExecutor, _drain_to_log


def _ep(name: str, value: str) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(
        name=name, value=value, group="vllm.general_plugins"
    )


def _mock_eps(
    monkeypatch: pytest.MonkeyPatch, eps: list[importlib.metadata.EntryPoint]
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: eps if group == "vllm.general_plugins" else [],
    )


class TestVLLMServeExecutorInit:
    def test_supported_task_types(self) -> None:
        assert TaskType.SERVE in VLLMServeExecutor.supported_task_types

    def test_only_serve_task_type(self) -> None:
        assert VLLMServeExecutor.supported_task_types == frozenset({TaskType.SERVE})

    def test_is_available_false_without_vllm(self) -> None:
        cfg = make_worker_config()
        with patch.dict("sys.modules", {"vllm": None}):
            result = VLLMServeExecutor.is_available(cfg)
        assert result is False

    def test_is_available_true_with_vllm(self) -> None:
        cfg = make_worker_config()
        fake_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            result = VLLMServeExecutor.is_available(cfg)
        assert result is True


class TestPluginFiltering:
    _OMNI_EP_NAME = "vllm_omni_register_models"
    _OMNI_EP_VALUE = "vllm_omni.engine.arg_utils:register_omni_models_to_vllm"

    def test_excludes_omni_keeps_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_eps(
            monkeypatch,
            [
                _ep(self._OMNI_EP_NAME, self._OMNI_EP_VALUE),
                _ep("some_other_plugin", "some_pkg.plugins:register"),
            ],
        )
        result = VLLMServeExecutor._vllm_plugins_excluding_omni()
        assert result == "some_other_plugin"

    def test_empty_when_only_omni(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_eps(monkeypatch, [_ep(self._OMNI_EP_NAME, self._OMNI_EP_VALUE)])
        assert VLLMServeExecutor._vllm_plugins_excluding_omni() == ""

    def test_no_eps_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_eps(monkeypatch, [])
        assert VLLMServeExecutor._vllm_plugins_excluding_omni() == ""

    def test_filters_by_module_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_eps(
            monkeypatch,
            [
                _ep("renamed", "vllm_omni.something.else:fn"),
                _ep("keep", "another_pkg:fn"),
            ],
        )
        assert VLLMServeExecutor._vllm_plugins_excluding_omni() == "keep"


class TestServeSpecStrict:
    def test_minimal_spec(self) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE)
        assert spec.model is None
        assert spec.model_name is None
        assert spec.ttlSeconds is None
        assert spec.readinessTimeoutSeconds is None
        assert spec.accessMode is None
        assert spec.port is None

    def test_spec_with_all_fields(self) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="meta-llama/Llama-3-8B"),
                vllm={"tensor_parallel_size": 2},
            ),
            ttlSeconds=7200.0,
            readinessTimeoutSeconds=300.0,
            accessMode="forward",
            port=8001,
        )
        assert spec.model_name == "meta-llama/Llama-3-8B"
        assert spec.model is not None
        assert spec.model.vllm == {"tensor_parallel_size": 2}
        assert spec.ttlSeconds == 7200.0
        assert spec.readinessTimeoutSeconds == 300.0
        assert spec.accessMode == "forward"
        assert spec.port == 8001

    def test_parses_inference_style_model_block(self) -> None:
        """ServeSpecStrict accepts the same model block as InferenceSpecStrict."""
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(
                    type="huggingface",
                    identifier="Qwen/Qwen3-0.6B",
                    revision="main",
                    trust_remote_code=True,
                ),
                vllm={
                    "gpu_memory_utilization": 0.9,
                    "trust_remote_code": True,
                    "max_model_len": 4096,
                },
            ),
        )
        assert spec.model_name == "Qwen/Qwen3-0.6B"
        assert spec.model_revision == "main"
        assert spec.model_trust_remote_code is True
        assert spec.model is not None
        assert spec.model.vllm == {
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "max_model_len": 4096,
        }

    def test_readiness_timeout_accepted(self) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE, readinessTimeoutSeconds=900.0)
        assert spec.readinessTimeoutSeconds == 900.0

    def test_task_type_is_serve(self) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE)
        assert spec.taskType == TaskType.SERVE

    def test_invalid_access_mode(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, accessMode="invalid")  # type: ignore[arg-type]

    def test_ttl_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, ttlSeconds=0.0)
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, ttlSeconds=-1.0)

    def test_readiness_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, readinessTimeoutSeconds=0.0)

    def test_port_must_be_in_range(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, port=0)
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, port=65536)


class TestServeExecutorCmdBuilding:
    """Executor maps model.vllm + model_name + revision to vllm api_server flags."""

    def _make_executor(self) -> VLLMServeExecutor:
        return VLLMServeExecutor(make_worker_config(), make_worker_hardware())

    def _run_capture_cmd(self, spec: ServeSpecStrict, tmp_path: Path) -> list[str]:
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        captured: list[list[str]] = []

        def fake_popen(cmd: list[str], **_: object) -> MagicMock:
            captured.append(list(cmd))
            m = MagicMock()
            m.stdout = io.StringIO("")
            m.poll.return_value = 0
            m.returncode = 0
            m.pid = 12345
            return m

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch.object(ex, "_poll_health"),
            patch.object(ex, "_wait_for_serve"),
            patch.object(ex, "emit_update"),
            patch.object(ex, "_terminate_process_group"),
        ):
            ex.run(task, tmp_path)

        return captured[0]

    def test_model_name_and_revision_in_cmd(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="Qwen/Qwen3-0.6B", revision="main"),
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-0.6B"
        assert "--revision" in cmd
        assert cmd[cmd.index("--revision") + 1] == "main"

    def test_vllm_dict_keys_become_flags(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--tensor-parallel-size" in cmd
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
        assert "--gpu-memory-utilization" in cmd
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"

    def test_trust_remote_code_from_vllm_dict(self, tmp_path: Path) -> None:
        """trust_remote_code: true in model.vllm renders as a bare flag."""
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"trust_remote_code": True},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--trust-remote-code" in cmd
        assert cmd.count("--trust-remote-code") == 1

    def test_trust_remote_code_from_source_not_duplicated(self, tmp_path: Path) -> None:
        """trust_remote_code in source but not model.vllm still renders once."""
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m", trust_remote_code=True),
                vllm={},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--trust-remote-code" in cmd
        assert cmd.count("--trust-remote-code") == 1

    def test_revision_omitted_when_not_set(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--revision" not in cmd

    def test_missing_model_identifier_raises(self, tmp_path: Path) -> None:
        from worker.executors.base_executor import ExecutionError

        spec = ServeSpecStrict(taskType=TaskType.SERVE)
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        with pytest.raises(ExecutionError, match="model.source.identifier"):
            ex.run(task, tmp_path)


class TestDefaultReadinessTimeout:
    def test_default_is_at_least_600s(self) -> None:
        from worker.executors import vllm_serve_executor as mod

        assert mod._DEFAULT_READINESS_TIMEOUT_SEC >= 600.0


class TestVLLMServeExecutorCancelStop:
    def _make_executor(self) -> VLLMServeExecutor:
        cfg = make_worker_config()
        hw = make_worker_hardware()
        return VLLMServeExecutor(cfg, hw)

    def test_cancel_sets_event(self) -> None:
        ex = self._make_executor()
        assert not ex._cancel_event.is_set()
        ex.cancel("tsk-test")
        assert ex._cancel_event.is_set()

    def test_stop_sets_event(self) -> None:
        ex = self._make_executor()
        assert not ex._stop_event.is_set()
        ex.stop("tsk-test")
        assert ex._stop_event.is_set()

    def test_cancel_terminates_proc(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        ex._proc = mock_proc
        with patch.object(ex, "_terminate_process_group") as mock_term:
            ex.cancel("tsk-test")
            mock_term.assert_called_once_with(mock_proc)

    def test_stop_terminates_proc(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        ex._proc = mock_proc
        with patch.object(ex, "_terminate_process_group") as mock_term:
            ex.stop("tsk-test")
            mock_term.assert_called_once_with(mock_proc)

    def test_cancel_no_proc_is_safe(self) -> None:
        ex = self._make_executor()
        ex._proc = None
        ex.cancel("tsk-test")  # must not raise


class TestWaitForServe:
    def _make_executor(self) -> VLLMServeExecutor:
        cfg = make_worker_config()
        return VLLMServeExecutor(cfg, make_worker_hardware())

    def test_exits_on_cancel(self) -> None:
        from worker.executors.base_executor import TaskCancelledError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        ex._cancel_event.set()
        with pytest.raises(TaskCancelledError):
            ex._wait_for_serve(mock_proc, ttl_sec=60.0)

    def test_exits_on_stop(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        ex._stop_event.set()
        ex._wait_for_serve(mock_proc, ttl_sec=60.0)

    def test_raises_on_unexpected_proc_exit(self) -> None:
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        with pytest.raises(ExecutionError):
            ex._wait_for_serve(mock_proc, ttl_sec=60.0)

    def test_exits_when_ttl_expires(self) -> None:
        from worker.executors import vllm_serve_executor as mod

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        original = mod._POLL_INTERVAL_SEC
        mod._POLL_INTERVAL_SEC = 0.01
        try:
            start = time.time()
            ex._wait_for_serve(mock_proc, ttl_sec=0.02)
            elapsed = time.time() - start
        finally:
            mod._POLL_INTERVAL_SEC = original
        assert elapsed < 2.0


class TestPollHealth:
    def _make_executor(self) -> VLLMServeExecutor:
        return VLLMServeExecutor(make_worker_config(), make_worker_hardware())

    def _empty_tail(self) -> "collections.deque[str]":
        return collections.deque(maxlen=200)

    def test_timeout_error_includes_subprocess_output(self) -> None:
        """Timeout error message contains the last lines captured from vLLM."""
        from worker.executors import vllm_serve_executor as mod
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        tail: collections.deque[str] = collections.deque(
            ["Loading weights...", "CUDA graph capture OOM"], maxlen=200
        )

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch("requests.get", side_effect=requests.ConnectionError()):
                with pytest.raises(ExecutionError) as exc_info:
                    ex._poll_health(
                        mock_proc, 8000, "tsk-test", timeout_sec=0.01, tail=tail
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig

        msg = str(exc_info.value)
        assert "Loading weights..." in msg
        assert "CUDA graph capture OOM" in msg

    def test_early_exit_error_includes_subprocess_output(self) -> None:
        """Proc-died error message contains the last captured lines."""
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        tail: collections.deque[str] = collections.deque(
            ["CUDA error: device-side assert triggered"], maxlen=200
        )

        with patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(ExecutionError) as exc_info:
                ex._poll_health(
                    mock_proc, 8000, "tsk-test", timeout_sec=30.0, tail=tail
                )

        assert "CUDA error: device-side assert triggered" in str(exc_info.value)

    def test_timeout_message_reflects_timeout_sec_argument(self) -> None:
        """Error message reports the actual timeout used, not a hardcoded value."""
        from worker.executors import vllm_serve_executor as mod
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch("requests.get", side_effect=requests.ConnectionError()):
                with pytest.raises(ExecutionError, match=r"within 42s"):
                    ex._poll_health(
                        mock_proc, 8000, "tsk-x", timeout_sec=42.0, tail=tail
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig

    def test_returns_when_health_200(self) -> None:
        """Returns without raising when /health responds 200."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            ex._poll_health(mock_proc, 8000, "tsk-ok", timeout_sec=30.0, tail=tail)

    def test_cancel_during_poll_raises(self) -> None:
        from worker.executors.base_executor import TaskCancelledError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()
        ex._cancel_event.set()

        with patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(TaskCancelledError):
                ex._poll_health(
                    mock_proc, 8000, "tsk-cancel", timeout_sec=60.0, tail=tail
                )

    def test_stop_during_poll_raises(self) -> None:
        from worker.executors.base_executor import TaskCancelledError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()
        ex._stop_event.set()

        with patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(TaskCancelledError):
                ex._poll_health(
                    mock_proc, 8000, "tsk-stop", timeout_sec=60.0, tail=tail
                )

    def test_empty_tail_no_snippet_in_message(self) -> None:
        """When subprocess produced no output, the error omits the snippet block."""
        from worker.executors import vllm_serve_executor as mod
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch("requests.get", side_effect=requests.ConnectionError()):
                with pytest.raises(ExecutionError) as exc_info:
                    ex._poll_health(
                        mock_proc, 8000, "tsk-empty", timeout_sec=0.01, tail=tail
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig

        assert "last vLLM output" not in str(exc_info.value)


class TestDrainToLog:
    def test_streams_lines_to_logger_and_tail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("Loading model...\nReady.\n")
        eof_event = threading.Event()

        with caplog.at_level(
            logging.INFO, logger="worker.executors.vllm_serve_executor"
        ):
            _drain_to_log(mock_proc, tail, eof_event)

        assert list(tail) == ["Loading model...", "Ready."]
        messages = [r.message for r in caplog.records]
        assert any("Loading model..." in m for m in messages)
        assert any("Ready." in m for m in messages)
        assert eof_event.is_set()

    def test_strips_trailing_newline(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("line with newline\n")

        _drain_to_log(mock_proc, tail, threading.Event())

        assert list(tail) == ["line with newline"]

    def test_respects_deque_maxlen(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=3)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("a\nb\nc\nd\ne\n")

        _drain_to_log(mock_proc, tail, threading.Event())

        assert list(tail) == ["c", "d", "e"]

    def test_empty_output_leaves_tail_empty(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("")

        _drain_to_log(mock_proc, tail, threading.Event())

        assert list(tail) == []

    def test_sets_eof_event_on_completion(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("some line\n")
        eof_event = threading.Event()

        _drain_to_log(mock_proc, tail, eof_event)

        assert eof_event.is_set()

    def test_sets_eof_event_on_empty_output(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("")
        eof_event = threading.Event()

        _drain_to_log(mock_proc, tail, eof_event)

        assert eof_event.is_set()


class TestPollHealthEofFastFail:
    """Regression: _poll_health must raise promptly on stdout-pipe EOF.

    vLLM is multiprocess. When the APIServer child crashes, the top-level
    Popen may stay alive (proc.poll() returns None), so the old "process
    exited" check missed it and the task hung for the full readiness
    timeout. The drain thread signals EOF via an event; _poll_health detects
    it and fails immediately instead.
    """

    def _make_executor(self) -> VLLMServeExecutor:
        return VLLMServeExecutor(make_worker_config(), make_worker_hardware())

    def _empty_tail(self) -> "collections.deque[str]":
        return collections.deque(maxlen=200)

    def test_fails_promptly_when_pipe_eof(self) -> None:
        """Raises ExecutionError well before timeout when eof_event is set."""
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # top-level process still "alive"
        mock_proc.returncode = 1

        tail: collections.deque[str] = collections.deque(
            ["AttributeError: 'NoneType' object has no attribute 'serve'"],
            maxlen=200,
        )
        eof_event = threading.Event()
        eof_event.set()  # simulate pipe EOF arriving before readiness timeout

        start = time.time()
        with patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(ExecutionError):
                ex._poll_health(
                    mock_proc,
                    8000,
                    "tsk-eof",
                    timeout_sec=600.0,
                    tail=tail,
                    eof_event=eof_event,
                )
        elapsed = time.time() - start

        assert elapsed < 5.0  # must not wait anywhere near 600s

    def test_error_includes_captured_output_on_eof(self) -> None:
        """ExecutionError raised on EOF carries the tail snippet."""
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = 1

        tail: collections.deque[str] = collections.deque(
            ["Starting APIServer...", "AttributeError: bad attribute"], maxlen=200
        )
        eof_event = threading.Event()
        eof_event.set()

        with patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(ExecutionError) as exc_info:
                ex._poll_health(
                    mock_proc,
                    8000,
                    "tsk-eof-output",
                    timeout_sec=600.0,
                    tail=tail,
                    eof_event=eof_event,
                )

        msg = str(exc_info.value)
        assert "Starting APIServer..." in msg
        assert "AttributeError: bad attribute" in msg

    def test_no_false_trigger_when_eof_not_set(self) -> None:
        """Health poll reaches normal timeout when eof_event is not set."""
        from worker.executors import vllm_serve_executor as mod
        from worker.executors.base_executor import ExecutionError

        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        eof_event = threading.Event()  # NOT set

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch("requests.get", side_effect=requests.ConnectionError()):
                with pytest.raises(ExecutionError, match=r"within 1s"):
                    ex._poll_health(
                        mock_proc,
                        8000,
                        "tsk-no-eof",
                        timeout_sec=1.0,
                        tail=self._empty_tail(),
                        eof_event=eof_event,
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig
