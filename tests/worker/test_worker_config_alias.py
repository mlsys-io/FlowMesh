import hmac
import uuid
from hashlib import sha256
from pathlib import Path

import pytest

from worker.config import WorkerConfig


def _external_token(name: str) -> str:
    return f"{name}.{hmac.new(b'secret', name.encode(), sha256).hexdigest()}"


@pytest.fixture(autouse=True)
def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "localhost:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.delenv("WORKER_ALIAS", raising=False)


def test_worker_alias_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_TOKEN", _external_token("from-token"))
    monkeypatch.setenv("WORKER_ALIAS", "from-env")
    assert WorkerConfig.from_env().alias == "from-env"


def test_external_token_supplies_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_TOKEN", _external_token("gpu-box.lab"))
    assert WorkerConfig.from_env().alias == "gpu-box.lab"


def test_managed_token_without_alias_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_TOKEN", uuid.uuid4().hex)
    with pytest.raises(SystemExit, match="WORKER_ALIAS is required"):
        WorkerConfig.from_env()


def test_ssh_max_disk_becomes_a_session_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_TOKEN", _external_token("gpu-box.lab"))
    monkeypatch.setenv("SSH_MAX_DISK", "20Gi")
    limits = WorkerConfig.from_env().ssh_limits
    assert limits is not None and limits.max_disk_bytes == 20 * 1024**3


def test_invalid_ssh_max_disk_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_TOKEN", _external_token("gpu-box.lab"))
    monkeypatch.setenv("SSH_MAX_DISK", "lots")
    with pytest.raises(SystemExit, match="SSH_MAX_DISK"):
        WorkerConfig.from_env()
