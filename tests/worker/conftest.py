import pytest

from worker.utils import upload_retry


@pytest.fixture(autouse=True)
def _reset_unreachable_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_retry, "_unreachable_origins", set())
