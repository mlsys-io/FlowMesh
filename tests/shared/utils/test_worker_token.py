import hmac
from hashlib import sha256

import pytest

from shared.utils.worker_token import external_token_name

DIGEST = "a" * 64


@pytest.mark.parametrize("name", ["fm-worker-0", "host.example.com", "w"])
def test_reads_name_of_well_formed_token(name: str) -> None:
    digest = hmac.new(b"secret", name.encode(), sha256).hexdigest()
    assert external_token_name(f"{name}.{digest}") == name


@pytest.mark.parametrize(
    "token",
    [
        "",
        "0123456789abcdef0123456789abcdef",
        DIGEST,
        f".{DIGEST}",
        f"w.{DIGEST[:-1]}",
        f"w.{DIGEST}0",
        f"w.{'g' * 64}",
        f"w.{'A' * 64}",
        "w.",
    ],
)
def test_rejects_malformed_tokens(token: str) -> None:
    assert external_token_name(token) is None
