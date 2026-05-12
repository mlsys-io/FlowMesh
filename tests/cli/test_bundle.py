from pathlib import Path

from flowmesh_cli_stack import bundle as bundle_module
from flowmesh_cli_stack.bundle import (
    _TLS_REDIS_SUBDIR,
    _TLS_SERVER_SUBDIR,
    _WORKER_CONFIG_FILE,
    _copy_server_assets,
    _scaffold_server_assets,
    bundle_init,
)


def test_scaffold_creates_full_layout(tmp_path: Path) -> None:
    _scaffold_server_assets(tmp_path, include_tls=True)
    assert (tmp_path / _TLS_SERVER_SUBDIR).is_dir()
    assert (tmp_path / _TLS_REDIS_SUBDIR).is_dir()
    worker_config = tmp_path / _WORKER_CONFIG_FILE
    assert worker_config.is_file()
    assert worker_config.read_bytes() == b""


def test_scaffold_skips_tls_when_disabled(tmp_path: Path) -> None:
    _scaffold_server_assets(tmp_path, include_tls=False)
    assert not (tmp_path / _TLS_SERVER_SUBDIR).exists()
    assert not (tmp_path / _TLS_REDIS_SUBDIR).exists()
    assert (tmp_path / _WORKER_CONFIG_FILE).is_file()


def test_scaffold_preserves_existing_worker_config(tmp_path: Path) -> None:
    worker_config = tmp_path / _WORKER_CONFIG_FILE
    worker_config.parent.mkdir(parents=True)
    worker_config.write_text("user_data: true\n")
    _scaffold_server_assets(tmp_path, include_tls=False)
    assert worker_config.read_text() == "user_data: true\n"


def test_scaffold_preserves_existing_tls_dirs(tmp_path: Path) -> None:
    pre_existing = tmp_path / _TLS_SERVER_SUBDIR / "server.pem"
    pre_existing.parent.mkdir(parents=True)
    pre_existing.write_text("CERT")
    _scaffold_server_assets(tmp_path, include_tls=True)
    assert pre_existing.read_text() == "CERT"


def test_bundle_init_writes_env_via_stack_init(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=False,
        env_file=Path(".env"),
        force=False,
    )
    assert (tmp_path / _WORKER_CONFIG_FILE).is_file()
    assert (tmp_path / _TLS_SERVER_SUBDIR).is_dir()
    env_text = (tmp_path / ".env").read_text()
    assert "FLOWMESH_VERSION" in env_text


def test_bundle_init_env_defaults_point_at_scaffolded_paths(
    tmp_path: Path, monkeypatch
) -> None:
    # The whole point of init is that `stack up` works out of the box
    # against the scaffolded layout. Pin the env-vs-layout contract.
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=False,
        env_file=Path(".env"),
        force=False,
    )
    env_text = (tmp_path / ".env").read_text()
    assert f"SERVER_TLS_DIR=./{_TLS_SERVER_SUBDIR}" in env_text
    assert f"REDIS_TLS_DIR=./{_TLS_REDIS_SUBDIR}" in env_text
    assert f"SERVER_WORKER_CONFIG=./{_WORKER_CONFIG_FILE}" in env_text


def test_bundle_init_force_overwrites_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    env.write_text("stale=1\n")
    bundle_init(
        dest=Path("."),
        no_tls=True,
        env_file=Path(".env"),
        force=True,
    )
    assert "stale=1" not in env.read_text()


def test_bundle_init_env_file_in_missing_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=True,
        env_file=Path("config/.env"),
        force=False,
    )
    assert (tmp_path / "config" / ".env").is_file()


def test_bundle_init_dest_subdirectory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "deploy"
    bundle_init(
        dest=target,
        no_tls=False,
        env_file=Path(".env"),
        force=False,
    )
    assert (target / _WORKER_CONFIG_FILE).is_file()
    assert (target / ".env").is_file()
    assert not (tmp_path / ".env").exists()


def test_copy_server_assets_stages_worker_config_under_configs(
    tmp_path: Path, monkeypatch
) -> None:
    # `_copy_server_assets` runs against a fresh temp staging dir; the
    # worker_config destination is nested under configs/ which doesn't
    # pre-exist, so the copy has to create the parent itself.
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "worker_config.yaml").write_text("scheduler: round_robin\n")
    monkeypatch.chdir(repo)
    staging = tmp_path / "stage"
    staging.mkdir()
    _copy_server_assets(staging, include_tls=False)
    staged = staging / _WORKER_CONFIG_FILE
    assert staged.is_file()
    assert staged.read_text() == "scheduler: round_robin\n"


def test_module_constants_match_env_defaults() -> None:
    # The scaffolded layout has to match the paths the shipped .env.example
    # and compose.yml defaults reference, otherwise `stack up` would point
    # somewhere other than where `bundle init` / `bundle export` wrote.
    assert bundle_module._TLS_SERVER_SUBDIR == "secrets/tls/server"
    assert bundle_module._TLS_REDIS_SUBDIR == "secrets/tls/redis"
    assert bundle_module._WORKER_CONFIG_FILE == "configs/worker_config.yaml"
