from pathlib import Path

from flowmesh.models.nodes import NodeRole
from flowmesh_cli_stack import bundle as bundle_module
from flowmesh_cli_stack.bundle import (
    _TLS_REDIS_SUBDIR,
    _TLS_SERVER_SUBDIR,
    _WORKER_CONFIG_FILE,
    _copy_server_assets,
    _scaffold_server_assets,
    bundle_init,
)
from flowmesh_cli_stack.env_schema import STACK_ENV_SCHEMA, role_overrides
from flowmesh_stack.env_schema import render_env_example, validate_env_values


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
        role=NodeRole.ROOT.value,
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
        role=NodeRole.ROOT.value,
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
        role=NodeRole.ROOT.value,
    )
    assert "stale=1" not in env.read_text()


def test_bundle_init_env_file_in_missing_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=True,
        env_file=Path("config/.env"),
        force=False,
        role=NodeRole.ROOT.value,
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
        role=NodeRole.ROOT.value,
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


def _seed_redis_tls_sources(repo: Path) -> None:
    tls = repo / _TLS_REDIS_SUBDIR
    tls.mkdir(parents=True)
    (tls / "redis-ca.pem").write_text("CA")
    (tls / "redis-server.pem").write_text("CERT")
    (tls / "redis-server.key").write_text("KEY")


def test_copy_server_assets_root_stages_redis_cert_and_key(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    _seed_redis_tls_sources(repo)
    monkeypatch.chdir(repo)
    staging = tmp_path / "stage"
    staging.mkdir()
    _copy_server_assets(staging, include_tls=True, role=NodeRole.ROOT)
    redis_dir = staging / _TLS_REDIS_SUBDIR
    assert (redis_dir / "redis-ca.pem").read_text() == "CA"
    assert (redis_dir / "redis-server.pem").read_text() == "CERT"
    assert (redis_dir / "redis-server.key").read_text() == "KEY"


def test_copy_server_assets_worker_skips_redis_cert_and_key(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    _seed_redis_tls_sources(repo)
    monkeypatch.chdir(repo)
    staging = tmp_path / "stage"
    staging.mkdir()
    _copy_server_assets(staging, include_tls=True, role=NodeRole.WORKER)
    redis_dir = staging / _TLS_REDIS_SUBDIR
    assert (redis_dir / "redis-ca.pem").read_text() == "CA"
    assert not (redis_dir / "redis-server.pem").exists()
    assert not (redis_dir / "redis-server.key").exists()


def test_bundle_init_next_steps_include_custom_env_file(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=False,
        env_file=Path("config/.env"),
        force=False,
        role=NodeRole.ROOT.value,
    )
    out = capsys.readouterr().out
    assert "flowmesh stack pull --env-file config/.env" in out
    assert "flowmesh stack up --env-file config/.env" in out


def test_bundle_init_next_steps_omit_env_flag_for_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=False,
        env_file=Path(".env"),
        force=False,
        role=NodeRole.ROOT.value,
    )
    out = capsys.readouterr().out
    assert "flowmesh stack pull\n" in out
    assert "flowmesh stack up" in out
    assert "--env-file" not in out


def test_bundle_init_worker_role_writes_worker_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=False,
        env_file=Path(".env"),
        force=False,
        role=NodeRole.WORKER.value,
    )
    env_text = (tmp_path / ".env").read_text()
    assert "NODE_ROLE=worker" in env_text
    assert "NODE_ROLE=root" not in env_text
    # Cert/key keys are blanked so the rendered file doesn't suggest
    # config the worker operator would have to maintain.
    for key in ("REDIS_TLS_CERT_FILE", "REDIS_TLS_KEY_FILE"):
        assert f"{key}=\n" in env_text, f"expected {key}= (blank) in worker env"


def test_install_script_passes_role_to_stack_init(tmp_path: Path) -> None:
    from flowmesh_cli_stack.bundle import _write_install_script

    _write_install_script(
        tmp_path,
        package_spec="flowmesh[cli]==0.1.0",
        include_wheels=False,
        role=NodeRole.WORKER,
    )
    script = (tmp_path / "install.sh").read_text()
    assert 'flowmesh stack init --env-file "$ENV_FILE" --role worker' in script


def test_install_script_omits_role_flag_for_root(tmp_path: Path) -> None:
    from flowmesh_cli_stack.bundle import _write_install_script

    _write_install_script(
        tmp_path,
        package_spec="flowmesh[cli]==0.1.0",
        include_wheels=False,
        role=NodeRole.ROOT,
    )
    script = (tmp_path / "install.sh").read_text()
    assert "--role" not in script
    assert 'flowmesh stack init --env-file "$ENV_FILE"' in script


def test_bundle_init_no_tls_drops_cert_guidance(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle_init(
        dest=Path("."),
        no_tls=True,
        env_file=Path(".env"),
        force=False,
        role=NodeRole.ROOT.value,
    )
    out = capsys.readouterr().out
    assert "drop TLS certs" not in out


def _parse_env_body(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def test_root_role_render_passes_schema_validation() -> None:
    body = render_env_example(STACK_ENV_SCHEMA, overrides=role_overrides(NodeRole.ROOT))
    errors, _ = validate_env_values(STACK_ENV_SCHEMA, _parse_env_body(body))
    assert errors == []


def test_worker_role_render_passes_schema_validation() -> None:
    # Pin the contract that a scaffolded worker .env is considered valid
    # by the schema's own validators — i.e. the blanked overrides don't
    # trip required/min_value/conditional checks.
    body = render_env_example(
        STACK_ENV_SCHEMA, overrides=role_overrides(NodeRole.WORKER)
    )
    errors, _ = validate_env_values(STACK_ENV_SCHEMA, _parse_env_body(body))
    assert errors == []


def test_module_constants_match_env_defaults() -> None:
    # The scaffolded layout has to match the paths the shipped .env.example
    # and compose.yml defaults reference, otherwise `stack up` would point
    # somewhere other than where `bundle init` / `bundle export` wrote.
    assert bundle_module._TLS_SERVER_SUBDIR == "secrets/tls/server"
    assert bundle_module._TLS_REDIS_SUBDIR == "secrets/tls/redis"
    assert bundle_module._WORKER_CONFIG_FILE == "configs/worker_config.yaml"
