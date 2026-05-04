import json
from pathlib import Path

import typer
from flowmesh.client import FlowMesh, resolve_config
from flowmesh.config import DEFAULT_CONFIG_PATH, FlowMeshConfig
from flowmesh.exceptions import FlowMeshError

from ..core import logging
from ..core.typer import get_typer

app = get_typer()


def _redact_api_key(
    api_key: str | None, prefix: int = 4, suffix: int = 4
) -> str | None:
    if api_key is None:
        return None

    length = len(api_key)
    if length <= prefix + suffix:
        return "*" * length

    masked_middle = "*" * (length - prefix - suffix)
    return f"{api_key[:prefix]}{masked_middle}{api_key[-suffix:]}"


@app.command()
def init(
    url: str = typer.Argument(
        "http://localhost:8000",
        help="FlowMesh server API URL (e.g., http://localhost:8000)",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Optional API key for authentication"
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to save configuration file"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config file without confirmation",
        show_default=False,
    ),
) -> None:
    """Initialize configuration for FlowMesh CLI."""
    if config_path.exists():
        if force:
            logging.warning(f"Overwriting existing config at {config_path}")
        else:
            if not typer.confirm(
                f"Config file {config_path} already exists. "
                "Do you want to overwrite it?"
            ):
                raise typer.Exit()
    config = FlowMeshConfig(url, api_key)
    config.save(config_path)
    logging.success(f"Config saved to {config_path}")


@app.command()
def deinit(config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config")) -> None:
    """Delete saved configuration file."""
    if config_path.exists():
        config_path.unlink()
        logging.success(f"Deleted config file {config_path}")
    else:
        logging.warning(f"No config file found at {config_path}")


@app.command()
def config(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
    source: str = typer.Option(
        "auto", "--source", help="Configuration source: auto, file, or env"
    ),
    show_api_key: bool = typer.Option(
        False, "--show-api-key", help="Show API key in plain text"
    ),
) -> None:
    """Display current configuration."""
    source = source.strip().lower()

    match source:
        case "file":
            try:
                cfg = FlowMeshConfig.from_file(config_path)
            except Exception as exc:
                logging.error(f"Error loading config from file: {exc}")
                raise typer.Exit(code=1)
        case "env":
            try:
                cfg = FlowMeshConfig.from_env()
            except Exception as exc:
                logging.error(f"Error loading config from environment: {exc}")
                raise typer.Exit(code=1)
        case "auto":
            try:
                cfg = resolve_config(
                    base_url=None, api_key=None, config_path=config_path
                )
            except Exception as exc:
                logging.error(f"Error resolving config: {exc}")
                raise typer.Exit(code=1)
        case _:
            logging.error("Invalid --source value. Expected one of: auto, file, env")
            raise typer.Exit(code=2)

    cfg.api_key = cfg.api_key if show_api_key else _redact_api_key(cfg.api_key)
    logging.log(json.dumps(cfg.to_mapping(), indent=2))


@app.command()
def info() -> None:
    """Query server health status and basic system information."""
    client = FlowMesh()
    try:
        resp = client.system.health()
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(resp.model_dump_json(indent=2))


@app.command()
def health() -> None:
    """Check if the FlowMesh server is reachable and healthy."""
    info()
