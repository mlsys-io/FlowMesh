"""FlowMesh CLI entrypoint."""

from importlib import import_module
from importlib.util import find_spec

import typer

from ._version import resolve_cli_version
from .core.typer import get_typer


def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the FlowMesh CLI version and exit.",
    ),
) -> None:
    if version:
        typer.echo(f"flowmesh {resolve_cli_version()}")
        raise typer.Exit()


def _register_optional(
    module_name: str, app: typer.Typer, register_func: str = "register"
) -> bool:
    """Import and register a command module if it is installed.

    Extras gate availability by deciding which modules are packaged. If a module
    is missing, we silently skip it; unexpected import errors still bubble up.
    """
    package = __package__ if module_name.startswith(".") else None
    spec = find_spec(module_name, package=package)
    if spec is None:
        return False
    module = import_module(module_name, package=package)
    register = getattr(module, register_func, None)
    if register is None:
        return False
    register(app)
    return True


def build_cli_app() -> typer.Typer:
    """Construct the CLI app by attaching available command groups."""
    app = get_typer(
        help="FlowMesh command line interface.",
        invoke_without_command=True,
        no_args_is_help=True,
    )
    app.callback()(_root)

    _register_optional(".commands", app)
    _register_optional("flowmesh_cli_stack", app)

    return app


def main() -> None:
    build_cli_app()()


if __name__ == "__main__":
    main()
