"""Stack management package for FlowMesh CLI."""

import typer

from .bundle import app as bundle_app
from .image import app as image_app
from .stack import app as stack_app
from .worker import app as worker_app


def register(root: typer.Typer) -> None:
    root.add_typer(stack_app, name="stack")
    stack_app.add_typer(worker_app, name="worker")
    stack_app.add_typer(bundle_app, name="bundle")
    stack_app.add_typer(image_app, name="image")
