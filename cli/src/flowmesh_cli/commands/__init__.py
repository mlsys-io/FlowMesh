"""Command modules for the FlowMesh CLI."""

import typer

from .base import app as base_app
from .logs import app as logs_app
from .node import app as node_app
from .profile import app as profile_app
from .result import app as results_app
from .ssh import app as ssh_app
from .system import app as system_app
from .task import app as task_app
from .worker import app as worker_app
from .workflow import app as workflow_app


def register(app: typer.Typer) -> None:
    """Register command groups on the root app."""
    app.add_typer(base_app)
    app.add_typer(task_app, name="task")
    app.add_typer(results_app, name="result")
    app.add_typer(workflow_app, name="workflow")
    app.add_typer(system_app, name="system")
    app.add_typer(node_app, name="node")
    app.add_typer(worker_app, name="worker")
    app.add_typer(ssh_app, name="ssh")
    app.add_typer(logs_app, name="logs")
    app.add_typer(profile_app, name="profile")
