from pathlib import Path

import typer
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError

from ..core import logging
from ..core.typer import get_typer

app = get_typer(help="Retrieve task execution results and output artifacts.")


@app.command("fetch")
def fetch(
    task_id: str = typer.Argument(..., help="Task identifier"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Directory to write result JSON"
    ),
) -> None:
    """Download task result JSON and optionally save it to a local file."""
    client = FlowMesh()
    if output:
        try:
            payload, target, downloaded = client.results.materialize(task_id, output)
        except FlowMeshError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1)
        if downloaded:
            logging.log(f"Downloaded images to {output / f'{task_id}-artifacts'}")
        logging.log(f"Wrote result to {target}")
        return

    try:
        result = client.results.retrieve(task_id)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(result.model_dump_json(indent=2))


@app.command("download")
def download_result_files(
    task_id: str = typer.Argument(..., help="Task identifier"),
    file_paths: list[str] = typer.Argument(..., help="List of file paths to download"),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Directory to save downloaded files"
    ),
) -> None:
    """Download specified result files for a task."""
    client = FlowMesh()
    try:
        for path in client.results.download_files(task_id, file_paths, output_dir):
            logging.log(f"Wrote file to {path}")
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
