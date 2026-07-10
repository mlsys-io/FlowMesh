"""Local Docker image management commands."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from flowmesh_cli.core import logging
from flowmesh_cli.core.typer import get_typer
from flowmesh_stack.docker import (
    DockerError,
    ManagedImage,
    RemovalResult,
    container_image_refs,
    ensure_docker_available,
    list_managed_images,
    remove_images,
)
from flowmesh_stack.env import load_env
from flowmesh_stack.image_prune import PrunePlan, parse_duration, select_prune_targets
from flowmesh_stack.images import BUILD_TARGETS, get_image_ref

from .utils import DEFAULT_ENV_FILE, STACK_PATH_KEYS

app = get_typer(help="Manage FlowMesh Docker images on the local daemon.")

_BUILDER_TARGET = "flowmesh_worker_gpu_builder"
_DEFAULT_REGISTRY = "ghcr.io/mlsys-io"


def _prepare(env_file: Path) -> str:
    """Guard Docker, load the env file, and return the configured registry."""
    try:
        ensure_docker_available()
    except DockerError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    load_env(env_file, base_dir=Path.cwd(), path_keys=STACK_PATH_KEYS)
    return os.getenv("FLOWMESH_REGISTRY", _DEFAULT_REGISTRY)


def _resolve_targets(targets: list[str] | None, include_builder: bool) -> list[str]:
    if targets:
        invalid = [t for t in targets if t not in BUILD_TARGETS]
        if invalid:
            logging.error(
                f"Invalid target(s): {', '.join(invalid)}. "
                f"Known: {', '.join(BUILD_TARGETS)}."
            )
            raise typer.Exit(code=1)
        resolved = list(dict.fromkeys(targets))
    else:
        resolved = [t for t in BUILD_TARGETS if t != _BUILDER_TARGET]
    if include_builder and _BUILDER_TARGET not in resolved:
        resolved.append(_BUILDER_TARGET)
    return resolved


def _human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def _image_row(image: ManagedImage) -> dict[str, Any]:
    return {
        "repo": image.repo,
        "tag": image.tag,
        "target": image.target,
        "version": image.version,
        "image_id": image.image_id,
        "size_bytes": image.size_bytes,
        "created": image.created.isoformat(),
        "dangling": image.dangling,
        "in_use": image.in_use,
    }


def _short(image: ManagedImage) -> dict[str, Any]:
    return {
        "tag": image.tag,
        "target": image.target,
        "version": image.version,
        "image_id": image.image_id,
    }


def _render_table(images: list[ManagedImage]) -> None:
    headers = ("REPOSITORY", "TAG", "VERSION", "SIZE", "CREATED", "IN-USE")
    rows: list[tuple[str, ...]] = []
    for image in images:
        tag = image.tag.split(":", 1)[-1] if image.tag else "<none>"
        rows.append(
            (
                image.repo,
                tag,
                image.version or "-",
                _human_size(image.size_bytes),
                image.created.date().isoformat(),
                "yes" if image.in_use else "no",
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    logging.log("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    for row in rows:
        logging.log("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


@app.command("list")
@app.command("ls")
def list_images(
    targets: list[str] | None = typer.Option(
        None, "--target", "-t", help="Restrict to specific build targets."
    ),
    version: str | None = typer.Option(
        None, "--version", help="Restrict to a single version tag."
    ),
    in_use: bool = typer.Option(
        False, "--in-use", help="Only images backing a container (running or stopped)."
    ),
    dangling: bool = typer.Option(
        False, "--dangling", help="Only untagged FlowMesh layers."
    ),
    include_builder: bool = typer.Option(
        False, "--include-builder", help="Include the GPU builder image."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit rows as JSON."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for docker"
    ),
) -> None:
    """List FlowMesh Docker images on the local daemon."""
    registry = _prepare(env_file)
    resolved = _resolve_targets(targets, include_builder)
    in_use_ids = container_image_refs()
    images = list_managed_images(
        registry, include_dangling=dangling, in_use_ids=in_use_ids
    )

    resolved_set = set(resolved)
    explicit = bool(targets)
    selected: list[ManagedImage] = []
    for image in images:
        if dangling and not image.dangling:
            continue
        if image.dangling:
            selected.append(image)
        elif image.target in resolved_set:
            selected.append(image)
        elif image.target is None and not explicit:
            selected.append(image)

    if version is not None:
        selected = [i for i in selected if i.version == version]
    if in_use:
        selected = [i for i in selected if i.in_use]

    selected.sort(key=lambda i: (i.repo, i.version or "", i.tag or ""))

    if as_json:
        logging.log(json.dumps([_image_row(i) for i in selected], indent=2))
        return
    if not selected:
        logging.info("No matching FlowMesh images found.")
        return
    _render_table(selected)


@app.command("prune")
def prune(
    keep_last: int | None = typer.Option(
        None, "--keep-last", help="Protect the N newest versions per target."
    ),
    keep_active: bool = typer.Option(
        False,
        "--keep-active",
        help="Protect versions backing a container (running or stopped).",
    ),
    keep: list[str] | None = typer.Option(
        None, "--keep", help="Protect explicit version(s)."
    ),
    older_than: str | None = typer.Option(
        None, "--older-than", help="Only images created before e.g. 30d, 12h."
    ),
    dangling: bool = typer.Option(
        False, "--dangling", help="Also remove dangling FlowMesh layers."
    ),
    targets: list[str] | None = typer.Option(
        None, "--target", "-t", help="Restrict to specific build targets."
    ),
    include_builder: bool = typer.Option(
        False, "--include-builder", help="Include the GPU builder image."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the deletion plan without removing anything."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the plan/result as JSON."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for docker"
    ),
) -> None:
    """Remove stale FlowMesh images by policy.

    Requires at least one selection policy (--keep-last, --older-than, or
    --dangling); --keep and --keep-active only protect images and never select.
    """
    if keep_last is None and older_than is None and not dangling:
        logging.error(
            "Refusing to prune without a selection policy. "
            "Pass at least one of --keep-last, --older-than, or --dangling."
        )
        raise typer.Exit(code=1)
    if as_json and not yes and not dry_run:
        logging.error(
            "Refusing to delete in --json mode without confirmation. "
            "Add --yes to proceed non-interactively, or --dry-run to preview."
        )
        raise typer.Exit(code=1)

    registry = _prepare(env_file)
    resolved = set(_resolve_targets(targets, include_builder))
    duration = None
    if older_than is not None:
        try:
            duration = parse_duration(older_than)
        except ValueError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1)

    in_use_ids = container_image_refs()
    images = list_managed_images(
        registry, include_dangling=dangling, in_use_ids=in_use_ids
    )
    scoped = [i for i in images if i.dangling or i.target in resolved]

    plan = select_prune_targets(
        scoped,
        keep_last=keep_last,
        keep_versions=set(keep or []),
        keep_active=keep_active,
        older_than=duration,
        include_dangling=dangling,
        now=datetime.now(UTC),
    )

    if dry_run:
        _report_plan(plan, as_json=as_json)
        return
    if not plan.deleted:
        if as_json:
            logging.log(json.dumps(_result_payload(False, [], []), indent=2))
        else:
            logging.info("Nothing to prune.")
        return

    if not as_json and not yes:
        logging.info(f"About to remove {len(plan.deleted)} image(s):")
        for image in plan.deleted:
            logging.log(f"  {image.removal_ref}")
        if not typer.confirm("Proceed?", default=False):
            logging.info("Aborted.")
            return

    results = remove_images([i.removal_ref for i in plan.deleted])
    _report_results(results, as_json=as_json)


@app.command("rm")
def rm(
    versions: list[str] = typer.Argument(..., help="Version(s) to remove."),
    targets: list[str] | None = typer.Option(
        None, "--target", "-t", help="Restrict to specific build targets."
    ),
    include_builder: bool = typer.Option(
        False, "--include-builder", help="Include the GPU builder image."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Remove even if the image is referenced."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed without removing it."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", help="Env file for docker"
    ),
) -> None:
    """Remove FlowMesh images for one or more explicit versions."""
    if as_json and not yes and not dry_run:
        logging.error(
            "Refusing to delete in --json mode without confirmation. "
            "Add --yes to proceed non-interactively, or --dry-run to preview."
        )
        raise typer.Exit(code=1)
    registry = _prepare(env_file)
    resolved = _resolve_targets(targets, include_builder)

    present = {
        image.tag for image in list_managed_images(registry) if image.tag is not None
    }
    refs: list[str] = []
    for version in dict.fromkeys(versions):
        for target in resolved:
            ref = get_image_ref(registry=registry, version=version, target=target)
            if ref in present:
                refs.append(ref)
            else:
                logging.warning(f"Image not found: {ref}")

    if not refs:
        logging.info("No matching images to remove.")
        return

    if dry_run:
        if as_json:
            logging.log(
                json.dumps(
                    {"dry_run": True, "deleted": [{"tag": r} for r in refs]}, indent=2
                )
            )
        else:
            logging.info("Images to be removed:")
            for ref in refs:
                logging.log(f"  {ref}")
        return

    if not as_json and not yes:
        logging.info(f"About to remove {len(refs)} image(s):")
        for ref in refs:
            logging.log(f"  {ref}")
        if not typer.confirm("Proceed?", default=False):
            logging.info("Aborted.")
            return

    results = remove_images(refs, force=force)
    _report_results(results, as_json=as_json)


def _result_payload(
    dry_run: bool,
    deleted: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"dry_run": dry_run, "deleted": deleted, "failed": failed}


def _report_plan(plan: PrunePlan, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "dry_run": True,
            "deleted": [_short(i) for i in plan.deleted],
            "protected": [
                {"tag": i.tag, "reason": reason} for i, reason in plan.protected
            ],
            "failed": [],
        }
        logging.log(json.dumps(payload, indent=2))
        return
    if not plan.deleted:
        logging.info("Nothing to prune.")
        return
    logging.info(f"Would remove {len(plan.deleted)} image(s):")
    for image in plan.deleted:
        logging.log(f"  {image.removal_ref}")


def _report_results(results: list[RemovalResult], *, as_json: bool) -> None:
    removed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    if as_json:
        payload = _result_payload(
            False,
            [{"tag": r.ref} for r in removed],
            [{"tag": r.ref, "error": r.error} for r in failed],
        )
        logging.log(json.dumps(payload, indent=2))
    else:
        for result in removed:
            logging.success(f"Removed image: {result.ref}")
        for result in failed:
            logging.error(f"Failed to remove image: {result.ref}")
            if result.error:
                logging.log(result.error, err=True)
    if failed:
        raise typer.Exit(code=1)
