"""Worker lifecycle helpers for direct node operations."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from flowmesh.exceptions import FlowMeshError

from .docker import ensure_docker_available
from .node_client import NodeClient

_MAX_PARALLEL_REQUESTS = 16


def operate_workers(
    client: NodeClient,
    names: list[str],
    operation: str,
) -> list[str]:
    """Apply a start/stop/destroy operation to one or more workers."""
    if not names:
        return []
    if "all" in names:
        if len(names) != 1:
            raise FlowMeshError("Use either 'all' or worker names, not both.")
        names = client.worker_names()
        if not names:
            return []

    def _apply(name: str) -> str:
        match operation:
            case "start":
                client.start_worker(name)
            case "stop":
                client.stop_worker(name)
            case "destroy":
                client.destroy_worker(name)
            case _:
                raise FlowMeshError(f"Unsupported worker operation: {operation}")
        return name

    successes: list[str] = []
    errors: list[str] = []
    max_workers = min(len(names), _MAX_PARALLEL_REQUESTS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_apply, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                successes.append(future.result())
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    if errors:
        raise FlowMeshError("; ".join(errors))
    return successes


def create_workers(
    client: NodeClient,
    kind: str = "cpu",
    count: int = 1,
    targets: str = "all",
    config_paths: list[Path] | None = None,
    config_raw: list[str] | None = None,
    name_template: str | None = None,
    slug: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Create node workers from configs or built-in cpu/gpu presets.

    When ``kind`` is "gpu", if ``count`` is equal to 1, a single worker with the
    specified GPU targets will be created. If ``count`` is greater than 1, one worker
    will be created per GPU target, and the number of targets must match ``count``.

    For the built-in cpu/gpu presets, ``slug`` prefixes the generated worker
    aliases so names are scoped to a stack, and ``name_template`` overrides the
    naming entirely (placeholders ``{slug}``, ``{kind}``, ``{idx}``, ``{gpu}``).
    Both are ignored when ``config_paths`` / ``config_raw`` are provided, since
    those payloads carry their own aliases.
    """
    payloads = _payloads_for_worker_create(
        kind=kind,
        count=count,
        targets=targets,
        config_paths=config_paths,
        config_raw=config_raw,
        name_template=name_template,
        slug=slug,
    )
    if not payloads:
        return []

    created: list[tuple[str, dict[str, Any]]] = []
    errors: list[str] = []
    max_workers = min(len(payloads), _MAX_PARALLEL_REQUESTS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(client.create_worker, payload): label
            for payload, label in payloads
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                created.append((label, future.result()))
            except Exception as exc:
                errors.append(f"{label}: {exc}")
    if errors:
        raise FlowMeshError("; ".join(errors))
    return created


def select_worker_images(
    kinds: list[str],
    images: dict[str, str],
    builder_images: dict[str, str] | None = None,
    builder: bool = False,
) -> list[str]:
    """Resolve requested worker image kinds into concrete image refs."""
    normalized = [kind.strip().lower() for kind in kinds if kind.strip()]
    if not normalized:
        raise FlowMeshError("worker pull expects cpu|gpu|ssh-cpu|ssh-gpu|all")

    selected_images = builder_images if builder else images
    valid_source = builder_images if builder and builder_images is not None else images
    valid = set([*valid_source, "all"])
    invalid = [kind for kind in normalized if kind not in valid]
    if invalid:
        unique = ", ".join(sorted(set(invalid)))
        raise FlowMeshError(f"Invalid kind(s): {unique}")

    if "all" in normalized:
        return list((selected_images or {}).values())

    requested = set(normalized)
    if builder and builder_images is not None:
        invalid_builder = [kind for kind in requested if kind not in builder_images]
        if invalid_builder:
            unique = ", ".join(sorted(invalid_builder))
            raise FlowMeshError(f"No builder image for: {unique}")
        return [builder_images[kind] for kind in builder_images if kind in requested]

    return [images[kind] for kind in images if kind in requested]


def pull_images(images: list[str]) -> None:
    """Pull one or more worker images via docker."""
    ensure_docker_available()
    for image in images:
        result = subprocess.run(["docker", "pull", image], text=True, check=False)
        if result.returncode != 0:
            raise FlowMeshError(f"Failed to pull image: {image}")


def detect_gpu_targets(targets: str) -> list[str]:
    """Resolve requested GPU ids into a concrete target list."""
    if targets != "all":
        return [item.strip() for item in targets.split(",") if item.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [item.strip() for item in result.stdout.splitlines() if item.strip()]


_ALIAS_FIELDS = "{slug}, {kind}, {idx}, {gpu}"


class _StrictFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


def _worker_alias(
    kind: str,
    idx: int,
    slug: str | None,
    template: str | None,
    gpu: str | None = None,
) -> str:
    if template is not None:
        fields: dict[str, Any] = {"slug": slug or "", "kind": kind, "idx": idx}
        if gpu is not None:
            fields["gpu"] = gpu
        try:
            return template.format_map(_StrictFormatDict(fields))
        except (KeyError, IndexError, ValueError) as exc:
            raise FlowMeshError(
                f"invalid --name-template ({exc}); "
                f"available placeholders: {_ALIAS_FIELDS}"
            ) from exc
    base = f"{slug}_" if slug else ""
    if gpu is not None:
        return f"{base}worker_{kind}_{gpu}"
    return f"{base}worker_{kind}_{idx}"


def _payloads_for_worker_create(
    kind: str,
    count: int,
    targets: str,
    config_paths: list[Path] | None,
    config_raw: list[str] | None,
    name_template: str | None = None,
    slug: str | None = None,
) -> list[tuple[str, str]]:
    if count < 1:
        raise FlowMeshError("Worker count must be at least 1.")
    if config_paths is not None or config_raw is not None:
        payloads: list[tuple[str, str]] = []
        for config_path in config_paths or []:
            if not config_path.exists():
                raise FlowMeshError(f"Config not found: {config_path}")
            _extend_payloads(
                payloads, f"worker from {config_path.name}", config_path.read_text()
            )
        for idx, raw in enumerate(config_raw or []):
            _extend_payloads(payloads, f"worker from raw#{idx}", raw)
        return payloads

    specs: list[tuple[str, dict[str, Any], str]] = []

    if kind == "cpu":
        for idx in range(count):
            alias = _worker_alias("cpu", idx, slug, name_template)
            specs.append(
                (
                    alias,
                    {"worker_type": "cpu", "worker_alias": alias},
                    "CPU worker",
                )
            )
    elif kind == "gpu":
        raw_gpu_ids = detect_gpu_targets(targets)
        gpu_ids: list[int] = []
        for raw_gpu_id in raw_gpu_ids:
            if not raw_gpu_id.isdigit():
                raise FlowMeshError(f"Invalid GPU id '{raw_gpu_id}'")
            gpu_ids.append(int(raw_gpu_id))

        if count > 1:
            if count != len(gpu_ids):
                raise FlowMeshError(
                    f"GPU worker count {count} does not match "
                    f"detected GPU targets: {gpu_ids}. "
                    f"Consider setting count={len(gpu_ids)} or specifying exactly "
                    f"{count} GPU targets."
                )
            for idx, gpu_id in enumerate(gpu_ids):
                alias = _worker_alias("gpu", idx, slug, name_template, gpu=str(gpu_id))
                specs.append(
                    (
                        alias,
                        {
                            "worker_type": "gpu",
                            "cuda_devices": [gpu_id],
                            "worker_alias": alias,
                        },
                        f"GPU worker for GPU {gpu_id}",
                    )
                )
        else:
            worker_suffix = "all" if targets == "all" else "_".join(raw_gpu_ids)
            alias = _worker_alias("gpu", 0, slug, name_template, gpu=worker_suffix)
            specs.append(
                (
                    alias,
                    {
                        "worker_type": "gpu",
                        "cuda_devices": gpu_ids,
                        "worker_alias": alias,
                    },
                    f"GPU worker for GPUs {', '.join(raw_gpu_ids)}",
                )
            )
    else:
        raise FlowMeshError("worker up expects kind cpu|gpu or use --config")

    aliases = [alias for alias, _, _ in specs]
    if len(set(aliases)) != len(aliases):
        raise FlowMeshError(
            "--name-template produced duplicate worker names; "
            "include {idx} or {gpu} to disambiguate"
        )

    return [
        (
            json.dumps(
                {
                    "provider": "docker",
                    "init_on_start": True,
                    "worker_config": worker_config,
                }
            ),
            label,
        )
        for _, worker_config, label in specs
    ]


def _extend_payloads(
    payloads: list[tuple[str, str]], label_prefix: str, payload_text: str
) -> None:
    try:
        payload_obj = yaml.safe_load(payload_text)
    except Exception:
        payload_obj = None

    if isinstance(payload_obj, list):
        for idx, item in enumerate(payload_obj):
            payloads.append((json.dumps(item), f"{label_prefix}#{idx}"))
        return
    if isinstance(payload_obj, dict):
        payloads.append((json.dumps(payload_obj), label_prefix))
        return
    payloads.append((payload_text, label_prefix))
