"""Rendering of Kubernetes manifest assets from stack environment values.

Assets are ordinary multi-document YAML carrying a small set of extensions:

``${VAR}`` / ``${VAR:-default}``
    Compose-compatible substitution, so one ``.env`` drives both stack
    backends. A reference with neither a value nor a default is an error, so a
    misconfigured stack fails while rendering rather than while applying.

``x-flowmesh-when: <expr>``
    Conditional inclusion. The mapping carrying the key is dropped when the
    expression is falsey. A dropped document leaves the stream, a dropped
    sequence entry leaves its sequence, and a dropped mapping value takes its
    key with it. Expressions are ``VAR``, ``!VAR``, or ``VAR==value``.

``x-flowmesh-value: <node>``
    Renders to the node itself, so ``x-flowmesh-when`` can make a single scalar
    conditional inside a list of arguments.

``x-flowmesh-int: <scalar>``
    Renders to an integer. Ports and replica counts are rejected by the API as
    strings, and substitution otherwise always yields a string.

``x-flowmesh-file: <VAR>``
    Renders to the contents of the file named by the environment variable
    ``VAR``, for configuration the compose backend bind-mounts.

``x-flowmesh-env-values``
    The mapping carrying the key is replaced by the environment file's own
    key/value pairs, which is how the server receives the configuration compose
    passes through ``env_file``. Values come from the environment file alone,
    never from the caller's process environment.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

WHEN_KEY = "x-flowmesh-when"
VALUE_KEY = "x-flowmesh-value"
INT_KEY = "x-flowmesh-int"
FILE_KEY = "x-flowmesh-file"
ENV_VALUES_KEY = "x-flowmesh-env-values"

_VAR_PATTERN = re.compile(
    r"""
    \$\{
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        (?::-(?P<default>[^}]*))?
    \}
    """,
    re.VERBOSE,
)
_FALSEY = {"", "0", "false", "no", "off"}

_DROP = object()
"""Sentinel marking a node whose condition evaluated false."""


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be rendered."""


def substitute(value: str, env: Mapping[str, str]) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references in ``value``."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        if name in env:
            return env[name]
        if default is not None:
            return default
        raise ManifestError(
            f"{name} is not set and has no default in the manifest reference "
            f"{match.group(0)}"
        )

    return _VAR_PATTERN.sub(replace, value)


def evaluate_condition(expression: str, env: Mapping[str, str]) -> bool:
    """Evaluate an ``x-flowmesh-when`` expression against ``env``."""
    expr = expression.strip()
    if not expr:
        raise ManifestError("Empty x-flowmesh-when expression")

    if expr.startswith("!"):
        return not evaluate_condition(expr[1:], env)

    if "==" in expr:
        name, _, expected = expr.partition("==")
        return env.get(name.strip(), "").strip() == expected.strip()

    return env.get(expr, "").strip().lower() not in _FALSEY


def _render_int(node: Any, env: Mapping[str, str]) -> int:
    rendered = substitute(str(node), env).strip()
    try:
        return int(rendered)
    except ValueError:
        raise ManifestError(
            f"{INT_KEY} expected an integer but {node!r} rendered to {rendered!r}"
        ) from None


def _render_file(node: Any, env: Mapping[str, str]) -> str:
    name = substitute(str(node), env).strip()
    path_value = env.get(name, "").strip()
    if not path_value:
        raise ManifestError(f"{FILE_KEY} requires {name} to name a readable file")
    try:
        return Path(path_value).read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"Failed to read {name} file {path_value}: {exc}") from exc


def _render_node(
    node: Any, env: Mapping[str, str], env_values: Mapping[str, str]
) -> Any:
    if isinstance(node, dict):
        condition = node.get(WHEN_KEY)
        if condition is not None and not evaluate_condition(
            substitute(str(condition), env), env
        ):
            return _DROP
        if VALUE_KEY in node:
            return _render_node(node[VALUE_KEY], env, env_values)
        if INT_KEY in node:
            return _render_int(node[INT_KEY], env)
        if FILE_KEY in node:
            return _render_file(node[FILE_KEY], env)
        if ENV_VALUES_KEY in node:
            return dict(env_values)

        rendered: dict[str, Any] = {}
        for key, value in node.items():
            if key == WHEN_KEY:
                continue
            child = _render_node(value, env, env_values)
            if child is _DROP:
                continue
            rendered[key] = child
        return rendered

    if isinstance(node, list):
        items = [_render_node(item, env, env_values) for item in node]
        return [item for item in items if item is not _DROP]

    if isinstance(node, str):
        return substitute(node, env)

    return node


def render_documents(
    source: str,
    env: Mapping[str, str],
    env_values: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Render a multi-document manifest string into resource dictionaries.

    Document order is preserved because ``kubectl apply`` processes a stream in
    order and later resources depend on earlier ones.
    """
    documents: list[dict[str, Any]] = []
    for raw in yaml.safe_load_all(source):
        if raw is None:
            continue
        rendered = _render_node(raw, env, env_values or {})
        if rendered is _DROP or not rendered:
            continue
        documents.append(rendered)
    return documents


def render_manifests(
    paths: list[Path],
    env: Mapping[str, str],
    env_values: Mapping[str, str] | None = None,
) -> str:
    """Render manifest assets into a single YAML stream."""
    documents: list[dict[str, Any]] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"Failed to read manifest {path}: {exc}") from exc
        try:
            documents.extend(render_documents(source, env, env_values))
        except yaml.YAMLError as exc:
            raise ManifestError(f"Failed to parse manifest {path}: {exc}") from exc
    if not documents:
        raise ManifestError("No manifests to apply")
    return yaml.safe_dump_all(documents, sort_keys=False, default_flow_style=False)
