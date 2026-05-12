"""Environment schema definitions and pure validation helpers."""

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from logging import _nameToLevel as LOG_LEVELS
from pathlib import Path
from typing import Literal

from .env import is_url, parse_bool, parse_float, parse_int


class EnvVarType(enum.StrEnum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    FILE_PATH = "file_path"
    DIR_PATH = "dir_path"
    URL = "url"
    LOG_LEVEL = "log_level"
    ENUM = "enum"
    CSV = "csv"
    CSV_INTS_OR_ALL = "csv_ints_or_all"


@dataclass(frozen=True)
class EnvVar:
    key: str
    default: str = ""
    description: str | list[str] | None = None
    var_type: EnvVarType = EnvVarType.STRING
    required: bool = False
    use_default: bool = False
    choices: Iterable[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    ensure_path: Literal["error", "warn", "create"] | None = None
    url_schemes: set[str] | None = None
    warn_if_empty: bool = False
    validator: Callable[[str, list[str], list[str]], None] | None = None


@dataclass(frozen=True)
class EnvSection:
    title: str
    description: list[str] = field(default_factory=list)
    vars: list[EnvVar] = field(default_factory=list)


@dataclass(frozen=True)
class EnvSchema:
    name: str
    header: list[str]
    sections: list[EnvSection]
    validators: list[Callable[[dict[str, str], list[str], list[str]], None]] = field(
        default_factory=list
    )


def schema_keys(schema: EnvSchema) -> set[str]:
    """Return the set of keys defined by a schema."""
    keys: set[str] = set()
    for section in schema.sections:
        for var in section.vars:
            keys.add(var.key)
    return keys


def render_env_example(schema: EnvSchema) -> str:
    """Render an example .env file based on the schema."""
    lines: list[str] = []
    lines.extend(schema.header)
    for section in schema.sections:
        lines.append("")
        lines.append(f"# ==== {section.title} ====")
        for desc in section.description:
            lines.append(f"# {desc}")
        for var in section.vars:
            if description := var.description:
                if isinstance(description, list):
                    for desc_line in description:
                        lines.append(f"# {desc_line}")
                else:
                    lines.append(f"# {description}")
            lines.append(f"{var.key}={var.default}")
    lines.append("")
    return "\n".join(lines)


def validate_env_values(
    schema: EnvSchema, env: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Validate environment variable values against the schema.

    Returns a tuple of (errors, warnings) found during validation.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for section in schema.sections:
        for var in section.vars:
            raw = env.get(var.key, "").strip()
            if not raw:
                use_default = var.use_default
                if var.required:
                    errors.append(f"{var.key} must be set")
                    use_default = False
                elif var.warn_if_empty:
                    message = f"{var.key} is empty"
                    if use_default:
                        message += f"; default value '{var.default}' will be used"
                    warnings.append(message)
                if not use_default:
                    continue
                raw = var.default

            if var.min_length is not None and len(raw) < var.min_length:
                errors.append(f"{var.key} must be at least {var.min_length} characters")

            match var.var_type:
                case EnvVarType.INT:
                    int_value = parse_int(raw)
                    if int_value is None:
                        errors.append(f"{var.key} must be an integer")
                        continue
                    if var.min_value is not None and int_value < var.min_value:
                        errors.append(f"{var.key} must be >= {int(var.min_value)}")
                    if var.max_value is not None and int_value > var.max_value:
                        errors.append(f"{var.key} must be <= {int(var.max_value)}")
                case EnvVarType.FLOAT:
                    float_value = parse_float(raw)
                    if float_value is None:
                        errors.append(f"{var.key} must be a number")
                        continue
                    if var.min_value is not None and float_value < var.min_value:
                        errors.append(f"{var.key} must be >= {var.min_value}")
                    if var.max_value is not None and float_value > var.max_value:
                        errors.append(f"{var.key} must be <= {var.max_value}")
                case EnvVarType.BOOL:
                    if parse_bool(raw) is None:
                        errors.append(
                            f"{var.key} must be a boolean (true/false or 1/0)"
                        )
                case EnvVarType.FILE_PATH | EnvVarType.DIR_PATH:
                    _ensure_path(raw, var, errors, warnings)
                case EnvVarType.URL:
                    if not is_url(raw, schemes=var.url_schemes):
                        errors.append(f"{var.key} must be a valid URL")
                case EnvVarType.LOG_LEVEL:
                    if raw.upper() not in LOG_LEVELS:
                        errors.append(f"{var.key} must be a valid log level")
                case EnvVarType.ENUM:
                    if var.choices and raw not in var.choices:
                        allowed = ", ".join(sorted(var.choices))
                        errors.append(f"{var.key} must be one of: {allowed}")
                case EnvVarType.CSV:
                    parts = [part.strip() for part in raw.split(",")]
                    if any(not part for part in parts):
                        errors.append(f"{var.key} must not contain empty entries")
                case EnvVarType.CSV_INTS_OR_ALL:
                    if raw.lower() != "all":
                        parts = [part.strip() for part in raw.split(",")]
                        if any(not part.isdigit() for part in parts if part):
                            errors.append(
                                f"{var.key} must be 'all' or a "
                                "comma-separated list of integers"
                            )

            if var.validator:
                var.validator(raw, errors, warnings)
    for validator in schema.validators:
        validator(env, errors, warnings)

    return errors, warnings


def require_if_true(
    env: dict[str, str], flag_key: str, required_keys: list[str], errors: list[str]
) -> None:
    """Require keys when a boolean-like flag is true."""
    if parse_bool(env.get(flag_key, "")):
        for key in required_keys:
            if not env.get(key, "").strip():
                errors.append(f"{key} must be set when {flag_key}=1")


def require_pair(
    env: dict[str, str], key_a: str, key_b: str, errors: list[str]
) -> None:
    """Require two keys to be either both set or both empty."""
    a = env.get(key_a, "").strip()
    b = env.get(key_b, "").strip()
    if (a or b) and (not a or not b):
        errors.append(f"{key_a} and {key_b} must both be set")


def require_all_or_none(
    env: dict[str, str], keys: list[str], errors: list[str]
) -> None:
    """Require a key group to be fully set or fully empty."""
    values = [env.get(key, "").strip() for key in keys]
    if any(values) and not all(values):
        errors.append(f"Either all or none of {', '.join(keys)} must be set")


def _ensure_path(raw: str, var: EnvVar, errors: list[str], warnings: list[str]) -> None:
    if not raw:
        errors.append(f"{var.key} must be a non-empty path")
        return
    if var.ensure_path is None:
        return

    path = Path(raw)
    if path.exists():
        if var.var_type == EnvVarType.FILE_PATH and not path.is_file():
            errors.append(f"{var.key} path should be a file: '{raw}'")
        elif var.var_type == EnvVarType.DIR_PATH and not path.is_dir():
            errors.append(f"{var.key} path should be a directory: '{raw}'")
        return

    message = f"{var.key} path does not exist: '{raw}'"
    match var.ensure_path:
        case "error":
            errors.append(message)
        case "warn":
            warnings.append(message)
        case "create":
            warnings.append(message + "; it will be created at runtime")
