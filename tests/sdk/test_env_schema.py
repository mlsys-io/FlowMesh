import pytest
from flowmesh_stack.env_schema import (
    EnvSchema,
    EnvSection,
    EnvVar,
    EnvVarType,
    render_env_example,
    validate_env_values,
)


def _toy_schema() -> EnvSchema:
    return EnvSchema(
        name="toy",
        header=["# toy header"],
        sections=[
            EnvSection(
                title="Role",
                vars=[
                    EnvVar("NODE_ROLE", "root"),
                    EnvVar("OTHER", "value"),
                ],
            ),
        ],
    )


def test_render_env_example_uses_schema_default_without_overrides() -> None:
    body = render_env_example(_toy_schema())
    assert "NODE_ROLE=root" in body
    assert "OTHER=value" in body


def test_render_env_example_applies_overrides() -> None:
    body = render_env_example(_toy_schema(), overrides={"NODE_ROLE": "worker"})
    assert "NODE_ROLE=worker" in body
    assert "NODE_ROLE=root" not in body
    # Non-overridden keys still use the schema default.
    assert "OTHER=value" in body


def test_render_env_example_ignores_overrides_for_unknown_keys() -> None:
    body = render_env_example(_toy_schema(), overrides={"NOT_A_KEY": "x"})
    assert "NOT_A_KEY" not in body
    assert "NODE_ROLE=root" in body


def _range_schema(var: EnvVar) -> EnvSchema:
    return EnvSchema(
        name="bounds", header=[], sections=[EnvSection(title="bounds", vars=[var])]
    )


class TestValidateRangeBounds:
    @pytest.mark.parametrize(
        "var,raw,expect_error",
        [
            # min_value inclusive (default)
            (EnvVar("X", var_type=EnvVarType.INT, min_value=1), "1", False),
            (EnvVar("X", var_type=EnvVarType.INT, min_value=1), "0", True),
            # min_value exclusive
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.INT,
                    min_value=0,
                    min_inclusive=False,
                ),
                "0",
                True,
            ),
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.INT,
                    min_value=0,
                    min_inclusive=False,
                ),
                "1",
                False,
            ),
            # max_value inclusive (default)
            (EnvVar("X", var_type=EnvVarType.INT, max_value=10), "10", False),
            (EnvVar("X", var_type=EnvVarType.INT, max_value=10), "11", True),
            # max_value exclusive
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.INT,
                    max_value=10,
                    max_inclusive=False,
                ),
                "10",
                True,
            ),
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.INT,
                    max_value=10,
                    max_inclusive=False,
                ),
                "9",
                False,
            ),
            # FLOAT path mirrors INT
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                "0",
                True,
            ),
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                "0.5",
                False,
            ),
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.FLOAT,
                    max_value=1.5,
                    max_inclusive=True,
                ),
                "1.5",
                False,
            ),
            (
                EnvVar(
                    "X",
                    var_type=EnvVarType.FLOAT,
                    max_value=1.5,
                    max_inclusive=False,
                ),
                "1.5",
                True,
            ),
        ],
    )
    def test_bounds(self, var: EnvVar, raw: str, expect_error: bool) -> None:
        errors, _ = validate_env_values(_range_schema(var), {var.key: raw})
        assert bool(errors) is expect_error

    def test_exclusive_error_message_uses_strict_comparator(self) -> None:
        var = EnvVar(
            "SSH_MAX_CPU",
            var_type=EnvVarType.FLOAT,
            min_value=0,
            min_inclusive=False,
        )
        errors, _ = validate_env_values(_range_schema(var), {"SSH_MAX_CPU": "0"})
        assert errors == ["SSH_MAX_CPU must be > 0"]

    def test_inclusive_error_message_uses_default_comparator(self) -> None:
        var = EnvVar("PORT", var_type=EnvVarType.INT, min_value=1024)
        errors, _ = validate_env_values(_range_schema(var), {"PORT": "80"})
        assert errors == ["PORT must be >= 1024"]
