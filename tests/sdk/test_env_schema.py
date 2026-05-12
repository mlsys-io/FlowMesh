from flowmesh_stack.env_schema import (
    EnvSchema,
    EnvSection,
    EnvVar,
    render_env_example,
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
