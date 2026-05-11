"""Validation coverage for checked-in workflow templates."""

from pathlib import Path

import pytest

from server.task.parser import parse_workflow

TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "examples" / "templates"
TEMPLATE_PATHS = sorted(TEMPLATE_DIR.glob("*.yaml"))


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS, ids=lambda path: path.name)
def test_template_parses(template_path: Path) -> None:
    parsed = parse_workflow(template_path.read_text(encoding="utf-8"), "native")

    assert parsed.tasks
