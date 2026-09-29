"""Validation of the ``python`` task spec."""

import pytest
from pydantic import ValidationError

from shared.tasks.envelope import TaskEnvelopeStrict
from shared.tasks.specs import PythonSpecStrict, PythonSpecTemplate
from shared.tasks.specs.python import MAX_CODE_BYTES

CODE = "def main(inputs):\n    return {'metrics': {'score': 1.0}}\n"


def _spec(**updates: object) -> PythonSpecStrict:
    return PythonSpecStrict.model_validate(
        {"taskType": "python", "code": CODE, **updates}
    )


def test_defaults_are_strict() -> None:
    spec = _spec()
    assert spec.entrypoint == "main"
    assert spec.network == "none"
    assert spec.requirements is None
    assert not spec.uses_gpu()


def test_envelope_discriminates_python() -> None:
    env = TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "v1",
            "kind": "Task",
            "spec": {"taskType": "python", "code": CODE},
        }
    )
    assert isinstance(env.spec, PythonSpecStrict)


@pytest.mark.parametrize(
    "updates,needle",
    [
        ({"code": "   "}, "non-empty"),
        ({"code": "x" * (MAX_CODE_BYTES + 1)}, "limit"),
        ({"entrypoint": "not an identifier"}, "identifier"),
        ({"timeoutSeconds": 0}, "timeoutSeconds"),
        ({"timeoutSeconds": 99999}, "timeoutSeconds"),
        ({"requirements": ["numpy"]}, "network: bridge"),
        ({"emits": [" "]}, "emits"),
        ({"network": "host"}, "network"),
        ({"inputs": [{"stage": "a"}, {"stage": "a"}]}, "unique"),
        ({"inputs": [{"stage": "a"}], "dependsOn": ["b"]}, "dependsOn"),
        (
            {"inputs": [{"stage": "a", "mountPath": "/mnt/flowmesh/output"}]},
            "output directory",
        ),
        (
            {"inputs": [{"stage": "a", "mountPath": "/mnt/flowmesh/output/a/"}]},
            "output directory",
        ),
    ],
)
def test_rejects(updates: dict[str, object], needle: str) -> None:
    with pytest.raises(ValidationError, match=needle):
        _spec(**updates)


def test_requirements_allowed_with_bridge() -> None:
    assert _spec(requirements=["numpy==2.1.0"], network="bridge").requirements


def test_gpu_request_uses_gpu() -> None:
    spec = _spec(resources={"hardware": {"gpu": {"count": 1}}})
    assert spec.uses_gpu()


def test_env_credentials_are_redacted() -> None:
    spec = _spec(env={"API_KEY": "secret-value"})
    redacted = spec.redact_credentials()
    assert redacted.env != spec.env
    assert redacted.has_redacted_credentials()


def test_template_validates_the_same_way() -> None:
    with pytest.raises(ValidationError, match="network: bridge"):
        PythonSpecTemplate.model_validate(
            {"taskType": "python", "code": CODE, "requirements": ["x"]}
        )
