from __future__ import annotations

from helix_proto.blind_inference import BlindInferencePolicy
from helix_proto.blind_inference import TokenVault
from helix_proto.blind_inference import blind_rehydrate_response
from helix_proto.blind_inference import blind_transform_request


def test_token_vault_is_deterministic_within_task_and_distinct_across_tasks() -> None:
    first = TokenVault(task_id="task-a")
    second = TokenVault(task_id="task-b")

    placeholder_a1 = first.issue_placeholder(sensitive_type="PERSON", original="Juan Perez", origin_rule="person-rule")
    placeholder_a2 = first.issue_placeholder(sensitive_type="PERSON", original="Juan Perez", origin_rule="person-rule")
    placeholder_b = second.issue_placeholder(sensitive_type="PERSON", original="Juan Perez", origin_rule="person-rule")

    assert placeholder_a1 == placeholder_a2
    assert placeholder_a1 != placeholder_b


def test_blind_transform_request_redacts_explicit_rules_and_rehydrates_response() -> None:
    policy = BlindInferencePolicy.from_payload(
        {
            "enabled": True,
            "scope": "cloud_proxy",
            "placeholder_stability": "per_task",
            "rules": [
                {"name": "person-rule", "type": "PERSON", "values": ["Juan Perez"]},
                {"name": "doc-rule", "type": "DOC_ID", "pattern": r"\b\d{8}\b"},
            ],
        }
    )
    transformed = blind_transform_request(
        [{"role": "user", "content": "Analiza a Juan Perez con DNI 12345678."}],
        policy=policy,
        task_id="task-redact-1",
    )

    outbound = transformed["messages"][0]["content"]
    assert "Juan Perez" not in outbound
    assert "12345678" not in outbound
    assert transformed["span_count"] == 2
    rehydrated = blind_rehydrate_response(
        f"Reporte listo para PERSON__T{transformed['vault'].task_tag}__001 y DOC_ID__T{transformed['vault'].task_tag}__001.",
        transformed["vault"],
    )
    assert rehydrated == "Reporte listo para Juan Perez y 12345678."


def test_blind_transform_request_detector_warning_does_not_force_redaction() -> None:
    policy = BlindInferencePolicy.from_payload(
        {
            "enabled": True,
            "scope": "cloud_proxy",
            "placeholder_stability": "per_task",
            "rules": [],
            "detectors": {"email": True},
        }
    )
    transformed = blind_transform_request(
        [{"role": "user", "content": "Escribile a persona@example.com pero no lo tapes sin policy."}],
        policy=policy,
        task_id="task-warning-1",
    )

    assert transformed["messages"][0]["content"] == "Escribile a persona@example.com pero no lo tapes sin policy."
    assert transformed["warnings"]
    assert transformed["warnings"][0]["detector"] == "email"
