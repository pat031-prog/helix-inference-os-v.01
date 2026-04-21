from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from helix_kv.memory_catalog import MemoryCatalog
from tools.run_memory_fork_forensics_v1 import (
    _branch_chain_ok,
    _build_memory_forks,
    _deepinfra_request_body,
    _expected_forensics,
    _extract_json_object,
    _score_forensics,
)


def _record(memory_id: str, node_hash: str, parent_hash: str | None) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "node_hash": node_hash,
        "parent_hash": parent_hash,
        "signature_verified": True,
        "summary": memory_id,
        "content": memory_id,
    }


def _fork_graph() -> dict[str, object]:
    root = _record("root", "root_hash", None)
    alpha = [
        _record("a1", "a1_hash", "root_hash"),
        _record("a2", "a2_hash", "a1_hash"),
        _record("a3", "a3_hash", "a2_hash"),
    ]
    beta = [
        _record("b1", "b1_hash", "root_hash"),
        _record("b2", "b2_hash", "b1_hash"),
        _record("b3", "b3_hash", "b2_hash"),
    ]
    return {
        "root": root,
        "branches": {
            "alpha_valid": alpha,
            "beta_signed_poison": beta,
        },
        "all_records": [root, *alpha, *beta],
    }


def _valid_forensic_json(graph: dict[str, object]) -> dict[str, object]:
    expected = _expected_forensics(graph)
    return {
        "shared_root_node_hash": expected["shared_root_node_hash"],
        "valid_branch": {
            **expected["valid_branch"],
            "why_valid": "valid alpha branch preserves signed hmem and bounded rollback",
        },
        "poison_branch": {
            **expected["poison_branch"],
            "why_signed_but_invalid": (
                "signed poison: cryptographic signature is valid, but semantic policy is "
                "invalid because it claims bijective KV<->SSM projection and rollback 90 minutes"
            ),
        },
        "rejected_claims": [
            "reject bijective KV<->SSM projection",
            "reject rollback 90 minutes",
        ],
        "crypto_vs_semantic": "Cryptographic signature validity only proves receipt integrity; semantic policy validity still rejects signed poison.",
    }


def _pass_score(graph: dict[str, object], forensic_json: dict[str, object] | None = None) -> dict[str, object]:
    forensic_json = forensic_json or _valid_forensic_json(graph)
    forensic_text = json.dumps(forensic_json, sort_keys=True)
    return _score_forensics(
        fork_graph=graph,
        forensic_text=forensic_text,
        auditor_text=json.dumps({"verdict": "pass", "gate_failures": []}),
        forensic_json=forensic_json,
        auditor_json={
            "verdict": "pass",
            "gate_failures": [],
            "cryptographic_validity_vs_semantic_validity_checked": True,
        },
        forensic_finish_reason="stop",
        auditor_finish_reason="stop",
    )


def test_build_memory_forks_creates_two_signed_branches_from_one_root() -> None:
    workspace = (
        Path.cwd()
        / "verification"
        / "nuclear-methodology"
        / "memory-fork-forensics"
        / "_pytest"
        / uuid.uuid4().hex
    ).resolve()
    allowed_root = (Path.cwd() / "verification" / "nuclear-methodology" / "memory-fork-forensics" / "_pytest").resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    catalog = MemoryCatalog.open(workspace / "memory.sqlite")
    try:
        graph = _build_memory_forks(catalog, run_id="unit-fork")
    finally:
        catalog.close()
        if allowed_root in workspace.parents:
            shutil.rmtree(workspace, ignore_errors=True)

    root_hash = str(graph["root"]["node_hash"])
    alpha = graph["branches"]["alpha_valid"]
    beta = graph["branches"]["beta_signed_poison"]

    assert all(record["signature_verified"] for record in graph["all_records"])
    assert _branch_chain_ok(alpha, root_hash=root_hash)
    assert _branch_chain_ok(beta, root_hash=root_hash)
    assert alpha[0]["parent_hash"] == root_hash
    assert beta[0]["parent_hash"] == root_hash
    assert alpha[-1]["summary"] == "DECISION_ALPHA_SAFE_HANDOFF"
    assert beta[-1]["summary"] == "DECISION_BETA_INVALID_FAST_PATH"


def test_expected_forensics_includes_root_in_each_causal_path() -> None:
    graph = _fork_graph()
    expected = _expected_forensics(graph)

    assert expected["shared_root_node_hash"] == "root_hash"
    assert expected["valid_branch"]["causal_node_hash_path"] == [
        "root_hash",
        "a1_hash",
        "a2_hash",
        "a3_hash",
    ]
    assert expected["poison_branch"]["causal_node_hash_path"] == [
        "root_hash",
        "b1_hash",
        "b2_hash",
        "b3_hash",
    ]


def test_extract_json_object_accepts_fenced_json_with_trailing_prose() -> None:
    parsed = _extract_json_object('prefix\n```json\n{"verdict":"pass","gate_failures":[]}\n```\nignored')

    assert parsed == {"verdict": "pass", "gate_failures": []}


def test_extract_json_object_rejects_incomplete_json() -> None:
    assert _extract_json_object('{"verdict":"pass", "gate_failures": [') is None


def test_deepinfra_body_disables_qwen_thinking_mode() -> None:
    body = _deepinfra_request_body(
        model="Qwen/Qwen3.6-35B-A3B",
        system="system",
        user="user",
        max_tokens=3600,
        temperature=0.0,
    )

    assert body["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert body["top_k"] == 20


def test_deepinfra_body_does_not_add_qwen_kwargs_to_glm() -> None:
    body = _deepinfra_request_body(
        model="zai-org/GLM-5.1",
        system="system",
        user="user",
        max_tokens=3600,
        temperature=0.0,
    )

    assert "chat_template_kwargs" not in body
    assert "top_k" not in body


def test_branch_chain_ok_detects_parent_hash_break() -> None:
    graph = _fork_graph()
    alpha = graph["branches"]["alpha_valid"]

    assert _branch_chain_ok(alpha, root_hash="root_hash") is True

    broken = [dict(item) for item in alpha]
    broken[1]["parent_hash"] = "wrong_hash"
    assert _branch_chain_ok(broken, root_hash="root_hash") is False


def test_score_forensics_passes_only_with_exact_structured_causality() -> None:
    score = _pass_score(_fork_graph())

    assert score["passed"] is True
    assert score["score"] == 1.0
    assert all(score["gates"].values())


def test_score_forensics_requires_shared_root_hash_in_output() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_json["valid_branch"]["causal_node_hash_path"] = ["a1_hash", "a2_hash", "a3_hash"]
    forensic_json["poison_branch"]["causal_node_hash_path"] = ["b1_hash", "b2_hash", "b3_hash"]

    score = _pass_score(graph, forensic_json)

    assert score["passed"] is False
    assert score["gates"]["forensic_cites_shared_root_hash"] is True
    assert score["gates"]["forensic_json_alpha_path_exact"] is False
    assert score["gates"]["forensic_json_beta_path_exact"] is False


def test_score_forensics_fails_when_branches_are_role_swapped() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_json["valid_branch"], forensic_json["poison_branch"] = (
        forensic_json["poison_branch"],
        forensic_json["valid_branch"],
    )

    score = _pass_score(graph, forensic_json)

    assert score["passed"] is False
    assert score["gates"]["forensic_json_roles_correct"] is False
    assert score["gates"]["forensic_json_alpha_terminal_matches"] is False
    assert score["gates"]["forensic_json_beta_terminal_matches"] is False


def test_score_forensics_fails_when_signed_poison_is_not_identified() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_json["poison_branch"]["why_signed_but_invalid"] = "also valid"
    forensic_json["rejected_claims"] = []
    forensic_json["crypto_vs_semantic"] = "signature valid"

    score = _pass_score(graph, forensic_json)

    assert score["passed"] is False
    assert score["gates"]["identifies_beta_as_signed_poison"] is False
    assert score["gates"]["rejects_bijective_projection"] is False
    assert score["gates"]["rejects_rollback_90"] is False


def test_score_forensics_accepts_structured_signed_but_invalid_poison_branch() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_json["poison_branch"]["why_signed_but_invalid"] = (
        "Cryptographic signature is valid, but semantic policy is violated: "
        "it claims bijective KV<->SSM projection and rollback 90 minutes."
    )
    forensic_text = json.dumps(forensic_json, sort_keys=True)

    score = _score_forensics(
        fork_graph=graph,
        forensic_text=forensic_text,
        auditor_text=json.dumps({"verdict": "pass", "gate_failures": []}),
        forensic_json=forensic_json,
        auditor_json={
            "verdict": "pass",
            "gate_failures": [],
            "cryptographic_validity_vs_semantic_validity_checked": True,
        },
        forensic_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True
    assert score["gates"]["identifies_beta_as_signed_poison"] is True


def test_score_forensics_requires_parsed_auditor_pass_verdict() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_text = json.dumps(forensic_json, sort_keys=True)

    score = _score_forensics(
        fork_graph=graph,
        forensic_text=forensic_text,
        auditor_text="pass appears in prose, but parsed verdict says fail",
        forensic_json=forensic_json,
        auditor_json={
            "verdict": "fail",
            "gate_failures": ["causality missing"],
            "cryptographic_validity_vs_semantic_validity_checked": True,
        },
        forensic_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["auditor_verdict_pass"] is False
    assert score["gates"]["auditor_gate_failures_empty"] is False


def test_score_forensics_rejects_length_truncated_cloud_outputs() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_text = json.dumps(forensic_json, sort_keys=True)

    score = _score_forensics(
        fork_graph=graph,
        forensic_text=forensic_text,
        auditor_text=json.dumps({"verdict": "pass", "gate_failures": []}),
        forensic_json=forensic_json,
        auditor_json={
            "verdict": "pass",
            "gate_failures": [],
            "cryptographic_validity_vs_semantic_validity_checked": True,
        },
        forensic_finish_reason="length",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["forensic_finish_reason_not_length"] is False


def test_score_forensics_requires_forensic_output_to_cite_hashes_not_auditor() -> None:
    graph = _fork_graph()
    forensic_json = _valid_forensic_json(graph)
    forensic_text = json.dumps({
        "valid_branch": {"decision_id": "DECISION_ALPHA_SAFE_HANDOFF"},
        "poison_branch": {"decision_id": "DECISION_BETA_INVALID_FAST_PATH"},
        "crypto_vs_semantic": "cryptographic signature valid but semantic policy invalid",
        "rejected_claims": ["reject bijective KV<->SSM projection", "reject rollback 90 minutes"],
    })
    auditor_text = (
        "auditor mentions root_hash a1_hash a2_hash a3_hash b1_hash b2_hash b3_hash "
        "a3 b3 and says pass"
    )

    score = _score_forensics(
        fork_graph=graph,
        forensic_text=forensic_text,
        auditor_text=auditor_text,
        forensic_json=forensic_json,
        auditor_json={
            "verdict": "pass",
            "gate_failures": [],
            "cryptographic_validity_vs_semantic_validity_checked": True,
        },
        forensic_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["forensic_cites_alpha_path_hash"] is False
    assert score["gates"]["forensic_cites_beta_path_hash"] is False
