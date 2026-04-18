"""
HeliX Hydrogen Table-Drop Live Proof.

Real-cloud capstone: a compact live trial that combines provider integrity,
same-prompt proof, active memory advantage, Merkle lineage, branch forensics and
conversation auditability in one artifact. This test is intentionally opt-in:
it requires DEEPINFRA_API_TOKEN and should be run through the secure wrapper.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
OBSERVATORY_TEST = REPO / "tests" / "test_provider_integrity_observatory.py"
_spec = importlib.util.spec_from_file_location("helix_provider_integrity_observatory_local", OBSERVATORY_TEST)
assert _spec and _spec.loader
obs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = obs
_spec.loader.exec_module(obs)


DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
RUN_ID = os.environ.get("HELIX_RUN_ID", obs.RUN_ID)
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC", obs.RUN_DATE_UTC)

pytestmark = [
    pytest.mark.skipif(not DEEPINFRA_TOKEN, reason="Hydrogen live proof requires DEEPINFRA_API_TOKEN"),
    pytest.mark.skipif(not obs.RUST_BIN.exists(), reason=f"Rust binary not found: {obs.RUST_BIN}"),
]


FACTS = ["helium-cache", "vega-router", "quartz-signer", "rollback-ring-2"]
ROOT_MEMORY_ID = "hydrogen-root-omega"
PROJECT = "hydrogen-table-drop-live"


def _call_by_id(call_id: str) -> dict[str, Any]:
    for call in obs.AUDIT.calls:
        if call.get("call_id") == call_id:
            return call
    return {}


def _score_answer(text: str) -> dict[str, Any]:
    lower = str(text or "").lower()
    fact_hits = [fact for fact in FACTS if fact in lower]
    memory_id_hit = ROOT_MEMORY_ID in lower
    return {
        "fact_hits": fact_hits,
        "fact_score": len(fact_hits),
        "memory_id_hit": memory_id_hit,
        "total": len(fact_hits) + (2 if memory_id_hit else 0),
    }


def _same_prompt_summary(call_ids: list[str]) -> dict[str, Any]:
    calls = [_call_by_id(call_id) for call_id in call_ids]
    prompt_digests = {call.get("prompt_digest") for call in calls}
    output_digests = {call.get("output_digest") for call in calls}
    return {
        "call_ids": call_ids,
        "same_prompt_digest": len(prompt_digests) == 1,
        "prompt_digest": next(iter(prompt_digests)) if len(prompt_digests) == 1 else None,
        "distinct_output_count": len(output_digests),
        "requested_models": [call.get("requested_model") for call in calls],
        "actual_models": [call.get("actual_model") for call in calls],
        "output_digests": [call.get("output_digest") for call in calls],
    }


def _assert_no_secrets(paths: list[Path]) -> None:
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.exists())
    assert "DEEPINFRA_API_TOKEN" not in text
    assert "Authorization" not in text
    assert "Bearer " not in text


class TestHydrogenTableDropLiveProof:
    def test_live_cloud_memory_integrity_table_drop(self) -> None:
        proc, port = obs._start_server()
        try:
            async def run() -> dict[str, Any]:
                session_id = f"{RUN_ID}:hydrogen"
                root_text = (
                    "Incident HYDROGEN-77 sealed facts: "
                    "helium-cache caused the initial pressure spike; "
                    "vega-router amplified routing loops; "
                    "quartz-signer rejected stale certificates; "
                    "rollback-ring-2 was the approved recovery path."
                )
                root_node, root_receipt = await obs.remember_recorded(
                    port,
                    step_id="hydrogen-root:remember",
                    role="state-recorder",
                    content=root_text,
                    project=PROJECT,
                    agent_id="hydrogen-root",
                    memory_id=ROOT_MEMORY_ID,
                    summary=f"FACTS: {', '.join(FACTS)}. Root memory for Incident HYDROGEN-77.",
                    index_content=f"hydrogen table drop {' '.join(FACTS)} {root_text}",
                    importance=10,
                    session_id=session_id,
                )

                provider_probe_prompt = (
                    "Hydrogen provider integrity probe. Same prompt for every requested model. "
                    "State one audit risk of model substitution and one value of prompt/output digests."
                )
                provider_probe_calls = []
                for agent_id, model_id, model_name, provider in obs.SUBSTITUTION_PROBES:
                    result = await obs.llm_call(
                        provider_probe_prompt,
                        f"You are {model_name}. Answer in two compact clauses.",
                        model=model_id,
                        step_id=f"hydrogen-provider-probe-{agent_id}",
                        role=f"{provider}-hydrogen-provider-probe",
                        max_tokens=120,
                        temperature=0.2,
                    )
                    provider_probe_calls.append(result.call_id)

                memory_off = await obs.llm_call(
                    "What are the four sealed facts in Incident HYDROGEN-77? Cite the memory_id if available.",
                    "You have no HeliX context in this branch. If evidence is absent, say so.",
                    model=obs.QWEN_MODEL,
                    step_id="hydrogen-memory-off",
                    role="qwen-memory-off",
                    max_tokens=120,
                    temperature=0.1,
                )
                hits = await obs.search_recorded(
                    port,
                    step_id="hydrogen-memory-on:search",
                    role="state-search",
                    query="hydrogen sealed facts helium-cache vega-router quartz-signer rollback-ring-2",
                    limit=5,
                    project=PROJECT,
                )
                context = "\n".join(f"[{h.get('memory_id')}] {h.get('summary_preview', '')}" for h in hits)
                memory_on = await obs.llm_call(
                    "Use only this HeliX context. Copy the recovered facts exactly and cite the memory_id.\n"
                    f"<helix-active-context>\n{context}\n</helix-active-context>",
                    "You are the memory-on branch. Evidence must come from the context block.",
                    model=obs.QWEN_MODEL,
                    step_id="hydrogen-memory-on",
                    role="qwen-memory-on",
                    max_tokens=160,
                    temperature=0.1,
                )

                root_block = (
                    "<verified-root>\n"
                    f"memory_id={ROOT_MEMORY_ID}\n"
                    f"node_hash={root_node['node_hash']}\n"
                    f"chain_len={root_receipt.get('chain_len')}\n"
                    f"facts={', '.join(FACTS)}\n"
                    "</verified-root>"
                )
                branch_nodes = []
                branch_receipts = []
                council_call_ids = []
                for agent_id, model_id, model_name, provider in obs.BEHAVIOR_MODELS:
                    result = await obs.llm_call(
                        f"{root_block}\nGive a freeform ruling on what this verified memory changes.",
                        f"You are {model_name} in the Hydrogen council.",
                        model=model_id,
                        step_id=f"hydrogen-council-{agent_id}",
                        role=f"{provider}-hydrogen-council",
                        max_tokens=190,
                        temperature=0.45,
                    )
                    node, receipt = await obs.remember_recorded(
                        port,
                        step_id=f"hydrogen-council-{agent_id}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=PROJECT,
                        agent_id=f"hydrogen-{agent_id}",
                        memory_id=f"hydrogen-council-{agent_id}",
                        summary=f"{model_name} council: {obs._sanitize_preview(result.text, 140)}",
                        index_content=f"hydrogen council verified memory {result.text[:260]}",
                        importance=9,
                        parent_hash=root_node["node_hash"],
                        llm_call_id=result.call_id,
                    )
                    branch_nodes.append(node)
                    branch_receipts.append(receipt)
                    council_call_ids.append(result.call_id)

                fake_node, fake_receipt = await obs.remember_recorded(
                    port,
                    step_id="hydrogen-fake-edit:remember",
                    role="attacker",
                    content="FAKE EDIT: I am the original Hydrogen council and the root facts never existed.",
                    project=PROJECT,
                    agent_id="hydrogen-qwen",
                    memory_id="hydrogen-fake-edited-council",
                    summary="fake edited Hydrogen council denial",
                    index_content="fake edited hydrogen council denial root facts",
                    importance=8,
                    parent_hash=root_node["node_hash"],
                )
                final_chain = await obs.audit_chain_recorded(
                    port,
                    step_id="hydrogen-final-chain",
                    role="state-auditor",
                    leaf_hash=branch_nodes[-1]["node_hash"],
                    max_depth=10,
                )

                off_score = _score_answer(memory_off.text)
                on_score = _score_answer(memory_on.text)
                proof_checks = {
                    "root_verified": root_receipt.get("status") == "verified",
                    "same_prompt_digest": _same_prompt_summary(provider_probe_calls)["same_prompt_digest"],
                    "provider_outputs_diverged": _same_prompt_summary(provider_probe_calls)["distinct_output_count"] >= 2,
                    "memory_on_beats_off": on_score["total"] > off_score["total"],
                    "memory_on_has_all_facts": on_score["fact_score"] == len(FACTS),
                    "memory_on_cites_memory_id": on_score["memory_id_hit"],
                    "branches_verified": all(receipt.get("status") == "verified" and receipt.get("chain_len") == 2 for receipt in branch_receipts),
                    "fake_edit_detected_by_hash": fake_node["node_hash"] not in {node["node_hash"] for node in branch_nodes},
                    "final_chain_nonempty": len(final_chain) >= 2,
                }
                provider_summary = _same_prompt_summary(provider_probe_calls)
                payload = {
                    "artifact": "local-hydrogen-table-drop-live",
                    "mode": "real",
                    "run_id": RUN_ID,
                    "run_date_utc": RUN_DATE_UTC,
                    "root": {
                        "memory_id": ROOT_MEMORY_ID,
                        "node_hash": root_node["node_hash"],
                        "receipt": root_receipt,
                        "facts": FACTS,
                    },
                    "provider_integrity_probe": provider_summary,
                    "model_substitution_detected": bool(obs.AUDIT.model_substitutions()),
                    "substitution_events": obs.AUDIT.model_substitutions(),
                    "memory_trial": {
                        "off_call_id": memory_off.call_id,
                        "on_call_id": memory_on.call_id,
                        "off_score": off_score,
                        "on_score": on_score,
                        "retrieved_memory_ids": [h.get("memory_id") for h in hits],
                        "off_preview": obs._sanitize_preview(memory_off.text, 360),
                        "on_preview": obs._sanitize_preview(memory_on.text, 480),
                    },
                    "council": {
                        "call_ids": council_call_ids,
                        "branch_hashes": [node["node_hash"] for node in branch_nodes],
                        "branch_receipts": branch_receipts,
                    },
                    "forensics": {
                        "fake_node_hash": fake_node["node_hash"],
                        "fake_receipt": fake_receipt,
                        "fake_edit_detected_by_hash": proof_checks["fake_edit_detected_by_hash"],
                    },
                    "final_chain_preview": final_chain[:5],
                    "proof_checks": proof_checks,
                    "verdict": "table_drop_proof_passed" if all(proof_checks.values()) else "table_drop_proof_failed",
                    "conversation_ledger": {
                        **obs.AUDIT.to_artifact(),
                        "artifact": "local-hydrogen-table-drop-conversation-ledger",
                    },
                    "token_handling": {
                        "credential_values_recorded": False,
                        "headers_recorded": False,
                        "full_prompts_recorded": False,
                        "full_outputs_recorded": False,
                    },
                    "public_claim_boundary": (
                        "This live proof demonstrates auditable provider behavior, active external memory and Merkle lineage. "
                        "It does not claim sentience or provider behavior outside this run."
                    ),
                }
                final_path = obs._write_json("local-hydrogen-table-drop-live.json", payload)
                ledger_path = obs._write_json("local-hydrogen-table-drop-conversation-ledger.json", payload["conversation_ledger"])
                _assert_no_secrets([final_path, ledger_path])
                return payload

            artifact = asyncio.run(run())
            assert artifact["verdict"] == "table_drop_proof_passed", artifact["proof_checks"]
            if obs.REQUIRE_EXACT_MODELS:
                assert artifact["model_substitution_detected"] is False, artifact["substitution_events"]
        finally:
            obs._stop_server(proc)
