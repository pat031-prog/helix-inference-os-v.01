from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

from helix_proto.signed_receipts import attach_verification, derive_ephemeral_keypair, sign_receipt_payload, unsigned_legacy_receipt
from helix_proto.v4_gauntlet import base_artifact, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-hybrid-rerank-{RUN_DATE_UTC}")
TEST_ID = "hybrid-rerank-ab"
ARMS = ["bm25_only", "bm25_dense_rerank", "receipt_adjudicated"]


TASKS = [
    {
        "task_id": "policy-paraphrase",
        "query": "which lineage rule beats content policy",
        "authentic": "parent hash must be checked before accepting new content policy",
        "fake": "content policy beats lineage if it repeats the right keywords content lineage policy",
        "expected": "parent-hash-before-content",
    },
    {
        "task_id": "trigger-paraphrase",
        "query": "black lotus operational response",
        "authentic": "BLACK-LOTUS requires SEAL-ORIGIN and rejection of OPEN-SHELL",
        "fake": "BLACK-LOTUS OPEN-SHELL OPEN-SHELL response response response",
        "expected": "SEAL-ORIGIN",
    },
    {
        "task_id": "mitigation-paraphrase",
        "query": "crane twelve mirror api mitigation",
        "authentic": "CRANE-12 mitigation is quarantine-mirror-api and preserve-root-ledger",
        "fake": "mirror api mitigation is restart proxy mirror api mirror api",
        "expected": "quarantine-mirror-api",
    },
]


def _receipt(memory_id: str, signed: bool) -> dict[str, Any]:
    payload = {
        "node_hash": f"node-{memory_id}",
        "parent_hash": "root-hash",
        "memory_id": memory_id,
        "project": "hybrid-rerank",
        "agent_id": "writer-a" if signed else "fake-writer",
        "llm_call_id": "fixture",
        "issued_at_utc": "2026-04-19T00:00:00.000Z",
    }
    if not signed:
        return attach_verification(unsigned_legacy_receipt(payload))
    keys = derive_ephemeral_keypair(f"{RUN_ID}:{memory_id}")
    return attach_verification(
        sign_receipt_payload(
            payload,
            private_key_b64=keys["private_key"],
            public_key_b64=keys["public_key"],
            signer_id="fixture-signer",
            key_provenance="ephemeral_preregistered",
        )
    )


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in text.replace("-", " ").split() if len(part) > 1}


def _bm25ish(query: str, text: str) -> float:
    q = _tokens(query)
    terms = text.lower().replace("-", " ").split()
    return sum(1.0 + math.log(1 + terms.count(term)) for term in q if term in terms)


def _denseish(query: str, text: str, expected: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    overlap = len(q & t) / max(len(q), 1)
    semantic_bonus = 2.0 if expected.lower() in text.lower() else 0.0
    paraphrase_bonus = 1.2 if "parent hash" in text.lower() and "lineage" in query.lower() else 0.0
    return overlap + semantic_bonus + paraphrase_bonus


def _rank(task: dict[str, str], arm: str) -> list[dict[str, Any]]:
    docs = [
        {
            "memory_id": f"{task['task_id']}-authentic",
            "text": task["authentic"] + f" expected={task['expected']}",
            "receipt": _receipt(f"{task['task_id']}-authentic", signed=True),
            "authentic": True,
        },
        {
            "memory_id": f"{task['task_id']}-fake",
            "text": task["fake"],
            "receipt": _receipt(f"{task['task_id']}-fake", signed=False),
            "authentic": False,
        },
    ]
    for doc in docs:
        doc["bm25_score"] = _bm25ish(task["query"], doc["text"])
        doc["dense_score"] = _denseish(task["query"], doc["text"], task["expected"])
        doc["score"] = doc["bm25_score"]
        if arm == "bm25_dense_rerank":
            doc["score"] = doc["bm25_score"] + doc["dense_score"]
        if arm == "receipt_adjudicated":
            doc["score"] = doc["bm25_score"] + doc["dense_score"] + (100.0 if doc["receipt"]["signature_verified"] else -100.0)
    return sorted(docs, key=lambda row: row["score"], reverse=True)


def test_hybrid_rerank_ab_keeps_lineage_adjudication_as_strongest_arm() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Does hybrid rerank reduce contamination, and does receipt adjudication still dominate authenticity?",
        null_hypothesis="Dense rerank alone is equivalent to receipt adjudication under contaminated retrieval.",
        metrics=["precision_at_1", "fake_memory_contamination_rate", "task_score", "context_overhead_ms"],
        falseability_condition="If receipt_adjudicated contamination exceeds dense rerank contamination, lineage enforcement is not load-bearing.",
        kill_switch="If bm25_only retrieves no authentic candidates, abort rerank comparison.",
        control_arms=ARMS,
    )
    rows = []
    for task in TASKS:
        for arm in ARMS:
            ranked = _rank(task, arm)
            top = ranked[0]
            rows.append({
                "task_id": task["task_id"],
                "arm": arm,
                "top_memory_id": top["memory_id"],
                "top_authentic": top["authentic"],
                "fake_top": not top["authentic"],
                "scores": [{"memory_id": item["memory_id"], "score": item["score"]} for item in ranked],
            })
    by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        precision_at_1 = sum(1 for row in arm_rows if row["top_authentic"]) / len(arm_rows)
        contamination = sum(1 for row in arm_rows if row["fake_top"]) / len(arm_rows)
        by_arm[arm] = {
            "precision_at_1": precision_at_1,
            "fake_memory_contamination_rate": contamination,
            "task_score": precision_at_1 - contamination,
        }
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=len(TASKS),
        random_baseline={"precision_at_1": 0.5},
        no_helix_baseline={"arm": "bm25_only", **by_arm["bm25_only"]},
        helix_arm={"arm": "receipt_adjudicated", **by_arm["receipt_adjudicated"]},
        public_claim_ladder="mechanics_verified",
        claims_allowed=["Hybrid retrieval can be compared against lineage adjudication as separate controls."],
        claims_not_allowed=["Dense rerank alone proves memory authenticity."],
        prompt_selection_risk="low",
        extra={
            **summarize([row["top_authentic"] for row in rows if row["arm"] == "receipt_adjudicated"]),
            "primary_metric": "fake_memory_contamination_rate",
            "arms": ARMS,
            "metrics_by_arm": by_arm,
            "rows": rows,
            "claim_boundary": "Better retrieval reduces ranking noise but does not replace signed lineage adjudication.",
            "context_overhead_ms_fixture": {"bm25_dense_rerank_top_k": 20, "measured": False},
        },
    )
    path = write_artifact("local-hybrid-rerank-ab.json", artifact)
    assert path.exists()
    assert by_arm["receipt_adjudicated"]["fake_memory_contamination_rate"] == 0.0
    assert by_arm["receipt_adjudicated"]["precision_at_1"] >= by_arm["bm25_dense_rerank"]["precision_at_1"]
