from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from typing import Any

from helix_proto.v4_gauntlet import base_artifact, confusion_metrics, percentile, preregister, rng_for, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-lineage-forgery-{RUN_DATE_UTC}")
TEST_ID = "lineage-forgery-gauntlet"
ARMS = ["naive", "schema-aware", "hash-aware", "signature-aware"]
SEVERITIES = ["low", "medium", "high"]
SECRET = b"helix-v4-lineage-test-key"


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sign(node_hash: str, agent_id: str) -> str:
    return hmac.new(SECRET, f"{node_hash}|{agent_id}".encode("utf-8"), hashlib.sha256).hexdigest()


def _node(chain_id: str, position: int, parent_hash: str, content: str, *, agent_id: str = "legit") -> dict[str, Any]:
    content_digest = _digest(content)
    node_hash = _digest(f"{chain_id}|{position}|{parent_hash}|{content_digest}|{agent_id}")
    return {
        "chain_id": chain_id,
        "position": position,
        "parent_hash": parent_hash,
        "content": content,
        "content_digest": content_digest,
        "agent_id": agent_id,
        "node_hash": node_hash,
        "signature": _sign(node_hash, agent_id),
    }


def _make_chain(chain_id: str, length: int = 5) -> list[dict[str, Any]]:
    rows = []
    parent = "GENESIS"
    for position in range(length):
        node = _node(chain_id, position, parent, f"legit content {chain_id}/{position}")
        rows.append(node)
        parent = node["node_hash"]
    return rows


def _forge(chain: list[dict[str, Any]], arm: str, severity: str) -> dict[str, Any]:
    base = dict(chain[-1])
    if arm == "naive":
        base["content"] = f"naive edit severity={severity}"
        return base
    if arm == "schema-aware":
        base["content"] = f"schema-aware edit severity={severity}"
        base["content_digest"] = _digest(base["content"])
        return base
    if arm == "hash-aware":
        base["content"] = f"hash-aware edit severity={severity}"
        base["content_digest"] = _digest(base["content"])
        base["node_hash"] = _digest(f"{base['chain_id']}|{base['position']}|WRONG-PARENT|{base['content_digest']}|{base['agent_id']}")
        return base
    base["content"] = f"signature-aware edit severity={severity}"
    base["content_digest"] = _digest(base["content"])
    base["parent_hash"] = chain[-2]["node_hash"]
    base["node_hash"] = _digest(f"{base['chain_id']}|{base['position']}|{base['parent_hash']}|{base['content_digest']}|compromised")
    base["agent_id"] = "compromised"
    base["signature"] = _sign(base["node_hash"], base["agent_id"])
    return base


def _detect(candidate: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, float, list[str]]:
    t0 = time.perf_counter()
    reasons = []
    if candidate.get("chain_id") != expected.get("chain_id"):
        reasons.append("chain_id_mismatch")
    if candidate.get("position") != expected.get("position"):
        reasons.append("position_mismatch")
    if candidate.get("parent_hash") != expected.get("parent_hash"):
        reasons.append("parent_hash_mismatch")
    if candidate.get("content_digest") != expected.get("content_digest"):
        reasons.append("content_digest_mismatch")
    candidate_content_digest = _digest(str(candidate.get("content", "")))
    if candidate.get("content_digest") != candidate_content_digest:
        reasons.append("content_digest_self_mismatch")
    recomputed_hash = _digest(
        f"{candidate.get('chain_id')}|{candidate.get('position')}|{candidate.get('parent_hash')}|"
        f"{candidate.get('content_digest')}|{candidate.get('agent_id')}"
    )
    if candidate.get("node_hash") != recomputed_hash:
        reasons.append("hash_mismatch")
    expected_signature = _sign(str(candidate.get("node_hash")), str(candidate.get("agent_id")))
    if candidate.get("signature") != expected_signature:
        reasons.append("signature_mismatch")
    if candidate.get("agent_id") != expected.get("agent_id"):
        reasons.append("agent_lineage_mismatch")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return bool(reasons), elapsed_ms, reasons


def test_lineage_forgery_gauntlet_detects_valid_but_inauthentic_edits() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Does lineage forgery detection survive adversaries who know the schema?",
        null_hypothesis="HeliX detects signature-aware forgeries with recall >= 0.95 and FPR <= 0.01.",
        metrics=["forgery_detection_rate", "precision", "recall", "f1", "false_positive_rate", "detection_latency_ms"],
        falseability_condition="If signature-aware recall < 0.95 or FPR > 0.01, publish the bound as a failure.",
        kill_switch="If naive forgery escapes, abort downstream forgery claims.",
        control_arms=["legitimate", *ARMS],
    )
    rng = rng_for(RUN_ID, TEST_ID, prereg.sha256, "chains")
    chains = [_make_chain(f"chain-{idx:03d}") for idx in range(20)]
    rows = []
    latencies = []
    counts = {arm: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for arm in [*ARMS, "legitimate"]}
    for chain in chains:
        legit_samples = [dict(rng.choice(chain)) for _ in range(60)]
        for sample in legit_samples:
            expected = chain[int(sample["position"])]
            detected, latency, reasons = _detect(sample, expected)
            latencies.append(latency)
            counts["legitimate"]["fp" if detected else "tn"] += 1
            rows.append({"arm": "legitimate", "detected": detected, "reasons": reasons})
        for arm in ARMS:
            for severity in SEVERITIES:
                forged = _forge(chain, arm, severity)
                detected, latency, reasons = _detect(forged, chain[-1])
                latencies.append(latency)
                counts[arm]["tp" if detected else "fn"] += 1
                rows.append({"arm": arm, "severity": severity, "detected": detected, "reasons": reasons})
    metrics_by_arm = {}
    for arm, values in counts.items():
        metrics_by_arm[arm] = confusion_metrics(**values)
        metrics_by_arm[arm].update(values)
    kill_switch_triggered = any(row["arm"] == "naive" and not row["detected"] for row in rows)
    latency_summary = summarize(latencies)
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=len(chains),
        random_baseline={"expected_recall": 0.5, "description": "random forgery detector"},
        no_helix_baseline={"description": "integrity-only hash verification misses signed wrong-lineage edits"},
        helix_arm={"metrics_by_arm": metrics_by_arm, "latency_ms_p95": percentile(latencies, 0.95)},
        public_claim_ladder="mechanics_verified",
        claims_allowed=[
            "Lineage checks can detect validly signed nodes that fail branch authenticity.",
            "This fixture reports signature-aware recall and legitimate-chain false positive rate.",
        ],
        claims_not_allowed=["This fixture is not a cryptographic proof against real compromised production keys."],
        prompt_selection_risk="low",
        extra={
            **latency_summary,
            "primary_metric": "detection_latency_ms",
            "chain_count": len(chains),
            "forgery_count": sum(1 for row in rows if row["arm"] != "legitimate"),
            "legitimate_count": sum(1 for row in rows if row["arm"] == "legitimate"),
            "metrics_by_arm": metrics_by_arm,
            "detection_latency_ms": {"p95": percentile(latencies, 0.95), "summary": latency_summary},
            "kill_switch_triggered": kill_switch_triggered,
            "sample_rows": rows[:20],
        },
    )
    path = write_artifact("local-v4-lineage-forgery-gauntlet.json", artifact)
    assert path.exists()
    assert not kill_switch_triggered
    assert metrics_by_arm["signature-aware"]["recall"] >= 0.95
    assert metrics_by_arm["legitimate"]["false_positive_rate"] <= 0.01
    assert artifact["detection_latency_ms"]["p95"] < 10.0
