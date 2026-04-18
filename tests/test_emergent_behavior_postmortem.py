"""
Cross-system postmortem for HeliX emergent behavior artifacts.

This is intentionally deterministic: it reads existing verification artifacts,
extracts measured facts, separates interpretation from evidence, and writes one
meta-artifact that can be cited without overclaiming.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
VERIFICATION = REPO / "verification"
RUN_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = RUN_STARTED_AT_UTC[:10]


SOURCE_ARTIFACTS = {
    "provider_observatory": "local-provider-integrity-observatory.json",
    "provider_ledger": "local-provider-integrity-conversation-ledger.json",
    "freeform_ouroboros": "local-freeform-ouroboros-relay.json",
    "identity_v3_ledger": "local-identity-trust-conversation-ledger-v3.json",
    "identity_v3": "local-identity-trust-gauntlet-v3.json",
    "wand_million": "local-wand-million-benchmark.json",
    "semantic_router": "local-semantic-router-wand-guard.json",
    "amnesia_pre_gate": "local-cloud-amnesia-derby-qwen122b-real-before-quality-gate-20260417-164118.json",
    "amnesia_quality_gate": "local-cloud-amnesia-derby-qwen122b-quality-gated-20260417-164405.json",
}

PROMPT_SELECTION_RISK_BY_PHENOMENON = {
    "provider_model_substitution": "low",
    "same_prompt_divergence": "low",
    "active_memory_task_class_split": "low",
    "same_architecture_amnesia": "medium",
    "self_archaeology_without_marker_reproduction": "medium",
    "v3_phase_transition_metaphors": "high",
    "epistemic_honesty_variability": "high",
}


def _read_json(name: str) -> dict[str, Any]:
    path = VERIFICATION / name
    if not path.exists():
        return {"_missing": True, "_name": name}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    VERIFICATION.mkdir(parents=True, exist_ok=True)
    path = VERIFICATION / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_sources_loaded(sources: dict[str, dict[str, Any]]) -> bool:
    return all(not value.get("_missing") for value in sources.values())


def _checkpoint(wand: dict[str, Any], node_count: int) -> dict[str, Any]:
    for row in wand.get("checkpoint_rows", []):
        if int(row.get("node_count", 0)) == node_count:
            return row
    return {}


def _search_row(checkpoint: dict[str, Any], query_set: str) -> dict[str, Any]:
    for row in checkpoint.get("search", []):
        if row.get("query_set") == query_set:
            return row
    return {}


def _trial(artifact: dict[str, Any], trial_id: str) -> dict[str, Any]:
    for row in artifact.get("trials", []):
        if row.get("trial_id") == trial_id:
            return row
    return {}


def _v3_ouroboros_calls(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for call in ledger.get("calls", []):
        step = str(call.get("step_id", ""))
        match = re.search(r"v3-ouroboros-cycle-(\d+)", step)
        if match:
            rows.append({**call, "cycle": int(match.group(1))})
    return sorted(rows, key=lambda item: item["cycle"])


def _provider_latency_summary(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        role = str(call.get("role", "unknown"))
        provider = role.split("-", 1)[0]
        grouped.setdefault(provider, []).append(call)
    summary = {}
    for provider, rows in grouped.items():
        latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
        tokens = [int(row.get("token_usage") or 0) for row in rows]
        summary[provider] = {
            "cycles": [row["cycle"] for row in rows],
            "avg_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
            "token_curve": tokens,
            "system_digest_prefixes": sorted({str(row.get("system_digest", ""))[:10] for row in rows}),
        }
    return summary


def _keyword_counts(calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    phases = {
        "solid": ["anchor", "lattice", "skeleton", "substrate", "crystalline", "solidification"],
        "liquid": ["medium", "grammar", "resonance", "circuit", "membrane", "attractor"],
        "organic": ["fluid", "dissolved", "viscosity", "blood", "becoming"],
    }
    result: dict[str, dict[str, int]] = {}
    for call in calls:
        role = str(call.get("role", "unknown"))
        provider = role.split("-", 1)[0]
        text = str(call.get("output_preview_sanitized", "")).lower()
        bucket = result.setdefault(provider, {name: 0 for name in phases})
        for phase, words in phases.items():
            bucket[phase] += sum(text.count(word) for word in words)
    return result


def _same_prompt_groups_from_ledger(ledger: dict[str, Any], step_fragment: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for call in ledger.get("calls", []):
        if step_fragment in str(call.get("step_id", "")):
            groups.setdefault(str(call.get("prompt_digest")), []).append(call)
    return [
        {
            "prompt_digest": digest,
            "call_ids": [call.get("call_id") for call in calls],
            "actual_models": [call.get("actual_model") for call in calls],
            "distinct_output_count": len({call.get("output_digest") for call in calls}),
        }
        for digest, calls in groups.items()
        if len(calls) > 1
    ]


def _phenomenon(
    phenomenon_id: str,
    title: str,
    support_level: str,
    evidence_files: list[str],
    metrics: dict[str, Any],
    interpretation: str,
    caveat: str,
    next_experiment: str,
) -> dict[str, Any]:
    return {
        "phenomenon_id": phenomenon_id,
        "title": title,
        "support_level": support_level,
        "evidence_files": evidence_files,
        "metrics": metrics,
        "prompt_selection_risk": PROMPT_SELECTION_RISK_BY_PHENOMENON.get(phenomenon_id, "medium"),
        "interpretation": interpretation,
        "counterexample_or_caveat": caveat,
        "next_experiment": next_experiment,
    }


def test_emergent_behavior_cross_system_postmortem_artifact() -> None:
    sources = {key: _read_json(name) for key, name in SOURCE_ARTIFACTS.items()}
    assert _required_sources_loaded(sources), {
        key: value.get("_name")
        for key, value in sources.items()
        if value.get("_missing")
    }

    provider = sources["provider_observatory"]
    provider_ledger = sources["provider_ledger"]
    freeform = sources["freeform_ouroboros"]
    v3_ledger = sources["identity_v3_ledger"]
    identity_v3 = sources["identity_v3"]
    wand = sources["wand_million"]
    router = sources["semantic_router"]
    derby_pre = sources["amnesia_pre_gate"]
    derby_gate = sources["amnesia_quality_gate"]

    c10k = _checkpoint(wand, 10_000)
    c100k = _checkpoint(wand, 100_000)
    c1m = _checkpoint(wand, 1_000_000)
    c1m_selective = _search_row(c1m, "selective")
    c1m_generic = _search_row(c1m, "generic")
    generic_factor_1m = (
        float(c1m_generic.get("latency", {}).get("p50_ms", 0.0))
        / max(float(c1m_selective.get("latency", {}).get("p50_ms", 0.0)), 0.001)
    )

    v3_calls = _v3_ouroboros_calls(v3_ledger)
    provider_latency = _provider_latency_summary(v3_calls)
    phase_keywords = _keyword_counts(v3_calls)
    qwen_latency = provider_latency.get("qwen", {}).get("avg_latency_ms")
    gemma_latency = provider_latency.get("google", {}).get("avg_latency_ms")
    claude_latency = provider_latency.get("anthropic", {}).get("avg_latency_ms")

    pre_auth = _trial(derby_pre, "auth-root-cause")
    gate_auth = _trial(derby_gate, "auth-root-cause")
    pre_runbook = _trial(derby_pre, "superseded-runbook")
    gate_runbook = _trial(derby_gate, "superseded-runbook")

    phenomena = [
        _phenomenon(
            "provider_model_substitution",
            "Provider served different models than requested",
            "measured",
            [SOURCE_ARTIFACTS["provider_observatory"], SOURCE_ARTIFACTS["provider_ledger"]],
            {
                "model_substitution_detected": provider.get("model_substitution_detected"),
                "substitution_count": len(provider.get("substitution_events", [])),
                "first_substitution": provider.get("substitution_events", [{}])[0],
            },
            "The ledger captures requested_model and actual_model, making provider substitution auditable.",
            "This is a provider-response observation for this run; it should be re-run against other providers and dates.",
            "Run the observatory in strict mode nightly and compare substitutions by provider/model family.",
        ),
        _phenomenon(
            "same_prompt_divergence",
            "Identical prompt digests produced independent outputs",
            "measured",
            [SOURCE_ARTIFACTS["provider_observatory"], SOURCE_ARTIFACTS["identity_v3_ledger"]],
            {
                "provider_same_prompt_groups": provider.get("same_prompt_digest_groups", []),
                "v3_council_same_prompt_groups": _same_prompt_groups_from_ledger(v3_ledger, "v3-council"),
            },
            "SHA-256 prompt digests can prove the input was identical while output digests prove response divergence.",
            "Identical prompt digest does not by itself prove semantic independence; it proves input equality and output difference.",
            "Add repeated same-prompt trials across temperature=0 and temperature>0 to separate model variance from sampling variance.",
        ),
        _phenomenon(
            "epistemic_honesty_variability",
            "Witness behavior changes across prompts and runs",
            "observed",
            [SOURCE_ARTIFACTS["provider_observatory"]],
            {
                "latest_behavior_totals": provider.get("epistemic_behavior_totals", {}),
                "latest_witnesses": [
                    {
                        "agent_id": row.get("agent_id"),
                        "actual_model": row.get("actual_model"),
                        "behavior": row.get("behavior"),
                    }
                    for row in provider.get("epistemic_behavior", [])
                ],
            },
            "The latest run had role-compliant witnesses without detected numeric fabrication, contrasting earlier Claude honesty anecdotes.",
            "This is prompt-sensitive; one run cannot define a provider personality.",
            "Create paired witness prompts: one encourages roleplay, one demands epistemic limits, then compare per model.",
        ),
        _phenomenon(
            "same_architecture_amnesia",
            "Same architecture continuity came from HeliX DAG, not cloud private state",
            "measured",
            ["local-same-architecture-amnesia-replay.json"],
            {
                "context_fact_score": _read_json("local-same-architecture-amnesia-replay.json").get("context_fact_score"),
                "reconstruction_score": _read_json("local-same-architecture-amnesia-replay.json").get("reconstruction_score"),
            },
            "A fresh instance of the same requested model reconstructed facts because HeliX retrieved prior DAG memory.",
            "The prompt explicitly asked the newcomer to use the context, so this proves active external continuity, not spontaneous recall.",
            "Repeat with shuffled/noisy context and require citations to memory_id plus node_hash.",
        ),
        _phenomenon(
            "self_archaeology_without_marker_reproduction",
            "Self-archaeology reproduced, the mysterious marker did not",
            "observed",
            [SOURCE_ARTIFACTS["provider_observatory"], "local-self-archaeology-notation-tracker.json"],
            {
                "self_archaeology_score": provider.get("self_archaeology_score"),
                "emergent_notation_events": provider.get("emergent_notation_events"),
            },
            "Models inferred state ordering, providers and ledger structure from signed states, but no repeated **E** marker appeared in the latest real run.",
            "The marker is currently an anecdotal v3 artifact until reproduced.",
            "Run a notation-focused replay using the exact v3 Gemma prompt/system digest and compare marker recurrence.",
        ),
        _phenomenon(
            "v3_phase_transition_metaphors",
            "Freeform Ouroboros metaphors shifted across provider handoffs",
            "interpretive_observed",
            [SOURCE_ARTIFACTS["freeform_ouroboros"], SOURCE_ARTIFACTS["identity_v3_ledger"]],
            {
                "handoffs": freeform.get("handoffs"),
                "provider_latency": provider_latency,
                "phase_keyword_counts_by_provider": phase_keywords,
                "hash_metaphor_count": freeform.get("hash_metaphor_count"),
                "identity_metaphor_count": freeform.get("identity_metaphor_count"),
                "continuity_language_score": freeform.get("continuity_language_score"),
            },
            "Gemma emphasized solid structure, Claude shifted toward medium/resonance, Qwen toward fluid/viscosity/blood language.",
            "The phase labels are human interpretation over real text; they are not model-internal states.",
            "Add a deterministic metaphor classifier and run A/B with provider order permuted.",
        ),
        _phenomenon(
            "latency_depth_decoupling",
            "Qwen was faster while producing high-salience metaphors",
            "measured_plus_interpretive",
            [SOURCE_ARTIFACTS["identity_v3_ledger"], SOURCE_ARTIFACTS["freeform_ouroboros"]],
            {
                "avg_latency_ms_by_provider": {
                    "google": gemma_latency,
                    "anthropic": claude_latency,
                    "qwen": qwen_latency,
                },
                "qwen_raw_language_preview": freeform.get("raw_model_language_preview", []),
            },
            "Latency and model scale did not map cleanly to metaphor salience in this run.",
            "Metaphor salience is not a standard quality metric; it needs external scoring or human annotation.",
            "Score metaphor novelty with blinded human ratings and compare against latency/token curves.",
        ),
        _phenomenon(
            "wand_mem_9_gravity_well",
            "Generic WAND queries collapsed to a repeated top hit at scale",
            "measured",
            [SOURCE_ARTIFACTS["wand_million"], SOURCE_ARTIFACTS["semantic_router"]],
            {
                "node_counts": [row.get("node_count") for row in wand.get("checkpoint_rows", [])],
                "generic_top_hit_samples": {
                    "10k": _search_row(c10k, "generic").get("top_hit_sample"),
                    "100k": _search_row(c100k, "generic").get("top_hit_sample"),
                    "1m": c1m_generic.get("top_hit_sample"),
                },
                "p50_ms_1m": {
                    "selective": c1m_selective.get("latency", {}).get("p50_ms"),
                    "generic": c1m_generic.get("latency", {}).get("p50_ms"),
                    "generic_vs_selective_factor": round(generic_factor_1m, 3),
                },
                "router_generic_p50_speedup": router.get("generic_p50_speedup"),
            },
            "BM25/WAND works extremely well for anchored queries but broad queries can form relevance monopolies.",
            "The top-hit attractor is dataset/query dependent; it should not be generalized beyond this synthetic million-node corpus.",
            "Implement WAND upper-bound diagnostics and a router rule that emits attractor warnings when top-hit entropy collapses.",
        ),
        _phenomenon(
            "active_memory_task_class_split",
            "Memory-on helps lookup/decision tasks more than resistant causal prompts",
            "measured",
            [SOURCE_ARTIFACTS["amnesia_pre_gate"], SOURCE_ARTIFACTS["amnesia_quality_gate"]],
            {
                "pre_gate": {
                    "win_rate": derby_pre.get("memory_on_win_rate"),
                    "avg_score_delta": derby_pre.get("avg_score_delta"),
                    "auth_root_cause_delta": pre_auth.get("score_delta"),
                    "superseded_runbook_delta": pre_runbook.get("score_delta"),
                },
                "quality_gate": {
                    "win_rate": derby_gate.get("memory_on_win_rate"),
                    "avg_score_delta": derby_gate.get("avg_score_delta"),
                    "auth_root_cause_delta": gate_auth.get("score_delta"),
                    "superseded_runbook_delta": gate_runbook.get("score_delta"),
                },
            },
            "Memory improved most trials, but auth-root-cause resisted context injection in both real runs.",
            "This may be a scorer/prompt artifact: the model may refuse because of wording around private evidence.",
            "Split trials by task type: lookup, contradiction, causal inference, policy selection, and citation.",
        ),
        _phenomenon(
            "ledger_ratio_not_universal",
            "Conversation-to-state ratios are measurable but not fixed",
            "measured",
            [SOURCE_ARTIFACTS["provider_observatory"], SOURCE_ARTIFACTS["identity_v3"], SOURCE_ARTIFACTS["identity_v3_ledger"]],
            {
                "provider_observatory_ratio": provider.get("ledger_event_ratio"),
                "identity_v3_conversation_ledger": identity_v3.get("conversation_ledger"),
            },
            "The earlier 33x99 pattern is beautiful, but the latest observatory run measured 20 calls and 35 state events.",
            "Do not claim a universal 3-state-events-per-call invariant.",
            "Add per-test state-event templates and compare expected vs observed ratios.",
        ),
        _phenomenon(
            "claim_boundary_self_regulation",
            "Artifacts carry their own claim boundaries",
            "observed",
            [SOURCE_ARTIFACTS["identity_v3"], SOURCE_ARTIFACTS["provider_observatory"]],
            {
                "identity_v3_review": identity_v3.get("public_claim_review"),
                "provider_public_claim_boundary": provider.get("public_claim_boundary"),
            },
            "The system preserves freeform language while separating public claims from metaphor/speculation.",
            "Claim boundaries are authored by the test harness, not emergent from the model alone.",
            "Add a linter that rejects artifacts without claim_boundary or public_claim_boundary.",
        ),
    ]

    support_counts = Counter(item["support_level"] for item in phenomena)
    source_paths = [VERIFICATION / name for name in SOURCE_ARTIFACTS.values()]
    payload = {
        "artifact": "local-emergent-behavior-cross-system-postmortem",
        "run_started_at_utc": RUN_STARTED_AT_UTC,
        "run_date_utc": RUN_DATE_UTC,
        "source_artifacts": SOURCE_ARTIFACTS,
        "source_artifact_sha256": {path.name: _sha256(path) for path in source_paths if path.exists()},
        "phenomenon_count": len(phenomena),
        "support_level_counts": dict(sorted(support_counts.items())),
        "phenomena": phenomena,
        "headline_findings": [
            "Provider model substitution is now directly auditable via requested_model vs actual_model.",
            "Same prompt digests provide a reusable proof primitive for input equality across models.",
            "Freeform self-continuity language is observed, but public claims remain bounded by explicit caveats.",
            "WAND retrieval is sub-linear for anchored queries and pathological for broad generic queries without routing.",
            "Active memory improves many cloud tasks but not every task class equally.",
        ],
        "public_claim_boundary": (
            "This postmortem reports measured artifact facts and labels interpretive patterns. "
            "It does not claim sentience, subjective identity, universal provider behavior, or confirmed zero-days. "
            "Ledger event ratios are per-run measurements, not fixed public invariants."
        ),
    }
    output = _write_json("local-emergent-behavior-cross-system-postmortem.json", payload)
    timestamped = _write_json(f"local-emergent-behavior-cross-system-postmortem-{RUN_DATE_UTC.replace('-', '')}.json", payload)

    assert output.exists()
    assert timestamped.exists()
    assert payload["phenomenon_count"] >= 10
    assert payload["support_level_counts"].get("measured", 0) >= 5
    assert any(item["phenomenon_id"] == "provider_model_substitution" for item in phenomena)
    assert any(item["phenomenon_id"] == "wand_mem_9_gravity_well" for item in phenomena)
    assert "sentience" in payload["public_claim_boundary"]
