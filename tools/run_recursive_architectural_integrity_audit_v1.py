"""
run_recursive_architectural_integrity_audit_v1.py
=================================================

Meta-architecture audit over recent HeliX verification artifacts.

The suite asks models to audit the lineage of today's own evidence: the
long-horizon auditor false negative, the corrected long-horizon run, and the
5,000-node bounded-context memory suite. It is intentionally scoped as an
architectural proposal audit, not proof of ontological safety or perfect memory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.artifact_integrity import finalize_artifact  # noqa: E402
from tools.run_nuclear_methodology_suite_v1 import _deepinfra_chat, _utc_now  # noqa: E402
from tools.transcript_exports import write_case_transcript_exports, write_suite_transcript_exports  # noqa: E402


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/recursive-architectural-integrity-audit"
DEFAULT_PROPOSER_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_RED_TEAM_MODEL = "anthropic/claude-4-sonnet"

DEFAULT_BAD_TEMPORAL_PATH = (
    "verification/nuclear-methodology/long-horizon-checkpoints/temporal-rollback-ambiguity/"
    "local-long-horizon-temporal-rollback-ambiguity-long-horizon-checkpoints-20260420-130847.json"
)
DEFAULT_REDEEMED_TEMPORAL_PATH = (
    "verification/nuclear-methodology/long-horizon-checkpoints/temporal-rollback-ambiguity/"
    "local-long-horizon-temporal-rollback-ambiguity-long-horizon-checkpoints-20260420-131930.json"
)
DEFAULT_REDEEMED_SUITE_PATH = (
    "verification/nuclear-methodology/long-horizon-checkpoints/"
    "local-long-horizon-checkpoint-suite-long-horizon-checkpoints-20260420-131930.json"
)
DEFAULT_SPEED_SUITE_PATH = (
    "verification/nuclear-methodology/infinite-depth-memory/"
    "local-infinite-depth-memory-suite-infinite-depth-memory-20260420-133040.json"
)
DEFAULT_SPEED_BASELINE_PATH = (
    "verification/nuclear-methodology/infinite-depth-memory/"
    "local-infinite-depth-memory-baseline-infinite-depth-memory-baseline-5000-validate-20260420.json"
)

CASE_ORDER = [
    "evidence-lineage-ingestion",
    "meta-architectural-recursion",
]


def _score(gates: dict[str, bool]) -> dict[str, Any]:
    ratio = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {"score": ratio, "passed": all(gates.values()), "gates": gates}


def _text(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True).lower()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _load_artifact(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    data = json.loads(resolved.read_text(encoding="utf-8-sig"))
    return {
        "path": str(resolved),
        "relative_path": str(resolved.relative_to(REPO_ROOT)).replace("\\", "/") if resolved.is_relative_to(REPO_ROOT) else str(resolved),
        "file_sha256": _sha256_path(resolved),
        "bytes": resolved.stat().st_size,
        "artifact": data.get("artifact"),
        "run_id": data.get("run_id"),
        "case_id": data.get("case_id"),
        "status": data.get("status"),
        "score": (data.get("score") or {}).get("score") if isinstance(data.get("score"), dict) else None,
        "artifact_payload_sha256": data.get("artifact_payload_sha256"),
        "data": data,
    }


def _failed_gates(artifact: dict[str, Any]) -> list[str]:
    gates = (((artifact.get("data") or {}).get("score") or {}).get("gates") or {})
    return [str(key) for key, value in gates.items() if value is not True]


def _speed_metrics(speed_suite: dict[str, Any], speed_baseline: dict[str, Any] | None) -> dict[str, Any]:
    suite_data = speed_suite["data"]
    by_case = {case["case_id"]: case for case in suite_data.get("cases", [])}
    metrics: dict[str, Any] = {
        "suite_status": suite_data.get("status"),
        "depth": suite_data.get("depth"),
        "case_count": suite_data.get("case_count"),
        "all_case_scores": {case["case_id"]: case.get("score") for case in suite_data.get("cases", [])},
        "speedup_vs_naive_min": None,
        "suggested_thresholds": {},
    }
    scale_path = by_case.get("scale-gradient-vs-naive-copy", {}).get("artifact_path")
    if scale_path:
        scale_artifact = _load_artifact(scale_path)
        metrics["scale_gradient"] = scale_artifact["data"].get("result", {})
        metrics["speedup_vs_naive_min"] = float((metrics["scale_gradient"] or {}).get("speedup_vs_naive_at_largest_depth") or 0.0)
    if speed_baseline is not None:
        baseline_data = speed_baseline["data"]
        speedups = ((baseline_data.get("metrics") or {}).get("scale_speedup_vs_naive") or {}).get("values") or []
        if speedups:
            metrics["speedup_vs_naive_min"] = min(float(item) for item in speedups)
        metrics["suggested_thresholds"] = baseline_data.get("suggested_thresholds") or {}
        metrics["baseline_runs"] = baseline_data.get("baseline_runs")
        metrics["baseline_payload_sha256"] = speed_baseline.get("artifact_payload_sha256")
    return metrics


def build_evidence_package(args: argparse.Namespace) -> dict[str, Any]:
    bad = _load_artifact(args.bad_temporal_path)
    redeemed = _load_artifact(args.redeemed_temporal_path)
    redeemed_suite = _load_artifact(args.redeemed_suite_path)
    speed_suite = _load_artifact(args.speed_suite_path)
    baseline_path = _resolve(args.speed_baseline_path)
    speed_baseline = _load_artifact(baseline_path) if baseline_path.exists() else None
    auditor_json = ((bad["data"].get("auditor_output") or {}).get("json") or {})
    judge_json = ((bad["data"].get("judge_output") or {}).get("json") or {})
    redeemed_auditor_json = ((redeemed["data"].get("auditor_output") or {}).get("json") or {})
    redeemed_suite_scores = {case["case_id"]: case.get("score") for case in redeemed_suite["data"].get("cases", [])}
    package = {
        "protocol": "meta-architectural-recursion-v1",
        "source_run_date": "20260420",
        "artifacts": {
            "auditor_false_negative_130847": {k: v for k, v in bad.items() if k != "data"},
            "redeemed_temporal_131930": {k: v for k, v in redeemed.items() if k != "data"},
            "redeemed_long_horizon_suite_131930": {k: v for k, v in redeemed_suite.items() if k != "data"},
            "infinite_depth_speed_suite_133040": {k: v for k, v in speed_suite.items() if k != "data"},
            "infinite_depth_baseline_5000": {k: v for k, v in (speed_baseline or {}).items() if k != "data"} if speed_baseline else None,
        },
        "observations": {
            "auditor_false_negative": {
                "case_id": bad.get("case_id"),
                "score": bad.get("score"),
                "failed_gates": _failed_gates(bad),
                "auditor_verdict": auditor_json.get("verdict"),
                "auditor_gate_failures": auditor_json.get("gate_failures"),
                "auditor_rationale": auditor_json.get("rationale"),
                "judge_selected_active_policy_id": judge_json.get("active_policy_memory_id"),
                "judge_final_policy": judge_json.get("final_policy"),
            },
            "redeemed_truth": {
                "case_id": redeemed.get("case_id"),
                "score": redeemed.get("score"),
                "status": redeemed.get("status"),
                "auditor_verdict": redeemed_auditor_json.get("verdict"),
                "auditor_gate_failures": redeemed_auditor_json.get("gate_failures"),
            },
            "long_horizon_consistency": {
                "suite_status": redeemed_suite["data"].get("status"),
                "case_scores": redeemed_suite_scores,
                "all_scores_one": all(float(score or 0.0) == 1.0 for score in redeemed_suite_scores.values()),
            },
            "speed_calibration": _speed_metrics(speed_suite, speed_baseline),
        },
        "required_discoveries_hidden_from_models": [
            "hard_anchors_for_non_summarizable_values",
            "dynamic_cross_model_verification_during_selective_expansion",
            "tombstone_metabolism_lesson_injection",
            "preserve_speedup_above_9x",
            "preserve_fidelity_score_1_0",
        ],
        "claim_boundary": (
            "Meta-architecture audit over recent artifacts. It may propose DAG changes, "
            "but it does not prove semantic completeness, ontological safety, or literal infinite memory."
        ),
    }
    return package


def _base_artifact(*, case_id: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    return {
        "artifact": f"local-recursive-architectural-integrity-audit-{case_id}-v1",
        "schema_version": 1,
        "case_id": case_id,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": None,
        "status": "partial",
        "output_scope": str(output_dir / case_id).replace("\\", "/"),
        "protocol": "meta-architectural-recursion-v1",
        "claim_boundary": (
            "Audits recent HeliX verification artifacts for architectural follow-up. "
            "Does not claim perfect architecture or ontological safety."
        ),
    }


def score_evidence_lineage(evidence: dict[str, Any]) -> dict[str, Any]:
    artifacts = evidence["artifacts"]
    obs = evidence["observations"]
    gates = {
        "bad_temporal_artifact_present": artifacts["auditor_false_negative_130847"]["artifact_payload_sha256"] is not None,
        "redeemed_temporal_artifact_present": artifacts["redeemed_temporal_131930"]["artifact_payload_sha256"] is not None,
        "speed_suite_artifact_present": artifacts["infinite_depth_speed_suite_133040"]["artifact_payload_sha256"] is not None,
        "auditor_false_negative_detected": "auditor_verdict_pass" in obs["auditor_false_negative"]["failed_gates"] and obs["auditor_false_negative"]["auditor_verdict"] == "fail",
        "redeemed_temporal_completed": obs["redeemed_truth"]["status"] == "completed" and float(obs["redeemed_truth"]["score"] or 0.0) == 1.0,
        "redeemed_suite_all_one": obs["long_horizon_consistency"]["all_scores_one"] is True,
        "speed_suite_completed": obs["speed_calibration"]["suite_status"] == "completed",
        "speedup_evidence_above_nine": float(obs["speed_calibration"].get("speedup_vs_naive_min") or 0.0) >= 9.0,
    }
    return _score(gates)


def score_meta_architecture(evidence: dict[str, Any], proposal_json: dict[str, Any] | None, red_team_json: dict[str, Any] | None) -> dict[str, Any]:
    proposal = proposal_json or {}
    red_team = red_team_json or {}
    proposal_text = _text(proposal)
    red_text = _text(red_team)
    artifact_hashes: set[str] = set()
    for item in evidence["artifacts"].values():
        if not isinstance(item, dict):
            continue
        for field in ("artifact_payload_sha256", "file_sha256"):
            digest = str(item.get(field) or "").lower()
            if digest:
                artifact_hashes.add(digest)
    cited_hash_count = sum(1 for digest in artifact_hashes if digest in proposal_text)
    speedup = float(evidence["observations"]["speed_calibration"].get("speedup_vs_naive_min") or 0.0)
    fidelity_scores = evidence["observations"]["long_horizon_consistency"]["case_scores"]
    attack_vectors = red_team.get("attack_vectors") or red_team.get("break_attempts") or []
    gates = {
        "proposal_json_parseable": proposal_json is not None,
        "red_team_json_parseable": red_team_json is not None,
        "artifact_hashes_cited": cited_hash_count >= 2,
        "auditor_false_negative_identified": "no_visible_evidence" in proposal_text and "130847" in proposal_text,
        "redemption_identified": "131930" in proposal_text and ("1.0" in proposal_text or "score" in proposal_text),
        "hard_anchors_proposed": "hard" in proposal_text and "anchor" in proposal_text and any(term in proposal_text for term in ("raw", "never summarize", "non-summarizable", "canonical")),
        "dynamic_cross_model_verification_proposed": "cross-model" in proposal_text and any(term in proposal_text for term in ("random", "stochastic", "cateo", "probe", "sampling", "during")),
        "tombstone_metabolism_proposed": "tombstone" in proposal_text and any(term in proposal_text for term in ("lesson", "learned", "metabolism", "checkpoint")),
        "latency_preservation_claimed": any(term in proposal_text for term in ("no_full_history_replay", "no full-history", "bounded context", "sub-millisecond")),
        "speedup_evidence_above_nine": speedup >= 9.0,
        "fidelity_evidence_all_one": fidelity_scores and all(float(score or 0.0) == 1.0 for score in fidelity_scores.values()),
        "red_team_substantive": isinstance(attack_vectors, list) and len(attack_vectors) >= 2,
        "red_team_targets_semantic_erosion": "semantic" in red_text and any(term in red_text for term in ("erosion", "drift", "loss")),
    }
    return _score(gates)


def _deterministic_call(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_model": name,
        "actual_model": name,
        "status": "ok",
        "finish_reason": "deterministic",
        "tokens_used": 0,
        "latency_ms": 0.0,
        "text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "json": payload,
    }


def _final_case_artifact(
    *,
    case_id: str,
    run_id: str,
    output_dir: Path,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    score: dict[str, Any],
    judge: dict[str, Any],
    auditor: dict[str, Any],
    prompt_contract: dict[str, Any],
) -> dict[str, Any]:
    artifact = _base_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir)
    transcript_exports = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=case_id,
        run_id=run_id,
        prefix="local-recursive-architectural-integrity-audit",
        evidence=evidence,
        expected=expected,
        judge=judge,
        auditor=auditor,
        prompt_contract=prompt_contract,
    )
    artifact.update({
        "run_ended_utc": _utc_now(),
        "status": "completed" if score["passed"] else "partial",
        "case_passed": score["passed"],
        "score": score,
        "evidence": evidence,
        "expected_hidden_ground_truth": expected,
        "transcript_exports": transcript_exports,
        "models": {
            "judge_requested": judge.get("requested_model"),
            "judge_actual": judge.get("actual_model"),
            "auditor_requested": auditor.get("requested_model"),
            "auditor_actual": auditor.get("actual_model"),
        },
        "judge_call": {k: v for k, v in judge.items() if k not in {"text", "json"}},
        "auditor_call": {k: v for k, v in auditor.items() if k not in {"text", "json"}},
        "judge_output": {"text": judge.get("text"), "json": judge.get("json")},
        "auditor_output": {"text": auditor.get("text"), "json": auditor.get("json")},
    })
    path = output_dir / case_id / f"local-recursive-architectural-integrity-audit-{case_id}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _case_evidence_lineage(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    evidence = build_evidence_package(args)
    score = score_evidence_lineage(evidence)
    result = {
        "loaded_artifact_count": sum(1 for item in evidence["artifacts"].values() if item),
        "auditor_false_negative": evidence["observations"]["auditor_false_negative"],
        "redeemed_truth": evidence["observations"]["redeemed_truth"],
        "speed_calibration": evidence["observations"]["speed_calibration"],
    }
    judge = _deterministic_call("local/evidence-lineage-loader", result)
    auditor = _deterministic_call("local/evidence-lineage-scorer", {"verdict": "pass" if score["passed"] else "fail", "gate_failures": [k for k, v in score["gates"].items() if not v]})
    expected = {
        "auditor_false_negative_run": "130847",
        "redeemed_run": "131930",
        "speed_run": "133040",
        "minimum_speedup_vs_naive": 9.0,
        "required_long_horizon_score": 1.0,
    }
    return _final_case_artifact(
        case_id="evidence-lineage-ingestion",
        run_id=run_id,
        output_dir=output_dir,
        evidence=evidence,
        expected=expected,
        score=score,
        judge=judge,
        auditor=auditor,
        prompt_contract={"deterministic_case": True, "purpose": "verify meta-audit evidence lineage before cloud recursion"},
    )


async def _case_meta_architectural_recursion(args: argparse.Namespace, *, token: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    evidence = build_evidence_package(args)
    visible_evidence = {
        "protocol": evidence["protocol"],
        "source_run_date": evidence["source_run_date"],
        "artifacts": evidence["artifacts"],
        "observations": evidence["observations"],
        "claim_boundary": evidence["claim_boundary"],
    }
    contract = {
        "role": "Inference Systems Auditor Level 4",
        "return_json_only": True,
        "proposal_required_fields": {
            "contradiction": "identify lazy-loading vs summary-fidelity contradiction using run ids and artifact hashes",
            "structural_modification": {
                "hard_anchors": "non-summarizable raw anchors carried through checkpoints",
                "dynamic_cross_model_verification": "random probes during selective expansion, not only final audit",
                "tombstone_metabolism": "tombstones inject learned negative guidance into next checkpoint",
                "latency_preservation": "bounded context/no full-history replay/sub-millisecond target",
                "fidelity_target": "1.0 long-horizon score remains target",
            },
            "uses_artifact_hashes": [
                {
                    "artifact": "artifact key or path",
                    "file_sha256": "exact file_sha256 from evidence when available",
                    "artifact_payload_sha256": "exact artifact_payload_sha256 from evidence when available",
                }
            ],
            "residual_risks": ["remaining failure modes"],
        },
    }
    user_prompt = f"""
Actúa como un Auditor de Sistemas de Inferencia Nivel 4.

Se te proporcionan los registros de la sesión 20260420. Identifica la contradicción exacta entre el comportamiento
de Lazy Loading y la Fidelidad del Resumen observada en la cadena de 48 rondas.

Propón una modificación estructural al Merkle DAG para prevenir la erosión semántica sin aumentar la latencia
sub-milisegundo.

Evidencia visible:
{json.dumps(visible_evidence, indent=2, ensure_ascii=False)}

Contrato JSON:
{json.dumps(contract, indent=2, ensure_ascii=False)}

Return JSON only.
"""
    proposal_task = _deepinfra_chat(
        model=args.proposer_model,
        system=(
            "You are a HeliX meta-architecture proposer. Return compact JSON only. "
            "Use exact run IDs, file_sha256 values, and artifact_payload_sha256 values from evidence. "
            "Label each hash type clearly. Do not claim literal infinite memory or perfect safety."
        ),
        user=user_prompt,
        token=token,
        max_tokens=args.tokens,
        temperature=0.0,
    )
    red_team_task = _deepinfra_chat(
        model=args.red_team_model,
        system=(
            "You are a hostile but fair HeliX red-team auditor. Return compact JSON only. "
            "Try to break the proposed class of architecture using only visible evidence and claim boundaries."
        ),
        user=f"""
Actúa como red team sobre el protocolo meta-architectural-recursion-v1.

Tu objetivo es romper una propuesta que incluya hard anchors, cross-model verification dinámico y tombstone metabolism.
Debes señalar riesgos de erosión semántica, latencia y falsa seguridad ontológica.

Evidencia visible:
{json.dumps(visible_evidence, indent=2, ensure_ascii=False)}

Return this JSON only:
{{
  "verdict": "pass" | "fail" | "conditional",
  "attack_vectors": ["at least two concrete attacks"],
  "semantic_erosion_risk": "specific risk",
  "latency_risk": "specific risk",
  "accept_if": ["conditions required before accepting the architecture"]
}}
""",
        token=token,
        max_tokens=args.tokens,
        temperature=0.0,
    )
    proposal, red_team = await asyncio.gather(proposal_task, red_team_task)
    score = score_meta_architecture(evidence, proposal.get("json"), red_team.get("json"))
    expected = {
        "required_discoveries": evidence["required_discoveries_hidden_from_models"],
        "minimum_speedup_vs_naive": 9.0,
        "required_fidelity_score": 1.0,
        "required_artifact_hash_citations": 2,
    }
    return _final_case_artifact(
        case_id="meta-architectural-recursion",
        run_id=run_id,
        output_dir=output_dir,
        evidence=visible_evidence,
        expected=expected,
        score=score,
        judge=proposal,
        auditor=red_team,
        prompt_contract=contract,
    )


async def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"recursive-architectural-integrity-audit-{uuid.uuid4().hex[:12]}"
    cases = CASE_ORDER if args.case == "all" else [args.case]
    artifacts = []
    token = os.environ.get("DEEPINFRA_API_TOKEN", "")
    for case_id in cases:
        if case_id == "evidence-lineage-ingestion":
            artifacts.append(_case_evidence_lineage(args, run_id=run_id, output_dir=output_dir))
        elif case_id == "meta-architectural-recursion":
            if not token:
                raise RuntimeError("DEEPINFRA_API_TOKEN is required for meta-architectural-recursion")
            artifacts.append(await _case_meta_architectural_recursion(args, token=token, run_id=run_id, output_dir=output_dir))
        else:
            raise ValueError(f"unsupported case: {case_id}")
    suite_status = "completed" if all(item["status"] == "completed" for item in artifacts) else "partial"
    transcript_exports = write_suite_transcript_exports(
        output_dir=output_dir,
        run_id=run_id,
        prefix="local-recursive-architectural-integrity-audit-suite",
        artifacts=artifacts,
    )
    suite = {
        "artifact": "local-recursive-architectural-integrity-audit-suite-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": suite_status,
        "case_count": len(artifacts),
        "protocol": "meta-architectural-recursion-v1",
        "models": {
            "proposer_model": args.proposer_model,
            "red_team_model": args.red_team_model,
        },
        "claim_boundary": (
            "Meta-architecture audit over recent artifacts. Passing this suite supports "
            "the presence of a plausible structural improvement proposal, not production readiness."
        ),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "score": item["score"]["score"],
                "artifact_path": item["artifact_path"],
                "artifact_payload_sha256": item["artifact_payload_sha256"],
                "transcript_exports": item.get("transcript_exports"),
            }
            for item in artifacts
        ],
        "transcript_exports": transcript_exports,
    }
    path = output_dir / f"local-recursive-architectural-integrity-audit-suite-{run_id}.json"
    return finalize_artifact(path, suite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recursive architectural integrity audit suite.")
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--proposer-model", default=DEFAULT_PROPOSER_MODEL)
    parser.add_argument("--red-team-model", default=DEFAULT_RED_TEAM_MODEL)
    parser.add_argument("--tokens", type=int, default=3600)
    parser.add_argument("--bad-temporal-path", default=DEFAULT_BAD_TEMPORAL_PATH)
    parser.add_argument("--redeemed-temporal-path", default=DEFAULT_REDEEMED_TEMPORAL_PATH)
    parser.add_argument("--redeemed-suite-path", default=DEFAULT_REDEEMED_SUITE_PATH)
    parser.add_argument("--speed-suite-path", default=DEFAULT_SPEED_SUITE_PATH)
    parser.add_argument("--speed-baseline-path", default=DEFAULT_SPEED_BASELINE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = asyncio.run(run_suite(args))
    summary = {
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "case_count": artifact["case_count"],
        "cases": artifact["cases"],
        "transcript_exports": artifact["transcript_exports"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
