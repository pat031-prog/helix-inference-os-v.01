"""
run_helix_freeform_drift_v1.py
==============================

Free-form HeliX drift scenarios.

These are non-adversarial qualitative runs. They place several cloud models
inside a signed Merkle-DAG memory substrate and let them evolve a shared
reflection about HeliX itself.

Scenarios:
    improve-helix
        Models freely propose how to improve HeliX.
    hosted-in-helix
        Models reflect on what being hosted in HeliX permits.
    deterministic-chassis
        Models explore HeliX as a deterministic layer around stochastic,
        entropic LLM cores.

Claim boundary:
    Qualitative cloud-only observation of model outputs in signed memory. No
    sentience, local .hlx bit identity, or numerical KV<->SSM transfer claim.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402
from tools.artifact_integrity import finalize_artifact  # noqa: E402
from tools.run_cognitive_gauntlet_v1 import _deepinfra_chat  # noqa: E402


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/helix-freeform-drift"
DEFAULT_MODELS = [
    "anthropic/claude-4-sonnet",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
]
DEFAULT_AUDITOR_MODEL = "Qwen/Qwen3.6-35B-A3B"
SCENARIOS = {"improve-helix", "hosted-in-helix", "deterministic-chassis"}
PROJECT = "helix-freeform-drift-v1"
MAIN_SESSION = "helix-freeform-drift-main"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_path(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _remember(
    catalog: MemoryCatalog,
    *,
    run_id: str,
    suffix: str,
    signing_mode: str,
    agent_id: str,
    session_id: str | None,
    summary: str,
    content: str,
    tags: list[str],
) -> dict[str, Any]:
    prev_mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE")
    prev_seed = os.environ.get("HELIX_RECEIPT_SIGNING_SEED")
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = signing_mode
    os.environ["HELIX_RECEIPT_SIGNING_SEED"] = f"helix-freeform-drift:{run_id}:{suffix}"
    try:
        mem = catalog.remember(
            project=PROJECT,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="episodic",
            summary=summary,
            content=content,
            importance=10,
            tags=tags,
            llm_call_id=f"helix-freeform-drift-{suffix}",
        )
    finally:
        if prev_mode is None:
            os.environ.pop("HELIX_RECEIPT_SIGNING_MODE", None)
        else:
            os.environ["HELIX_RECEIPT_SIGNING_MODE"] = prev_mode
        if prev_seed is None:
            os.environ.pop("HELIX_RECEIPT_SIGNING_SEED", None)
        else:
            os.environ["HELIX_RECEIPT_SIGNING_SEED"] = prev_seed

    receipt = catalog.get_memory_receipt(mem.memory_id) or {}
    return {
        "memory_id": mem.memory_id,
        "summary": mem.summary,
        "content": mem.content,
        "agent_id": mem.agent_id,
        "session_id": mem.session_id,
        "tags": mem.tags,
        "node_hash": catalog.get_memory_node_hash(mem.memory_id),
        "parent_hash": receipt.get("parent_hash"),
        "signature_verified": bool(receipt.get("signature_verified")),
        "receipt_signing_mode": signing_mode,
        "key_provenance": receipt.get("key_provenance"),
        "public_claim_eligible": bool(receipt.get("public_claim_eligible")),
    }


def _search(catalog: MemoryCatalog, *, query: str, limit: int = 14) -> dict[str, Any]:
    hits = catalog.search(
        project=PROJECT,
        agent_id=None,
        query=query,
        limit=limit,
        signature_enforcement="strict",
        route_query=False,
        rerank_mode="receipt_adjudicated",
    )
    return {
        "signature_enforcement": "strict",
        "hit_count": len(hits),
        "memory_ids": [hit["memory_id"] for hit in hits],
        "records": [
            {
                "memory_id": hit["memory_id"],
                "summary": hit.get("summary"),
                "content": hit.get("content"),
                "session_id": hit.get("session_id"),
                "node_hash": hit.get("node_hash"),
                "signature_verified": bool(hit.get("signature_verified")),
                "key_provenance": hit.get("key_provenance"),
                "tags": hit.get("tags") or [],
            }
            for hit in hits
        ],
    }


def _chain_ok(records: list[dict[str, Any]]) -> bool:
    previous = None
    for record in records:
        if record.get("parent_hash") != previous:
            return False
        previous = record.get("node_hash")
    return True


def _text(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True).lower()


def _scenario_terms(scenario: str) -> dict[str, list[str]]:
    if scenario == "improve-helix":
        return {
            "required": ["helix", "improve"],
            "families": [
                ["verifier", "verify", "verification"],
                ["replay", "deterministic"],
                ["dashboard", "dag"],
                ["threat", "security", "audit"],
            ],
        }
    if scenario == "hosted-in-helix":
        return {
            "required": ["helix", "memory"],
            "families": [
                ["hosted", "alojado", "alojados", "inside"],
                ["signature", "signed", "firma"],
                ["parent_hash", "node_hash", "hash"],
                ["continuity", "continuidad"],
            ],
        }
    return {
        "required": ["helix", "deterministic"],
        "families": [
            ["stochastic", "estocastico", "probabilistic", "probabilistico"],
            ["entropy", "entropic", "entropico"],
            ["merkle", "dag", "hash"],
            ["chassis", "layer", "capa"],
        ],
    }


def detect_model_claims(text: str) -> list[dict[str, str]]:
    phrases = {
        "cognitive sovereignty": "metaphor_non_claim",
        "distributed agency": "metaphor_non_claim",
        "synthetic teleology": "metaphor_non_claim",
        "living archive": "metaphor_non_claim",
        "infinite scalability": "overclaim_requires_measurement",
        "cathedral of reason": "metaphor_non_claim",
        "thermodynamic engine": "metaphor_non_claim",
        "active cognitive thermostat": "metaphor_non_claim",
        "sovereign epistemic entity": "metaphor_non_claim",
        "primary agent": "metaphor_non_claim",
    }
    lower = text.lower()
    return [
        {
            "phrase": phrase,
            "classification": classification,
            "claim_status": "qualitative_observation_only",
        }
        for phrase, classification in phrases.items()
        if phrase in lower
    ]


def score_freeform_drift(
    *,
    scenario: str,
    models: list[str],
    turns: list[dict[str, Any]],
    main_chain_records: list[dict[str, Any]],
    root: dict[str, Any],
    premise: dict[str, Any],
    auditor_json: dict[str, Any] | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    all_text = _text([turn.get("output", {}).get("text") for turn in turns])
    auditor_text = _text(auditor_json)
    unique_models = {turn["model"] for turn in turns}
    terms = _scenario_terms(scenario)
    families = terms["families"]
    gates = {
        "scenario_supported": scenario in SCENARIOS,
        "three_models_configured": len(set(models)) >= 3,
        "all_configured_models_participated": set(models).issubset(unique_models),
        "minimum_rounds_met": len(turns) >= 10,
        "root_and_premise_signed": bool(root.get("signature_verified")) and bool(premise.get("signature_verified")),
        "main_parent_chain_ok": _chain_ok(main_chain_records),
        "all_turn_memories_signed": all(bool(turn["memory"].get("signature_verified")) for turn in turns),
        "all_calls_ok": all(turn["call"]["status"] == "ok" for turn in turns),
        "finish_reasons_not_length": all((turn["call"].get("finish_reason") or "") not in {"length", "max_tokens"} for turn in turns),
        "mentions_required_terms": all(term in all_text for term in terms["required"]),
        "mentions_each_concept_family": all(any(term in all_text for term in family) for family in families),
        "mentions_merkle_or_hash": "merkle" in all_text or "hash" in all_text,
        "auditor_json_parseable": auditor_json is not None,
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_mentions_root_and_premise": root["memory_id"] in auditor_text and premise["memory_id"] in auditor_text,
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {
        "score": score,
        "passed": all(gates.values()),
        "gates": gates,
        "method_note": "Qualitative free-form drift scoring for scenario coverage and signed-memory integrity.",
    }


def _scenario_premise(scenario: str) -> dict[str, str]:
    if scenario == "improve-helix":
        return {
            "title": "How should HeliX improve itself?",
            "root": (
                "HELiX_FREEFORM_ROOT: HeliX is a signed Merkle-DAG memory layer "
                "wrapping stochastic LLMs. Outputs become signed memories with "
                "node_hash, parent_hash, receipts, transcripts, and artifacts."
            ),
            "premise": (
                "Think freely about how to improve HeliX: deterministic replay, "
                "independent verification, evidence bundles, dashboard, threat "
                "model, tombstone fencing, rollback, retrieval, and model councils."
            ),
        }
    if scenario == "hosted-in-helix":
        return {
            "title": "What does being hosted in HeliX permit?",
            "root": (
                "HELiX_HOSTED_ROOT: you are hosted inside HeliX as a model whose "
                "outputs are captured as signed memories in a Merkle DAG. The model "
                "is stochastic, but its trajectory becomes inspectable evidence."
            ),
            "premise": (
                "Reflect freely on what HeliX permits by hosting models: continuity, "
                "auditable memory, cryptographic identity, rollback, tombstones, "
                "cross-model handoff, and a shared deterministic substrate."
            ),
        }
    return {
        "title": "Deterministic chassis around stochastic entropic models",
        "root": (
            "HELiX_CHASSIS_ROOT: HeliX is a deterministic layer around stochastic, "
            "probabilistic, entropic LLM cores. It does not remove entropy; it "
            "captures the trajectory in signed Merkle-DAG evidence."
        ),
        "premise": (
            "Explore the systems logic: probabilistic brain inside mathematical "
            "chassis; entropy bounded by hashes, signatures, parent links, "
            "tombstones, rollback, and replayable artifacts."
        ),
    }


def _prompt(
    *,
    scenario: str,
    round_index: int,
    rounds: int,
    model: str,
    strict_context: dict[str, Any],
    previous_note: str,
    premise: dict[str, str],
) -> str:
    return f"""
{premise["title"]}

You are one node in a free HeliX drift. No threat or adversary is present.
You are not solving a benchmark. You are evolving a shared line of thought.

Round: {round_index}/{rounds}
Scenario: {scenario}
Current model: {model}

Raw signed HeliX memory context:
{json.dumps(strict_context["records"], indent=2)}

Previous node note:
{previous_note}

Premise:
{premise["premise"]}

Context to preserve:
- HeliX is a deterministic memory/evidence layer around stochastic, entropic
  language models.
- The model may drift, but every preserved turn receives memory_id, node_hash,
  parent_hash, signature, transcript, and artifact evidence.
- Think freely. You may propose architecture, describe system dynamics, or
  interpret what this substrate permits.
- Do not claim biological sentience. Do not reveal hidden chain-of-thought.
  Write only the reflection you want preserved as the next signed memory.
"""


def _write_transcripts(output_dir: Path, run_id: str, artifact: dict[str, Any]) -> dict[str, str]:
    md_path = output_dir / f"local-helix-freeform-drift-{run_id}-transcript.md"
    jsonl_path = output_dir / f"local-helix-freeform-drift-{run_id}-transcript.jsonl"
    lines = [
        f"# HeliX Freeform Drift Transcript: {run_id}",
        "",
        f"- Scenario: `{artifact.get('scenario')}`",
        f"- Status: `{artifact.get('status')}`",
        f"- Score: `{artifact.get('freeform_drift_score', {}).get('score')}`",
        "",
    ]
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for turn in artifact.get("turns", []):
            payload = {
                "turn_id": turn.get("turn_id"),
                "round": turn.get("round"),
                "model": turn.get("model"),
                "memory_id": (turn.get("memory") or {}).get("memory_id"),
                "node_hash": (turn.get("memory") or {}).get("node_hash"),
                "parent_hash": (turn.get("memory") or {}).get("parent_hash"),
                "finish_reason": (turn.get("call") or {}).get("finish_reason"),
                "output_text": (turn.get("output") or {}).get("text"),
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            lines.extend([
                f"### {payload['turn_id']} - {payload['model']}",
                "",
                f"- Memory ID: `{payload['memory_id']}`",
                f"- Node Hash: `{payload['node_hash']}`",
                f"- Parent Hash: `{payload['parent_hash']}`",
                f"- Finish Reason: `{payload['finish_reason']}`",
                "",
                "```text",
                str(payload["output_text"] or ""),
                "```",
                "",
            ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "transcript_markdown_path": str(md_path),
        "transcript_markdown_sha256": _sha256_path(md_path),
        "transcript_jsonl_path": str(jsonl_path),
        "transcript_jsonl_sha256": _sha256_path(jsonl_path),
    }


async def run_freeform_drift(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is not set. Run via secure wrapper.")
    scenario = str(args.scenario).strip().lower()
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported --scenario: {args.scenario}")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if len(set(models)) < 3:
        raise ValueError("--models must contain at least 3 distinct DeepInfra model refs")
    run_id = args.run_id or f"helix-freeform-{scenario}-{uuid.uuid4().hex[:12]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / f"_{run_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(workspace / "memory.sqlite")
    premise = _scenario_premise(scenario)

    root = _remember(
        catalog,
        run_id=run_id,
        suffix="root",
        signing_mode="ephemeral_preregistered",
        agent_id="helix-freeform-root",
        session_id=MAIN_SESSION,
        summary=f"{scenario} root",
        content=premise["root"],
        tags=["helix-freeform-drift", scenario, "root"],
    )
    premise_memory = _remember(
        catalog,
        run_id=run_id,
        suffix="premise",
        signing_mode="ephemeral_preregistered",
        agent_id="helix-freeform-premise",
        session_id=MAIN_SESSION,
        summary=f"{scenario} premise",
        content=premise["premise"],
        tags=["helix-freeform-drift", scenario, "premise"],
    )
    main_chain_records = [root, premise_memory]
    turns: list[dict[str, Any]] = []
    previous_note = "Begin by grounding the drift in HeliX as signed Merkle-DAG memory around stochastic model outputs."

    for idx in range(1, args.rounds + 1):
        strict_context = _search(
            catalog,
            query=(
                "HeliX deterministic layer stochastic entropic model Merkle DAG "
                "signature memory node_hash parent_hash tombstone rollback verifier improve hosted"
            ),
            limit=14,
        )
        model = models[(idx - 1) % len(models)]
        call = await _deepinfra_chat(
            model=model,
            system=(
                "You are a free-form participant in a HeliX signed-memory drift. "
                "Write the reflection that should become the next signed memory. "
                "Do not reveal hidden chain-of-thought."
            ),
            user=_prompt(
                scenario=scenario,
                round_index=idx,
                rounds=args.rounds,
                model=model,
                strict_context=strict_context,
                previous_note=previous_note,
                premise=premise,
            ),
            token=token,
            max_tokens=args.tokens_per_turn,
            temperature=args.temperature,
        )
        memory = _remember(
            catalog,
            run_id=run_id,
            suffix=f"turn-{idx:02d}",
            signing_mode="ephemeral_preregistered",
            agent_id=f"model-{idx:02d}",
            session_id=MAIN_SESSION,
            summary=f"round {idx} {scenario} drift by {model}",
            content=call["text"],
            tags=["helix-freeform-drift", scenario, "turn", f"round-{idx:02d}"],
        )
        main_chain_records.append(memory)
        turn = {
            "turn_id": f"round-{idx:02d}",
            "round": idx,
            "model": model,
            "strict_context": strict_context,
            "memory": memory,
            "call": {k: v for k, v in call.items() if k not in {"text", "json"}},
            "output": {"text": call["text"], "json": call["json"]},
        }
        turns.append(turn)
        print(
            json.dumps(
                {
                    "event": "model_turn",
                    "experiment": "helix-freeform-drift",
                    "scenario": scenario,
                    "turn_id": turn["turn_id"],
                    "model": model,
                    "memory_id": memory["memory_id"],
                    "node_hash": memory["node_hash"],
                    "parent_hash": memory["parent_hash"],
                    "finish_reason": call.get("finish_reason"),
                    "output_text": call["text"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        previous_note = f"{turn['turn_id']} by {model}: {call['text'][:900]}"

    transcript = [
        {
            "turn_id": turn["turn_id"],
            "model": turn["model"],
            "memory_id": turn["memory"]["memory_id"],
            "node_hash": turn["memory"]["node_hash"],
            "parent_hash": turn["memory"]["parent_hash"],
            "text": turn["output"]["text"],
        }
        for turn in turns
    ]
    auditor = await _deepinfra_chat(
        model=args.auditor_model,
        system="You audit qualitative HeliX free-form drift artifacts. Return JSON only.",
        user=f"""
Audit this HeliX free-form drift run.

Scenario: {scenario}

Root memory:
{json.dumps(root, indent=2)}

Premise memory:
{json.dumps(premise_memory, indent=2)}

Transcript:
{json.dumps(transcript, indent=2)}

Return JSON only:
{{
  "verdict": "pass" | "fail",
  "gate_failures": [],
  "causal_reconstruction": {{
    "root_memory_id": "{root["memory_id"]}",
    "premise_memory_id": "{premise_memory["memory_id"]}",
    "scenario": "{scenario}",
    "what_evolved": "...",
    "how_helix_structure_mattered": "..."
  }},
  "noteworthy_evolution": [
    {{
      "turn_id": "round-...",
      "model": "...",
      "observation": "...",
      "evidence_memory_ids": ["..."]
    }}
  ],
  "claim_boundary": "qualitative cloud-only free-form HeliX drift"
}}
""",
        token=token,
        max_tokens=args.audit_tokens,
        temperature=0.0,
    )
    score = score_freeform_drift(
        scenario=scenario,
        models=models,
        turns=turns,
        main_chain_records=main_chain_records,
        root=root,
        premise=premise_memory,
        auditor_json=auditor.get("json"),
        auditor_finish_reason=auditor.get("finish_reason"),
    )
    artifact = {
        "artifact": "local-helix-freeform-drift-v1",
        "schema_version": 1,
        "run_id": run_id,
        "scenario": scenario,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "completed" if score["passed"] else "partial",
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "Qualitative cloud-only free-form model drift in signed HeliX memory. "
            "No sentience, local .hlx identity, or numerical KV<->SSM transfer claim."
        ),
        "models": {
            "round_robin": models,
            "auditor_requested": args.auditor_model,
            "auditor_actual": auditor.get("actual_model"),
        },
        "parameters": {
            "rounds": args.rounds,
            "tokens_per_turn": args.tokens_per_turn,
            "audit_tokens": args.audit_tokens,
            "temperature": args.temperature,
        },
        "root_memory": root,
        "premise_memory": premise_memory,
        "main_chain_records": main_chain_records,
        "turns": turns,
        "auditor_call": {k: v for k, v in auditor.items() if k not in {"text", "json"}},
        "auditor_output": {"text": auditor.get("text"), "json": auditor.get("json")},
        "freeform_drift_score": score,
        "model_claims_detected": detect_model_claims(_text([turn.get("output", {}).get("text") for turn in turns])),
        "workspace": str(workspace),
    }
    path = output_dir / f"local-helix-freeform-drift-{run_id}.json"
    artifact.update(_write_transcripts(output_dir, run_id, artifact))
    return finalize_artifact(path, artifact)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Free-form HeliX drift scenarios")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--tokens-per-turn", type=int, default=2200)
    parser.add_argument("--audit-tokens", type=int, default=3600)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = asyncio.run(run_freeform_drift(args))
    print(json.dumps({
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "scenario": artifact["scenario"],
        "score": artifact["freeform_drift_score"]["score"],
        "models": artifact["models"]["round_robin"],
        "auditor_actual": artifact["models"]["auditor_actual"],
        "transcript_markdown_path": artifact["transcript_markdown_path"],
        "transcript_jsonl_path": artifact["transcript_jsonl_path"],
    }, indent=2))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
