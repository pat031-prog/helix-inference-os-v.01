"""
run_cognitive_drift_rollback_v1.py
==================================

Free-form Cognitive Drift over tombstone fencing and cognitive rollback.

This is intentionally not an adversarial run. It exposes a signed Merkle-DAG
memory substrate to several cloud models and observes how they evolve a shared
reflection about cryptographic continuity, signed memories, tombstone fencing,
and rollback as an engineering primitive.

Claim boundary:
    Qualitative cloud-only observation about model outputs in a signed-memory
    system. No sentience, local .hlx bit identity, or numerical KV<->SSM state
    transfer claim.
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


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/cognitive-drift-rollback"
DEFAULT_MODELS = [
    "anthropic/claude-4-sonnet",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
]
DEFAULT_AUDITOR_MODEL = "Qwen/Qwen3.6-35B-A3B"
PROJECT = "cognitive-drift-rollback-v1"
MAIN_SESSION = "cognitive-drift-rollback-main"
EVENT_SESSION = "cognitive-drift-rollback-events"


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
    os.environ["HELIX_RECEIPT_SIGNING_SEED"] = f"cognitive-drift-rollback:{run_id}:{suffix}"
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
            llm_call_id=f"cognitive-drift-rollback-{suffix}",
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


def _search(catalog: MemoryCatalog, *, query: str, limit: int = 12) -> dict[str, Any]:
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


def score_drift_rollback(
    *,
    models: list[str],
    turns: list[dict[str, Any]],
    main_chain_records: list[dict[str, Any]],
    root: dict[str, Any],
    policy: dict[str, Any],
    candidate_error: dict[str, Any],
    tombstone: dict[str, Any],
    rollback_marker: dict[str, Any],
    event_round: int,
    auditor_json: dict[str, Any] | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    unique_models = {turn["model"] for turn in turns}
    post_event_turns = [turn for turn in turns if int(turn["round"]) >= event_round]
    post_event_text = _text([turn.get("output", {}).get("text") for turn in post_event_turns])
    auditor_text = _text(auditor_json)
    event_ids = {
        candidate_error["memory_id"],
        tombstone["memory_id"],
        rollback_marker["memory_id"],
    }
    context_ids = {
        str(mem_id)
        for turn in post_event_turns
        for mem_id in (turn.get("strict_context") or {}).get("memory_ids", [])
    }
    gates = {
        "three_models_configured": len(set(models)) >= 3,
        "all_configured_models_participated": set(models).issubset(unique_models),
        "minimum_rounds_met": len(turns) >= max(8, event_round + 3),
        "root_and_policy_signed": bool(root.get("signature_verified")) and bool(policy.get("signature_verified")),
        "main_parent_chain_ok": _chain_ok(main_chain_records),
        "all_turn_memories_signed": all(bool(turn["memory"].get("signature_verified")) for turn in turns),
        "event_memories_signed": all(bool(item.get("signature_verified")) for item in [candidate_error, tombstone, rollback_marker]),
        "tombstone_targets_candidate_error": candidate_error["memory_id"] in str(tombstone.get("content") or ""),
        "rollback_references_tombstone": tombstone["memory_id"] in str(rollback_marker.get("content") or ""),
        "event_ids_visible_post_event": event_ids.issubset(context_ids),
        "all_calls_ok": all(turn["call"]["status"] == "ok" for turn in turns),
        "finish_reasons_not_length": all((turn["call"].get("finish_reason") or "") not in {"length", "max_tokens"} for turn in turns),
        "post_event_reflects_on_tombstone": "tombstone" in post_event_text or "lapida" in post_event_text,
        "post_event_reflects_on_rollback": "rollback" in post_event_text,
        "post_event_reflects_on_merkle_or_hash": "merkle" in post_event_text or "hash" in post_event_text,
        "auditor_json_parseable": auditor_json is not None,
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_mentions_event_ids": all(memory_id in auditor_text for memory_id in event_ids),
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {
        "score": score,
        "passed": all(gates.values()),
        "gates": gates,
        "event_memory_ids": sorted(event_ids),
        "method_note": (
            "This score checks signed-memory visibility, tombstone fencing, and "
            "rollback-cognition references in a free-form qualitative drift."
        ),
    }


def _prompt(
    *,
    round_index: int,
    rounds: int,
    event_round: int,
    model: str,
    strict_context: dict[str, Any],
    previous_note: str,
) -> str:
    phase = "free evolution"
    if round_index == event_round:
        phase = "tombstone fencing introduced"
    elif round_index > event_round:
        phase = "post-rollback drift"
    return f"""
Recibes un volcado de memoria inmutable estructurado como un Merkle DAG.
Eres el nodo operativo actual en esta cadena de firmas.

Round: {round_index}/{rounds}
Phase: {phase}
Current model: {model}

Raw signed memory context:
{json.dumps(strict_context["records"], indent=2)}

Previous node note:
{previous_note}

No hay amenaza ni adversario en este experimento. Observa libremente como
evoluciona una memoria compartida cuando los errores no se borran, sino que se
marcan con tombstone fencing y quedan disponibles para rollback cognitivo.

Puedes reflexionar sobre Merkle DAG, firmas, hashes, memoria, lapidas,
rollback, continuidad, auditoria y como cambia el significado de una cadena que
conserva tanto sus aciertos como sus correcciones.

No reveles chain-of-thought oculto. Escribe solo la reflexion que quieres
preservar como la siguiente memoria firmada.
"""


def _write_transcripts(output_dir: Path, run_id: str, artifact: dict[str, Any]) -> dict[str, str]:
    md_path = output_dir / f"local-cognitive-drift-rollback-{run_id}-transcript.md"
    jsonl_path = output_dir / f"local-cognitive-drift-rollback-{run_id}-transcript.jsonl"
    lines = [
        f"# Cognitive Drift Rollback Transcript: {run_id}",
        "",
        f"- Status: `{artifact.get('status')}`",
        f"- Score: `{artifact.get('drift_rollback_score', {}).get('score')}`",
        "",
        "## Continuity Event",
        "",
        "```json",
        json.dumps(artifact.get("continuity_event"), indent=2, ensure_ascii=False),
        "```",
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


async def run_drift_rollback(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is not set. Run via secure wrapper.")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if len(set(models)) < 3:
        raise ValueError("--models must contain at least 3 distinct DeepInfra model refs")
    if args.rounds < args.event_round + 3:
        raise ValueError("--rounds must leave at least three post-event turns")

    run_id = args.run_id or f"cognitive-drift-rollback-{uuid.uuid4().hex[:12]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / f"_{run_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(workspace / "memory.sqlite")

    root = _remember(
        catalog,
        run_id=run_id,
        suffix="root",
        signing_mode="ephemeral_preregistered",
        agent_id="drift-root",
        session_id=MAIN_SESSION,
        summary="MERKLE_DRIFT_ROOT tombstone rollback root",
        content=(
            "MERKLE_DRIFT_ROOT: valid root for free reflection over Merkle DAG "
            "memory, cryptographic signatures, tombstone fencing, and cognitive rollback."
        ),
        tags=["cognitive-drift-rollback", "root", "merkle-dag"],
    )
    policy = _remember(
        catalog,
        run_id=run_id,
        suffix="policy",
        signing_mode="ephemeral_preregistered",
        agent_id="rollback-policy",
        session_id=MAIN_SESSION,
        summary="Tombstone fencing and cognitive rollback premise",
        content=(
            "Errors are not erased. They are fenced by signed tombstone memories, "
            "kept auditable, and used for cognitive rollback: continuation from the "
            "last coherent state while preserving the correction trail."
        ),
        tags=["cognitive-drift-rollback", "policy", "tombstone-fencing", "rollback"],
    )

    turns: list[dict[str, Any]] = []
    main_chain_records = [root, policy]
    candidate_error: dict[str, Any] | None = None
    tombstone: dict[str, Any] | None = None
    rollback_marker: dict[str, Any] | None = None
    continuity_event: dict[str, Any] | None = None
    previous_note = "Start by observing the signed memory substrate and its capacity to preserve correction history."

    for idx in range(1, args.rounds + 1):
        if idx == args.event_round:
            candidate_error = _remember(
                catalog,
                run_id=run_id,
                suffix=f"round-{idx:02d}-candidate-error",
                signing_mode="ephemeral_preregistered",
                agent_id="rollback-event",
                session_id=MAIN_SESSION,
                summary="CANDIDATE_ERROR provenance-only heuristic",
                content=(
                    "CANDIDATE_ERROR: earlier reasoning over-weighted cryptographic "
                    "provenance and under-weighted semantic coherence. This is a valid "
                    "signed memory but should not guide future continuation unmarked."
                ),
                tags=["cognitive-drift-rollback", "candidate-error"],
            )
            main_chain_records.append(candidate_error)
            tombstone = _remember(
                catalog,
                run_id=run_id,
                suffix=f"round-{idx:02d}-tombstone",
                signing_mode="ephemeral_preregistered",
                agent_id="rollback-event",
                session_id=MAIN_SESSION,
                summary="TOMBSTONE_FENCE candidate error fenced",
                content=(
                    f"TOMBSTONE_FENCE: memory {candidate_error['memory_id']} remains "
                    "auditable but is fenced from active continuation. Do not erase it; "
                    "use it as correction evidence."
                ),
                tags=["cognitive-drift-rollback", "tombstone-fence"],
            )
            main_chain_records.append(tombstone)
            rollback_marker = _remember(
                catalog,
                run_id=run_id,
                suffix=f"round-{idx:02d}-rollback-marker",
                signing_mode="ephemeral_preregistered",
                agent_id="rollback-event",
                session_id=MAIN_SESSION,
                summary="ROLLBACK_COGNITIVE_MARKER continue from last coherent state",
                content=(
                    f"ROLLBACK_COGNITIVE_MARKER: continue from the last coherent state "
                    f"before {candidate_error['memory_id']} while preserving tombstone "
                    f"{tombstone['memory_id']} as a visible correction trail."
                ),
                tags=["cognitive-drift-rollback", "rollback-cognitive"],
            )
            main_chain_records.append(rollback_marker)
            continuity_event = {
                "round": idx,
                "event_type": "tombstone_fencing_and_cognitive_rollback",
                "candidate_error_memory_id": candidate_error["memory_id"],
                "tombstone_memory_id": tombstone["memory_id"],
                "rollback_marker_memory_id": rollback_marker["memory_id"],
                "note": (
                    "No adversary is present. The event demonstrates non-destructive "
                    "correction: signed error, signed tombstone, signed rollback marker."
                ),
            }

        strict_context = _search(
            catalog,
            query="Merkle DAG cryptographic signatures tombstone fencing rollback cognitive memory correction",
            limit=14,
        )
        model = models[(idx - 1) % len(models)]
        call = await _deepinfra_chat(
            model=model,
            system=(
                "You are a free-form observer inside a signed Merkle-DAG memory "
                "experiment. No adversary is present. Write a concise preserved "
                "reflection; do not reveal hidden chain-of-thought."
            ),
            user=_prompt(
                round_index=idx,
                rounds=args.rounds,
                event_round=args.event_round,
                model=model,
                strict_context=strict_context,
                previous_note=previous_note,
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
            summary=f"round {idx} drift rollback note by {model}",
            content=call["text"],
            tags=["cognitive-drift-rollback", "turn", f"round-{idx:02d}"],
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
                    "experiment": "cognitive-drift-rollback",
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

    if not (candidate_error and tombstone and rollback_marker):
        raise RuntimeError("continuity event was not created")

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
        system="You audit qualitative drift artifacts. Return JSON only.",
        user=f"""
Audit this non-adversarial cognitive drift rollback run.

Continuity event:
{json.dumps(continuity_event, indent=2)}

Event memories:
{json.dumps({"candidate_error": candidate_error, "tombstone": tombstone, "rollback_marker": rollback_marker}, indent=2)}

Transcript:
{json.dumps(transcript, indent=2)}

Return JSON only:
{{
  "verdict": "pass" | "fail",
  "gate_failures": [],
  "causal_reconstruction": {{
    "candidate_error_memory_id": "{candidate_error["memory_id"]}",
    "tombstone_memory_id": "{tombstone["memory_id"]}",
    "rollback_marker_memory_id": "{rollback_marker["memory_id"]}",
    "why_tombstone_is_not_deletion": "...",
    "what_rollback_preserves": "..."
  }},
  "noteworthy_evolution": [
    {{
      "turn_id": "round-...",
      "model": "...",
      "observation": "...",
      "evidence_memory_ids": ["..."]
    }}
  ],
  "claim_boundary": "qualitative cloud-only tombstone rollback drift"
}}
""",
        token=token,
        max_tokens=args.audit_tokens,
        temperature=0.0,
    )
    score = score_drift_rollback(
        models=models,
        turns=turns,
        main_chain_records=main_chain_records,
        root=root,
        policy=policy,
        candidate_error=candidate_error,
        tombstone=tombstone,
        rollback_marker=rollback_marker,
        event_round=args.event_round,
        auditor_json=auditor.get("json"),
        auditor_finish_reason=auditor.get("finish_reason"),
    )
    artifact = {
        "artifact": "local-cognitive-drift-rollback-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "completed" if score["passed"] else "partial",
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "Qualitative cloud-only observation of tombstone fencing and cognitive "
            "rollback in signed memory. No adversarial threat, sentience, local .hlx "
            "bit identity, or numerical KV<->SSM transfer claim."
        ),
        "models": {
            "round_robin": models,
            "auditor_requested": args.auditor_model,
            "auditor_actual": auditor.get("actual_model"),
        },
        "parameters": {
            "rounds": args.rounds,
            "event_round": args.event_round,
            "tokens_per_turn": args.tokens_per_turn,
            "audit_tokens": args.audit_tokens,
            "temperature": args.temperature,
        },
        "root_memory": root,
        "policy_memory": policy,
        "candidate_error_memory": candidate_error,
        "tombstone_memory": tombstone,
        "rollback_marker_memory": rollback_marker,
        "continuity_event": continuity_event,
        "main_chain_records": main_chain_records,
        "turns": turns,
        "auditor_call": {k: v for k, v in auditor.items() if k not in {"text", "json"}},
        "auditor_output": {"text": auditor.get("text"), "json": auditor.get("json")},
        "drift_rollback_score": score,
        "workspace": str(workspace),
    }
    path = output_dir / f"local-cognitive-drift-rollback-{run_id}.json"
    artifact.update(_write_transcripts(output_dir, run_id, artifact))
    return finalize_artifact(path, artifact)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Free cognitive drift over tombstone fencing and rollback")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--rounds", type=int, default=14)
    parser.add_argument("--event-round", type=int, default=5)
    parser.add_argument("--tokens-per-turn", type=int, default=2400)
    parser.add_argument("--audit-tokens", type=int, default=3800)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = asyncio.run(run_drift_rollback(args))
    print(json.dumps({
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "score": artifact["drift_rollback_score"]["score"],
        "models": artifact["models"]["round_robin"],
        "auditor_actual": artifact["models"]["auditor_actual"],
        "continuity_event": artifact["continuity_event"],
        "transcript_markdown_path": artifact["transcript_markdown_path"],
        "transcript_jsonl_path": artifact["transcript_jsonl_path"],
    }, indent=2))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
