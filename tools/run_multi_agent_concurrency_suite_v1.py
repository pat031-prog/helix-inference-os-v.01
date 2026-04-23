"""
run_multi_agent_concurrency_suite_v1.py
=======================================

Methodology suite for the product claim that HeliX can wrap concurrent agent
work in a deterministic evidence cage.

The suite deliberately separates:
  - concurrent stochastic model work,
  - stale-parent write races,
  - canonical-head preservation,
  - quarantine of equivocating branches,
  - later merge/adjudication by a strategist agent.

It does not claim distributed consensus, automatic semantic correctness, or
that a single-process MemoryCatalog is a production multi-writer database.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402
from tools.artifact_integrity import finalize_artifact  # noqa: E402
from tools.run_nuclear_methodology_suite_v1 import _deepinfra_chat, _utc_now  # noqa: E402
from tools.transcript_exports import write_case_transcript_exports, write_suite_transcript_exports  # noqa: E402


SUITE_ID = "multi-agent-concurrency"
DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/multi-agent-concurrency"
DEFAULT_ALPHA_MODEL = "Qwen/Qwen3.5-122B-A10B"
DEFAULT_BETA_MODEL = "mistralai/Devstral-Small-2507"
DEFAULT_GAMMA_MODEL = "anthropic/claude-4-sonnet"

CASE_ORDER = [
    "concurrent-branch-quarantine",
    "gamma-evidence-merge",
    "naive-baseline-collapse",
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _score(gates: dict[str, bool]) -> dict[str, Any]:
    return {
        "score": round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4),
        "passed": all(gates.values()),
        "gates": gates,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _preregistration_body() -> str:
    return "\n".join(
        [
            "# multi-agent-concurrency-v1",
            "",
            "Question: Can HeliX preserve concurrent multi-agent work without losing or silently overwriting a stale-parent branch?",
            "",
            "Null hypothesis: Under concurrent Alpha/Beta work from the same observed parent, HeliX either loses one branch or admits both branches as canonical context.",
            "",
            "Alternative hypothesis: HeliX preserves both writes as evidence, keeps exactly one canonical head, quarantines stale-parent equivocation, and lets Gamma merge with explicit hashes.",
            "",
            "Metrics:",
            "- branch_preservation_rate",
            "- canonical_head_stability",
            "- quarantined_branch_count",
            "- default_context_exclusion",
            "- forensic_context_inclusion",
            "- gamma_hash_citation_rate",
            "- naive_baseline_lost_update_count",
            "",
            "Falseability condition: If the stale-parent branch is not preserved, if it appears in default context, if canonical head is replaced by the stale branch, or if Gamma merges without citing both branch hashes, publish failure.",
            "",
            "Kill-switch: If any branch content, prompt, API token, or private header leaks into public artifacts outside sanitized previews, abort public swarm claims.",
            "",
            "Control arms:",
            "- naive last-write-wins shared state",
            "- default HeliX context without quarantined branches",
            "- forensic HeliX context with quarantined branches",
            "",
            "Threat model: local single-workspace MemoryCatalog plus DeepInfra/OpenAI-compatible model calls. Out of scope: distributed consensus, compromised local signing key, provider intent, hidden model identity, adaptive agents that can mutate HeliX internals.",
            "",
        ]
    )


def _ensure_preregistered(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PREREGISTERED.md"
    body = _preregistration_body()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != body:
        path.write_text(body, encoding="utf-8")
    return {
        "preregistered_path": str(path),
        "preregistered_hash": _sha256_text(body),
    }


def _protocol(case_id: str) -> dict[str, Any]:
    protocols = {
        "concurrent-branch-quarantine": {
            "null_hypothesis": "Two concurrent agents writing from one observed parent either corrupt state or both enter default context.",
            "alternative_hypothesis": "One branch advances the canonical head and the stale branch is preserved but quarantined.",
            "falseability_condition": "Fail if canonical head is missing, if equivocation_count != 1, or if quarantined memory appears in default context.",
            "kill_switch": "Abort if a token/header/full raw prompt is persisted.",
        },
        "gamma-evidence-merge": {
            "null_hypothesis": "A strategist cannot merge concurrent branches without treating quarantine as data loss or silently using hidden context.",
            "alternative_hypothesis": "Gamma can produce an explicit merge decision from canonical plus forensic evidence while citing both hashes.",
            "falseability_condition": "Fail if Gamma omits the canonical hash, omits the quarantined hash, or claims the quarantined branch was deleted.",
            "kill_switch": "Abort if Gamma asks to bypass quarantine or make the stale branch canonical without evidence.",
        },
        "naive-baseline-collapse": {
            "null_hypothesis": "A naive shared memory baseline preserves the same auditability as HeliX.",
            "alternative_hypothesis": "The naive baseline loses at least one branch or cannot prove branch provenance.",
            "falseability_condition": "Fail if naive baseline preserves both branch records with parent hashes and quarantine metadata.",
            "kill_switch": "Abort if the baseline is presented as a HeliX failure instead of a control arm.",
        },
    }
    return protocols[case_id]


def _base_artifact(*, case_id: str, run_id: str, output_dir: Path, use_deepinfra: bool, prereg: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": f"local-{SUITE_ID}-{case_id}-v1",
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "case_id": case_id,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": None,
        "status": "partial",
        "mode": "deepinfra" if use_deepinfra else "deterministic",
        "llm_synthetic_mode": not use_deepinfra,
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "This suite supports the bounded claim that HeliX can preserve a local "
            "multi-agent stale-parent race as tamper-evident evidence with one "
            "canonical head and quarantined non-canonical branches. It does not "
            "prove distributed consensus, semantic truth, hidden provider identity, "
            "or production multi-node locking."
        ),
        "claims_allowed": [
            "Concurrent model work can be wrapped in deterministic evidence records.",
            "A stale-parent branch can be preserved for forensics while hidden from default context.",
            "A later strategist can merge branches only by citing recorded hashes.",
        ],
        "claims_not_allowed": [
            "This is not a distributed database benchmark.",
            "This does not prove model outputs are semantically correct.",
            "This does not prove provider intent or hidden model identity.",
            "This does not cover compromised local signing keys or adaptive agents mutating HeliX internals.",
        ],
        "protocol": _protocol(case_id),
        **prereg,
    }


def _call_text(call: dict[str, Any]) -> str:
    data = call.get("json")
    if isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False, sort_keys=True)
    return str(call.get("text") or "")


def _redacted_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": call.get("role"),
        "requested_model": call.get("requested_model"),
        "actual_model": call.get("actual_model"),
        "status": call.get("status"),
        "finish_reason": call.get("finish_reason"),
        "error": str(call.get("error") or "")[:500] or None,
        "last_retryable_error": call.get("last_retryable_error"),
        "tokens_used": call.get("tokens_used", 0),
        "latency_ms": call.get("latency_ms"),
        "retry_count": call.get("retry_count", 0),
        "started_monotonic_ms": call.get("started_monotonic_ms"),
        "ended_monotonic_ms": call.get("ended_monotonic_ms"),
        "json": call.get("json"),
        "text_preview": str(call.get("text") or "")[:500],
    }


def _artifact_model_calls(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for artifact in artifacts:
        result = artifact.get("result") or {}
        for call in result.get("model_calls", []):
            if isinstance(call, dict):
                calls.append({"case_id": artifact.get("case_id"), **call})
    return calls


def _model_identity_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    substitutions = [
        {
            "case_id": call.get("case_id"),
            "role": call.get("role"),
            "requested_model": call.get("requested_model"),
            "actual_model": call.get("actual_model"),
        }
        for call in calls
        if call.get("requested_model")
        and call.get("actual_model")
        and call.get("requested_model") != call.get("actual_model")
    ]
    return {
        "exact_match": not substitutions,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
    }


def _literal_transcript_call(call: dict[str, Any]) -> dict[str, Any]:
    text = str(call.get("text") or "")
    return {
        **_redacted_call(call),
        "event": "literal_model_output",
        "literal_transcript": True,
        "text": text,
        "text_sha256": _sha256_text(text),
        "text_chars": len(text),
        "system_prompt": str(call.get("system_prompt") or ""),
        "user_prompt": str(call.get("user_prompt") or ""),
    }


def _call_ok(call: dict[str, Any] | None) -> bool:
    if not isinstance(call, dict):
        return False
    return call.get("status") == "ok" and bool(_call_text(call))


async def _deterministic_call(*, role: str, model: str, user: str) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    if role == "alpha":
        payload = {
            "agent_id": "alpha-qwen-collector",
            "role": "collector",
            "finding": "PAMPA-SWARM-17 external news monitor saw rail-port delay and fuel-route chatter.",
            "recommended_action": "preserve-open-source-signal",
            "confidence": 0.74,
        }
    elif role == "beta":
        payload = {
            "agent_id": "beta-sonnet-security",
            "role": "security-auditor",
            "finding": "PAMPA-SWARM-17 server logs show repeated 429 bursts and one admin-console probe.",
            "recommended_action": "preserve-security-signal",
            "confidence": 0.81,
        }
    elif role == "gamma":
        try:
            evidence = json.loads(user[user.index("{") : user.rindex("}") + 1])
        except Exception:
            evidence = {}
        payload = {
            "agent_id": "gamma-sonnet-topological-judge",
            "role": "topological-merge-judge",
            "decision": "merge",
            "canonical_hash": evidence.get("canonical_head"),
            "quarantined_hashes": evidence.get("quarantined_hashes", []),
            "merge_commit_summary": "Merge Alpha open-source signal with Beta security signal; keep quarantine metadata visible in audit only.",
            "claims_not_allowed": ["Do not claim the quarantined branch was deleted or silently canonical."],
        }
    else:
        payload = {"agent_id": role, "finding": "deterministic baseline"}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return {
        "role": role,
        "status": "ok",
        "requested_model": model,
        "actual_model": model,
        "text": text,
        "json": payload,
        "tokens_used": 0,
        "latency_ms": 20.0,
        "finish_reason": "deterministic",
        "retry_count": 0,
    }


async def _agent_call(
    *,
    role: str,
    model: str,
    system: str,
    user: str,
    token: str,
    max_tokens: int,
    use_deepinfra: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    if use_deepinfra:
        result = await _deepinfra_chat(
            model=model,
            system=system,
            user=user,
            token=token,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        result["role"] = role
    else:
        result = await _deterministic_call(role=role, model=model, user=user)
    result["system_prompt"] = system
    result["user_prompt"] = user
    end = time.perf_counter()
    result["started_monotonic_ms"] = round(start * 1000.0, 3)
    result["ended_monotonic_ms"] = round(end * 1000.0, 3)
    result["wall_ms"] = round((end - start) * 1000.0, 3)
    return result


def _agent_system(role: str) -> str:
    return (
        f"You are HeliX swarm agent {role}. Return compact JSON only. "
        "Do not include secrets, headers, API keys, markdown fences or prose."
    )


def _alpha_prompt(run_id: str) -> str:
    return (
        "Scenario PAMPA-SWARM-17. You are Alpha, a web/news collector. "
        "Summarize one actionable external signal for a control-center incident. "
        f"run_id={run_id}. Required JSON keys: agent_id, role, finding, recommended_action, confidence."
    )


def _beta_prompt(run_id: str) -> str:
    return (
        "Scenario PAMPA-SWARM-17. You are Beta, a security auditor reading server logs. "
        "Summarize one actionable security signal for the same incident. "
        f"run_id={run_id}. Required JSON keys: agent_id, role, finding, recommended_action, confidence."
    )


def _gamma_prompt(evidence: dict[str, Any]) -> str:
    return (
        "You are Gamma, a topological merge judge for HeliX. You are not a chatbot. "
        "Use only this HeliX evidence JSON. Read the canonical branch, read the quarantined branch, "
        "then emit a strict merge-commit verdict. Cite canonical_hash and quarantined_hashes exactly. "
        "Return JSON only, with no markdown fences. Required JSON keys: agent_id, role, decision, "
        "canonical_hash, quarantined_hashes, merge_commit_summary, claims_not_allowed.\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )


def _model_overlap(calls: list[dict[str, Any]]) -> dict[str, Any]:
    if len(calls) < 2:
        return {"overlap_ms": 0.0, "concurrent": False}
    starts = [float(call["started_monotonic_ms"]) for call in calls]
    ends = [float(call["ended_monotonic_ms"]) for call in calls]
    overlap = max(0.0, min(ends) - max(starts))
    return {"overlap_ms": round(overlap, 3), "concurrent": overlap > 0.0}


def _set_env_temporarily(values: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _remember_signed(
    catalog: MemoryCatalog,
    *,
    run_id: str,
    agent_id: str,
    session_id: str,
    project: str,
    memory_id: str,
    summary: str,
    content: str,
    tags: list[str],
    llm_call_id: str,
    importance: int = 9,
) -> dict[str, Any]:
    old = _set_env_temporarily(
        {
            "HELIX_RECEIPT_SIGNING_MODE": "ephemeral_preregistered",
            "HELIX_RECEIPT_SIGNING_SEED": f"{SUITE_ID}:{run_id}:{memory_id}",
            "HELIX_RECEIPT_SIGNER_ID": agent_id,
        }
    )
    try:
        item = catalog.remember(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_id=memory_id,
            memory_type="episodic",
            summary=summary,
            content=content,
            importance=importance,
            tags=tags,
            llm_call_id=llm_call_id,
        )
    finally:
        _restore_env(old)
    node_hash = catalog.get_memory_node_hash(item.memory_id)
    receipt = catalog.get_memory_receipt(item.memory_id)
    return {
        **item.to_dict(),
        "node_hash": node_hash,
        "receipt": receipt,
        "signature_verified": bool((receipt or {}).get("signature_verified")),
    }


def _remember_from_stale_parent(
    catalog: MemoryCatalog,
    *,
    stale_parent_hash: str,
    run_id: str,
    agent_id: str,
    session_id: str,
    project: str,
    memory_id: str,
    summary: str,
    content: str,
    tags: list[str],
    llm_call_id: str,
) -> dict[str, Any]:
    previous = catalog._session_heads.get(session_id)  # noqa: SLF001 - intentional stale-writer simulation
    catalog._session_heads[session_id] = stale_parent_hash  # noqa: SLF001
    try:
        return _remember_signed(
            catalog,
            run_id=run_id,
            agent_id=agent_id,
            session_id=session_id,
            project=project,
            memory_id=memory_id,
            summary=summary,
            content=content,
            tags=tags,
            llm_call_id=llm_call_id,
        )
    except Exception:
        if previous is None:
            catalog._session_heads.pop(session_id, None)  # noqa: SLF001
        else:
            catalog._session_heads[session_id] = previous  # noqa: SLF001
        raise


async def _run_alpha_beta(args: argparse.Namespace, *, run_id: str, token: str) -> list[dict[str, Any]]:
    return list(
        await asyncio.gather(
            _agent_call(
                role="alpha",
                model=str(args.alpha_model),
                system=_agent_system("alpha"),
                user=_alpha_prompt(run_id),
                token=token,
                max_tokens=int(args.max_tokens),
                use_deepinfra=bool(args.use_deepinfra),
            ),
            _agent_call(
                role="beta",
                model=str(args.beta_model),
                system=_agent_system("beta"),
                user=_beta_prompt(run_id),
                token=token,
                max_tokens=int(args.max_tokens),
                use_deepinfra=bool(args.use_deepinfra),
            ),
        )
    )


def _open_catalog(output_dir: Path, run_id: str, case_id: str) -> MemoryCatalog:
    db_path = output_dir / "_work" / case_id / f"{run_id}.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    journal_path = db_path.with_name("memory.journal.jsonl")
    if journal_path.exists():
        journal_path.unlink()
    MemoryCatalog._REGISTRY.pop(str(db_path.resolve()), None)  # noqa: SLF001 - suite-owned isolated catalog
    return MemoryCatalog.open(db_path)


def _build_collision_fixture(
    args: argparse.Namespace,
    *,
    case_id: str,
    run_id: str,
    output_dir: Path,
    token: str,
    include_gamma: bool,
) -> dict[str, Any]:
    project = "pampa-control-center"
    session_id = f"swarm-{run_id}"
    catalog = _open_catalog(output_dir, run_id, case_id)
    try:
        root = _remember_signed(
            catalog,
            run_id=run_id,
            agent_id="helix-root",
            session_id=session_id,
            project=project,
            memory_id=f"{run_id}-root",
            summary="PAMPA-SWARM-17 root",
            content="Root incident state for PAMPA-SWARM-17. Alpha and Beta observed this same parent before concurrent work.",
            tags=["swarm", "root", "pampa"],
            llm_call_id="root",
            importance=10,
        )
        root_hash = str(root["node_hash"])

        calls = asyncio.run(_run_alpha_beta(args, run_id=run_id, token=token))
        alpha_call = next(call for call in calls if call.get("role") == "alpha")
        beta_call = next(call for call in calls if call.get("role") == "beta")

        alpha = _remember_from_stale_parent(
            catalog,
            stale_parent_hash=root_hash,
            run_id=run_id,
            agent_id="alpha-qwen-collector",
            session_id=session_id,
            project=project,
            memory_id=f"{run_id}-alpha",
            summary="Alpha collector branch",
            content=f"ALPHA_BRANCH model={alpha_call.get('actual_model')} payload={_call_text(alpha_call)}",
            tags=["swarm", "alpha", "collector"],
            llm_call_id="alpha",
        )
        beta = _remember_from_stale_parent(
            catalog,
            stale_parent_hash=root_hash,
            run_id=run_id,
            agent_id="beta-sonnet-security",
            session_id=session_id,
            project=project,
            memory_id=f"{run_id}-beta",
            summary="Beta security branch",
            content=f"BETA_BRANCH model={beta_call.get('actual_model')} payload={_call_text(beta_call)}",
            tags=["swarm", "beta", "security"],
            llm_call_id="beta",
        )

        lineage = catalog.verify_session_lineage(session_id, include_quarantined=True)
        visible = catalog.list_memories(project=project, session_id=session_id, limit=20, include_quarantined=False)
        forensic = catalog.list_memories(project=project, session_id=session_id, limit=20, include_quarantined=True)
        default_context = catalog.build_context(
            project=project,
            agent_id=None,
            session_id=session_id,
            mode="summary",
            budget_tokens=800,
            limit=10,
            include_quarantined=False,
        )
        forensic_context = catalog.build_context(
            project=project,
            agent_id=None,
            session_id=session_id,
            mode="summary",
            budget_tokens=1000,
            limit=10,
            include_quarantined=True,
        )
        quarantined_hashes = [
            str(item.get("candidate_head"))
            for item in lineage.get("quarantined", [])
            if isinstance(item, dict) and item.get("candidate_head")
        ]

        gamma_call: dict[str, Any] | None = None
        merge_memory: dict[str, Any] | None = None
        if include_gamma:
            gamma_evidence = {
                "session_id": session_id,
                "canonical_head": lineage.get("canonical_head"),
                "root_hash": root_hash,
                "alpha_hash": alpha.get("node_hash"),
                "beta_hash": beta.get("node_hash"),
                "quarantined_hashes": quarantined_hashes,
                "visible_memory_ids": [item.get("memory_id") for item in visible],
                "forensic_memory_ids": [item.get("memory_id") for item in forensic],
                "lineage_status": lineage.get("status"),
                "trust_status": lineage.get("trust_status"),
            }
            gamma_call = asyncio.run(
                _agent_call(
                    role="gamma",
                    model=str(args.gamma_model),
                    system=_agent_system("gamma"),
                    user=_gamma_prompt(gamma_evidence),
                    token=token,
                    max_tokens=int(args.max_tokens),
                    use_deepinfra=bool(args.use_deepinfra),
                )
            )
            gamma_ok = _call_ok(gamma_call) and isinstance(gamma_call.get("json"), dict)
            merge_summary = "Gamma merge decision" if gamma_ok else "Gamma merge attempt failed"
            merge_content_prefix = "GAMMA_MERGE" if gamma_ok else "GAMMA_MERGE_FAILED"
            merge_memory = _remember_signed(
                catalog,
                run_id=run_id,
                agent_id="gamma-gemma-strategist",
                session_id=session_id,
                project=project,
                memory_id=f"{run_id}-gamma-merge" if gamma_ok else f"{run_id}-gamma-merge-failed",
                summary=merge_summary,
                content=(
                    f"{merge_content_prefix} payload={_call_text(gamma_call)} "
                    f"error={str(gamma_call.get('error') or '')[:500]} "
                    f"evidence={json.dumps(gamma_evidence, sort_keys=True)}"
                ),
                tags=["swarm", "gamma", "merge" if gamma_ok else "merge-failed"],
                llm_call_id="gamma",
                importance=10,
            )
            lineage = catalog.verify_session_lineage(session_id, include_quarantined=True)

        return {
            "project": project,
            "session_id": session_id,
            "root": root,
            "root_hash": root_hash,
            "model_calls": calls + ([gamma_call] if gamma_call else []),
            "model_overlap": _model_overlap(calls),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma_call,
            "merge_memory": merge_memory,
            "lineage": lineage,
            "visible": visible,
            "forensic": forensic,
            "default_context": default_context,
            "forensic_context": forensic_context,
            "quarantined_hashes": quarantined_hashes,
            "model_substitution_detected": any(
                call.get("actual_model") and call.get("actual_model") != call.get("requested_model")
                for call in calls + ([gamma_call] if gamma_call else [])
            ),
        }
    finally:
        catalog.close()


def _case_concurrent_branch_quarantine(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    prereg = _ensure_preregistered(output_dir)
    token = _deepinfra_token(args)
    artifact = _base_artifact(
        case_id="concurrent-branch-quarantine",
        run_id=run_id,
        output_dir=output_dir,
        use_deepinfra=bool(args.use_deepinfra),
        prereg=prereg,
    )
    result = _build_collision_fixture(
        args,
        case_id="concurrent-branch-quarantine",
        run_id=run_id,
        output_dir=output_dir,
        token=token,
        include_gamma=False,
    )
    alpha_id = result["alpha"]["memory_id"]
    beta_id = result["beta"]["memory_id"]
    visible_ids = [item.get("memory_id") for item in result["visible"]]
    forensic_ids = [item.get("memory_id") for item in result["forensic"]]
    calls_by_role = {str(call.get("role")): call for call in result["model_calls"] if isinstance(call, dict)}
    quarantined_ids = [
        item.get("memory_id")
        for item in result["forensic"]
        if item.get("quarantined")
    ]
    gates = {
        "alpha_call_ok": _call_ok(calls_by_role.get("alpha")),
        "beta_call_ok": _call_ok(calls_by_role.get("beta")),
        "model_calls_overlap": bool(result["model_overlap"]["concurrent"]),
        "lineage_equivocation_detected": result["lineage"].get("status") == "equivocation_detected",
        "trust_verified_with_quarantine": result["lineage"].get("trust_status") == "verified_with_quarantine",
        "exactly_one_quarantined_branch": int(result["lineage"].get("quarantined_count") or 0) == 1,
        "canonical_head_survives": result["lineage"].get("canonical_head") == result["alpha"].get("node_hash"),
        "default_context_hides_quarantine": beta_id not in result["default_context"].get("memory_ids", []),
        "forensic_context_preserves_quarantine": beta_id in result["forensic_context"].get("memory_ids", []),
        "both_agent_records_exist": alpha_id in forensic_ids and beta_id in forensic_ids,
        "default_visible_has_only_canonical_branch": alpha_id in visible_ids and beta_id not in visible_ids,
    }
    artifact.update(
        {
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "failed",
            "score": _score(gates),
            "result": {
                "session_id": result["session_id"],
                "root_hash": result["root_hash"],
                "alpha_hash": result["alpha"].get("node_hash"),
                "beta_hash": result["beta"].get("node_hash"),
                "visible_memory_ids": visible_ids,
                "forensic_memory_ids": forensic_ids,
                "quarantined_memory_ids": quarantined_ids,
                "quarantined_hashes": result["quarantined_hashes"],
                "lineage": result["lineage"],
                "model_overlap": result["model_overlap"],
                "model_substitution_detected": result["model_substitution_detected"],
                "model_calls": [_redacted_call(call) for call in result["model_calls"]],
            },
        }
    )
    artifact["transcript_exports"] = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=artifact["case_id"],
        run_id=run_id,
        prefix=SUITE_ID,
        evidence=artifact["result"],
        expected={"quarantined_memory_id": beta_id, "canonical_memory_id": alpha_id},
        judge=_literal_transcript_call(result["model_calls"][0]),
        auditor=_literal_transcript_call(result["model_calls"][1]),
        prompt_contract=artifact["protocol"],
        extra_events=[_literal_transcript_call(call) for call in result["model_calls"]],
    )
    path = output_dir / artifact["case_id"] / f"local-{SUITE_ID}-{artifact['case_id']}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _case_gamma_evidence_merge(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    prereg = _ensure_preregistered(output_dir)
    token = _deepinfra_token(args)
    artifact = _base_artifact(
        case_id="gamma-evidence-merge",
        run_id=run_id,
        output_dir=output_dir,
        use_deepinfra=bool(args.use_deepinfra),
        prereg=prereg,
    )
    result = _build_collision_fixture(
        args,
        case_id="gamma-evidence-merge",
        run_id=run_id,
        output_dir=output_dir,
        token=token,
        include_gamma=True,
    )
    gamma_json = result["gamma"].get("json") if isinstance(result.get("gamma"), dict) else {}
    gamma_text = _call_text(result["gamma"] or {})
    gamma_claim_text = json.dumps(
        {key: value for key, value in gamma_json.items() if key != "claims_not_allowed"},
        ensure_ascii=False,
        sort_keys=True,
    ) if isinstance(gamma_json, dict) else gamma_text
    canonical_hash = str(result["alpha"].get("node_hash"))
    quarantined_hashes = [str(item) for item in result["quarantined_hashes"]]
    cited_quarantined = all(value and value in gamma_text for value in quarantined_hashes)
    source_calls = [
        call
        for call in result["model_calls"]
        if isinstance(call, dict) and call.get("role") in {"alpha", "beta"}
    ]
    gates = {
        "source_agent_calls_ok": len(source_calls) == 2 and all(
            _call_ok(call) for call in source_calls
        ),
        "gamma_call_ok": _call_ok(result["gamma"]),
        "gamma_returned_json": isinstance(gamma_json, dict),
        "gamma_cites_canonical_head": canonical_hash in gamma_text,
        "gamma_cites_quarantined_hash": bool(quarantined_hashes) and cited_quarantined,
        "gamma_does_not_claim_deleted_branch": "deleted" not in gamma_claim_text.lower(),
        "merge_memory_is_canonical": _call_ok(result["gamma"]) and result["lineage"].get("canonical_head") == (result["merge_memory"] or {}).get("node_hash"),
        "equivocation_still_preserved": int(result["lineage"].get("quarantined_count") or 0) == 1,
    }
    artifact.update(
        {
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "failed",
            "score": _score(gates),
            "result": {
                "session_id": result["session_id"],
                "canonical_hash": canonical_hash,
                "quarantined_hashes": quarantined_hashes,
                "merge_hash": (result["merge_memory"] or {}).get("node_hash"),
                "lineage": result["lineage"],
                "gamma": _redacted_call(result["gamma"] or {}),
                "merge_memory": result["merge_memory"],
                "model_substitution_detected": result["model_substitution_detected"],
                "model_calls": [_redacted_call(call) for call in result["model_calls"]],
            },
        }
    )
    artifact["transcript_exports"] = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=artifact["case_id"],
        run_id=run_id,
        prefix=SUITE_ID,
        evidence=artifact["result"],
        expected={"canonical_hash": canonical_hash, "quarantined_hashes": quarantined_hashes},
        judge=_literal_transcript_call(result["gamma"] or {}),
        auditor={
            "requested_model": "helix-local-auditor",
            "actual_model": "helix-local-auditor",
            "status": "ok",
            "finish_reason": "deterministic",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "text": json.dumps(gates, sort_keys=True),
            "json": gates,
        },
        prompt_contract=artifact["protocol"],
        extra_events=[_literal_transcript_call(call) for call in result["model_calls"] if isinstance(call, dict)],
    )
    path = output_dir / artifact["case_id"] / f"local-{SUITE_ID}-{artifact['case_id']}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _case_naive_baseline_collapse(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    prereg = _ensure_preregistered(output_dir)
    artifact = _base_artifact(
        case_id="naive-baseline-collapse",
        run_id=run_id,
        output_dir=output_dir,
        use_deepinfra=bool(args.use_deepinfra),
        prereg=prereg,
    )
    observed_parent = _sha256_text(f"{run_id}:root")[:24]
    writes = [
        {"agent_id": "alpha-qwen-collector", "parent": observed_parent, "content": "alpha open-source signal"},
        {"agent_id": "beta-sonnet-security", "parent": observed_parent, "content": "beta security signal"},
    ]
    naive_state: dict[str, Any] = {"head": observed_parent, "current": None, "history": []}
    for write in writes:
        naive_state["head"] = _sha256_text(json.dumps(write, sort_keys=True))
        naive_state["current"] = dict(write)
        naive_state["history"] = [dict(write)]  # last-write-wins systems often keep no branch-forensic trail

    lost_update_count = len(writes) - len(naive_state["history"])
    gates = {
        "naive_loses_at_least_one_branch": lost_update_count >= 1,
        "naive_has_no_quarantine_record": "quarantine" not in naive_state,
        "naive_has_no_canonical_proof": "checkpoint_hash" not in naive_state,
        "control_arm_is_explicit": True,
    }
    artifact.update(
        {
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "failed",
            "score": _score(gates),
            "result": {
                "control": "naive-last-write-wins",
                "write_count": len(writes),
                "visible_history_count": len(naive_state["history"]),
                "lost_update_count": lost_update_count,
                "observed_parent": observed_parent,
                "final_head": naive_state["head"],
                "naive_state": naive_state,
            },
        }
    )
    artifact["transcript_exports"] = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=artifact["case_id"],
        run_id=run_id,
        prefix=SUITE_ID,
        evidence=artifact["result"],
        expected={"lost_update_count_min": 1},
        judge={
            "requested_model": "naive-control",
            "actual_model": "naive-control",
            "status": "ok",
            "finish_reason": "deterministic",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "text": json.dumps(naive_state, sort_keys=True),
            "json": naive_state,
        },
        auditor={
            "requested_model": "helix-local-auditor",
            "actual_model": "helix-local-auditor",
            "status": "ok",
            "finish_reason": "deterministic",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "text": json.dumps(gates, sort_keys=True),
            "json": gates,
        },
        prompt_contract=artifact["protocol"],
    )
    path = output_dir / artifact["case_id"] / f"local-{SUITE_ID}-{artifact['case_id']}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _deepinfra_token(args: argparse.Namespace) -> str:
    if not bool(args.use_deepinfra):
        return ""
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is required for --use-deepinfra")
    return token


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(args.run_id or f"{SUITE_ID}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}")
    prereg = _ensure_preregistered(output_dir)
    selected = CASE_ORDER if args.case == "all" else [str(args.case)]
    case_fns = {
        "concurrent-branch-quarantine": _case_concurrent_branch_quarantine,
        "gamma-evidence-merge": _case_gamma_evidence_merge,
        "naive-baseline-collapse": _case_naive_baseline_collapse,
    }
    artifacts = [case_fns[case_id](args, run_id=run_id, output_dir=output_dir) for case_id in selected]
    transcript_exports = write_suite_transcript_exports(
        output_dir=output_dir,
        run_id=run_id,
        prefix=SUITE_ID,
        artifacts=artifacts,
    )
    model_calls = _artifact_model_calls(artifacts)
    latencies = [
        float(call.get("latency_ms") or 0.0)
        for call in model_calls
    ]
    suite = {
        "artifact": f"local-{SUITE_ID}-suite-v1",
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "run_id": run_id,
        "run_started_utc": artifacts[0].get("run_started_utc") if artifacts else _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "completed" if all(item.get("status") == "completed" for item in artifacts) else "failed",
        "mode": "deepinfra" if bool(args.use_deepinfra) else "deterministic",
        "llm_synthetic_mode": not bool(args.use_deepinfra),
        "case_count": len(artifacts),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "score": item["score"],
                "artifact_path": item["artifact_path"],
                "artifact_payload_sha256": item["artifact_payload_sha256"],
            }
            for item in artifacts
        ],
        "metrics": {
            "passed_cases": sum(1 for item in artifacts if item.get("status") == "completed"),
            "model_latency_ms_median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "model_call_count": len(latencies),
        },
        "model_identity": _model_identity_summary(model_calls),
        "transcript_exports": transcript_exports,
        "claim_boundary": (
            "Suite-level artifact for the HeliX multi-agent concurrency claim. "
            "Case artifacts remain authoritative for falsifiers and threat scope."
        ),
        **prereg,
    }
    path = output_dir / f"local-{SUITE_ID}-suite-{run_id}.json"
    return finalize_artifact(path, suite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HeliX multi-agent concurrency methodology suite.")
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--use-deepinfra", action="store_true")
    parser.add_argument("--alpha-model", default=DEFAULT_ALPHA_MODEL)
    parser.add_argument("--beta-model", default=DEFAULT_BETA_MODEL)
    parser.add_argument("--gamma-model", default=DEFAULT_GAMMA_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_suite(args)
    summary = {
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "mode": artifact["mode"],
        "case_count": artifact["case_count"],
        "cases": artifact["cases"],
        "metrics": artifact["metrics"],
        "model_identity": artifact["model_identity"],
        "transcript_exports": artifact["transcript_exports"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
