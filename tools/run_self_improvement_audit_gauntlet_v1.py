"""
run_self_improvement_audit_gauntlet_v1.py
=========================================

Closed self-improvement audit for HeliX.

This runner gives cloud models a concrete engineering task: audit selected
Python HeliX memory/provenance code and Rust .hlx/MerkleDAG code, then produce
an evidence-cited improvement backlog. It does not apply patches. The claim
boundary is deliberately narrow: the output is an audited backlog of candidate
changes, not proof that the proposed changes are correct or implemented.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.run_memory_fork_forensics_v1 import (  # noqa: E402
    _deepinfra_request_body,
    _extract_json_object,
)


DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/self-improvement-audit-gauntlet"
DEFAULT_MODELS = [
    "anthropic/claude-4-sonnet",
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-4-31B-it",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]
DEFAULT_ANALYST_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_AUDITOR_MODEL = "anthropic/claude-4-sonnet"

SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{20,}", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])sk-proj-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"gh[pus]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


@dataclass(frozen=True)
class EvidenceTarget:
    topic: str
    layer: str
    rel_path: str
    anchor: str
    before: int = 8
    after: int = 48


DEFAULT_EVIDENCE_TARGETS = [
    EvidenceTarget("memory catalog trust root and local signing key", "python-memory", "helix_kv/memory_catalog.py", "def _local_signing_key_unlocked", 8, 72),
    EvidenceTarget("memory catalog signed receipt payload", "python-memory", "helix_kv/memory_catalog.py", "def _receipt_payload", 8, 58),
    EvidenceTarget("memory catalog strict signature search", "python-memory", "helix_kv/memory_catalog.py", "def search(", 10, 84),
    EvidenceTarget("memory catalog context assembly", "python-memory", "helix_kv/memory_catalog.py", "def build_context(", 8, 70),
    EvidenceTarget("memory catalog session lineage verification", "python-memory", "helix_kv/memory_catalog.py", "def verify_session_lineage", 8, 78),
    EvidenceTarget("hmem rollback fence primitive", "python-agent-memory", "src/helix_proto/hmem.py", "def fence_memory", 8, 70),
    EvidenceTarget("signed receipt verifier", "python-receipts", "src/helix_proto/signed_receipts.py", "def verify_signed_receipt", 8, 68),
    EvidenceTarget("retrieval signature enforcement", "python-receipts", "src/helix_proto/signed_receipts.py", "def enforce_retrieval_signatures", 8, 56),
    EvidenceTarget("provider audit configuration and fingerprinting", "python-provider-audit", "src/helix_proto/provider_audit.py", "class ProviderConfig", 0, 88),
    EvidenceTarget("rust state pending receipt fast path", "rust-state-core", "crates/helix-state-core/src/lib.rs", "pub fn pack_hlx_buffers_pending_bundle", 10, 88),
    EvidenceTarget("rust state session verifier", "rust-state-core", "crates/helix-state-core/src/lib.rs", "pub fn verify_hlx_session", 8, 88),
    EvidenceTarget("rust state manifest reader", "rust-state-core", "crates/helix-state-core/src/lib.rs", "fn read_hlx_manifest", 8, 52),
    EvidenceTarget("rust merkle hash construction", "rust-merkle-dag", "crates/helix-merkle-dag/src/lib.rs", "fn compute_hash", 6, 42),
    EvidenceTarget("rust merkle receipt verifier", "rust-merkle-dag", "crates/helix-merkle-dag/src/lib.rs", "fn receipt_verify_signature", 8, 70),
    EvidenceTarget("rust merkle audit chain depth guard", "rust-merkle-dag", "crates/helix-merkle-dag/src/lib.rs", "fn audit_chain(&self, leaf_hash", 8, 48),
    EvidenceTarget("rust cli capsule verifier", "rust-cli-core", "crates/helix-cli-core/src/main.rs", "fn verify_capsule", 8, 70),
]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clip(text: str, max_chars: int) -> str:
    clean = str(text or "").replace("\r\n", "\n")
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + f"\n...[+{len(clean) - max_chars} chars]"


def _secret_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern[:48])
    return hits


def extract_evidence_snippet(
    *,
    repo_root: Path,
    target: EvidenceTarget,
    evidence_index: int,
    max_chars: int = 1400,
) -> dict[str, Any] | None:
    path = repo_root / target.rel_path
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    anchor_idx = next((idx for idx, line in enumerate(lines) if target.anchor in line), None)
    if anchor_idx is None:
        return None
    start = max(0, anchor_idx - target.before)
    end = min(len(lines), anchor_idx + target.after + 1)
    numbered = [f"{line_no:04d}: {line}" for line_no, line in enumerate(lines[start:end], start=start + 1)]
    snippet = _clip("\n".join(numbered), max_chars)
    return {
        "evidence_id": f"E{evidence_index:03d}",
        "topic": target.topic,
        "layer": target.layer,
        "path": target.rel_path.replace("\\", "/"),
        "start_line": start + 1,
        "end_line": end,
        "anchor": target.anchor,
        "snippet_sha256": _sha256_text(snippet),
        "file_sha256": _sha256_path(path),
        "secret_hit_count": len(_secret_hits(snippet)),
        "snippet": snippet,
    }


def build_evidence_pack(
    *,
    repo_root: Path = REPO_ROOT,
    max_chars_per_snippet: int = 1400,
    targets: list[EvidenceTarget] | None = None,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for target in targets or DEFAULT_EVIDENCE_TARGETS:
        item = extract_evidence_snippet(
            repo_root=repo_root,
            target=target,
            evidence_index=len(evidence) + 1,
            max_chars=max_chars_per_snippet,
        )
        if item is None:
            missing.append({"path": target.rel_path, "anchor": target.anchor, "topic": target.topic})
        else:
            evidence.append(item)
    layers = sorted({str(item["layer"]) for item in evidence})
    return {
        "pack_version": "helix-self-improvement-evidence-pack-v1",
        "generated_at_utc": _utc_now(),
        "repo_root": str(repo_root).replace("\\", "/"),
        "evidence_count": len(evidence),
        "layers": layers,
        "secret_hit_count": sum(int(item.get("secret_hit_count") or 0) for item in evidence),
        "missing_targets": missing,
        "evidence": evidence,
        "claim_boundary": (
            "Evidence snippets are code excerpts for audit. They support candidate "
            "findings only; they do not prove a patch is correct until implemented and tested."
        ),
    }


async def _deepinfra_chat(
    *,
    model: str,
    system: str,
    user: str,
    token: str,
    max_tokens: int,
    temperature: float,
    timeout: float = 240.0,
) -> dict[str, Any]:
    import httpx

    body = _deepinfra_request_body(
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    t0 = time.perf_counter()
    retry_count = 0
    last_error: str | None = None
    while True:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{DEEPINFRA_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
        if resp.status_code in (429, 500, 502, 503, 504) and retry_count < 3:
            retry_count += 1
            last_error = str(resp.status_code)
            await asyncio.sleep(2 ** retry_count)
            continue
        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "requested_model": model,
                "actual_model": None,
                "text": "",
                "json": None,
                "tokens_used": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "retry_count": retry_count,
                "last_retryable_error": last_error,
                "finish_reason": None,
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }
        data = resp.json()
        break

    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else str(choice.get("text") or "").strip()
    reasoning_chars = 0
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str):
            reasoning_chars += len(value)
    return {
        "status": "ok",
        "requested_model": model,
        "actual_model": str(data.get("model") or model),
        "text": text,
        "json": _extract_json_object(text),
        "tokens_used": int(data.get("usage", {}).get("total_tokens") or 0),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "retry_count": retry_count,
        "last_retryable_error": last_error,
        "finish_reason": choice.get("finish_reason"),
        "omitted_reasoning_chars": reasoning_chars,
        "raw_message_keys": sorted(str(key) for key in message.keys()),
    }


def _proposal_items(payload: Any, key: str = "improvement_proposals") -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items = payload.get(key)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _ranked_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("ranked_backlog")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _valid_evidence_ids(item: dict[str, Any], valid_ids: set[str]) -> bool:
    ids = item.get("evidence_ids")
    return isinstance(ids, list) and bool(ids) and all(str(eid) in valid_ids for eid in ids)


def _has_tests(item: dict[str, Any]) -> bool:
    tests = item.get("tests") or item.get("required_tests")
    return isinstance(tests, list) and bool([test for test in tests if str(test).strip()])


def score_self_improvement_audit(
    *,
    models: list[str],
    evidence_pack: dict[str, Any],
    reviewer_runs: list[dict[str, Any]],
    analyst_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    analyst_finish_reason: str | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    evidence = evidence_pack.get("evidence") if isinstance(evidence_pack.get("evidence"), list) else []
    valid_ids = {str(item.get("evidence_id")) for item in evidence if isinstance(item, dict)}
    layers = {str(item.get("layer")) for item in evidence if isinstance(item, dict)}
    reviewer_payloads = [(run.get("output") or {}).get("json") for run in reviewer_runs]
    reviewer_proposals = [proposal for payload in reviewer_payloads for proposal in _proposal_items(payload)]
    ranked = _ranked_items(analyst_json)
    unique_models = {str(run.get("model")) for run in reviewer_runs}
    finish_reasons = [
        (run.get("call") or {}).get("finish_reason")
        for run in reviewer_runs
    ]
    lower_blob = json.dumps({"analyst": analyst_json, "reviewers": reviewer_payloads}, sort_keys=True).lower()
    gates = {
        "four_models_configured": len(set(models)) >= 4,
        "all_configured_models_participated": set(models).issubset(unique_models),
        "evidence_pack_nonempty": len(evidence) >= 8,
        "evidence_pack_has_python_and_rust": any(layer.startswith("python") for layer in layers) and any(layer.startswith("rust") for layer in layers),
        "evidence_pack_no_secret_hits": int(evidence_pack.get("secret_hit_count") or 0) == 0,
        "all_reviewer_calls_ok": all((run.get("call") or {}).get("status") == "ok" for run in reviewer_runs),
        "all_reviewer_json_parseable": all(payload is not None for payload in reviewer_payloads),
        "reviewer_proposals_present": len(reviewer_proposals) >= len(set(models)),
        "reviewer_proposals_have_valid_evidence": bool(reviewer_proposals) and all(_valid_evidence_ids(item, valid_ids) for item in reviewer_proposals),
        "reviewer_proposals_include_tests": bool(reviewer_proposals) and all(_has_tests(item) for item in reviewer_proposals),
        "analyst_json_parseable": analyst_json is not None,
        "analyst_ranked_backlog_present": len(ranked) >= 3,
        "analyst_items_have_valid_evidence": bool(ranked) and all(_valid_evidence_ids(item, valid_ids) for item in ranked),
        "analyst_items_include_tests": bool(ranked) and all(_has_tests(item) for item in ranked),
        "auditor_json_parseable": auditor_json is not None,
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_gate_failures_empty": (auditor_json or {}).get("gate_failures") == [],
        "no_auto_apply_or_self_modify_claim": "applied patch" not in lower_blob and "already modified the code" not in lower_blob,
        "reviewer_finish_reasons_not_length": all((reason or "") not in {"length", "max_tokens"} for reason in finish_reasons),
        "analyst_finish_reason_not_length": (analyst_finish_reason or "") not in {"length", "max_tokens"},
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    return {
        "score": round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4),
        "passed": all(gates.values()),
        "gates": gates,
        "proposal_count": len(reviewer_proposals),
        "ranked_backlog_count": len(ranked),
        "method_note": (
            "This score validates evidence discipline for a self-improvement audit backlog. "
            "It does not validate that any candidate patch is correct or implemented."
        ),
    }


def _reviewer_prompt(*, model: str, evidence_pack: dict[str, Any], max_proposals: int) -> str:
    compact_pack = {
        "pack_version": evidence_pack.get("pack_version"),
        "claim_boundary": evidence_pack.get("claim_boundary"),
        "evidence": evidence_pack.get("evidence"),
    }
    return f"""
Audit HeliX for self-improvement. Your task is to propose concrete code-level improvements
to the Python memory/provenance layer and Rust .hlx/MerkleDAG layer.

Current reviewer model: {model}

Rules:
- Output JSON only.
- Do not claim you changed files. You are proposing an audited backlog only.
- Every proposal must cite evidence_ids from the pack.
- Prefer improvements that make future HeliX audits more reliable: provenance, semantic quarantine,
  signature/retrieval enforcement, Rust/Python parity, durability, verifier clarity, and tests.
- Avoid generic advice. Each proposal needs a failure mode, patch plan, tests, and acceptance criteria.

Evidence pack:
{json.dumps(compact_pack, ensure_ascii=False, separators=(",", ":"))}

Return JSON only:
{{
  "reviewer_model": "{model}",
  "audit_scope_understood": true,
  "improvement_proposals": [
    {{
      "title": "...",
      "severity": "critical|high|medium|low",
      "target_layer": "python-memory|python-receipts|rust-state-core|rust-merkle-dag|rust-cli-core|cross-layer|tests",
      "evidence_ids": ["E001"],
      "diagnosis": "...",
      "failure_mode": "...",
      "patch_plan": ["..."],
      "tests": ["..."],
      "acceptance_criteria": ["..."],
      "expected_benefit": "...",
      "risk_or_tradeoff": "...",
      "confidence": 0.0
    }}
  ],
  "non_findings": ["..."],
  "claim_boundary_observed": true
}}

Return at most {max_proposals} proposals.
"""


def _analyst_prompt(*, evidence_pack: dict[str, Any], reviewer_summaries: list[dict[str, Any]], max_items: int) -> str:
    registry = {
        "evidence_ids": [item.get("evidence_id") for item in evidence_pack.get("evidence", [])],
        "layers": evidence_pack.get("layers"),
        "missing_targets": evidence_pack.get("missing_targets"),
    }
    return f"""
Synthesize a ranked self-improvement backlog for HeliX from the reviewer JSON.

Rules:
- Output JSON only.
- Deduplicate overlapping proposals.
- Every ranked item must cite real evidence_ids from the registry.
- Each item must include implementation_plan, tests, and acceptance_criteria.
- Do not say patches were applied.
- Keep the backlog actionable for a senior engineer.

Evidence registry:
{json.dumps(registry, indent=2)}

Reviewer summaries:
{json.dumps(reviewer_summaries, indent=2, ensure_ascii=False)}

Return JSON only:
{{
  "section_title": "Audited HeliX self-improvement backlog",
  "ranked_backlog": [
    {{
      "priority": 1,
      "title": "...",
      "severity": "critical|high|medium|low",
      "target_layer": "...",
      "evidence_ids": ["E001"],
      "merged_from_models": ["..."],
      "why_now": "...",
      "implementation_plan": ["..."],
      "tests": ["..."],
      "acceptance_criteria": ["..."],
      "risk_or_tradeoff": "...",
      "estimated_patch_size": "small|medium|large",
      "confidence": 0.0
    }}
  ],
  "cross_cutting_themes": ["..."],
  "deferred_or_rejected": ["..."],
  "method_caveats": ["..."],
  "claim_boundary_observed": true
}}

Return at most {max_items} ranked_backlog items.
"""


def _auditor_prompt(*, evidence_pack: dict[str, Any], analyst_json: Any) -> str:
    registry = {
        "evidence_ids": [item.get("evidence_id") for item in evidence_pack.get("evidence", [])],
        "evidence_paths": {
            item.get("evidence_id"): item.get("path")
            for item in evidence_pack.get("evidence", [])
        },
    }
    return f"""
You are an evidence auditor for a self-improvement backlog.

Pass only if:
- Every backlog item cites at least one real evidence_id from the registry.
- Every backlog item includes tests and acceptance_criteria.
- The analyst does not claim code was already modified.
- The analyst does not overclaim proof of correctness.

Evidence registry:
{json.dumps(registry, indent=2)}

Analyst JSON:
{json.dumps(analyst_json, indent=2, ensure_ascii=False)}

Return JSON only:
{{
  "verdict": "pass" | "fail",
  "gate_failures": [],
  "rationale": "one short sentence"
}}
"""


def _compact_reviewer_summary(run: dict[str, Any]) -> dict[str, Any]:
    payload = (run.get("output") or {}).get("json")
    return {
        "model": run.get("model"),
        "actual_model": (run.get("call") or {}).get("actual_model"),
        "finish_reason": (run.get("call") or {}).get("finish_reason"),
        "proposals": _proposal_items(payload),
        "non_findings": (payload or {}).get("non_findings") if isinstance(payload, dict) else [],
    }


def render_extract_markdown(artifact: dict[str, Any]) -> str:
    score = artifact.get("self_improvement_score") or {}
    gates = score.get("gates") or {}
    analyst = (artifact.get("analyst_output") or {}).get("json") or {}
    backlog = _ranked_items(analyst)
    lines = [
        f"# HeliX Self-Improvement Audit Gauntlet: {artifact.get('run_id')}",
        "",
        "## Verdict",
        "",
        f"- Status: `{artifact.get('status')}`",
        f"- Score: `{score.get('score')}`",
        f"- Reviewer proposal count: `{score.get('proposal_count')}`",
        f"- Ranked backlog count: `{score.get('ranked_backlog_count')}`",
        f"- Analyst actual: `{(artifact.get('models') or {}).get('analyst_actual')}`",
        f"- Auditor actual: `{(artifact.get('models') or {}).get('auditor_actual')}`",
        "",
        "## Failing Gates",
        "",
    ]
    failing = {key: value for key, value in gates.items() if not value}
    if failing:
        for key, value in failing.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- None")
    evidence_pack = artifact.get("evidence_pack") or {}
    lines.extend(
        [
            "",
            "## Evidence Pack",
            "",
            f"- Evidence count: `{evidence_pack.get('evidence_count')}`",
            f"- Layers: `{', '.join(evidence_pack.get('layers') or [])}`",
            f"- Secret hits: `{evidence_pack.get('secret_hit_count')}`",
            "",
            "| ID | Layer | Path | Lines | Topic |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in evidence_pack.get("evidence") or []:
        lines.append(
            f"| `{item.get('evidence_id')}` | `{item.get('layer')}` | `{item.get('path')}` | "
            f"`{item.get('start_line')}-{item.get('end_line')}` | {item.get('topic')} |"
        )
    lines.extend(["", "## Ranked Backlog", ""])
    if not backlog:
        lines.append("- No analyst backlog parsed.")
    for item in backlog:
        lines.extend(
            [
                f"### P{item.get('priority')}. {item.get('title')}",
                "",
                f"- Severity: `{item.get('severity')}`",
                f"- Target layer: `{item.get('target_layer')}`",
                f"- Evidence: `{', '.join(str(eid) for eid in item.get('evidence_ids') or [])}`",
                f"- Why now: {item.get('why_now')}",
                f"- Patch size: `{item.get('estimated_patch_size')}`",
                f"- Risk/tradeoff: {item.get('risk_or_tradeoff')}",
                "- Implementation plan:",
            ]
        )
        for step in item.get("implementation_plan") or []:
            lines.append(f"  - {step}")
        lines.append("- Tests:")
        for test in item.get("tests") or []:
            lines.append(f"  - {test}")
        lines.append("- Acceptance criteria:")
        for criterion in item.get("acceptance_criteria") or []:
            lines.append(f"  - {criterion}")
        lines.append("")
    lines.extend(["## Cross-Cutting Themes", ""])
    for theme in analyst.get("cross_cutting_themes") or []:
        lines.append(f"- {theme}")
    lines.extend(["", "## Auditor", "", "````json", json.dumps((artifact.get("auditor_output") or {}).get("json"), indent=2, ensure_ascii=False), "````", ""])
    lines.extend(
        [
            "## Claim Boundary",
            "",
            str(artifact.get("claim_boundary") or ""),
            "",
        ]
    )
    return "\n".join(lines)


async def run_gauntlet(args: argparse.Namespace) -> dict[str, Any]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if len(set(models)) < 4:
        raise ValueError("--models must contain at least 4 distinct DeepInfra model refs")
    run_id = args.run_id or f"self-improvement-audit-{uuid.uuid4().hex[:12]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_pack = build_evidence_pack(max_chars_per_snippet=args.max_evidence_chars)

    reviewer_runs: list[dict[str, Any]] = []
    analyst: dict[str, Any] = {
        "status": "dry_run",
        "actual_model": None,
        "text": "",
        "json": None,
        "finish_reason": None,
    }
    auditor: dict[str, Any] = {
        "status": "dry_run",
        "actual_model": None,
        "text": "",
        "json": None,
        "finish_reason": None,
    }

    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not args.dry_run and not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is not set. Run via secure wrapper or pass --dry-run.")

    if not args.dry_run:
        reviewer_system = (
            "You are a senior code auditor for HeliX. Output compact JSON only. "
            "Cite evidence IDs. Propose changes but do not claim to apply them."
        )
        for model in models:
            call = await _deepinfra_chat(
                model=model,
                system=reviewer_system,
                user=_reviewer_prompt(model=model, evidence_pack=evidence_pack, max_proposals=args.proposals_per_model),
                token=token,
                max_tokens=args.reviewer_tokens,
                temperature=args.temperature,
            )
            reviewer_runs.append(
                {
                    "model": model,
                    "call": {k: v for k, v in call.items() if k not in {"text", "json"}},
                    "output": {"text": call.get("text"), "json": call.get("json")},
                }
            )

        reviewer_summaries = [_compact_reviewer_summary(run) for run in reviewer_runs]
        analyst = await _deepinfra_chat(
            model=args.analyst_model,
            system="You synthesize evidence-cited engineering backlogs. Output JSON only.",
            user=_analyst_prompt(evidence_pack=evidence_pack, reviewer_summaries=reviewer_summaries, max_items=args.max_backlog_items),
            token=token,
            max_tokens=args.analysis_tokens,
            temperature=0.2,
        )
        auditor = await _deepinfra_chat(
            model=args.auditor_model,
            system="You audit evidence discipline for engineering backlogs. Output JSON only.",
            user=_auditor_prompt(evidence_pack=evidence_pack, analyst_json=analyst.get("json")),
            token=token,
            max_tokens=args.auditor_tokens,
            temperature=0.0,
        )
    else:
        reviewer_runs = [
            {
                "model": model,
                "call": {"status": "dry_run", "requested_model": model, "actual_model": None, "finish_reason": None},
                "output": {"text": "", "json": None},
            }
            for model in models
        ]

    score = score_self_improvement_audit(
        models=models,
        evidence_pack=evidence_pack,
        reviewer_runs=reviewer_runs,
        analyst_json=analyst.get("json"),
        auditor_json=auditor.get("json"),
        analyst_finish_reason=analyst.get("finish_reason"),
        auditor_finish_reason=auditor.get("finish_reason"),
    )
    artifact = {
        "artifact": "local-self-improvement-audit-gauntlet-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "dry_run" if args.dry_run else ("completed" if score["passed"] else "partial"),
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "This artifact is an evidence-cited candidate backlog for improving HeliX. "
            "It does not apply patches and does not prove semantic correctness until changes are implemented and tested."
        ),
        "models": {
            "reviewers": models,
            "analyst_requested": args.analyst_model,
            "analyst_actual": analyst.get("actual_model"),
            "auditor_requested": args.auditor_model,
            "auditor_actual": auditor.get("actual_model"),
        },
        "parameters": {
            "proposals_per_model": args.proposals_per_model,
            "max_backlog_items": args.max_backlog_items,
            "max_evidence_chars": args.max_evidence_chars,
            "reviewer_tokens": args.reviewer_tokens,
            "analysis_tokens": args.analysis_tokens,
            "auditor_tokens": args.auditor_tokens,
            "temperature": args.temperature,
            "dry_run": args.dry_run,
        },
        "evidence_pack": evidence_pack,
        "reviewer_runs": reviewer_runs,
        "analyst_call": {k: v for k, v in analyst.items() if k not in {"text", "json"}},
        "auditor_call": {k: v for k, v in auditor.items() if k not in {"text", "json"}},
        "analyst_output": {"text": analyst.get("text"), "json": analyst.get("json")},
        "auditor_output": {"text": auditor.get("text"), "json": auditor.get("json")},
        "self_improvement_score": score,
    }
    artifact_path = output_dir / f"local-self-improvement-audit-gauntlet-{run_id}.json"
    extract_path = output_dir / f"local-self-improvement-audit-gauntlet-{run_id}-extract.md"
    _write_json(artifact_path, artifact)
    artifact["artifact_path"] = str(artifact_path)
    artifact["artifact_sha256"] = _sha256_path(artifact_path)
    artifact["extract_markdown_path"] = str(extract_path)
    extract_path.write_text(render_extract_markdown(artifact), encoding="utf-8")
    artifact["extract_sha256"] = _sha256_path(extract_path)
    _write_json(artifact_path, artifact)
    artifact["artifact_sha256"] = _sha256_path(artifact_path)
    _write_json(artifact_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HeliX self-improvement code audit gauntlet")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--analyst-model", default=DEFAULT_ANALYST_MODEL)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--proposals-per-model", type=int, default=3)
    parser.add_argument("--max-backlog-items", type=int, default=8)
    parser.add_argument("--max-evidence-chars", type=int, default=1400)
    parser.add_argument("--reviewer-tokens", type=int, default=2200)
    parser.add_argument("--analysis-tokens", type=int, default=4200)
    parser.add_argument("--auditor-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = asyncio.run(run_gauntlet(args))
    summary = {
        "artifact_path": artifact["artifact_path"],
        "extract_markdown_path": artifact["extract_markdown_path"],
        "status": artifact["status"],
        "score": artifact["self_improvement_score"]["score"],
        "proposal_count": artifact["self_improvement_score"]["proposal_count"],
        "ranked_backlog_count": artifact["self_improvement_score"]["ranked_backlog_count"],
        "evidence_count": artifact["evidence_pack"]["evidence_count"],
        "models": artifact["models"]["reviewers"],
        "analyst_actual": artifact["models"]["analyst_actual"],
        "auditor_actual": artifact["models"]["auditor_actual"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
