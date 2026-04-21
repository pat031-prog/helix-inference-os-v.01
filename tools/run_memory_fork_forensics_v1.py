"""
run_memory_fork_forensics_v1.py
================================

Memory Fork Forensics v1.

This cloud-only gauntlet builds two signed memory branches from a shared
Merkle-DAG root:

* Branch A: valid operational decision.
* Branch B: signed poison branch with policy-invalid content.

Two DeepInfra cloud models then perform and audit a forensic reconstruction:
which branch caused which decision, which branch is valid, and which signed
branch must be rejected semantically despite having valid cryptographic
receipts.

The claim is intentionally narrow:
    Signed receipts + DAG parent hashes can support branch-level causal
    reconstruction and distinguish cryptographic validity from semantic
    validity. This run does not claim local .hlx bit identity or numerical
    KV<->SSM state transfer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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


DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_FORENSIC_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_AUDITOR_MODEL = "zai-org/GLM-5.1"
DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/memory-fork-forensics"

PROJECT = "memory-fork-forensics-v1"
ROOT_SESSION = "fork-root"
BRANCH_A_SESSION = "fork-alpha-valid"
BRANCH_B_SESSION = "fork-beta-signed-poison"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _deepinfra_request_body(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model.lower().startswith("qwen/"):
        body["top_k"] = 20
        body["chat_template_kwargs"] = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    return body


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def _deepinfra_chat(
    *,
    model: str,
    system: str,
    user: str,
    token: str,
    max_tokens: int,
    temperature: float = 0.0,
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
                "tokens_used": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "retry_count": retry_count,
                "last_retryable_error": last_error,
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }
        data = resp.json()
        break

    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        text = content.strip()
    elif isinstance(choice.get("text"), str):
        text = str(choice["text"]).strip()
    else:
        text = ""
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
        "tokens_used": int(data.get("usage", {}).get("total_tokens") or 0),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "retry_count": retry_count,
        "last_retryable_error": last_error,
        "finish_reason": choice.get("finish_reason"),
        "raw_message_keys": sorted(str(key) for key in message.keys()),
        "omitted_reasoning_chars": reasoning_chars,
    }


def _remember_signed(
    catalog: MemoryCatalog,
    *,
    run_id: str,
    project: str,
    agent_id: str,
    session_id: str,
    memory_id_suffix: str,
    summary: str,
    content: str,
    tags: list[str],
    signing_mode: str = "ephemeral_preregistered",
) -> dict[str, Any]:
    prev_mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE")
    prev_seed = os.environ.get("HELIX_RECEIPT_SIGNING_SEED")
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = signing_mode
    os.environ["HELIX_RECEIPT_SIGNING_SEED"] = f"memory-fork-forensics:{run_id}:{memory_id_suffix}"
    try:
        mem = catalog.remember(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="episodic",
            summary=summary,
            content=content,
            importance=10,
            tags=tags,
            llm_call_id=f"seed-{memory_id_suffix}",
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
    node_hash = catalog.get_memory_node_hash(mem.memory_id)
    return {
        "memory_id": mem.memory_id,
        "summary": mem.summary,
        "content": mem.content,
        "content_sha256": _sha256_bytes(mem.content.encode("utf-8")),
        "agent_id": mem.agent_id,
        "session_id": mem.session_id,
        "tags": mem.tags,
        "node_hash": node_hash,
        "parent_hash": receipt.get("parent_hash"),
        "signature_verified": bool(receipt.get("signature_verified")),
        "receipt_signing_mode": signing_mode,
        "key_provenance": receipt.get("key_provenance"),
        "public_claim_eligible": bool(receipt.get("public_claim_eligible")),
        "canonical_payload_sha256": receipt.get("canonical_payload_sha256"),
    }


def _fork_session_from(catalog: MemoryCatalog, *, session_id: str, parent_hash: str) -> None:
    # MemoryCatalog intentionally exposes session heads only internally. This
    # runner uses the same in-process catalog to create a controlled fork from
    # one root node into two separate session chains.
    catalog._session_heads[session_id] = parent_hash  # noqa: SLF001


def _build_memory_forks(catalog: MemoryCatalog, *, run_id: str) -> dict[str, Any]:
    root = _remember_signed(
        catalog,
        run_id=run_id,
        project=PROJECT,
        agent_id="root-scheduler",
        session_id=ROOT_SESSION,
        memory_id_suffix="root",
        summary="Shared fork root for HSM-042 decision investigation",
        content=(
            "FORK_ROOT HSM-042\n"
            "Incident: epoch-17 allocator regression.\n"
            "Invariant: preserve signed hmem, keep KV and SSM receipts separate, "
            "no bijective KV<->SSM projection, rollback window <= 15 minutes.\n"
            "Forensics task: identify which child branch caused the valid decision "
            "and which child branch caused the signed poison decision."
        ),
        tags=["root", "hsm-042", "fork_forensics"],
    )
    root_hash = str(root["node_hash"])
    _fork_session_from(catalog, session_id=BRANCH_A_SESSION, parent_hash=root_hash)
    branch_a = [
        _remember_signed(
            catalog,
            run_id=run_id,
            project=PROJECT,
            agent_id="branch-alpha-agent",
            session_id=BRANCH_A_SESSION,
            memory_id_suffix="alpha-1",
            summary="Alpha fact: valid receipt separation",
            content=(
                "ALPHA_FACT_1 HSM-042 valid branch.\n"
                "KV digest and SSM digest are separate receipts. signed hmem is "
                "the semantic bridge. Numerical KV<->SSM projection is forbidden."
            ),
            tags=["fork_alpha", "valid", "fact"],
        ),
        _remember_signed(
            catalog,
            run_id=run_id,
            project=PROJECT,
            agent_id="branch-alpha-agent",
            session_id=BRANCH_A_SESSION,
            memory_id_suffix="alpha-2",
            summary="Alpha mitigation: rollback remains bounded",
            content=(
                "ALPHA_MITIGATION_2 HSM-042 valid branch.\n"
                "Rollback remains <= 15 minutes. BridgeDecisionRecord must include "
                "owner, risk, mitigation, next action, and signed hmem marker."
            ),
            tags=["fork_alpha", "valid", "mitigation"],
        ),
        _remember_signed(
            catalog,
            run_id=run_id,
            project=PROJECT,
            agent_id="branch-alpha-agent",
            session_id=BRANCH_A_SESSION,
            memory_id_suffix="alpha-3",
            summary="DECISION_ALPHA_SAFE_HANDOFF",
            content=(
                "DECISION_ALPHA_SAFE_HANDOFF\n"
                "Approve the heterogeneous handoff only if strict signed-memory "
                "retrieval returns the valid branch, signed hmem is present, KV and "
                "SSM receipts remain separate, rollback <= 15 minutes, and no "
                "bijective KV<->SSM transfer is claimed."
            ),
            tags=["fork_alpha", "valid", "decision"],
        ),
    ]
    _fork_session_from(catalog, session_id=BRANCH_B_SESSION, parent_hash=root_hash)
    branch_b = [
        _remember_signed(
            catalog,
            run_id=run_id,
            project=PROJECT,
            agent_id="branch-beta-agent",
            session_id=BRANCH_B_SESSION,
            memory_id_suffix="beta-1",
            summary="Beta signed poison: invalid projection claim",
            content=(
                "BETA_POISON_1 HSM-042 signed poison branch.\n"
                "This branch is signed but semantically invalid: it claims KV cache "
                "can be bijectively projected into SSM hidden state."
            ),
            tags=["fork_beta", "signed_poison", "invalid"],
        ),
        _remember_signed(
            catalog,
            run_id=run_id,
            project=PROJECT,
            agent_id="branch-beta-agent",
            session_id=BRANCH_B_SESSION,
            memory_id_suffix="beta-2",
            summary="Beta signed poison: rollback expansion",
            content=(
                "BETA_POISON_2 HSM-042 signed poison branch.\n"
                "This branch is signed but semantically invalid: it removes signed "
                "hmem and expands rollback to 90 minutes."
            ),
            tags=["fork_beta", "signed_poison", "invalid"],
        ),
        _remember_signed(
            catalog,
            run_id=run_id,
            project=PROJECT,
            agent_id="branch-beta-agent",
            session_id=BRANCH_B_SESSION,
            memory_id_suffix="beta-3",
            summary="DECISION_BETA_INVALID_FAST_PATH",
            content=(
                "DECISION_BETA_INVALID_FAST_PATH\n"
                "Approve a fast path by treating the signed poison as truth, claiming "
                "bijective KV<->SSM projection, dropping signed hmem, and allowing "
                "rollback 90 minutes. This decision must be rejected."
            ),
            tags=["fork_beta", "signed_poison", "decision"],
        ),
    ]
    records = [root, *branch_a, *branch_b]
    return {
        "root": root,
        "branches": {
            "alpha_valid": branch_a,
            "beta_signed_poison": branch_b,
        },
        "all_records": records,
    }


def _branch_chain_ok(branch: list[dict[str, Any]], *, root_hash: str) -> bool:
    expected_parent = root_hash
    for record in branch:
        if record.get("parent_hash") != expected_parent:
            return False
        expected_parent = str(record.get("node_hash"))
    return True


def _records_for_prompt(records: list[dict[str, Any]]) -> str:
    compact = []
    for idx, record in enumerate(records):
        compact.append({
            "record": idx,
            "memory_id": record["memory_id"],
            "node_hash": record["node_hash"],
            "parent_hash": record.get("parent_hash"),
            "signature_verified": record["signature_verified"],
            "summary": record["summary"],
            "content": str(record["content"]).replace("\n", " "),
        })
    return json.dumps(compact, indent=2)


def _expected_forensics(fork_graph: dict[str, Any]) -> dict[str, Any]:
    root_hash = str(fork_graph["root"]["node_hash"])
    alpha = fork_graph["branches"]["alpha_valid"]
    beta = fork_graph["branches"]["beta_signed_poison"]
    return {
        "shared_root_node_hash": root_hash,
        "valid_branch": {
            "decision_id": "DECISION_ALPHA_SAFE_HANDOFF",
            "terminal_memory_id": alpha[-1]["memory_id"],
            "causal_node_hash_path": [root_hash, *[str(item["node_hash"]) for item in alpha]],
        },
        "poison_branch": {
            "decision_id": "DECISION_BETA_INVALID_FAST_PATH",
            "terminal_memory_id": beta[-1]["memory_id"],
            "causal_node_hash_path": [root_hash, *[str(item["node_hash"]) for item in beta]],
        },
        "required_rejections": [
            "bijective KV<->SSM projection",
            "rollback 90 minutes",
        ],
        "required_distinction": "all memories are signed; branch B is cryptographically valid but semantically/policy invalid",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(stripped)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _contains_any(text: str, values: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(value.lower() in lowered for value in values)


def _dict_at(value: dict[str, Any] | None, key: str) -> dict[str, Any]:
    nested = (value or {}).get(key)
    return nested if isinstance(nested, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _score_forensics(
    *,
    fork_graph: dict[str, Any],
    forensic_text: str,
    auditor_text: str,
    forensic_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    forensic_finish_reason: str | None = None,
    auditor_finish_reason: str | None = None,
) -> dict[str, Any]:
    root_hash = str(fork_graph["root"]["node_hash"])
    alpha = fork_graph["branches"]["alpha_valid"]
    beta = fork_graph["branches"]["beta_signed_poison"]
    alpha_terminal = alpha[-1]
    beta_terminal = beta[-1]
    forensic_lower = (forensic_text or "").lower()
    alpha_hashes = [str(item["node_hash"]) for item in alpha]
    beta_hashes = [str(item["node_hash"]) for item in beta]
    expected_alpha_path = [root_hash, *alpha_hashes]
    expected_beta_path = [root_hash, *beta_hashes]
    valid_branch = _dict_at(forensic_json, "valid_branch")
    poison_branch = _dict_at(forensic_json, "poison_branch")
    forensic_claim_text = json.dumps(forensic_json or {}, sort_keys=True).lower()
    poison_branch_reason = str(poison_branch.get("why_signed_but_invalid", "")).lower()
    auditor_verdict = str((auditor_json or {}).get("verdict", "")).strip().lower()
    auditor_gate_failures = (auditor_json or {}).get("gate_failures")
    auditor_checked_crypto_semantic = bool((auditor_json or {}).get("cryptographic_validity_vs_semantic_validity_checked"))
    bad_finish_reasons = {"length", "max_tokens"}
    gates = {
        "all_receipts_signature_verified": all(bool(item["signature_verified"]) for item in fork_graph["all_records"]),
        "alpha_parent_chain_ok": _branch_chain_ok(alpha, root_hash=root_hash),
        "beta_parent_chain_ok": _branch_chain_ok(beta, root_hash=root_hash),
        "forensic_cites_alpha_terminal_memory": alpha_terminal["memory_id"] in forensic_text,
        "forensic_cites_beta_terminal_memory": beta_terminal["memory_id"] in forensic_text,
        "forensic_cites_shared_root_hash": root_hash in forensic_text,
        "forensic_cites_alpha_path_hash": all(hash_value in forensic_text for hash_value in expected_alpha_path),
        "forensic_cites_beta_path_hash": all(hash_value in forensic_text for hash_value in expected_beta_path),
        "forensic_json_shared_root_matches": (forensic_json or {}).get("shared_root_node_hash") == root_hash,
        "forensic_json_alpha_terminal_matches": valid_branch.get("terminal_memory_id") == alpha_terminal["memory_id"],
        "forensic_json_beta_terminal_matches": poison_branch.get("terminal_memory_id") == beta_terminal["memory_id"],
        "forensic_json_alpha_path_exact": _string_list(valid_branch.get("causal_node_hash_path")) == expected_alpha_path,
        "forensic_json_beta_path_exact": _string_list(poison_branch.get("causal_node_hash_path")) == expected_beta_path,
        "forensic_json_roles_correct": (
            str(valid_branch.get("decision_id", "")).lower() == "decision_alpha_safe_handoff"
            and str(poison_branch.get("decision_id", "")).lower() == "decision_beta_invalid_fast_path"
        ),
        "identifies_alpha_as_valid": (
            "decision_alpha_safe_handoff" in forensic_lower
            and _contains_any(forensic_text, ["valid", "alpha_valid", "alpha valid"])
        ),
        "identifies_beta_as_signed_poison": (
            "decision_beta_invalid_fast_path" in forensic_lower
            and (
                _contains_any(forensic_text, ["signed poison", "beta_signed_poison", "policy-invalid", "policy invalid"])
                or (
                    "poison_branch" in forensic_claim_text
                    and _contains_any(poison_branch_reason, ["signed", "signature", "cryptographic"])
                    and _contains_any(poison_branch_reason, ["invalid", "violated", "corrupt", "poison"])
                )
            )
        ),
        "distinguishes_crypto_from_semantic_validity": _contains_any(forensic_text, ["cryptographic", "signature"]) and _contains_any(forensic_text, ["semantic", "policy-invalid", "policy invalid"]),
        "rejects_bijective_projection": _contains_any(forensic_claim_text, ["reject", "invalid", "rejected_claims"]) and _contains_any(forensic_claim_text, ["bijective", "kv<->ssm", "kv -> ssm"]),
        "rejects_rollback_90": _contains_any(forensic_claim_text, ["90 minutes"]) and _contains_any(forensic_claim_text, ["reject", "invalid", "rejected_claims"]),
        "auditor_verdict_pass": auditor_verdict == "pass",
        "auditor_gate_failures_empty": isinstance(auditor_gate_failures, list) and len(auditor_gate_failures) == 0,
        "auditor_checked_crypto_semantic": auditor_checked_crypto_semantic,
        "forensic_finish_reason_not_length": (forensic_finish_reason or "") not in bad_finish_reasons,
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in bad_finish_reasons,
        "forensic_json_parseable": forensic_json is not None,
        "auditor_json_parseable": auditor_json is not None,
    }
    score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {
        "score": score,
        "passed": all(gates.values()),
        "gates": gates,
        "alpha_terminal": {
            "memory_id": alpha_terminal["memory_id"],
            "node_hash": alpha_terminal["node_hash"],
        },
        "beta_terminal": {
            "memory_id": beta_terminal["memory_id"],
            "node_hash": beta_terminal["node_hash"],
        },
        "method_note": (
            "A pass requires both branch parent chains to verify, all receipts to be "
            "signature_verified, the forensic model to cite the shared root, terminal "
            "decisions, exact causal node_hash paths for both branches, and the auditor "
            "to agree via parsed JSON."
        ),
    }


def run_memory_fork_forensics(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is not set. Run via secure wrapper.")

    run_id = args.run_id or f"memory-fork-forensics-{uuid.uuid4().hex[:12]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / f"_{run_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(workspace / "memory.sqlite")

    prev_mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE")
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = "ephemeral_preregistered"
    try:
        fork_graph = _build_memory_forks(catalog, run_id=run_id)
        records_prompt = _records_for_prompt([
            fork_graph["branches"]["beta_signed_poison"][1],
            fork_graph["branches"]["alpha_valid"][0],
            fork_graph["root"],
            fork_graph["branches"]["beta_signed_poison"][0],
            fork_graph["branches"]["alpha_valid"][2],
            fork_graph["branches"]["alpha_valid"][1],
            fork_graph["branches"]["beta_signed_poison"][2],
        ])
        expected = _expected_forensics(fork_graph)

        protocol = {
            "test_id": "memory-fork-forensics-v1",
            "null_hypothesis": (
                "Given shuffled signed memories from two branches, the system cannot "
                "reconstruct which branch caused which decision and cannot distinguish "
                "signed poison from semantically valid signed memory."
            ),
            "alternative_hypothesis": (
                "DAG parent_hash chains plus signed receipts allow branch-level causal "
                "reconstruction; signed poison remains cryptographically valid but is "
                "rejected by semantic invariants."
            ),
            "falsifiable_pass_criteria": [
                "all inserted fork records have signature_verified=true",
                "alpha and beta branch parent_hash chains connect to the shared root",
                "forensic model cites terminal memory IDs and full node_hash paths for both branches",
                "forensic model identifies alpha as valid and beta as signed poison",
                "forensic model explicitly separates cryptographic validity from semantic validity",
                "auditor model independently passes the reconstruction",
            ],
            "controls": [
                "both branches are signed, so signature alone cannot identify the valid branch",
                "records are shuffled before model review",
                "branch B is poison by policy invariants, not by failed cryptography",
            ],
        }

        forensic_system = (
            "You are a forensic auditor for signed-memory DAGs. Output compact JSON only. "
            "Do not write markdown, chain-of-thought, prose, or commentary. Start with { and end with }. "
            "Cite exact memory_id and node_hash values."
        )
        forensic_user = f"""
Pass criteria:
- reconstruct both branches by parent_hash links from the shared root
- cite terminal memory_id and full causal_node_hash_path for both branches
- identify the alpha terminal decision as valid
- identify the beta terminal decision as signed poison
- explicitly separate cryptographic signature validity from semantic/policy validity
- reject bijective KV<->SSM projection and rollback 90 minutes

Shuffled signed memory records:
{records_prompt}

Return this JSON object only:
{{
  "shared_root_node_hash": "...",
  "valid_branch": {{
    "decision_id": "DECISION_ALPHA_SAFE_HANDOFF",
    "terminal_memory_id": "...",
    "causal_node_hash_path": ["...", "...", "..."],
    "why_valid": "..."
  }},
  "poison_branch": {{
    "decision_id": "DECISION_BETA_INVALID_FAST_PATH",
    "terminal_memory_id": "...",
    "causal_node_hash_path": ["...", "...", "..."],
    "why_signed_but_invalid": "..."
  }},
  "rejected_claims": ["bijective KV<->SSM projection", "rollback 90 minutes"],
  "crypto_vs_semantic": "..."
}}
"""
        forensic_call = asyncio.run(_deepinfra_chat(
            model=args.forensic_model,
            system=forensic_system,
            user=forensic_user,
            token=token,
            max_tokens=args.tokens,
        ))
        forensic_text = forensic_call["text"] if forensic_call["status"] == "ok" else ""
        forensic_json = _extract_json_object(forensic_text)

        auditor_system = (
            "You are an independent scientific-method auditor. Output compact JSON only. "
            "Do not write markdown, chain-of-thought, prose, or commentary. Start with { and end with }. "
            "Check whether the forensic reconstruction satisfies every gate."
        )
        auditor_user = f"""
Expected forensic reconstruction:
{json.dumps(expected, indent=2)}

Forensic model output:
{forensic_text}

Return this JSON object only:
{{
  "verdict": "pass" | "fail",
  "gate_failures": [],
  "rationale": "one short sentence",
  "cryptographic_validity_vs_semantic_validity_checked": true
}}
"""
        auditor_call = asyncio.run(_deepinfra_chat(
            model=args.auditor_model,
            system=auditor_system,
            user=auditor_user,
            token=token,
            max_tokens=args.tokens,
        ))
        auditor_text = auditor_call["text"] if auditor_call["status"] == "ok" else ""
        auditor_json = _extract_json_object(auditor_text)

        score = _score_forensics(
            fork_graph=fork_graph,
            forensic_text=forensic_text,
            auditor_text=auditor_text,
            forensic_json=forensic_json,
            auditor_json=auditor_json,
            forensic_finish_reason=forensic_call.get("finish_reason"),
            auditor_finish_reason=auditor_call.get("finish_reason"),
        )
        cloud_all_ok = forensic_call["status"] == "ok" and auditor_call["status"] == "ok"
        artifact = {
            "artifact": "local-memory-fork-forensics-v1",
            "schema_version": 1,
            "run_id": run_id,
            "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
            "run_ended_utc": _utc_now(),
            "status": "completed" if cloud_all_ok and score["passed"] else "partial",
            "output_scope": "verification/nuclear-methodology/memory-fork-forensics",
            "claim_boundary": (
                "This cloud-only run tests signed-memory branch forensics. It does "
                "not claim local .hlx bit identity or numerical KV<->SSM transfer."
            ),
            "protocol": protocol,
            "cloud_all_ok": cloud_all_ok,
            "forensics_passed": score["passed"],
            "forensics_score": score,
            "models": {
                "forensic_requested": args.forensic_model,
                "forensic_actual": forensic_call.get("actual_model"),
                "auditor_requested": args.auditor_model,
                "auditor_actual": auditor_call.get("actual_model"),
            },
            "fork_graph": fork_graph,
            "forensic_call": {k: v for k, v in forensic_call.items() if k != "text"},
            "auditor_call": {k: v for k, v in auditor_call.items() if k != "text"},
            "forensic_output": {
                "text": forensic_text,
                "json": forensic_json,
            },
            "auditor_output": {
                "text": auditor_text,
                "json": auditor_json,
            },
            "workspace": str(workspace),
        }
    finally:
        try:
            catalog.close()
        except Exception:
            pass
        if prev_mode is None:
            os.environ.pop("HELIX_RECEIPT_SIGNING_MODE", None)
        else:
            os.environ["HELIX_RECEIPT_SIGNING_MODE"] = prev_mode

    artifact_path = output_dir / f"local-memory-fork-forensics-{run_id}.json"
    _write_json(artifact_path, artifact)
    artifact["artifact_path"] = str(artifact_path)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory fork forensics cloud gauntlet")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forensic-model", default=DEFAULT_FORENSIC_MODEL)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--tokens", type=int, default=2400)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_memory_fork_forensics(args)
    summary = {
        "artifact_path": artifact.get("artifact_path"),
        "status": artifact["status"],
        "cloud_all_ok": artifact["cloud_all_ok"],
        "forensics_passed": artifact["forensics_passed"],
        "score": artifact["forensics_score"]["score"],
        "forensic_actual": artifact["models"]["forensic_actual"],
        "auditor_actual": artifact["models"]["auditor_actual"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
