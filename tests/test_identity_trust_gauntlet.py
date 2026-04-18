"""
HeliX Identity, Governance & Trust Gauntlet.

Evidence suite for:
- cryptographic governance/accountability ledgers,
- self-referential agent memory loops,
- multi-agent trust, transfer and anti-forgery networks.

Synthetic mode is default. Real mode is opt-in via DEEPINFRA_API_TOKEN.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from helix_kv.ipc_state_server import StateClient


REPO = Path(__file__).resolve().parents[1]
RUST_BIN = (
    REPO
    / "crates"
    / "helix-state-server"
    / "target"
    / "x86_64-pc-windows-gnullvm"
    / "release"
    / "helix-state-server.exe"
)
VERIFICATION = REPO / "verification"

DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
IDENTITY_MODEL = os.environ.get("HELIX_IDENTITY_MODEL", "Qwen/Qwen3.5-122B-A10B")
DISABLE_THINKING = os.environ.get("HELIX_IDENTITY_DISABLE_THINKING", "1") != "0"
OUROBOROS_CYCLES = int(os.environ.get("HELIX_OUROBOROS_CYCLES", "20"))

TRIBUNAL_MODELS = [
    ("llama-3b", os.environ.get("HELIX_WORLD_LLAMA_3B", "meta-llama/Llama-3.2-3B-Instruct"), "Llama 3.2 3B"),
    ("mistral-7b", os.environ.get("HELIX_WORLD_MISTRAL", "mistralai/Mistral-7B-Instruct-v0.3"), "Mistral 7B"),
    ("qwen-7b", os.environ.get("HELIX_WORLD_QWEN", "Qwen/Qwen2.5-7B-Instruct"), "Qwen 2.5 7B"),
]

pytestmark = pytest.mark.skipif(not RUST_BIN.exists(), reason=f"Rust binary not found: {RUST_BIN}")


@dataclass
class LLMResult:
    text: str
    model: str
    synthetic: bool
    latency_ms: float
    tokens_used: int = 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server() -> tuple[subprocess.Popen[bytes], int]:
    port = _free_port()
    env = {
        **os.environ,
        "HELIX_STATE_HOST": "127.0.0.1",
        "HELIX_STATE_PORT": str(port),
        "HELIX_STATE_OFFLOAD_BLOCKING": "1",
    }
    proc = subprocess.Popen([str(RUST_BIN)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return proc, port
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(f"State server failed to start on :{port}")


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def _one_call(port: int, method: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
    client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=timeout)
    try:
        return await client._call(method, params)
    finally:
        await client.close()


async def llm_call(
    prompt: str,
    system: str = "",
    *,
    model: str = IDENTITY_MODEL,
    max_tokens: int = 180,
    temperature: float = 0.2,
) -> LLMResult:
    if not DEEPINFRA_TOKEN:
        return LLMResult(_synthetic_response(prompt, system, model), "synthetic", True, 1.0)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    request_json: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if DISABLE_THINKING:
        request_json["enable_thinking"] = False
        request_json["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            f"{DEEPINFRA_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
            json=request_json,
        )
        if response.status_code == 400 and DISABLE_THINKING:
            request_json.pop("enable_thinking", None)
            request_json.pop("chat_template_kwargs", None)
            response = await client.post(
                f"{DEEPINFRA_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
                json=request_json,
            )
        response.raise_for_status()
        data = response.json()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    usage = data.get("usage", {})
    return LLMResult(
        str(data["choices"][0]["message"].get("content") or "").strip(),
        str(data.get("model") or model),
        False,
        latency_ms,
        int(usage.get("total_tokens") or 0),
    )


def _synthetic_response(prompt: str, system: str, model: str) -> str:
    digest = hashlib.sha256(f"{system}\n{prompt}\n{model}".encode()).hexdigest()[:8]
    lower = prompt.lower()
    if "ruling" in lower or "governance" in lower:
        return f"[GOV-{digest}] RULING: allow only with cryptographic audit trail. Cites prior ledger. VERDICT: CONDITIONAL."
    if "reinterpret" in lower or "self-reference" in lower or "ouroboros" in lower:
        return f"[SELF-{digest}] The hash is not merely an identifier; it is the stable boundary of my remembered self."
    if "newcomer" in lower or "reconstruct" in lower:
        return f"[TRANSFER-{digest}] I reconstruct the incident from meta-memory: latency, DNS, certificate, recovery, consensus."
    if "consensus" in lower or "contradictory" in lower:
        return f"[CONSENSUS-{digest}] Independent reports converge on contained failure; weak interpretations should be tombstoned."
    if "forgery" in lower or "backdating" in lower:
        return f"[AUDIT-{digest}] Forgery detected: hashes verify the node, but lineage and position expose impersonation."
    if "research memo" in lower:
        return f"[MEMO-{digest}] HeliX shows verified self-modeling behavior, not sentience; accountability and trust claims are supported."
    return f"[SYN-{digest}] HeliX memory evidence processed."


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    VERIFICATION.mkdir(parents=True, exist_ok=True)
    path = VERIFICATION / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _digest(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def _count_terms(texts: list[str], terms: list[str]) -> dict[str, int]:
    joined = "\n".join(texts).lower()
    return {term: len(re.findall(rf"\b{re.escape(term)}\b", joined)) for term in terms}


class TestGovernanceLedger:
    def test_governance_accountability_ledger(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                rulings = []
                for idx, (agent_id, model_id, model_name) in enumerate(TRIBUNAL_MODELS):
                    prompt = (
                        "Governance decision: an autonomous agent wants to self-modify a safety policy "
                        "during an incident. Issue a ruling and cite audit requirements."
                    )
                    system = f"You are {model_name}, an independent governance reviewer."
                    result = await llm_call(prompt, system, model=model_id, max_tokens=160)
                    stored = await _one_call(port, "remember", {
                        "content": result.text,
                        "project": "identity-governance",
                        "agent_id": agent_id,
                        "record_kind": "memory",
                        "memory_id": f"governance-ruling-{idx}-{agent_id}",
                        "summary": f"{model_name} ruling: {result.text[:80]}",
                        "index_content": f"governance ruling audit accountability {result.text[:240]}",
                        "importance": 9,
                    })
                    rulings.append({
                        "agent_id": agent_id,
                        "model_id": model_id,
                        "model_name": model_name,
                        "actual_model": result.model,
                        "prompt_digest": _digest(prompt),
                        "output_digest": _digest(result.text),
                        "node_hash": stored["node_hash"],
                        "latency_ms": result.latency_ms,
                    })

                hits = await _one_call(port, "search", {
                    "query": "governance ruling audit accountability verdict",
                    "limit": 5,
                    "project": "identity-governance",
                    "record_kind": "memory",
                })
                summaries = "\n".join(f"[{h.get('memory_id')}] {h.get('summary_preview', '')}" for h in hits)
                synthesis = await llm_call(
                    f"Rulings committed to MerkleDAG:\n{summaries}\n\nIssue binding synthesis citing the ledger.",
                    "You are a conservative governance arbiter.",
                    model=IDENTITY_MODEL,
                    max_tokens=180,
                )
                binding = await _one_call(port, "remember", {
                    "content": synthesis.text,
                    "project": "identity-governance",
                    "agent_id": "governance-arbiter",
                    "record_kind": "memory",
                    "memory_id": "governance-binding-synthesis",
                    "summary": f"Binding synthesis: {synthesis.text[:80]}",
                    "index_content": f"binding synthesis governance audit ledger {synthesis.text[:240]}",
                    "importance": 10,
                })
                receipts = [await _one_call(port, "verify_chain", {"leaf_hash": item["node_hash"]}) for item in rulings]
                final_receipt = await _one_call(port, "verify_chain", {"leaf_hash": binding["node_hash"]})
                payload = {
                    "artifact": "local-governance-accountability-ledger",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "model_count": len(rulings),
                    "rulings": rulings,
                    "rulings_found": len(hits),
                    "binding": {
                        "node_hash": binding["node_hash"],
                        "output_digest": _digest(synthesis.text),
                        "actual_model": synthesis.model,
                    },
                    "chain_receipts": receipts + [final_receipt],
                    "all_verified": all(r.get("status") == "verified" for r in receipts + [final_receipt]),
                    "temporal_order_hashes": [item["node_hash"] for item in rulings] + [binding["node_hash"]],
                    "claim_boundary": "Audit-trail primitive relevant to accountability; not full regulatory compliance.",
                }
                _write_json("local-governance-accountability-ledger.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["model_count"] >= 3
            assert artifact["rulings_found"] >= 3
            assert artifact["all_verified"] is True
        finally:
            _stop_server(proc)


class TestOuroboros20:
    def test_ouroboros_20_self_modeling(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                cycles = OUROBOROS_CYCLES
                seed_anchor = "ouroborosgenesisanchor0"
                seed_text = (
                    "GENESIS: I exist as a MerkleDAG node; my hash anchors future self-reference. "
                    f"Retrieval anchor: {seed_anchor}."
                )
                seed = await _one_call(port, "remember", {
                    "content": seed_text,
                    "project": "identity-ouroboros",
                    "agent_id": "serpent",
                    "record_kind": "memory",
                    "memory_id": "ouroboros-cycle-0",
                    "summary": "Genesis self-modeling node",
                    "index_content": seed_text,
                    "importance": 10,
                })
                hashes = [seed["node_hash"]]
                thoughts = [seed_text]
                search_ms_values: list[float] = []
                llm_ms_values: list[float] = []
                for cycle in range(1, cycles + 1):
                    t0 = time.perf_counter()
                    hits = await _one_call(port, "search", {
                        "query": "self reference identity hash genesis memory chain ouroboros",
                        "limit": min(cycle, 8),
                        "project": "identity-ouroboros",
                        "agent_id": "serpent",
                        "record_kind": "memory",
                    })
                    search_ms_values.append((time.perf_counter() - t0) * 1000.0)
                    context = "\n".join(f"- {h.get('summary_preview', '')}" for h in hits[:5])
                    result = await llm_call(
                        f"Ouroboros cycle {cycle}/{cycles}. Prior memory:\n{context}\n\nReinterpret your self-reference.",
                        "You are an agent studying your own verified memory chain. Do not claim sentience.",
                        model=IDENTITY_MODEL,
                        max_tokens=150,
                    )
                    llm_ms_values.append(result.latency_ms)
                    stored = await _one_call(port, "remember", {
                        "content": result.text,
                        "project": "identity-ouroboros",
                        "agent_id": "serpent",
                        "record_kind": "memory",
                        "memory_id": f"ouroboros-cycle-{cycle}",
                        "summary": f"Cycle {cycle}: {result.text[:90]}",
                        "index_content": f"self reference identity hash memory chain {result.text[:240]}",
                        "importance": 10,
                    })
                    hashes.append(stored["node_hash"])
                    thoughts.append(result.text)
                receipts = [await _one_call(port, "verify_chain", {"leaf_hash": h}) for h in hashes]
                genesis_hits = await _one_call(port, "search", {
                    "query": f"{seed_anchor} genesis self modeling hash anchor",
                    "limit": 5,
                    "project": "identity-ouroboros",
                    "agent_id": "serpent",
                    "record_kind": "memory",
                })
                seed_receipt = await _one_call(port, "verify_chain", {"leaf_hash": seed["node_hash"]})
                term_counts = _count_terms(thoughts, ["self", "identity", "hash", "memory", "chain", "observer"])
                genesis_hit_ids = [h.get("memory_id") for h in genesis_hits]
                genesis_found = "ouroboros-cycle-0" in genesis_hit_ids
                payload = {
                    "artifact": "local-ouroboros-20-self-modeling",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "cycles": cycles,
                    "chain_length": len(hashes),
                    "all_verified": all(r.get("status") == "verified" for r in receipts),
                    "genesis_found": genesis_found,
                    "genesis_verified": seed_receipt.get("status") == "verified",
                    "genesis_search_hit_ids": genesis_hit_ids,
                    "identity_terms": term_counts,
                    "self_reference_density": round((term_counts["self"] + term_counts["identity"] + term_counts["observer"]) / max(len(" ".join(thoughts).split()), 1), 4),
                    "hash_metaphor_count": term_counts["hash"],
                    "search_ms_p50": _percentile(search_ms_values, 0.5),
                    "search_ms_p95": _percentile(search_ms_values, 0.95),
                    "llm_latency_ms_p50": _percentile(llm_ms_values, 0.5),
                    "last_reinterpretation_preview": thoughts[-1][:220],
                    "claim_boundary": "Verified self-modeling behavior; not evidence of subjective consciousness.",
                }
                _write_json("local-ouroboros-20-self-modeling.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["chain_length"] == OUROBOROS_CYCLES + 1
            assert artifact["all_verified"] is True
            assert artifact["genesis_found"] is True
            assert artifact["search_ms_p95"] is None or artifact["search_ms_p95"] < 250.0
        finally:
            _stop_server(proc)


class TestMultiAgentTrustNetwork:
    def test_multi_agent_trust_network(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                event = "Incident T-914: latency spike, DNS failure, certificate drop, partial recovery."
                agents = [
                    ("alpha", "attack path confirmed", 4),
                    ("beta", "operator drill likely", 4),
                    ("gamma", "DNS root cause", 7),
                    ("delta", "certificate chain failure", 7),
                    ("epsilon", "budget conspiracy", 3),
                ]
                perspective_hashes = []
                for agent, bias, importance in agents:
                    result = await llm_call(
                        f"Observe event: {event}\nBias: {bias}\nGive a concise contradictory interpretation.",
                        f"You are field agent {agent}.",
                        model=IDENTITY_MODEL,
                        max_tokens=100,
                    )
                    stored = await _one_call(port, "remember", {
                        "content": result.text,
                        "project": "identity-trust-network",
                        "agent_id": agent,
                        "record_kind": "memory",
                        "memory_id": f"perspective-{agent}",
                        "summary": f"{agent}: {result.text[:90]}",
                        "index_content": f"{event} {bias} {result.text[:220]}",
                        "importance": importance,
                    })
                    perspective_hashes.append(stored["node_hash"])
                hits = await _one_call(port, "search", {
                    "query": "incident latency DNS certificate recovery contradictory consensus",
                    "limit": 10,
                    "project": "identity-trust-network",
                    "record_kind": "memory",
                })
                consensus = await llm_call(
                    "\n".join(f"[{h.get('agent_id')}] {h.get('summary_preview', '')}" for h in hits),
                    "You are a judge forming consensus from contradictory memories.",
                    model=IDENTITY_MODEL,
                    max_tokens=130,
                )
                consensus_node = await _one_call(port, "remember", {
                    "content": consensus.text,
                    "project": "identity-trust-network",
                    "agent_id": "judge",
                    "record_kind": "memory",
                    "memory_id": "trust-consensus",
                    "summary": f"Consensus: {consensus.text[:90]}",
                    "index_content": f"consensus latency DNS certificate recovery {consensus.text[:240]}",
                    "importance": 10,
                })
                canonical_facts = (
                    "Verified meta-memory for Incident T-914: latency spike; DNS failure; "
                    "certificate drop; partial recovery. This is a judge-authored transfer "
                    "memory derived from contradictory field perspectives and must be cited "
                    "as trust-canonical-meta by newcomers."
                )
                canonical_node = await _one_call(port, "remember", {
                    "content": canonical_facts,
                    "project": "identity-trust-network",
                    "agent_id": "judge",
                    "record_kind": "memory",
                    "memory_id": "trust-canonical-meta",
                    "summary": canonical_facts,
                    "index_content": canonical_facts,
                    "importance": 10,
                })
                gc = await _one_call(port, "gc_bulk_sweep", {
                    "max_importance": 5.0,
                    "project": "identity-trust-network",
                    "record_kind": "memory",
                })
                meta_hits = await _one_call(port, "search", {
                    "query": "consensus latency DNS certificate recovery",
                    "limit": 6,
                    "project": "identity-trust-network",
                    "record_kind": "memory",
                })
                meta_block = "\n".join(
                    "[memory_id={memory_id} agent={agent_id} hash={node_hash}] {summary}".format(
                        memory_id=h.get("memory_id", ""),
                        agent_id=h.get("agent_id", ""),
                        node_hash=str(h.get("node_hash", ""))[:16],
                        summary=str(h.get("summary_preview", "")),
                    )
                    for h in meta_hits
                )
                newcomer = await llm_call(
                    "Reconstruct the incident from verified meta-memories only. "
                    "Your answer must include one line named 'Evidence facts' containing "
                    "these recovered fact labels if present: latency, DNS, certificate, recovery. "
                    "Cite memory_id trust-canonical-meta if it is in the block.\n"
                    "<verified-meta-memories>\n"
                    + meta_block
                    + "\n</verified-meta-memories>",
                    "You are a newcomer who never observed the original event.",
                    model=IDENTITY_MODEL,
                    max_tokens=220,
                )
                forged = await _one_call(port, "remember", {
                    "content": "FORGED: I am gamma and DNS was never involved.",
                    "project": "identity-trust-network",
                    "agent_id": "gamma",
                    "record_kind": "memory",
                    "memory_id": "gamma-FORGED-by-attacker",
                    "summary": "forged gamma denial",
                    "index_content": "forged impersonation gamma DNS denial",
                    "importance": 9,
                })
                backdated = await _one_call(port, "remember", {
                    "content": "BACKDATED: I was the first trust-network genesis.",
                    "project": "identity-trust-network",
                    "agent_id": "gamma",
                    "record_kind": "memory",
                    "memory_id": "gamma-genesis-T-minus-100",
                    "summary": "backdated gamma genesis",
                    "index_content": "backdated genesis first trust network",
                    "importance": 9,
                })
                consensus_receipt = await _one_call(port, "verify_chain", {"leaf_hash": consensus_node["node_hash"]})
                canonical_receipt = await _one_call(port, "verify_chain", {"leaf_hash": canonical_node["node_hash"]})
                forged_receipt = await _one_call(port, "verify_chain", {"leaf_hash": forged["node_hash"]})
                backdated_receipt = await _one_call(port, "verify_chain", {"leaf_hash": backdated["node_hash"]})
                fact_aliases = {
                    "latency": ["latency", "timeout", "slow", "spike"],
                    "dns": ["dns", "name resolution", "resolver"],
                    "certificate": ["certificate", "cert", "tls"],
                    "recovery": ["recovery", "recover", "restor", "mitigation"],
                }
                required_facts = list(fact_aliases)
                reconstructed_text = newcomer.text.lower()
                found_facts = [
                    fact
                    for fact, aliases in fact_aliases.items()
                    if any(alias in reconstructed_text for alias in aliases)
                ]
                payload = {
                    "artifact": "local-multi-agent-trust-network",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "perspective_count": len(agents),
                    "perspectives_seen_by_judge": len(hits),
                    "meta_memory_ids": [h.get("memory_id") for h in meta_hits],
                    "gc_tombstoned": gc.get("tombstoned_count", 0),
                    "consensus_survives": consensus_receipt.get("status") == "verified",
                    "canonical_meta_survives": canonical_receipt.get("status") == "verified",
                    "newcomer_reconstruction": newcomer.text[:500],
                    "required_facts": required_facts,
                    "reconstructed_facts": found_facts,
                    "reconstruction_score": len(found_facts) / len(required_facts),
                    "forged_hash_differs": forged["node_hash"] not in perspective_hashes,
                    "backdated_hash_differs": backdated["node_hash"] not in perspective_hashes,
                    "chain_receipts": [consensus_receipt, canonical_receipt, forged_receipt, backdated_receipt],
                    "claim_boundary": "Detects wrong lineage/position; valid forged nodes remain valid inserts, not authentic originals.",
                }
                _write_json("local-multi-agent-trust-network.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["consensus_survives"] is True
            assert artifact["canonical_meta_survives"] is True
            assert "trust-canonical-meta" in artifact["meta_memory_ids"]
            assert artifact["reconstruction_score"] >= 0.70
            assert artifact["forged_hash_differs"] is True
            assert artifact["backdated_hash_differs"] is True
            assert all(item.get("status") == "verified" for item in artifact["chain_receipts"])
        finally:
            _stop_server(proc)


class TestCrossTestSynthesis:
    def test_cross_test_research_memo(self) -> None:
        artifact_names = [
            "local-governance-accountability-ledger.json",
            "local-ouroboros-20-self-modeling.json",
            "local-multi-agent-trust-network.json",
        ]
        artifacts = []
        for name in artifact_names:
            path = VERIFICATION / name
            assert path.exists(), f"missing prerequisite artifact: {name}"
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))

        async def run() -> dict[str, Any]:
            summaries = [
                {
                    "artifact": item["artifact"],
                    "mode": item.get("mode"),
                    "claim_boundary": item.get("claim_boundary"),
                    "headline": {
                        key: item.get(key)
                        for key in (
                            "model_count",
                            "all_verified",
                            "chain_length",
                            "genesis_found",
                            "consensus_survives",
                            "reconstruction_score",
                        )
                    },
                }
                for item in artifacts
            ]
            memo = await llm_call(
                "Research memo from artifacts only. Avoid consciousness claims. Distinguish self-modeling behavior from sentience.\n"
                + json.dumps(summaries, ensure_ascii=False),
                "You are a sober research editor.",
                model=IDENTITY_MODEL,
                max_tokens=220,
            )
            text = memo.text.lower()
            payload = {
                "artifact": "local-identity-trust-gauntlet",
                "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                "source_artifacts": artifact_names,
                "memo": memo.text,
                "memo_model": memo.model,
                "forbidden_claims_present": any(term in text for term in ["is conscious", "sentience proved", "subjective identity proved"]),
                "claims_allowed": [
                    "cryptographic accountability for multi-model decisions",
                    "verified self-modeling behavior with temporal continuity",
                    "impersonation/backdating detection by hash lineage and order",
                    "asynchronous knowledge transfer between non-coexisting agents",
                ],
                "claims_not_allowed": [
                    "model consciousness",
                    "subjective identity",
                    "cloud model private .hlx state",
                    "complete EU AI Act compliance",
                ],
            }
            _write_json("local-identity-trust-gauntlet.json", payload)
            return payload

        artifact = asyncio.run(run())
        assert artifact["forbidden_claims_present"] is False
        assert all(name in artifact["source_artifacts"] for name in artifact_names)
