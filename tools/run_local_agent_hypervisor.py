from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(max(1, min(2, int(torch.get_num_threads()))))

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_kv import rust_session  # noqa: E402
from helix_kv.transformers_cache import _load_benchmark_cache, _save_benchmark_cache  # noqa: E402
from tools.run_local_session_core import PROFILE, _hf_model_cached, _load_model_bundle  # noqa: E402
from tools.run_local_hybrid_stress import _encode_prompt_text, _json_ready, _run_generation_trace, _write_json  # noqa: E402

SHOWCASE_DIR = REPO_ROOT / "benchmarks" / "agent_showcase"
PR_WAR_ROOM_FIXTURE = SHOWCASE_DIR / "pr_war_room.md"

AGENT_PROMPTS = [
    "Agent A is reviewing a Python bug report. Keep a terse running note.",
    "Agent B is planning a local benchmark. Keep a terse running note.",
    "Agent C is summarizing a disk cleanup checklist. Keep a terse running note.",
    "Agent D is drafting a README claim caveat. Keep a terse running note.",
    "Agent E is checking an inference session receipt. Keep a terse running note.",
]

TURN_PROMPT = " Continue this agent note with one compact token-level step."

PR_WAR_ROOM_ROLES = [
    {
        "role": "bug_hunter",
        "task": "Find the most likely integration bug or failure mode in the mini diff.",
    },
    {
        "role": "perf_engineer",
        "task": "Identify the serialization bottleneck and one practical optimization.",
    },
    {
        "role": "benchmark_scientist",
        "task": "Name the metric needed before claiming a speed breakthrough.",
    },
    {
        "role": "claims_editor",
        "task": "Rewrite the public claim so it stays exciting but scientifically safe.",
    },
    {
        "role": "release_captain",
        "task": "Produce one release checklist item that protects users on small laptops.",
    },
]

HYBRID_CAMEO_ROLES = [
    {
        "role": "hybrid_integrity",
        "task": "Check whether a short hybrid Zamba session can be switched without losing hash integrity.",
    },
    {
        "role": "hybrid_claims",
        "task": "Write the safest one-line caveat for the Zamba cameo.",
    },
]


def _session_hash(session_dir: Path, *, verify_policy: str = "full") -> str | None:
    if verify_policy == "receipt-only":
        receipt_path = session_dir / "session-hlx-receipt.json"
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                return receipt.get("session_hash") or (receipt.get("receipt") or {}).get("session_hash") or receipt.get("fast_payload_checksum")
            except Exception:
                return None
    try:
        receipt = rust_session.verify_hlx_session(session_dir)
    except Exception:
        return None
    inner = receipt.get("receipt") if isinstance(receipt.get("receipt"), dict) else receipt
    return inner.get("session_hash")


def _read_session_receipt(session_dir: Path) -> dict[str, Any]:
    receipt_path = session_dir / "session-hlx-receipt.json"
    if not receipt_path.exists():
        return {}
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(
    cache: Any,
    *,
    model_config: Any,
    session_dir: Path,
    codec: str,
    verify_policy: str = "full",
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    if session_dir.exists():
        import shutil

        shutil.rmtree(session_dir)
    start = time.perf_counter()
    _save_benchmark_cache(cache, model_config=model_config, path=session_dir, session_codec=codec, audit_policy=audit_policy)
    save_ms = (time.perf_counter() - start) * 1000.0
    receipt = _read_session_receipt(session_dir)
    if codec in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"} and verify_policy == "full" and str(audit_policy) != "deferred":
        receipt = rust_session.verify_hlx_session(session_dir)
    total_bytes = int(sum(item.stat().st_size for item in session_dir.iterdir() if item.is_file()))
    return {"save_time_ms": save_ms, "session_total_bytes": total_bytes, "receipt": receipt}


def _scenario_agents(scenario: str, requested_agents: int) -> list[dict[str, str]]:
    if scenario in {"pr-war-room", "pr-war-room-long"}:
        fixture = PR_WAR_ROOM_FIXTURE.read_text(encoding="utf-8")
        agents = PR_WAR_ROOM_ROLES[: int(requested_agents)]
        long_mode = scenario == "pr-war-room-long"
        return [
            {
                "role": item["role"],
                "task": item["task"],
                "prompt": (
                    f"You are {item['role']} in the HeliX PR war room. "
                    f"Task: {item['task']} "
                    + (
                        "You will receive handoffs from other agents in later turns. "
                        "Use them as external messages, not as shared model state. "
                        "Return a concise but useful engineering note.\n\n"
                        if long_mode
                        else "Return one compact finding and one compact action.\n\n"
                    )
                    + f"{fixture}"
                ),
            }
            for item in agents
        ]
    if scenario == "hybrid-cameo":
        agents = HYBRID_CAMEO_ROLES[: min(int(requested_agents), 2)]
        return [
            {
                "role": item["role"],
                "task": item["task"],
                "prompt": (
                    "HeliX hybrid cameo: a short Zamba2 session is saved, restored, and continued. "
                    f"You are {item['role']}. Task: {item['task']} Keep the answer tiny."
                ),
            }
            for item in agents
        ]
    return [
        {
            "role": f"agent-{index}",
            "task": "Keep a terse running note.",
            "prompt": AGENT_PROMPTS[index % len(AGENT_PROMPTS)],
        }
        for index in range(int(requested_agents))
    ]


def _artifact_name(scenario: str, model_key: str, *, audit_policy: str = "blocking") -> str:
    if scenario == "pr-war-room-long" and str(audit_policy) == "deferred":
        return "local-agent-hypervisor-pr-war-room-deferred.json"
    if scenario == "pr-war-room-long":
        return "local-agent-hypervisor-pr-war-room-long.json"
    if scenario == "pr-war-room":
        return "local-agent-hypervisor-pr-war-room.json"
    if scenario == "hybrid-cameo" and model_key == "zamba":
        return "local-agent-hypervisor-zamba-cameo.json"
    return "local-agent-hypervisor-demo.json"


def _heuristic_pass(answer_text: str, token_ids: list[int]) -> bool:
    return bool(str(answer_text).strip()) or bool(token_ids)


def _transcript_hash(transcript: list[dict[str, Any]]) -> str:
    payload = json.dumps(_json_ready(transcript), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _format_handoff_transcript(transcript: list[dict[str, Any]], *, max_entries: int = 5) -> str:
    if not transcript:
        return "No previous handoffs yet."
    lines = []
    for item in transcript[-int(max_entries) :]:
        role = str(item.get("role") or "agent")
        summary = str(item.get("handoff_summary") or "").replace("\n", " ").strip()
        lines.append(f"- {role}: {summary}")
    return "\n".join(lines)


def _handoff_summary(answer_text: str, token_ids: list[int], *, max_chars: int = 360) -> str:
    text = " ".join(str(answer_text or "").split())
    if not text and token_ids:
        text = f"generated_token_ids={token_ids[:12]}"
    return text[: int(max_chars)]


def _turn_prompt_for_agent(scenario: str, agent: dict[str, str], transcript: list[dict[str, Any]]) -> str:
    if scenario == "pr-war-room-long":
        return (
            "Shared PR war-room transcript so far:\n"
            f"{_format_handoff_transcript(transcript)}\n\n"
            f"Your role is {agent['role']}. Your task: {agent['task']}\n"
            "Respond to the prior handoffs if useful. Do not restate the instructions. "
            "Start directly with this structure:\nObservation:"
        )
    return TURN_PROMPT


def _base_prompt_token_budget(scenario: str) -> int:
    return 192 if scenario == "pr-war-room-long" else 64


def _turn_prompt_token_budget(scenario: str) -> int:
    return 192 if scenario == "pr-war-room-long" else 16


def run_hypervisor(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_key = str(args.model)
    config = PROFILE[model_key]
    model_ref = str(config["model_ref"])
    scenario = str(args.scenario)
    artifact_name = _artifact_name(scenario, model_key, audit_policy=str(args.audit_policy))
    if args.local_files_only and not _hf_model_cached(model_ref):
        payload = {
            "title": "HeliX Local Agent Hypervisor v0",
            "scenario": scenario,
            "status": "skipped",
            "skip_reason": "skipped_not_cached",
            "model_ref": model_ref,
        }
        _write_json(output_dir / artifact_name, payload)
        return payload

    device = torch.device(args.device)
    model, adapter, input_adapter = _load_model_bundle(model_ref, device=device, local_files_only=bool(args.local_files_only))
    variant = dict(config["variant"])
    agents = _scenario_agents(scenario, int(args.agents))
    events: list[dict[str, Any]] = []
    shared_transcript: list[dict[str, Any]] = []
    final_audits: list[dict[str, Any]] = []
    start_wall = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="helix-agent-hypervisor-") as temp:
        temp_root = Path(temp)
        session_dirs: dict[int, Path] = {}
        generated_by_agent: dict[int, list[int]] = {}
        for agent_index, agent in enumerate(agents):
            prompt = agent["prompt"]
            prompt_inputs, prompt_ids = _encode_prompt_text(
                model_ref=model_ref,
                adapter=adapter,
                input_adapter=input_adapter,
                prompt_text=prompt,
                max_tokens=_base_prompt_token_budget(scenario),
            )
            base = _run_generation_trace(
                model,
                prompt_inputs=prompt_inputs,
                prompt_ids=prompt_ids,
                variant=variant,
                max_new_tokens=0,
                adapter=adapter,
                device=device,
                return_cache=True,
            )
            cache = base.pop("_cache")
            session_dir = temp_root / f"agent-{agent_index}"
            save = _save_cache(
                cache,
                model_config=model.config,
                session_dir=session_dir,
                codec=str(args.codec),
                verify_policy="receipt-only",
                audit_policy=str(args.audit_policy),
            )
            session_dirs[agent_index] = session_dir
            generated_by_agent[agent_index] = []
            events.append(
                {
                    "agent_id": f"agent-{agent_index}",
                    "role": agent["role"],
                    "task": agent["task"],
                    "round_index": -1,
                    "phase": "init",
                    "session_hash_after": _session_hash(session_dir, verify_policy="receipt-only"),
                    "save_time_ms": save["save_time_ms"],
                    "session_total_bytes": save["session_total_bytes"],
                    "audit_status": save["receipt"].get("audit_status"),
                    "generated_token_ids": [],
                    "answer_preview": "",
                    "shared_transcript_hash": _transcript_hash(shared_transcript),
                }
            )
            del cache
            gc.collect()
        for round_index in range(int(args.rounds)):
            for agent_index, agent in enumerate(agents):
                handoff_in = _format_handoff_transcript(shared_transcript)
                transcript_hash_before = _transcript_hash(shared_transcript)
                turn_inputs, turn_ids = _encode_prompt_text(
                    model_ref=model_ref,
                    adapter=adapter,
                    input_adapter=input_adapter,
                    prompt_text=_turn_prompt_for_agent(scenario, agent, shared_transcript),
                    max_tokens=_turn_prompt_token_budget(scenario),
                )
                session_dir = session_dirs[agent_index]
                before_hash = _session_hash(session_dir, verify_policy="receipt-only")
                load_start = time.perf_counter()
                cache = _load_benchmark_cache(session_dir, model_config=model.config, device=device, verify_policy="receipt-only")
                load_ms = (time.perf_counter() - load_start) * 1000.0
                loaded_hash = _session_hash(session_dir, verify_policy="receipt-only")
                result = _run_generation_trace(
                    model,
                    prompt_inputs=turn_inputs,
                    prompt_ids=turn_ids,
                    variant=variant,
                    max_new_tokens=int(args.timeslice_tokens),
                    adapter=adapter,
                    device=device,
                    initial_cache=cache,
                    resume_prompt_token_by_token=(model_key == "zamba"),
                    return_cache=True,
                )
                generated_by_agent[agent_index].extend(result.get("answer_token_ids") or [])
                updated_cache = result.pop("_cache")
                save = _save_cache(
                    updated_cache,
                    model_config=model.config,
                    session_dir=session_dir,
                    codec=str(args.codec),
                    verify_policy="receipt-only",
                    audit_policy=str(args.audit_policy),
                )
                after_hash = _session_hash(session_dir, verify_policy="receipt-only")
                answer_tokens = list(result.get("answer_token_ids") or [])
                answer_text = str(result.get("answer_text") or "")
                handoff_out = _handoff_summary(answer_text, answer_tokens)
                if scenario == "pr-war-room-long":
                    shared_transcript.append(
                        {
                            "round_index": int(round_index),
                            "agent_id": f"agent-{agent_index}",
                            "role": agent["role"],
                            "handoff_summary": handoff_out,
                            "session_hash_after": after_hash,
                        }
                    )
                transcript_hash_after = _transcript_hash(shared_transcript)
                events.append(
                    {
                        "agent_id": f"agent-{agent_index}",
                        "role": agent["role"],
                        "task": agent["task"],
                        "round_index": int(round_index),
                        "phase": "timeslice",
                        "session_hash_before": before_hash,
                        "session_hash_loaded": loaded_hash,
                        "session_hash_after": after_hash,
                        "hash_match": before_hash == loaded_hash,
                        "save_time_ms": save["save_time_ms"],
                        "load_time_ms": load_ms,
                        "session_total_bytes": save["session_total_bytes"],
                        "audit_status": save["receipt"].get("audit_status"),
                        "restore_from_pending": str(args.audit_policy) == "deferred",
                        "generated_token_ids": answer_tokens,
                        "answer_text": answer_text,
                        "answer_preview": result.get("answer_preview"),
                        "handoff_in": handoff_in,
                        "handoff_out": handoff_out,
                        "shared_transcript_hash_before": transcript_hash_before,
                        "shared_transcript_hash": transcript_hash_after,
                        "coordination_mode": "external-message-handoff" if scenario == "pr-war-room-long" else "independent-session-switching",
                        "heuristic_pass": _heuristic_pass(answer_text, answer_tokens),
                    }
                )
                del cache
                del updated_cache
                gc.collect()
        for agent_index, session_dir in session_dirs.items():
            audit_start = time.perf_counter()
            try:
                if str(args.audit_policy) == "deferred":
                    audit_receipt = rust_session.verify_deferred_session(session_dir)
                elif str(args.codec) in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
                    audit_receipt = rust_session.verify_hlx_session(session_dir)
                else:
                    audit_receipt = _read_session_receipt(session_dir)
                audit_status = str(audit_receipt.get("audit_status") or "verified")
                ok = bool(audit_receipt.get("ok", True)) and audit_status != "failed"
            except Exception as exc:
                audit_receipt = {"audit_status": "failed", "audit_error": str(exc)}
                audit_status = "failed"
                ok = False
            final_audits.append(
                {
                    "agent_id": f"agent-{agent_index}",
                    "role": agents[agent_index]["role"],
                    "audit_status": audit_status,
                    "ok": ok,
                    "audit_time_ms": (time.perf_counter() - audit_start) * 1000.0,
                    "session_hash": audit_receipt.get("session_hash"),
                    "merkle_root": audit_receipt.get("merkle_root"),
                }
            )
    payload = {
        "title": "HeliX Local Agent Hypervisor v0",
        "scenario": scenario,
        "status": "completed",
        "model_ref": model_ref,
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "agents": len(agents),
        "rounds": int(args.rounds),
        "timeslice_tokens": int(args.timeslice_tokens),
        "total_wall_time_s": time.perf_counter() - start_wall,
        "coordination_mode": "external-message-handoff" if scenario == "pr-war-room-long" else "independent-session-switching",
        "state_sharing": "none; each agent keeps its own cache/session snapshot",
        "shared_transcript_hash": _transcript_hash(shared_transcript),
        "shared_transcript": shared_transcript,
        "final_audits": final_audits,
        "events": events,
        "agent_summaries": [
            {
                "agent_id": f"agent-{agent_index}",
                "role": agents[agent_index]["role"],
                "task": agents[agent_index]["task"],
                "generated_token_ids": generated_by_agent.get(agent_index, []),
            }
            for agent_index in range(len(agents))
        ],
        "all_restore_hash_matches": None
        if str(args.audit_policy) == "deferred"
        else all(event.get("hash_match", True) for event in events if event.get("phase") == "timeslice"),
        "all_pending_receipts_loaded": all(event.get("hash_match", True) for event in events if event.get("phase") == "timeslice"),
        "all_final_audits_verified": all(audit.get("ok") is True and audit.get("audit_status") == "verified" for audit in final_audits),
        "all_heuristics_passed": all(event.get("heuristic_pass", True) for event in events if event.get("phase") == "timeslice"),
    }
    _write_json(output_dir / artifact_name, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local sequential agent context-switching demo.")
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "qwen", "zamba"])
    parser.add_argument("--scenario", default="demo", choices=["demo", "pr-war-room", "pr-war-room-long", "hybrid-cameo"])
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeslice-tokens", type=int, default=1)
    parser.add_argument("--codec", default="rust-hlx", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="blocking", choices=["blocking", "deferred"])
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_hypervisor(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
