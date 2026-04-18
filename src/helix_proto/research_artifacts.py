from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CANONICAL_RESEARCH_ARTIFACTS = {
    "remote-transformers-gpu-summary.json",
    "remote-qwen25-1.5b-transformers-gpu.json",
    "remote-smollm2-1.7b-transformers-gpu.json",
    "remote-qwen25-3b-transformers-gpu.json",
    "hybrid-memory-frontier-summary.json",
    "local-zamba2-hxq-vs-vanilla-summary.json",
    "local-zamba2-prompt-suite-code-daily.json",
    "local-zamba2-qmamba-runtime-cpu-smoke.json",
    "local-zamba2-vanilla-hybrid-cpu-smoke.json",
    "local-zamba2-stress-dashboard.json",
    "local-zamba2-stress-state-juggler.json",
    "local-zamba2-stress-context-switcher.json",
    "local-zamba2-stress-long-context.json",
    "local-zamba2-stress-restore-equivalence.json",
    "local-session-core-summary.json",
    "local-session-core-gpt2.json",
    "local-session-core-qwen.json",
    "local-session-core-zamba.json",
    "local-session-core-toolchain.json",
    "local-session-core-zero-copy-summary.json",
    "local-session-core-flattened-summary.json",
    "local-session-core-deferred-audit-summary.json",
    "local-agent-hypervisor-demo.json",
    "local-agent-hypervisor-pr-war-room.json",
    "local-agent-hypervisor-pr-war-room-long.json",
    "local-agent-hypervisor-pr-war-room-deferred.json",
    "local-agent-hypervisor-zamba-cameo.json",
    "local-multimodel-hypervisor-demo.json",
    "local-session-catalog-smoke.json",
    "local-prefix-reuse-ttft-summary.json",
    "local-session-os-demo.json",
    "local-segmented-session-summary.json",
    "local-hybrid-prefix-checkpoint-summary.json",
    "local-session-branching-summary.json",
    "local-agent-framework-showcase.json",
    "local-openai-compatible-smoke.json",
    "local-agent-memory-catalog-smoke.json",
    "local-hlx-layer-slice-smoke.json",
    "local-airllm-bridge-smoke.json",
    "local-memory-augmented-openai-smoke.json",
    "local-hmem-wiring-smoke.json",
    "local-inference-os-architecture-summary.json",
    "local-blueprint-stack-catalog.json",
    "local-blueprint-meta-microsite-demo.json",
    "local-blueprint-meta-microsite-real-cached.json",
    "local-blueprint-frontend-factory-smoke.json",
    "local-blueprint-framework-showcase.json",
    "local-memory-catalog-concurrency.json",
    "local-memory-decay-selection.json",
    "local-hlx-layer-chaos.json",
    "local-rust-python-layer-slice-soak.json",
    "local-airllm-real-smoke.json",
    "local-ttft-cold-warm-summary.json",
    "local-agent-capacity-budget.json",
    "agent-memory-comparison-summary.json",
    "helix-claims-matrix.json",
}

_ARTIFACT_TITLE_OVERRIDES = {
    "remote-transformers-gpu-summary.json": "Transformer GPU summary",
    "hybrid-memory-frontier-summary.json": "Hybrid memory frontier summary",
    "local-zamba2-hxq-vs-vanilla-summary.json": "Zamba2 local frontier summary",
    "local-zamba2-prompt-suite-code-daily.json": "Zamba2 prompt suite",
    "local-zamba2-hxq-direct-diagnostics.json": "HXQ direct diagnostics",
    "local-gemma-attempts.json": "Local Gemma attempts",
    "local-hxq-qwen3b-cpu-smoke.json": "HXQ Qwen3B CPU smoke",
    "local-zamba2-stress-dashboard.json": "Zamba2 stress missions dashboard",
    "local-zamba2-stress-state-juggler.json": "Stress mission: State Juggler",
    "local-zamba2-stress-context-switcher.json": "Stress mission: Context Switcher",
    "local-zamba2-stress-long-context.json": "Stress mission: Long-Context Coder",
    "local-zamba2-stress-restore-equivalence.json": "Stress mission: Restore Equivalence",
    "local-session-core-summary.json": "Local session core summary",
    "local-session-core-gpt2.json": "Session core: GPT-2",
    "local-session-core-qwen.json": "Session core: Qwen2.5 1.5B",
    "local-session-core-zamba.json": "Session core: Zamba2 hybrid",
    "local-session-core-toolchain.json": "Session core toolchain",
    "local-session-core-zero-copy-summary.json": "Session core zero-copy summary",
    "local-session-core-flattened-summary.json": "Session core tensor-flattening summary",
    "local-session-core-deferred-audit-summary.json": "Session core deferred audit summary",
    "local-agent-hypervisor-demo.json": "Agent hypervisor local demo",
    "local-agent-hypervisor-pr-war-room.json": "Agent hypervisor: PR War Room",
    "local-agent-hypervisor-pr-war-room-long.json": "Agent hypervisor: PR War Room Long",
    "local-agent-hypervisor-pr-war-room-deferred.json": "Agent hypervisor: PR War Room Deferred",
    "local-agent-hypervisor-zamba-cameo.json": "Agent hypervisor: Zamba cameo",
    "local-multimodel-hypervisor-demo.json": "Multimodel hypervisor demo",
    "local-session-catalog-smoke.json": "Session OS: catalog smoke",
    "local-prefix-reuse-ttft-summary.json": "Session OS: prefix reuse TTFT",
    "local-session-os-demo.json": "Session OS: scheduler demo",
    "local-segmented-session-summary.json": "Session OS: segmented sessions",
    "local-hybrid-prefix-checkpoint-summary.json": "Session OS: hybrid prefix checkpoint",
    "local-session-branching-summary.json": "Session OS: branching summary",
    "local-agent-framework-showcase.json": "Session OS: agent framework showcase",
    "local-openai-compatible-smoke.json": "Session OS: OpenAI-compatible smoke",
    "local-agent-memory-catalog-smoke.json": "Session OS: agent memory catalog",
    "local-hlx-layer-slice-smoke.json": "Session OS: .hlx layer-slice smoke",
    "local-airllm-bridge-smoke.json": "Session OS: AirLLM bridge smoke",
    "local-memory-augmented-openai-smoke.json": "Session OS: memory-augmented OpenAI smoke",
    "local-hmem-wiring-smoke.json": "Session OS: hmem wiring smoke",
    "local-inference-os-architecture-summary.json": "Inference OS architecture summary",
    "local-blueprint-stack-catalog.json": "Inference OS blueprint stack catalog",
    "local-blueprint-meta-microsite-demo.json": "Blueprint: Meta Microsite demo",
    "local-blueprint-meta-microsite-real-cached.json": "Blueprint: Meta Microsite real cached",
    "local-blueprint-frontend-factory-smoke.json": "Blueprint: Frontend Factory smoke",
    "local-blueprint-framework-showcase.json": "Blueprint: framework showcase",
    "local-memory-catalog-concurrency.json": "Session OS: MemoryCatalog concurrency",
    "local-memory-decay-selection.json": "Session OS: memory decay selection",
    "local-hlx-layer-chaos.json": "Session OS: .hlx layer chaos",
    "local-rust-python-layer-slice-soak.json": "Session OS: Rust/Python layer-slice soak",
    "local-airllm-real-smoke.json": "Session OS: optional AirLLM real smoke",
    "local-ttft-cold-warm-summary.json": "Agent memory: TTFT cold vs warm",
    "local-agent-capacity-budget.json": "Agent memory: capacity budget",
    "agent-memory-comparison-summary.json": "Agent memory comparison summary",
    "helix-claims-matrix.json": "HeliX claims matrix",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verification_root() -> Path:
    return _repo_root() / "verification"


def artifact_title(name: str) -> str:
    override = _ARTIFACT_TITLE_OVERRIDES.get(name)
    if override is not None:
        return override
    base = name.removesuffix(".json").replace("-", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in base.split())


def artifact_tags(name: str, payload: dict[str, Any]) -> list[str]:
    lowered = name.lower()
    tags: set[str] = set()
    payload_status = str(payload.get("status") or "")
    payload_blocked = payload.get("logits_finite") is False or payload_status.startswith("skipped") or payload_status == "failed"
    if name in _CANONICAL_RESEARCH_ARTIFACTS and not payload_blocked:
        tags.add("verified")
    elif name not in _CANONICAL_RESEARCH_ARTIFACTS:
        tags.add("exploratory")
    if lowered.startswith("remote-"):
        tags.add("gpu")
    if lowered.startswith("local-"):
        tags.add("local")
    if "hybrid" in lowered or "zamba2" in lowered:
        tags.add("hybrid")
    if "hxq" in lowered:
        tags.add("hxq")
    if "gemma" in lowered:
        tags.add("gemma")
    if "transformers" in lowered or "qwen" in lowered or "smollm" in lowered:
        tags.add("transformer")
    if "summary" in lowered:
        tags.add("summary")
    if "stress" in lowered:
        tags.add("stress")
    if "claims" in lowered:
        tags.add("claims")
    if "agent-memory" in lowered or payload.get("benchmark_kind", "").startswith("agent-memory"):
        tags.add("agent-memory")
    if "session-core" in lowered:
        tags.add("session-core")
    if "session-os" in lowered or str(payload.get("benchmark_kind", "")).startswith("session-os"):
        tags.add("session-os")
    if "hypervisor" in lowered:
        tags.add("hypervisor")
    if "airllm" in lowered:
        tags.add("airllm")
    if "blueprint" in lowered:
        tags.add("blueprint")
        tags.add("inference-os")
    if "inference-os" in lowered:
        tags.add("inference-os")
    if "layer-slice" in lowered:
        tags.add("layer-slice")
    if "concurrency" in lowered:
        tags.add("concurrency")
        tags.add("reliability")
    if "decay" in lowered:
        tags.add("memory-decay")
        tags.add("reliability")
    if "chaos" in lowered:
        tags.add("chaos")
        tags.add("reliability")
    if "soak" in lowered:
        tags.add("soak")
        tags.add("reliability")
    if "memory-catalog" in lowered or "memory-augmented" in lowered or "hmem" in lowered:
        tags.add("agent-recall")
    if "multimodel" in lowered or payload.get("demo") == "heterogeneous-multi-model":
        tags.add("multimodel")
    if "merkle" in lowered or "session" in lowered:
        tags.add("session")
    if "benchmark_kind" in payload:
        tags.add("benchmark")
    model_statuses: list[str] = []
    for key in ("models", "attempts"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    status = item.get("status")
                    if status is not None:
                        model_statuses.append(str(status))
    if (
        payload_blocked
        or any("blocked" in status or "failed" in status for status in model_statuses)
    ):
        tags.add("blocked")
    if lowered.startswith("remote-") and "summary" in lowered:
        tags.add("canonical")
    if lowered.startswith("local-zamba2") or lowered.startswith("hybrid-memory"):
        tags.add("frontier")
    return sorted(tags)


def artifact_model_refs(payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("model_ref", "vanilla_model_ref", "hxq_model_ref"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            refs.append(value)
    models = payload.get("models")
    if isinstance(models, dict):
        refs.extend(str(key) for key in models.keys())
    elif isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                model_ref = item.get("model_ref")
                if isinstance(model_ref, str) and model_ref:
                    refs.append(model_ref)
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        for item in attempts:
            if isinstance(item, dict):
                model_ref = item.get("model_ref")
                if isinstance(model_ref, str) and model_ref:
                    refs.append(model_ref)
    transformer_gpu = payload.get("transformer_gpu")
    if isinstance(transformer_gpu, dict):
        transformer_models = transformer_gpu.get("models")
        if isinstance(transformer_models, list):
            for item in transformer_models:
                if isinstance(item, dict):
                    model_ref = item.get("model_ref")
                    if isinstance(model_ref, str) and model_ref:
                        refs.append(model_ref)
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def artifact_headline_metrics(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "remote-transformers-gpu-summary.json":
        models = payload.get("models") or {}
        best_ratio = 0.0
        best_model = None
        for model_ref, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            row = model_data.get("turbo_int8k_4bitv") or {}
            ratio = float(row.get("kv_cache_ratio_vs_native") or 0.0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_model = str(model_ref)
        return {
            "best_fidelity_variant": (payload.get("best_fidelity_default") or {}).get("variant"),
            "best_compression_variant": (payload.get("best_compression_default") or {}).get("variant"),
            "best_transformer_kv_ratio_vs_native": best_ratio,
            "best_transformer_model_ref": best_model,
        }
    if name == "hybrid-memory-frontier-summary.json":
        hybrid_local = payload.get("hybrid_local") or {}
        combined = hybrid_local.get("combined_hybrid_gain") or {}
        return {
            "combined_runtime_cache_ratio_vs_native": combined.get("hybrid_total_runtime_cache_ratio_vs_native"),
            "combined_speedup_vs_native": combined.get("speedup_vs_native"),
            "hxq_comparison_available": (hybrid_local.get("vanilla_vs_hxq") or {}).get("comparison_available"),
        }
    if name == "local-zamba2-hxq-vs-vanilla-summary.json":
        combined = payload.get("combined hybrid gain") or {}
        return {
            "combined_runtime_cache_ratio_vs_native": combined.get("hybrid_total_runtime_cache_ratio_vs_native"),
            "combined_speedup_vs_native": combined.get("speedup_vs_native"),
            "combined_generated_match_vs_baseline": combined.get("generated_match_vs_baseline"),
        }
    if name == "local-zamba2-prompt-suite-code-daily.json":
        vanilla = (payload.get("aggregates") or {}).get("vanilla") or {}
        return {
            "code_avg_speedup_vs_native": (vanilla.get("code") or {}).get("avg_speedup_vs_native"),
            "daily_avg_speedup_vs_native": (vanilla.get("daily") or {}).get("avg_speedup_vs_native"),
        }
    if name == "local-zamba2-hxq-direct-diagnostics.json":
        return {
            "logits_finite": payload.get("logits_finite"),
            "nan_count": payload.get("nan_count"),
            "weight_runtime_source": payload.get("weight_runtime_source"),
        }
    if name == "local-gemma-attempts.json":
        attempts = payload.get("models") or payload.get("attempts") or []
        blocked = sum(1 for item in attempts if str(item.get("status")) == "blocked_before_benchmark")
        failed = sum(1 for item in attempts if "failed" in str(item.get("status")))
        return {
            "attempt_count": len(attempts),
            "blocked_attempt_count": blocked,
            "failed_attempt_count": failed,
        }
    if name == "local-zamba2-stress-dashboard.json":
        missions = payload.get("missions") or []
        return {
            "mission_count": len(missions),
            "profile": payload.get("profile"),
            "model_ref": payload.get("model_ref"),
        }
    if name == "local-zamba2-stress-state-juggler.json":
        metrics = payload.get("headline_metrics") or {}
        return {
            "hash_match": metrics.get("hash_match"),
            "session_total_bytes": metrics.get("session_total_bytes"),
            "load_time_ms": metrics.get("load_time_ms"),
        }
    if name == "local-zamba2-stress-context-switcher.json":
        metrics = payload.get("headline_metrics") or {}
        return {
            "logits_finite": metrics.get("logits_finite"),
            "promoted_block_total": metrics.get("promoted_block_total"),
            "runtime_cache_ratio": metrics.get("hybrid_total_runtime_cache_ratio_vs_native"),
        }
    if name == "local-zamba2-stress-long-context.json":
        metrics = payload.get("headline_metrics") or {}
        return {
            "identifier_recall_passes": metrics.get("identifier_recall_passes"),
            "best_runtime_ratio": metrics.get("best_runtime_ratio"),
            "best_speedup_vs_native": metrics.get("best_speedup_vs_native"),
        }
    if name == "local-zamba2-stress-restore-equivalence.json":
        metrics = payload.get("headline_metrics") or {}
        return {
            "hash_match": metrics.get("hash_match"),
            "generated_ids_match": metrics.get("generated_ids_match"),
            "max_abs_logit_delta": metrics.get("max_abs_logit_delta"),
        }
    if name == "local-session-core-summary.json":
        models = payload.get("models") or []
        completed = [model for model in models if isinstance(model, dict) and model.get("status") == "completed"]
        all_hash_match = all(model.get("hash_match") is True for model in completed) if completed else None
        all_generated_match = all(model.get("generated_ids_match") is True for model in completed) if completed else None
        return {
            "completed_count": payload.get("completed_count"),
            "failed_count": payload.get("failed_count"),
            "all_hash_match": all_hash_match,
            "all_generated_ids_match": all_generated_match,
        }
    if name in {
        "local-session-core-zero-copy-summary.json",
        "local-session-core-flattened-summary.json",
        "local-session-core-deferred-audit-summary.json",
    }:
        models = payload.get("models") or []
        completed = [model for model in models if isinstance(model, dict) and model.get("status") == "completed"]
        best_p50 = None
        best_flat_groups = None
        for model in completed:
            repeat = model.get("repeat_benchmark") or {}
            value = repeat.get("save_time_ms_p50")
            if isinstance(value, (int, float)):
                best_p50 = float(value) if best_p50 is None else min(best_p50, float(value))
            flat_groups = repeat.get("flat_group_count_median")
            if isinstance(flat_groups, (int, float)) and flat_groups > 0:
                best_flat_groups = int(flat_groups) if best_flat_groups is None else min(best_flat_groups, int(flat_groups))
        return {
            "completed_count": payload.get("completed_count"),
            "best_save_time_ms_p50": best_p50,
            "best_flat_group_count": best_flat_groups,
            "all_hash_match": all(model.get("hash_match") is True for model in completed) if completed else None,
            "audit_policy": payload.get("audit_policy"),
        }
    if name in {
        "local-session-core-gpt2.json",
        "local-session-core-qwen.json",
        "local-session-core-zamba.json",
    }:
        return {
            "status": payload.get("status"),
            "cache_kind": payload.get("cache_kind"),
            "hash_match": payload.get("hash_match"),
            "generated_ids_match": payload.get("generated_ids_match"),
            "rust_hlx_save_time_ms": payload.get("rust_hlx_save_time_ms"),
            "python_npz_save_time_ms": payload.get("python_npz_save_time_ms"),
        }
    if name == "local-session-core-toolchain.json":
        return {
            "pyo3_module_available": payload.get("pyo3_module_available"),
            "cli_available": payload.get("cli_available"),
            "gnu_target": payload.get("gnu_target"),
        }
    if name == "local-agent-hypervisor-demo.json":
        return {
            "status": payload.get("status"),
            "agents": payload.get("agents"),
            "rounds": payload.get("rounds"),
            "all_restore_hash_matches": payload.get("all_restore_hash_matches"),
            "total_wall_time_s": payload.get("total_wall_time_s"),
        }
    if name in {
        "local-agent-hypervisor-pr-war-room.json",
        "local-agent-hypervisor-pr-war-room-long.json",
        "local-agent-hypervisor-pr-war-room-deferred.json",
        "local-agent-hypervisor-zamba-cameo.json",
    }:
        return {
            "status": payload.get("status"),
            "scenario": payload.get("scenario"),
            "agents": payload.get("agents"),
            "rounds": payload.get("rounds"),
            "all_restore_hash_matches": payload.get("all_restore_hash_matches"),
            "all_pending_receipts_loaded": payload.get("all_pending_receipts_loaded"),
            "all_final_audits_verified": payload.get("all_final_audits_verified"),
            "all_heuristics_passed": payload.get("all_heuristics_passed"),
            "coordination_mode": payload.get("coordination_mode"),
        }
    if name == "local-multimodel-hypervisor-demo.json":
        return {
            "status": payload.get("status"),
            "models_used": payload.get("models_used"),
            "model_swaps": payload.get("model_swaps"),
            "restored_session_count": payload.get("restored_session_count"),
            "all_restore_hash_matches": payload.get("all_restore_hash_matches"),
            "all_final_audits_verified": payload.get("all_final_audits_verified"),
            "coordination_mode": payload.get("coordination_mode"),
        }
    if name == "local-session-catalog-smoke.json":
        return {
            "status": payload.get("status"),
            "session_count": (payload.get("catalog_stats") or {}).get("session_count"),
            "prefix_match_status": (payload.get("prefix_match") or {}).get("status"),
            "traversal_rejected": payload.get("traversal_rejected"),
        }
    if name == "local-prefix-reuse-ttft-summary.json":
        models = payload.get("models") or []
        completed = [model for model in models if isinstance(model, dict) and model.get("status") == "completed"]
        claim_rows = [
            model
            for model in completed
            if isinstance(model.get("claim_speedup_including_restore"), (int, float))
        ]
        best_claim = max(
            (float(model.get("claim_speedup_including_restore") or 0.0) for model in claim_rows),
            default=None,
        )
        return {
            "completed_count": len(completed),
            "best_claimed_prefix_reuse_speedup": best_claim,
            "claim_variants": [model.get("claim_variant") for model in claim_rows if model.get("claim_variant")],
            "scope": payload.get("claim_boundary"),
        }
    if name == "local-session-os-demo.json":
        return {
            "status": payload.get("status"),
            "models_used": payload.get("models_used"),
            "model_swaps": payload.get("model_swaps"),
            "all_final_audits_verified": payload.get("all_final_audits_verified"),
        }
    if name == "local-segmented-session-summary.json":
        return {
            "status": payload.get("status"),
            "segment_chain_length": payload.get("segment_chain_length"),
            "verify_chain_status": payload.get("verify_chain_status"),
            "rewrite_avoided_bytes": payload.get("rewrite_avoided_bytes"),
        }
    if name == "local-hybrid-prefix-checkpoint-summary.json":
        return {
            "status": payload.get("status"),
            "prefix_kind": payload.get("prefix_kind"),
            "top1_match_all": payload.get("top1_match_all"),
            "finite_all": payload.get("finite_all"),
            "best_speedup": payload.get("best_speedup"),
        }
    if name == "local-session-branching-summary.json":
        return {
            "status": payload.get("status"),
            "branch_count": payload.get("branch_count"),
            "verify_chain_status": payload.get("verify_chain_status"),
            "rewrite_avoided_bytes_estimate": payload.get("rewrite_avoided_bytes_estimate"),
        }
    if name == "local-agent-framework-showcase.json":
        return {
            "status": payload.get("status"),
            "client": payload.get("client"),
            "agent_count": payload.get("agent_count"),
            "server_mode": payload.get("server_mode"),
        }
    if name == "local-openai-compatible-smoke.json":
        return {
            "status": payload.get("status"),
            "object": payload.get("object"),
            "session_recorded": payload.get("session_recorded"),
        }
    if name == "local-agent-memory-catalog-smoke.json":
        return {
            "status": payload.get("status"),
            "fts_enabled": payload.get("fts_enabled"),
            "privacy_redaction_ok": payload.get("privacy_redaction_ok"),
            "search_hit_count": payload.get("search_hit_count"),
            "context_tokens": payload.get("context_tokens"),
        }
    if name == "local-hlx-layer-slice-smoke.json":
        return {
            "status": payload.get("status"),
            "read_mode": payload.get("read_mode"),
            "selected_array_count": payload.get("selected_array_count"),
            "unrelated_array_loaded": payload.get("unrelated_array_loaded"),
            "tamper_detected": payload.get("tamper_detected"),
        }
    if name == "local-airllm-bridge-smoke.json":
        return {
            "status": payload.get("status"),
            "bridge_mode": payload.get("bridge_mode"),
            "airllm_dependency_required": payload.get("airllm_dependency_required"),
            "all_layer_injections_hit": payload.get("all_layer_injections_hit"),
            "total_injected_arrays": payload.get("total_injected_arrays"),
        }
    if name == "local-memory-augmented-openai-smoke.json":
        return {
            "status": payload.get("status"),
            "client_surface": payload.get("client_surface"),
            "memory_context_injected": payload.get("memory_context_injected"),
            "memory_context_tokens": (payload.get("response_helix") or {}).get("memory_context_tokens"),
        }
    if name == "local-hmem-wiring-smoke.json":
        acceptance = payload.get("acceptance") or {}
        agent_run = payload.get("agent_run") or {}
        memory_api = payload.get("memory_api") or {}
        return {
            "status": payload.get("status"),
            "auto_tool_observe": acceptance.get("auto_tool_observe"),
            "startup_context_injected": acceptance.get("startup_context_injected"),
            "hybrid_search_returns_memory_and_knowledge": acceptance.get("hybrid_search_returns_memory_and_knowledge"),
            "memory_context_tokens": agent_run.get("memory_context_tokens"),
            "graph_node_count": memory_api.get("graph_node_count"),
        }
    if name == "local-inference-os-architecture-summary.json":
        return {
            "status": payload.get("status"),
            "layer_count": len(payload.get("layers") or []),
            "public_wording": payload.get("public_wording"),
        }
    if name == "local-blueprint-stack-catalog.json":
        return {
            "status": payload.get("status"),
            "stack_count": payload.get("stack_count"),
            "stack_ids": [item.get("id") for item in payload.get("stacks", [])],
        }
    if name in {"local-blueprint-meta-microsite-demo.json", "local-blueprint-meta-microsite-real-cached.json"}:
        quality = payload.get("quality_checks") or {}
        return {
            "status": payload.get("status"),
            "mode": payload.get("mode"),
            "quality_status": quality.get("status"),
            "html_bytes": quality.get("html_bytes"),
            "final_audit_status": payload.get("final_audit_status"),
            "fallback_content_used": payload.get("fallback_content_used"),
            "public_claim_level": payload.get("public_claim_level"),
            "real_generation_event_count": len(payload.get("real_generation_events") or []),
        }
    if name == "local-blueprint-frontend-factory-smoke.json":
        return {
            "status": payload.get("status"),
            "blueprint_id": payload.get("blueprint_id"),
            "smoke_scope": payload.get("smoke_scope"),
        }
    if name == "local-blueprint-framework-showcase.json":
        return {
            "status": payload.get("status"),
            "client_surface": payload.get("client_surface"),
            "blueprints_can_target_openai_compatible_api": payload.get("blueprints_can_target_openai_compatible_api"),
        }
    if name == "local-memory-catalog-concurrency.json":
        return {
            "status": payload.get("status"),
            "worker_count": payload.get("worker_count"),
            "lost_observations": payload.get("lost_observations"),
            "lost_memories": payload.get("lost_memories"),
            "write_errors": payload.get("write_errors"),
            "journal_mode": payload.get("journal_mode"),
        }
    if name == "local-memory-decay-selection.json":
        return {
            "status": payload.get("status"),
            "memory_count": payload.get("memory_count"),
            "critical_retained_count": payload.get("critical_retained_count"),
            "critical_retained_all": payload.get("critical_retained_all"),
            "noise_selected_count": payload.get("noise_selected_count"),
        }
    if name == "local-hlx-layer-chaos.json":
        return {
            "status": payload.get("status"),
            "tamper_detected": payload.get("tamper_detected"),
            "full_verify_blocked_injection": payload.get("full_verify_blocked_injection"),
            "tampered_array": (payload.get("tamper") or {}).get("array_name"),
        }
    if name == "local-rust-python-layer-slice-soak.json":
        return {
            "status": payload.get("status"),
            "iteration_count": payload.get("iteration_count"),
            "rss_delta_bytes": payload.get("rss_delta_bytes"),
            "rss_growth_pct": payload.get("rss_growth_pct"),
            "load_time_ms_p95": payload.get("load_time_ms_p95"),
            "error_count": payload.get("error_count"),
        }
    if name == "local-airllm-real-smoke.json":
        return {
            "status": payload.get("status"),
            "airllm_dependency_available": payload.get("airllm_dependency_available"),
            "model_path": payload.get("model_path"),
            "generated_token_count": payload.get("generated_token_count"),
            "real_airllm_injection_supported": payload.get("real_airllm_injection_supported"),
        }
    if name == "local-ttft-cold-warm-summary.json":
        models = payload.get("models") or []
        completed = [model for model in models if isinstance(model, dict) and model.get("status") == "completed"]
        best_including = None
        best_compute = None
        best_model = None
        for model in completed:
            including = model.get("ttft_speedup_including_restore")
            if isinstance(including, (int, float)) and (best_including is None or float(including) > best_including):
                best_including = float(including)
                compute = model.get("ttft_speedup_compute_only")
                best_compute = float(compute) if isinstance(compute, (int, float)) else None
                best_model = model.get("model_ref")
        return {
            "completed_count": payload.get("completed_count"),
            "best_ttft_speedup_including_restore": best_including,
            "best_ttft_speedup_compute_only": best_compute,
            "best_model_ref": best_model,
        }
    if name == "local-agent-capacity-budget.json":
        rows = payload.get("rows") or []
        measured = [row for row in rows if isinstance(row, dict) and row.get("projection") is False]
        projected = [row for row in rows if isinstance(row, dict) and row.get("projection") is True]
        best_measured = max((int(row.get("agents_fit") or 0) for row in measured), default=None)
        best_projected = max((int(row.get("agents_fit") or 0) for row in projected), default=None)
        return {
            "budget_bytes": payload.get("budget_bytes"),
            "row_count": len(rows),
            "best_measured_agents_fit": best_measured,
            "best_projected_agents_fit": best_projected,
        }
    if name == "agent-memory-comparison-summary.json":
        return {
            "competitor_arxiv_id": (payload.get("competitor_paper") or {}).get("arxiv_id"),
            "helix_best_ttft_speedup_including_restore": payload.get("helix_best_ttft_speedup_including_restore"),
            "helix_best_ttft_speedup_compute_only": payload.get("helix_best_ttft_speedup_compute_only"),
            "capacity_rows": payload.get("capacity_rows"),
        }
    if name == "helix-claims-matrix.json":
        claims = payload.get("claims") or []
        return {
            "claim_count": len(claims),
            "verified_claim_count": sum(1 for claim in claims if claim.get("status") == "verified"),
            "blocked_claim_count": sum(1 for claim in claims if claim.get("status") == "blocked"),
        }
    return {}


def load_research_artifact(name: str, verification_root: Path | None = None) -> dict[str, Any]:
    root = (verification_root or verification_root_default()).resolve()
    normalized = str(name).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.json", normalized):
        raise FileNotFoundError("artifact not found")
    path = (root / normalized).resolve()
    if not str(path).startswith(str(root)) or not path.exists():
        raise FileNotFoundError("artifact not found")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verification_root_default() -> Path:
    return verification_root()


def artifact_status(name: str, payload: dict[str, Any]) -> str:
    payload_status = str(payload.get("status") or "")
    if payload.get("logits_finite") is False or payload_status == "failed":
        return "blocked"
    if payload_status.startswith("skipped"):
        return "blocked"
    if name in _CANONICAL_RESEARCH_ARTIFACTS:
        return "verified"
    return "exploratory"


def research_artifact_manifest(verification_root: Path | None = None) -> list[dict[str, Any]]:
    root = (verification_root or verification_root_default()).resolve()
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = load_research_artifact(path.name, verification_root=root)
        artifacts.append(
            {
                "name": path.name,
                "title": artifact_title(path.name),
                "category": "canonical evidence" if path.name in _CANONICAL_RESEARCH_ARTIFACTS else "lab notebook",
                "status": artifact_status(path.name, payload),
                "benchmark_kind": payload.get("benchmark_kind"),
                "model_refs": artifact_model_refs(payload),
                "tags": artifact_tags(path.name, payload),
                "headline_metrics": artifact_headline_metrics(path.name, payload),
                "source_path": str(path),
                "raw_available": True,
            }
        )
    artifacts.sort(
        key=lambda item: (
            0 if item["category"] == "canonical evidence" else 1,
            0 if item["status"] == "verified" else 1,
            str(item["title"]).lower(),
        )
    )
    return artifacts


def wrapped_research_artifact(name: str, verification_root: Path | None = None) -> dict[str, Any]:
    return {
        "name": str(name),
        "title": artifact_title(str(name)),
        "payload": load_research_artifact(name, verification_root=verification_root),
    }
