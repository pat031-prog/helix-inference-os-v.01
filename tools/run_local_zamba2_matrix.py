from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from helix_kv.transformers_cache import (
    _DEFAULT_MAMBA_STATE_BLOCK_SIZE,
    _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT,
    _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD,
    _DEFAULT_MAMBA_STATE_SCALE_FLOOR,
    _canonical_model_ref,
    _load_text_adapter,
    build_transformers_hybrid_state_variants,
    run_transformers_kv_benchmark,
    run_transformers_model_diagnostics,
)


PROMPT_SUITE: dict[str, dict[str, str]] = {
    "code_completion": {
        "category": "code",
        "text": (
            "Write a Python function merge_intervals(intervals) that merges overlapping intervals. "
            "Handle empty input, already sorted input, and nested ranges. Keep the implementation short."
        ),
    },
    "code_debug": {
        "category": "code",
        "text": (
            "Explain and fix a bug in a simple LRU cache where get() forgets to refresh recency. "
            "Show the corrected Python code and mention one edge case."
        ),
    },
    "daily_planning": {
        "category": "daily",
        "text": (
            "Plan a simple 3-day trip in Buenos Aires for someone on a moderate budget. "
            "Include neighborhoods, public transport, and one food suggestion per day."
        ),
    },
    "daily_buying": {
        "category": "daily",
        "text": (
            "Evaluate whether buying a used laptop for studying is a good idea. "
            "Give a checklist covering battery, screen, thermals, charger, and SSD health."
        ),
    },
}
QUALITATIVE_PROMPT_KEYS = ("code_completion", "daily_planning")
SMOKE_PROMPT = (
    "Explain briefly why a hybrid Zamba2 cache must treat Mamba recurrent state separately from Transformer KV cache."
)
PUBLIC_LOCAL_PROFILES: dict[str, dict[str, Any]] = {
    "laptop-12gb": {
        "vanilla_model_ref": "Zyphra/Zamba2-1.2B-Instruct-v2",
        "hxq_model_ref": "EchoLabs33/zamba2-1.2b-hxq",
        "prompt_length": 16,
        "qualitative_max_new_tokens": 0,
        "kv_hot_window": 4,
        "kv_quant_seed": 7,
        "mamba_state_block_size": 64,
        "mamba_state_scale_floor": 1e-8,
        "mamba_state_clip_threshold_pct": 2.0,
        "mamba_state_rel_rmse_threshold": 0.2,
        "mamba_state_auto_promote": False,
    }
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _public_prompt_category(value: str) -> str:
    return "agentic" if str(value) == "code" else str(value)


def _apply_public_local_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile_name = str(getattr(args, "public_local_profile", "") or "laptop-12gb")
    profile = dict(PUBLIC_LOCAL_PROFILES.get(profile_name, PUBLIC_LOCAL_PROFILES["laptop-12gb"]))
    if bool(getattr(args, "budgeted_local", False)):
        profile["qualitative_max_new_tokens"] = 0
    return profile


def _arg_or_profile(args: argparse.Namespace, name: str, profile: dict[str, Any]) -> Any:
    value = getattr(args, name, None)
    return profile[name] if value is None else value


def _smoke_variants(*, kv_quant_seed: int, kv_hot_window: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "native-dense",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
            "mamba_state_precision": "native-dense",
        },
        {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "mamba_state_precision": "native-dense",
        },
    ]


def _combined_variants(
    *,
    kv_quant_seed: int,
    kv_hot_window: int,
    mamba_state_block_size: int = _DEFAULT_MAMBA_STATE_BLOCK_SIZE,  # type: ignore[name-defined]
    mamba_state_scale_floor: float = _DEFAULT_MAMBA_STATE_SCALE_FLOOR,  # type: ignore[name-defined]
    mamba_state_clip_threshold_pct: float = _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT,  # type: ignore[name-defined]
    mamba_state_rel_rmse_threshold: float = _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD,  # type: ignore[name-defined]
    mamba_state_auto_promote: bool = True,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "native-dense",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
            "mamba_state_precision": "native-dense",
        },
        {
            "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "mamba_state_precision": "q-mamba-dsq-int4",
            "mamba_state_block_size": int(mamba_state_block_size),
            "mamba_state_scale_floor": float(mamba_state_scale_floor),
            "mamba_state_clip_threshold_pct": float(mamba_state_clip_threshold_pct),
            "mamba_state_rel_rmse_threshold": float(mamba_state_rel_rmse_threshold),
            "mamba_state_auto_promote": bool(mamba_state_auto_promote),
        },
    ]


def _row_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    for row in report.get("rows", []):
        if str(row.get("name")) == str(name):
            return dict(row)
    raise KeyError(name)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_ref": report.get("model_ref"),
        "device": report.get("device"),
        "prompt_length": report.get("prompt_length"),
        "max_new_tokens": report.get("max_new_tokens"),
        "variant_order": report.get("variant_order"),
        "rows": report.get("rows"),
    }


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def _comparison_summary(report: dict[str, Any], *, baseline: str, target: str) -> dict[str, Any]:
    baseline_row = _row_by_name(report, baseline)
    target_row = _row_by_name(report, target)
    baseline_runtime = baseline_row.get("hybrid_total_runtime_cache_bytes") or baseline_row.get("hybrid_total_cache_bytes")
    target_runtime = target_row.get("hybrid_total_runtime_cache_bytes") or target_row.get("hybrid_total_cache_bytes")
    return {
        "baseline_variant": baseline,
        "target_variant": target,
        "speedup_vs_native": target_row.get("speedup_vs_native"),
        "kv_cache_ratio_vs_native": target_row.get("kv_cache_ratio_vs_native"),
        "mamba_state_runtime_ratio_vs_native": target_row.get("mamba_state_runtime_ratio_vs_native"),
        "hybrid_total_runtime_cache_ratio_vs_native": _safe_ratio(baseline_runtime, target_runtime),
        "prompt_perplexity_delta_pct_vs_native": target_row.get("prompt_perplexity_delta_pct_vs_native"),
        "generated_match_vs_baseline": target_row.get("generated_match_vs_baseline"),
    }


def _load_decoder(model_ref: str) -> Any | None:
    adapter, _, processor_used, _ = _load_text_adapter(
        model_ref,
        local_files_only=False,
        trust_remote_code=False,
    )
    if adapter is None:
        return None
    if processor_used and hasattr(adapter, "tokenizer"):
        return adapter.tokenizer
    return adapter


def _decode_ids(
    decoder: Any | None,
    token_ids: list[int],
    *,
    skip_special_tokens: bool,
) -> str | None:
    if decoder is None:
        return None
    if not token_ids:
        return ""
    try:
        return str(decoder.decode(list(token_ids), skip_special_tokens=skip_special_tokens))
    except Exception:
        return None


def _augment_with_decoded_suffixes(report: dict[str, Any], *, decoder: Any | None) -> dict[str, Any]:
    compact = _compact_report(report)
    prompt_ids = list(report.get("prompt_ids") or [])
    generated_ids_by_variant: dict[str, list[int]] = {}
    generated_suffix_ids_by_variant: dict[str, list[int]] = {}
    decoded: dict[str, str | None] = {}
    decoded_raw: dict[str, str | None] = {}
    decoded_full: dict[str, str | None] = {}
    decoded_full_raw: dict[str, str | None] = {}
    for row in report.get("rows", []):
        variant_name = str(row["name"])
        generated_ids = list(row.get("generated_ids") or [])
        suffix_ids = generated_ids[len(prompt_ids) :]
        generated_ids_by_variant[variant_name] = generated_ids
        generated_suffix_ids_by_variant[variant_name] = suffix_ids
        decoded[variant_name] = _decode_ids(decoder, suffix_ids, skip_special_tokens=True)
        decoded_raw[variant_name] = _decode_ids(decoder, suffix_ids, skip_special_tokens=False)
        decoded_full[variant_name] = _decode_ids(decoder, generated_ids, skip_special_tokens=True)
        decoded_full_raw[variant_name] = _decode_ids(decoder, generated_ids, skip_special_tokens=False)
    compact["prompt_ids"] = prompt_ids
    compact["benchmark_prompt_text"] = report.get("benchmark_prompt_text")
    compact["generated_ids_by_variant"] = generated_ids_by_variant
    compact["generated_suffix_ids_by_variant"] = generated_suffix_ids_by_variant
    compact["decoded_suffix_by_variant"] = decoded
    compact["decoded_suffix_raw_by_variant"] = decoded_raw
    compact["decoded_full_text_by_variant"] = decoded_full
    compact["decoded_full_text_raw_by_variant"] = decoded_full_raw
    return compact


def _aggregate_prompt_runs(prompt_runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in prompt_runs:
        grouped.setdefault(_public_prompt_category(str(item["category"])), []).append(item)
    aggregates: dict[str, Any] = {}
    for category, items in grouped.items():
        speedups: list[float] = []
        perplexity_deltas: list[float] = []
        runtime_ratios: list[float] = []
        for item in items:
            rows = item["report"]["rows"]
            native_row = _row_by_name({"rows": rows}, "native-dense")
            combined_row = _row_by_name({"rows": rows}, "turbo-int8-hadamard+q-mamba-dsq-int4")
            speedups.append(float(combined_row["speedup_vs_native"]))
            perplexity_deltas.append(float(combined_row["prompt_perplexity_delta_pct_vs_native"]))
            native_runtime = native_row.get("hybrid_total_runtime_cache_bytes") or native_row.get("hybrid_total_cache_bytes")
            combined_runtime = combined_row.get("hybrid_total_runtime_cache_bytes") or combined_row.get("hybrid_total_cache_bytes")
            if native_runtime and combined_runtime:
                runtime_ratios.append(float(native_runtime) / float(combined_runtime))
        aggregates[category] = {
            "prompt_count": len(items),
            "avg_speedup_vs_native": float(np.mean(speedups)) if speedups else None,
            "avg_prompt_perplexity_delta_pct_vs_native": float(np.mean(perplexity_deltas)) if perplexity_deltas else None,
            "avg_hybrid_total_runtime_cache_ratio_vs_native": float(np.mean(runtime_ratios)) if runtime_ratios else None,
        }
    return aggregates


def _build_summary(
    *,
    vanilla_smoke: dict[str, Any],
    qmamba_smoke: dict[str, Any],
    prompt_suite: dict[str, Any],
    hxq_diagnostics: dict[str, Any],
    hxq_smoke: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = {
        "KV-only gain": _comparison_summary(vanilla_smoke, baseline="native-dense", target="turbo-int8-hadamard"),
        "Mamba-state-only gain": _comparison_summary(qmamba_smoke, baseline="native-dense", target="q-mamba-dsq-int4"),
        "combined hybrid gain": _comparison_summary(
            qmamba_smoke,
            baseline="native-dense",
            target="turbo-int8-hadamard+q-mamba-dsq-int4",
        ),
        "prompt_category_aggregates": prompt_suite.get("aggregates"),
        "vanilla vs HXQ": {
            "hxq_logits_finite": hxq_diagnostics.get("logits_finite"),
            "hxq_load_error": hxq_diagnostics.get("load_error"),
            "hxq_forward_error": hxq_diagnostics.get("forward_error"),
            "comparison_available": hxq_smoke is not None,
        },
    }
    if hxq_smoke is not None:
        summary["vanilla vs HXQ"]["native_dense_total_time_ratio"] = _safe_ratio(
            _row_by_name(vanilla_smoke, "native-dense").get("total_time_s"),
            _row_by_name(hxq_smoke, "native-dense").get("total_time_s"),
        )
        summary["vanilla vs HXQ"]["native_dense_kv_ratio"] = _safe_ratio(
            _row_by_name(vanilla_smoke, "native-dense").get("hybrid_total_runtime_cache_bytes")
            or _row_by_name(vanilla_smoke, "native-dense").get("hybrid_total_cache_bytes"),
            _row_by_name(hxq_smoke, "native-dense").get("hybrid_total_runtime_cache_bytes")
            or _row_by_name(hxq_smoke, "native-dense").get("hybrid_total_cache_bytes"),
        )
    return summary


def run_local_zamba2_matrix(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = _apply_public_local_profile(args)
    vanilla_ref = str(_arg_or_profile(args, "vanilla_model_ref", profile))
    hxq_ref = _canonical_model_ref(_arg_or_profile(args, "hxq_model_ref", profile))
    kv_quant_seed = int(_arg_or_profile(args, "kv_quant_seed", profile))
    kv_hot_window = int(_arg_or_profile(args, "kv_hot_window", profile))
    prompt_length = int(_arg_or_profile(args, "prompt_length", profile))
    qualitative_max_new_tokens = int(_arg_or_profile(args, "qualitative_max_new_tokens", profile))
    receipts_path = output_dir / "local-zamba2-state-receipts.jsonl.gz"
    if receipts_path.exists():
        receipts_path.unlink()
    receipt_run_id = f"local-zamba2-{uuid.uuid4().hex[:12]}"
    qmamba_variants = build_transformers_hybrid_state_variants(
        kv_quant_seed=kv_quant_seed,
        kv_hot_window=kv_hot_window,
        mamba_state_block_size=int(profile["mamba_state_block_size"]),
        mamba_state_scale_floor=float(profile["mamba_state_scale_floor"]),
        mamba_state_clip_threshold_pct=float(profile["mamba_state_clip_threshold_pct"]),
        mamba_state_rel_rmse_threshold=float(profile["mamba_state_rel_rmse_threshold"]),
        mamba_state_auto_promote=bool(profile["mamba_state_auto_promote"]),
        mamba_receipts_enabled=True,
        mamba_receipts_path=receipts_path,
        mamba_receipt_run_id=receipt_run_id,
    )

    vanilla_smoke = run_transformers_kv_benchmark(
        vanilla_ref,
        prompt_text=SMOKE_PROMPT,
        prompt_length=prompt_length,
        max_new_tokens=1,
        warmup_max_new_tokens=0,
        kv_variants=_smoke_variants(kv_quant_seed=kv_quant_seed, kv_hot_window=kv_hot_window),
        device=args.device,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )

    qmamba_smoke = run_transformers_kv_benchmark(
        vanilla_ref,
        prompt_text=SMOKE_PROMPT,
        prompt_length=prompt_length,
        max_new_tokens=1,
        warmup_max_new_tokens=0,
        kv_variants=qmamba_variants,
        device=args.device,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )

    hxq_diagnostics = run_transformers_model_diagnostics(
        hxq_ref,
        prompt_text=SMOKE_PROMPT,
        prompt_length=prompt_length,
        device=args.device,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    hxq_smoke: dict[str, Any] | None = None
    if hxq_diagnostics.get("logits_finite") is True:
        hxq_smoke = run_transformers_kv_benchmark(
            hxq_ref,
            prompt_text=SMOKE_PROMPT,
            prompt_length=prompt_length,
            max_new_tokens=1,
            warmup_max_new_tokens=0,
            kv_variants=_smoke_variants(kv_quant_seed=kv_quant_seed, kv_hot_window=kv_hot_window),
            device=args.device,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
        )
    hxq_diag_payload = {
        "profile": str(getattr(args, "public_local_profile", "laptop-12gb") or "laptop-12gb"),
        "requested_model_ref": hxq_diagnostics.get("requested_model_ref"),
        "effective_model_ref": hxq_diagnostics.get("effective_model_ref"),
        "logits_finite": hxq_diagnostics.get("logits_finite"),
        "load_error": hxq_diagnostics.get("load_error"),
        "forward_error": hxq_diagnostics.get("forward_error"),
        "nan_count": hxq_diagnostics.get("nan_count"),
        "inf_count": hxq_diagnostics.get("inf_count"),
        "model_dtype": hxq_diagnostics.get("model_dtype"),
        "model_device": hxq_diagnostics.get("model_device"),
        "weight_runtime_source": hxq_diagnostics.get("weight_runtime_source"),
        "followup_smoke_report": None if hxq_smoke is None else _compact_report(hxq_smoke),
    }
    hxq_diag_path = _write_json(output_dir / "local-zamba2-hxq-finiteness.json", hxq_diag_payload)

    prompt_suite_payload: dict[str, Any] = {
        "profile": str(getattr(args, "public_local_profile", "laptop-12gb") or "laptop-12gb"),
        "vanilla_model_ref": vanilla_ref,
        "hxq_model_ref": hxq_ref,
        "device": args.device,
        "qualitative_prompt_keys": list(QUALITATIVE_PROMPT_KEYS) if qualitative_max_new_tokens > 0 else [],
        "prompt_suite": [],
        "qualitative_suite": [],
        "aggregates": {},
    }
    vanilla_decoder = _load_decoder(vanilla_ref)
    hxq_decoder = _load_decoder(hxq_ref) if hxq_smoke is not None else None
    for model_label, model_ref, decoder, enabled in [
        ("vanilla", vanilla_ref, vanilla_decoder, True),
        ("hxq", hxq_ref, hxq_decoder, hxq_smoke is not None),
    ]:
        if not enabled:
            continue
        quantitative_runs: list[dict[str, Any]] = []
        qualitative_runs: list[dict[str, Any]] = []
        for prompt_key, prompt in PROMPT_SUITE.items():
            report = run_transformers_kv_benchmark(
                model_ref,
                prompt_text=prompt["text"],
                prompt_length=prompt_length,
                max_new_tokens=1,
                warmup_max_new_tokens=0,
                kv_variants=_combined_variants(
                    kv_quant_seed=kv_quant_seed,
                    kv_hot_window=kv_hot_window,
                    mamba_state_block_size=int(profile["mamba_state_block_size"]),
                    mamba_state_scale_floor=float(profile["mamba_state_scale_floor"]),
                    mamba_state_clip_threshold_pct=float(profile["mamba_state_clip_threshold_pct"]),
                    mamba_state_rel_rmse_threshold=float(profile["mamba_state_rel_rmse_threshold"]),
                    mamba_state_auto_promote=bool(profile["mamba_state_auto_promote"]),
                ),
                device=args.device,
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
            )
            combined_row = _row_by_name(report, "turbo-int8-hadamard+q-mamba-dsq-int4")
            quantitative_runs.append(
                {
                    "model_label": model_label,
                    "prompt_key": prompt_key,
                    "category": _public_prompt_category(prompt["category"]),
                    "prompt_text": prompt["text"],
                    "report": {
                        "variant_order": report.get("variant_order"),
                        "rows": report.get("rows"),
                    },
                    "summary": {
                        "speedup_vs_native": combined_row.get("speedup_vs_native"),
                        "prompt_perplexity_delta_pct_vs_native": combined_row.get("prompt_perplexity_delta_pct_vs_native"),
                        "hybrid_total_runtime_cache_ratio_vs_native": _safe_ratio(
                            _row_by_name(report, "native-dense").get("hybrid_total_runtime_cache_bytes")
                            or _row_by_name(report, "native-dense").get("hybrid_total_cache_bytes"),
                            combined_row.get("hybrid_total_runtime_cache_bytes")
                            or combined_row.get("hybrid_total_cache_bytes"),
                        ),
                        "generated_match_vs_baseline": combined_row.get("generated_match_vs_baseline"),
                    },
                }
            )
            if qualitative_max_new_tokens > 0 and prompt_key in QUALITATIVE_PROMPT_KEYS:
                qualitative_report = run_transformers_kv_benchmark(
                    model_ref,
                    prompt_text=prompt["text"],
                    prompt_length=prompt_length,
                    max_new_tokens=qualitative_max_new_tokens,
                    warmup_max_new_tokens=0,
                    kv_variants=_combined_variants(
                        kv_quant_seed=kv_quant_seed,
                        kv_hot_window=kv_hot_window,
                        mamba_state_block_size=int(profile["mamba_state_block_size"]),
                        mamba_state_scale_floor=float(profile["mamba_state_scale_floor"]),
                        mamba_state_clip_threshold_pct=float(profile["mamba_state_clip_threshold_pct"]),
                        mamba_state_rel_rmse_threshold=float(profile["mamba_state_rel_rmse_threshold"]),
                        mamba_state_auto_promote=bool(profile["mamba_state_auto_promote"]),
                    ),
                    device=args.device,
                    local_files_only=bool(args.local_files_only),
                    trust_remote_code=bool(args.trust_remote_code),
                )
                qualitative_runs.append(
                    {
                        "model_label": model_label,
                        "prompt_key": prompt_key,
                        "category": _public_prompt_category(prompt["category"]),
                        "prompt_text": prompt["text"],
                        "report": _augment_with_decoded_suffixes(qualitative_report, decoder=decoder),
                    }
                )
        prompt_suite_payload["prompt_suite"].extend(quantitative_runs)
        prompt_suite_payload["qualitative_suite"].extend(qualitative_runs)
        prompt_suite_payload["aggregates"][model_label] = _aggregate_prompt_runs(quantitative_runs)
    prompt_suite_path = _write_json(output_dir / "local-zamba2-agentic-vs-daily.json", prompt_suite_payload)

    summary_payload = _build_summary(
        vanilla_smoke=vanilla_smoke,
        qmamba_smoke=qmamba_smoke,
        prompt_suite=prompt_suite_payload,
        hxq_diagnostics=hxq_diagnostics,
        hxq_smoke=hxq_smoke,
    )
    summary_payload["profile"] = str(getattr(args, "public_local_profile", "laptop-12gb") or "laptop-12gb")
    summary_payload["local_budget"] = {
        "budgeted_local": bool(getattr(args, "budgeted_local", False)),
        "qualitative_enabled": bool(qualitative_max_new_tokens > 0),
        "receipts_path": receipts_path.name,
        "max_artifact_size_mb": 100,
    }
    summary_payload["artifacts"] = {
        "vanilla_smoke": "embedded-in-summary",
        "hxq_diagnostics": hxq_diag_path.name,
        "qmamba_smoke": "embedded-in-summary",
        "prompt_suite": prompt_suite_path.name,
        "state_receipts": receipts_path.name,
    }
    summary_payload["stability_runs"] = {
        "vanilla_smoke": _compact_report(vanilla_smoke),
        "qmamba_smoke": _compact_report(qmamba_smoke),
    }
    summary_path = _write_json(output_dir / "local-zamba2-stability-summary.json", summary_payload)
    if receipts_path.exists() and receipts_path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError(f"receipts artifact exceeded local budget: {receipts_path}")
    return {
        "state_receipts": receipts_path,
        "vanilla_smoke": summary_path,
        "hxq_diagnostics": hxq_diag_path,
        "qmamba_smoke": summary_path,
        "prompt_suite": prompt_suite_path,
        "summary": summary_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Zamba2 vanilla/HXQ/Q-Mamba comparison matrix.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("verification"))
    parser.add_argument("--public-local-profile", default="laptop-12gb")
    parser.add_argument("--budgeted-local", action="store_true")
    parser.add_argument("--vanilla-model-ref")
    parser.add_argument("--hxq-model-ref")
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--qualitative-max-new-tokens", type=int)
    parser.add_argument("--kv-quant-seed", type=int)
    parser.add_argument("--kv-hot-window", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = run_local_zamba2_matrix(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
