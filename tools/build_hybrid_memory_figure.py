from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_hybrid_memory_summary(
    *,
    transformer_summary: dict[str, Any],
    hybrid_summary: dict[str, Any],
) -> dict[str, Any]:
    transformer_models: list[dict[str, Any]] = []
    for model_ref, model_data in (transformer_summary.get("models") or {}).items():
        fidelity = model_data.get("turbo_int8_hadamard") or {}
        compression = model_data.get("turbo_int8k_4bitv") or {}
        transformer_models.append(
            {
                "model_ref": model_ref,
                "best_fidelity_variant": "turbo-int8-hadamard",
                "best_fidelity_kv_ratio_vs_native": fidelity.get("kv_cache_ratio_vs_native"),
                "best_fidelity_prompt_perplexity_delta_pct_vs_native": fidelity.get(
                    "prompt_perplexity_delta_pct_vs_native"
                ),
                "best_fidelity_match_vs_baseline": fidelity.get("generated_match_vs_baseline"),
                "best_compression_variant": "turbo-int8k-4bitv",
                "best_compression_kv_ratio_vs_native": compression.get("kv_cache_ratio_vs_native"),
                "best_compression_prompt_perplexity_delta_pct_vs_native": compression.get(
                    "prompt_perplexity_delta_pct_vs_native"
                ),
                "best_compression_match_vs_baseline": compression.get("generated_match_vs_baseline"),
            }
        )

    return {
        "benchmark_kind": "hybrid-memory-frontier-summary-v1",
        "sources": {
            "transformer_gpu_summary": "verification/remote-transformers-gpu-summary.json",
            "hybrid_local_summary": "verification/local-zamba2-hxq-vs-vanilla-summary.json",
            "hybrid_local_prompt_suite": "verification/local-zamba2-prompt-suite-code-daily.json",
        },
        "transformer_gpu": {
            "best_fidelity_default": transformer_summary.get("best_fidelity_default"),
            "best_compression_default": transformer_summary.get("best_compression_default"),
            "models": transformer_models,
        },
        "hybrid_local": {
            "kv_only_gain": hybrid_summary.get("KV-only gain"),
            "mamba_state_only_gain": hybrid_summary.get("Mamba-state-only gain"),
            "combined_hybrid_gain": hybrid_summary.get("combined hybrid gain"),
            "prompt_category_aggregates": hybrid_summary.get("prompt_category_aggregates"),
            "vanilla_vs_hxq": hybrid_summary.get("vanilla vs HXQ"),
        },
    }


def _panel_bar(
    *,
    x: float,
    baseline_y: float,
    width: float,
    height: float,
    color: str,
    label: str,
    value_text: str,
) -> str:
    top = baseline_y - height
    label_y = baseline_y + 20
    value_y = top - 8
    return "\n".join(
        [
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10" fill="{color}" />',
            f'<text x="{x + width / 2:.1f}" y="{value_y:.1f}" text-anchor="middle" font-size="13" fill="#132238">{value_text}</text>',
            f'<text x="{x + width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="12" fill="#29405c">{label}</text>',
        ]
    )


def build_hybrid_memory_svg(summary: dict[str, Any]) -> str:
    transformer_models = list(summary["transformer_gpu"]["models"])
    hybrid = summary["hybrid_local"]
    kv_only = hybrid["kv_only_gain"]["hybrid_total_runtime_cache_ratio_vs_native"]
    state_only = hybrid["mamba_state_only_gain"]["hybrid_total_runtime_cache_ratio_vs_native"]
    combined = hybrid["combined_hybrid_gain"]["hybrid_total_runtime_cache_ratio_vs_native"]
    combined_speedup = hybrid["combined_hybrid_gain"]["speedup_vs_native"]
    code_speedup = hybrid["prompt_category_aggregates"]["vanilla"]["code"]["avg_speedup_vs_native"]
    daily_speedup = hybrid["prompt_category_aggregates"]["vanilla"]["daily"]["avg_speedup_vs_native"]

    transformer_max = max(
        max(float(model.get("best_fidelity_kv_ratio_vs_native") or 0.0), float(model.get("best_compression_kv_ratio_vs_native") or 0.0))
        for model in transformer_models
    )
    hybrid_max = max(float(kv_only or 0.0), float(state_only or 0.0), float(combined or 0.0))

    left_origin_x = 70.0
    right_origin_x = 610.0
    baseline_y = 380.0
    chart_height = 220.0
    bar_width = 42.0
    left_gap = 36.0
    group_gap = 42.0
    right_gap = 56.0

    elements: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="620" viewBox="0 0 1120 620" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Helix unified hybrid-memory frontier summary</title>",
        "<desc id=\"desc\">Transformer GPU KV ratios and hybrid Zamba2 local runtime-cache ratios extracted from verification JSON artifacts.</desc>",
        '<rect width="1120" height="620" fill="#f7f5ef" />',
        '<rect x="28" y="24" width="1064" height="572" rx="28" fill="#fffdf8" stroke="#d7d0bf" stroke-width="2" />',
        '<text x="56" y="72" font-size="28" font-weight="700" fill="#152238">Helix Memory Frontier Snapshot</text>',
        '<text x="56" y="100" font-size="15" fill="#50627a">Pure Transformers are still KV-bound; hybrid Zamba2 shifts the bottleneck to recurrent state.</text>',
        '<text x="70" y="138" font-size="18" font-weight="700" fill="#1d3552">Transformer GPU: verified KV ratios</text>',
        '<text x="610" y="138" font-size="18" font-weight="700" fill="#1d3552">Hybrid Zamba2 local: total runtime-cache ratios</text>',
        f'<line x1="{left_origin_x - 20:.1f}" y1="{baseline_y:.1f}" x2="540" y2="{baseline_y:.1f}" stroke="#d7d0bf" stroke-width="2" />',
        f'<line x1="{right_origin_x - 20:.1f}" y1="{baseline_y:.1f}" x2="1044" y2="{baseline_y:.1f}" stroke="#d7d0bf" stroke-width="2" />',
    ]

    current_x = left_origin_x
    for model in transformer_models:
        fidelity_ratio = float(model.get("best_fidelity_kv_ratio_vs_native") or 0.0)
        compression_ratio = float(model.get("best_compression_kv_ratio_vs_native") or 0.0)
        fidelity_height = chart_height * (fidelity_ratio / transformer_max) if transformer_max else 0.0
        compression_height = chart_height * (compression_ratio / transformer_max) if transformer_max else 0.0
        label = str(model["model_ref"]).replace("Qwen/", "Qwen ").replace("HuggingFaceTB/", "")
        elements.append(
            _panel_bar(
                x=current_x,
                baseline_y=baseline_y,
                width=bar_width,
                height=fidelity_height,
                color="#2f6fed",
                label="int8",
                value_text=f"{fidelity_ratio:.2f}x",
            )
        )
        elements.append(
            _panel_bar(
                x=current_x + bar_width + left_gap / 2,
                baseline_y=baseline_y,
                width=bar_width,
                height=compression_height,
                color="#e7903c",
                label="int8/4b",
                value_text=f"{compression_ratio:.2f}x",
            )
        )
        elements.append(
            f'<text x="{current_x + bar_width + left_gap / 4:.1f}" y="428" text-anchor="middle" font-size="12" fill="#31465d">{label}</text>'
        )
        current_x += (bar_width * 2) + left_gap + group_gap

    hybrid_bars = [
        ("KV-only", float(kv_only or 0.0), "#6e8fdb"),
        ("State-only", float(state_only or 0.0), "#3aa388"),
        ("Combined", float(combined or 0.0), "#d85d4d"),
    ]
    current_x = right_origin_x
    for label, value, color in hybrid_bars:
        height = chart_height * (value / hybrid_max) if hybrid_max else 0.0
        elements.append(
            _panel_bar(
                x=current_x,
                baseline_y=baseline_y,
                width=66.0,
                height=height,
                color=color,
                label=label,
                value_text=f"{value:.2f}x",
            )
        )
        current_x += 66.0 + right_gap

    elements.extend(
        [
            '<rect x="70" y="468" width="974" height="94" rx="18" fill="#f2efe5" stroke="#d7d0bf" stroke-width="1.5" />',
            '<text x="92" y="500" font-size="16" font-weight="700" fill="#20354f">Readout</text>',
            f'<text x="92" y="528" font-size="14" fill="#31465d">Combined hybrid mode reached {combined:.2f}x total runtime-cache reduction and {combined_speedup:.2f}x local speedup on Zamba2-1.2B.</text>',
            f'<text x="92" y="550" font-size="14" fill="#31465d">Prompt-suite averages: code {code_speedup:.2f}x speedup, daily {daily_speedup:.2f}x speedup, both at the same {combined:.2f}x memory ratio.</text>',
        ]
    )
    elements.append("</svg>")
    return "\n".join(elements)


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Helix unified hybrid-memory summary JSON and SVG figure.")
    parser.add_argument(
        "--transformer-summary",
        type=Path,
        default=Path("verification") / "remote-transformers-gpu-summary.json",
    )
    parser.add_argument(
        "--hybrid-summary",
        type=Path,
        default=Path("verification") / "local-zamba2-hxq-vs-vanilla-summary.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("verification") / "hybrid-memory-frontier-summary.json",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("docs") / "figures" / "hybrid-memory-frontier.svg",
    )
    args = parser.parse_args()

    summary = build_hybrid_memory_summary(
        transformer_summary=_load_json(args.transformer_summary.resolve()),
        hybrid_summary=_load_json(args.hybrid_summary.resolve()),
    )
    svg = build_hybrid_memory_svg(summary)

    _write_text(args.output_json.resolve(), json.dumps(summary, indent=2))
    _write_text(args.output_svg.resolve(), svg)
    print(f"summary_json: {args.output_json.resolve()}")
    print(f"summary_svg: {args.output_svg.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
