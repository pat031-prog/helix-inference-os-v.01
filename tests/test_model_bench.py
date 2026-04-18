from pathlib import Path

from helix_proto.model_bench import (
    compare_benchmark_reports,
    default_benchmark_suite_path,
    load_benchmark_prompts,
    render_markdown_summary,
    recommend_assistant_models,
)


def test_default_benchmark_suite_loads_prompts() -> None:
    prompts = load_benchmark_prompts(default_benchmark_suite_path())
    roles = {prompt.role for prompt in prompts}

    assert len(prompts) >= 6
    assert roles == {"general", "code", "legal"}


def test_recommend_assistant_models_prefers_best_role_score_then_speed() -> None:
    results = [
        {
            "model_ref": "model-a",
            "status": "ok",
            "role_scores": {"general": 0.8, "code": 0.4, "legal": 0.5},
            "quality_proxy_score": 0.56,
            "tokens_per_second": 10.0,
        },
        {
            "model_ref": "model-b",
            "status": "ok",
            "role_scores": {"general": 0.7, "code": 0.9, "legal": 0.6},
            "quality_proxy_score": 0.73,
            "tokens_per_second": 8.0,
        },
        {
            "model_ref": "model-c",
            "status": "ok",
            "role_scores": {"general": 0.7, "code": 0.9, "legal": 0.9},
            "quality_proxy_score": 0.83,
            "tokens_per_second": 12.0,
        },
    ]

    recommendations = recommend_assistant_models(results)

    assert recommendations["general"]["model_ref"] == "model-a"
    assert recommendations["code"]["model_ref"] == "model-c"
    assert recommendations["legal"]["model_ref"] == "model-c"


def test_compare_benchmark_reports_supports_explicit_model_pairs() -> None:
    baseline = {
        "generated_at_utc": "2026-03-24T00:00:00Z",
        "models": [
            {
                "model_ref": "Qwen/Qwen2.5-1.5B",
                "model_label": "Qwen2.5-1.5B",
                "status": "ok",
                "quality_proxy_score": 0.4,
                "tokens_per_second": 2.5,
                "role_scores": {"general": 0.5, "code": 0.3, "legal": 0.4},
            }
        ],
    }
    tuned = {
        "generated_at_utc": "2026-03-24T01:00:00Z",
        "models": [
            {
                "model_ref": str(Path("C:/tmp/qwen-legal-ft")),
                "model_label": "qwen-legal-ft",
                "status": "ok",
                "quality_proxy_score": 0.7,
                "tokens_per_second": 2.1,
                "role_scores": {"general": 0.5, "code": 0.4, "legal": 0.9},
            }
        ],
    }

    comparison = compare_benchmark_reports(
        baseline,
        tuned,
        model_pairs=[("Qwen/Qwen2.5-1.5B", str(Path("C:/tmp/qwen-legal-ft")))],
    )

    assert comparison["comparisons"][0]["quality_proxy_delta"] == 0.3
    assert comparison["comparisons"][0]["role_score_deltas"]["legal"] == 0.5


def test_render_markdown_summary_uses_model_labels() -> None:
    summary = render_markdown_summary(
        {
            "generated_at_utc": "2026-03-24T00:00:00Z",
            "prompt_suite_path": "benchmarks/local_assistant_prompts.json",
            "local_files_only": True,
            "recommendations": {"general": None, "code": None, "legal": None},
            "models": [
                {
                    "model_ref": "model-a",
                    "model_label": "My Local Model",
                    "status": "ok",
                    "quality_proxy_score": 0.5,
                    "tokens_per_second": 9.0,
                    "rss_peak_mb": 100.0,
                    "role_scores": {"general": 0.5},
                }
            ],
        }
    )

    assert "My Local Model" in summary
