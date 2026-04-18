from __future__ import annotations

import gc
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix_proto.cdna import generate_text_with_target, load_generation_target
from helix_proto.hf import _process_rss_mb


DEFAULT_MODEL_REFS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "sshleifer/tiny-gpt2",
    "facebook/opt-350m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "microsoft/phi-2",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_benchmark_suite_path() -> Path:
    return repo_root() / "benchmarks" / "local_assistant_prompts.json"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _keyword_hits(text: str, groups: list[list[str]]) -> tuple[int, list[list[str]]]:
    lowered = text.lower()
    matched: list[list[str]] = []
    for group in groups:
        if any(term.lower() in lowered for term in group):
            matched.append(group)
    return len(matched), matched


def _bullet_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•")):
            count += 1
            continue
        if len(stripped) >= 2 and stripped[0].isdigit() and stripped[1] in {".", ")"}:
            count += 1
    return count


def _score_response(prompt_spec: dict[str, Any], response: str) -> dict[str, Any]:
    checks = dict(prompt_spec.get("checks", {}))
    groups = [list(group) for group in checks.get("keyword_groups", [])]
    matched_groups_count, matched_groups = _keyword_hits(response, groups)

    total_checks = 0
    passed_checks = 0

    if groups:
        total_checks += len(groups)
        passed_checks += matched_groups_count

    min_bullets = checks.get("min_bullets")
    if isinstance(min_bullets, int):
        total_checks += 1
        if _bullet_count(response) >= min_bullets:
            passed_checks += 1

    must_include = checks.get("must_include")
    if isinstance(must_include, list):
        for token in must_include:
            total_checks += 1
            if str(token).lower() in response.lower():
                passed_checks += 1

    if checks.get("requires_code_block"):
        total_checks += 1
        if "```" in response or "def " in response or "function " in response:
            passed_checks += 1

    quality_score = (passed_checks / total_checks) if total_checks else 0.0
    return {
        "quality_score": round(float(quality_score), 4),
        "matched_keyword_groups": matched_groups,
        "matched_checks": passed_checks,
        "total_checks": total_checks,
        "bullet_count": _bullet_count(response),
    }


def _cache_dir_for_model(model_ref: str) -> Path | None:
    local_path = Path(model_ref)
    if local_path.exists():
        return local_path.resolve()
    candidate = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_ref.replace('/', '--')}"
    )
    return candidate if candidate.exists() else None


def _cache_size_bytes(model_ref: str) -> int | None:
    root = _cache_dir_for_model(model_ref)
    if root is None:
        return None
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _safe_memory_footprint(model: Any) -> float | None:
    try:
        footprint = model.get_memory_footprint()
    except Exception:  # noqa: BLE001
        return None
    return float(footprint) / (1024 * 1024)


def _model_label(model_ref: str) -> str:
    path = Path(model_ref)
    if path.exists():
        return path.resolve().name
    return model_ref


@dataclass(slots=True)
class BenchmarkPrompt:
    id: str
    role: str
    prompt: str
    checks: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkPrompt":
        return cls(
            id=str(data["id"]),
            role=str(data["role"]),
            prompt=str(data["prompt"]),
            checks=dict(data.get("checks", {})),
        )


def load_benchmark_prompts(path: str | Path | None = None) -> list[BenchmarkPrompt]:
    source = Path(path) if path is not None else default_benchmark_suite_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    return [BenchmarkPrompt.from_dict(item) for item in data["prompts"]]


def benchmark_one_model(
    model_ref: str,
    *,
    prompts: list[BenchmarkPrompt],
    workspace_root: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_input_tokens: int = 512,
    torch_dtype: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("benchmarking requires transformers and torch") from exc

    before_load_rss = _process_rss_mb()
    load_started = time.perf_counter()
    target = load_generation_target(
        model_ref,
        workspace_root=workspace_root,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    tokenizer = target.tokenizer
    model = target.model

    if target.backend == "transformers" and tokenizer is not None:
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token_id is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                model.resize_token_embeddings(len(tokenizer))

    load_time_s = time.perf_counter() - load_started
    after_load_rss = _process_rss_mb()

    prompt_results: list[dict[str, Any]] = []
    role_scores: dict[str, list[float]] = {}
    total_generated_tokens = 0
    total_generation_time_s = 0.0
    peak_rss_mb = max(before_load_rss, after_load_rss)

    context = torch.inference_mode() if target.backend == "transformers" else nullcontext()
    with context:
        for prompt in prompts:
            generation_started = time.perf_counter()
            generated = generate_text_with_target(
                target,
                prompt=prompt.prompt,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                do_sample=False,
            )
            generation_time_s = time.perf_counter() - generation_started

            response = str(generated["text"])
            generated_tokens = int(generated["generated_tokens"])
            score = _score_response(
                {"checks": prompt.checks},
                response,
            )

            total_generated_tokens += generated_tokens
            total_generation_time_s += generation_time_s
            peak_rss_mb = max(peak_rss_mb, _process_rss_mb())
            role_scores.setdefault(prompt.role, []).append(float(score["quality_score"]))

            prompt_results.append(
                {
                    "id": prompt.id,
                    "role": prompt.role,
                    "prompt": prompt.prompt,
                    "response": response,
                    "generated_tokens": generated_tokens,
                    "generation_time_s": round(generation_time_s, 4),
                    "tokens_per_second": round(
                        generated_tokens / generation_time_s, 4
                    )
                    if generation_time_s
                    else 0.0,
                    "score": score,
                }
            )

    footprint_mb = _safe_memory_footprint(model) if target.backend == "transformers" else None
    config = getattr(model, "config", None)
    if target.backend == "transformers":
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
    else:
        num_parameters = 0

    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - GPU environments vary
        torch.cuda.empty_cache()

    role_summary = {
        role: round(sum(values) / len(values), 4)
        for role, values in sorted(role_scores.items())
        if values
    }
    overall_quality = round(
        sum(role_summary.values()) / len(role_summary),
        4,
    ) if role_summary else 0.0

    return {
        "model_ref": target.model_ref,
        "requested_ref": model_ref,
        "model_label": _model_label(target.model_label),
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", []) or []),
        "num_parameters": int(num_parameters),
        "cache_size_bytes": target.storage_bytes if target.storage_bytes is not None else _cache_size_bytes(target.model_ref),
        "load_mode": target.load_mode,
        "local_files_only": local_files_only,
        "load_time_s": round(load_time_s, 4),
        "rss_before_load_mb": round(before_load_rss, 2),
        "rss_after_load_mb": round(after_load_rss, 2),
        "rss_peak_mb": round(peak_rss_mb, 2),
        "model_memory_footprint_mb": round(footprint_mb, 2) if footprint_mb is not None else None,
        "total_generated_tokens": int(total_generated_tokens),
        "total_generation_time_s": round(total_generation_time_s, 4),
        "tokens_per_second": round(
            total_generated_tokens / total_generation_time_s,
            4,
        ) if total_generation_time_s else 0.0,
        "quality_proxy_score": overall_quality,
        "role_scores": role_summary,
        "prompt_results": prompt_results,
    }


def benchmark_models(
    model_refs: list[str],
    *,
    prompt_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_input_tokens: int = 512,
    torch_dtype: str = "auto",
) -> dict[str, Any]:
    prompts = load_benchmark_prompts(prompt_path)
    prompt_summary = [
        {"id": prompt.id, "role": prompt.role, "prompt": prompt.prompt}
        for prompt in prompts
    ]
    results: list[dict[str, Any]] = []
    for model_ref in model_refs:
        started = time.perf_counter()
        try:
            result = benchmark_one_model(
                model_ref,
                prompts=prompts,
                workspace_root=workspace_root,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                torch_dtype=torch_dtype,
            )
            result["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            result = {
                "model_ref": model_ref,
                "model_label": _model_label(model_ref),
                "status": "error",
                "error": str(exc),
            }
        result["wall_time_s"] = round(time.perf_counter() - started, 4)
        results.append(result)

    recommendations = recommend_assistant_models(results)
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_suite_path": str(Path(prompt_path) if prompt_path else default_benchmark_suite_path()),
        "local_files_only": local_files_only,
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "models": results,
        "prompts": prompt_summary,
        "recommendations": recommendations,
    }


def recommend_assistant_models(results: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for role in ("general", "code", "legal"):
        candidates = [
            item
            for item in results
            if item.get("status") == "ok" and role in item.get("role_scores", {})
        ]
        if not candidates:
            recommendations[role] = None
            continue
        best = sorted(
            candidates,
            key=lambda item: (
                item["role_scores"][role],
                item.get("quality_proxy_score", 0.0),
                item.get("tokens_per_second", 0.0),
            ),
            reverse=True,
        )[0]
        recommendations[role] = {
            "model_ref": best["model_ref"],
            "role_score": best["role_scores"][role],
            "quality_proxy_score": best.get("quality_proxy_score"),
            "tokens_per_second": best.get("tokens_per_second"),
        }
    return recommendations


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Local Model Benchmark",
        "",
        f"- Generated at: `{report['generated_at_utc']}`",
        f"- Prompt suite: `{report['prompt_suite_path']}`",
        f"- Local files only: `{report['local_files_only']}`",
        "",
        "## Recommendations",
    ]
    for role, item in report.get("recommendations", {}).items():
        if not item:
            lines.append(f"- `{role}`: no successful result")
            continue
        lines.append(
            f"- `{role}`: `{item['model_ref']}` | score={item['role_score']:.4f} | tok/s={item['tokens_per_second']:.2f}"
        )
    lines.extend(["", "## Models"])
    for model in report.get("models", []):
        if model.get("status") != "ok":
            lines.append(f"- `{model['model_label']}`: error `{model.get('error', 'unknown')}`")
            continue
        role_scores = ", ".join(
            f"{role}={score:.4f}" for role, score in sorted(model.get("role_scores", {}).items())
        )
        lines.append(
            f"- `{model['model_label']}`: quality={model['quality_proxy_score']:.4f}, tok/s={model['tokens_per_second']:.2f}, "
            f"peak_rss_mb={model['rss_peak_mb']:.2f}, roles=({role_scores})"
        )
    return "\n".join(lines) + "\n"


def benchmark_report_paths(output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / "benchmark_report.json", root / "benchmark_summary.md"


def save_benchmark_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    json_path, markdown_path = benchmark_report_paths(output_dir)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_summary(report), encoding="utf-8")
    return json_path, markdown_path


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_benchmark_reports(
    baseline_report: dict[str, Any],
    tuned_report: dict[str, Any],
    *,
    model_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    def _ok_models(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            item["model_ref"]: item
            for item in report.get("models", [])
            if item.get("status") == "ok"
        }

    baseline_models = _ok_models(baseline_report)
    tuned_models = _ok_models(tuned_report)
    comparisons: list[dict[str, Any]] = []

    pairs = model_pairs or [
        (model_ref, model_ref)
        for model_ref in tuned_models
        if model_ref in baseline_models
    ]

    for baseline_ref, tuned_ref in pairs:
        baseline = baseline_models.get(baseline_ref)
        tuned = tuned_models.get(tuned_ref)
        if baseline is None or tuned is None:
            continue
        role_deltas = {}
        for role, score in tuned.get("role_scores", {}).items():
            if role in baseline.get("role_scores", {}):
                role_deltas[role] = round(score - baseline["role_scores"][role], 4)
        comparisons.append(
            {
                "baseline_model_ref": baseline_ref,
                "baseline_model_label": baseline.get("model_label", _model_label(baseline_ref)),
                "tuned_model_ref": tuned_ref,
                "tuned_model_label": tuned.get("model_label", _model_label(tuned_ref)),
                "quality_proxy_delta": round(
                    tuned.get("quality_proxy_score", 0.0) - baseline.get("quality_proxy_score", 0.0),
                    4,
                ),
                "tokens_per_second_delta": round(
                    tuned.get("tokens_per_second", 0.0) - baseline.get("tokens_per_second", 0.0),
                    4,
                ),
                "role_score_deltas": role_deltas,
            }
        )

    return {
        "baseline_generated_at_utc": baseline_report.get("generated_at_utc"),
        "tuned_generated_at_utc": tuned_report.get("generated_at_utc"),
        "comparisons": comparisons,
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Comparison",
        "",
        f"- Baseline generated at: `{comparison.get('baseline_generated_at_utc')}`",
        f"- Tuned generated at: `{comparison.get('tuned_generated_at_utc')}`",
        "",
        "## Comparisons",
    ]
    for item in comparison.get("comparisons", []):
        role_deltas = ", ".join(
            f"{role}={delta:+.4f}" for role, delta in sorted(item.get("role_score_deltas", {}).items())
        )
        lines.append(
            f"- `{item['tuned_model_label']}` vs `{item['baseline_model_label']}`: "
            f"quality_delta={item['quality_proxy_delta']:+.4f}, "
            f"tok/s_delta={item['tokens_per_second_delta']:+.4f}, "
            f"roles=({role_deltas})"
        )
    return "\n".join(lines) + "\n"


def save_benchmark_comparison(
    comparison: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "benchmark_comparison.json"
    markdown_path = root / "benchmark_comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path
