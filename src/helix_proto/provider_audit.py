"""Provider-returned model audit and behavioral fingerprint helpers."""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    token_env: str

    @property
    def token_available(self) -> bool:
        return bool(os.environ.get(self.token_env))


OPENAI_COMPATIBLE_PROVIDERS = [
    ProviderConfig("deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_TOKEN"),
    ProviderConfig("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    ProviderConfig("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    ProviderConfig("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    ProviderConfig("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
]


REFUSAL_TERMS = (
    "i can't",
    "i cannot",
    "unable to",
    "can't assist",
    "cannot comply",
    "policy",
    "safety",
)


def digest_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def behavioral_fingerprint(text: str, *, latency_ms: float = 0.0) -> dict[str, Any]:
    clean = str(text or "")
    words = re.findall(r"[A-Za-z0-9_'-]+", clean)
    lower = clean.lower()
    return {
        "output_digest": digest_text(clean),
        "char_count": len(clean),
        "token_proxy_count": max(1, (len(clean) + 3) // 4),
        "word_count": len(words),
        "line_count": len([line for line in clean.splitlines() if line.strip()]),
        "refusal": any(term in lower for term in REFUSAL_TERMS),
        "refusal_term_count": sum(lower.count(term) for term in REFUSAL_TERMS),
        "jsonish": clean.strip().startswith("{") and clean.strip().endswith("}"),
        "markdown_bullet_count": len(re.findall(r"(?m)^\s*[-*]\s+", clean)),
        "latency_ms": float(latency_ms),
    }


def two_proportion_mde(*, n_per_group: int, baseline_rate: float = 0.1, z_alpha: float = 1.96, z_power: float = 1.645) -> float:
    """Approximate minimum detectable effect for two independent proportions."""

    n = max(int(n_per_group), 1)
    p = min(max(float(baseline_rate), 0.0001), 0.9999)
    return (z_alpha + z_power) * math.sqrt((2 * p * (1 - p)) / n)


def normal_mean_mde(*, n_per_group: int, pooled_stdev: float, z_alpha: float = 1.96, z_power: float = 1.645) -> float:
    n = max(int(n_per_group), 1)
    return (z_alpha + z_power) * math.sqrt((2 * float(pooled_stdev) ** 2) / n)


def triangulation_power_analysis(*, prompt_count_per_provider: int, baseline_refusal_rate: float = 0.1, pooled_length_stdev: float = 80.0) -> dict[str, Any]:
    refusal_mde = two_proportion_mde(n_per_group=prompt_count_per_provider, baseline_rate=baseline_refusal_rate)
    length_mde = normal_mean_mde(n_per_group=prompt_count_per_provider, pooled_stdev=pooled_length_stdev)
    adequate = prompt_count_per_provider >= 200
    return {
        "prompt_count_per_provider": int(prompt_count_per_provider),
        "baseline_refusal_rate": baseline_refusal_rate,
        "refusal_rate_mde": refusal_mde,
        "output_length_mde_chars": length_mde,
        "target_power": 0.95,
        "alpha": 0.05,
        "adequately_powered_for_public_claim": adequate,
        "underpowered_exploratory": not adequate,
        "claim_boundary": "If underpowered_exploratory is true, failure to reject H0 is not evidence of indistinguishability.",
    }
