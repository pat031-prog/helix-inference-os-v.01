"""Shared helpers for falsable HeliX v4 verification gauntlets.

The v4 gauntlet deliberately treats negative results as first-class evidence:
each artifact records the null hypothesis, falseability condition, controls,
baselines, deterministic seed and claim ladder.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CLAIM_LADDER = (
    "mechanics_verified",
    "empirically_observed",
    "replicated",
    "longitudinal",
    "external_replication",
)


REPO = Path(__file__).resolve().parents[2]
VERIFICATION = REPO / "verification"
PREREGISTERED = VERIFICATION / "preregistered"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(run_id: str, test_id: str, preregistered_hash: str) -> int:
    return int(sha256_text(f"{run_id}|{test_id}|{preregistered_hash}")[:16], 16)


def rng_for(run_id: str, test_id: str, preregistered_hash: str, *parts: str) -> random.Random:
    seed = stable_seed(run_id, test_id, preregistered_hash)
    if parts:
        seed = int(sha256_text(f"{seed}|{'|'.join(parts)}")[:16], 16)
    return random.Random(seed)


@dataclass(frozen=True)
class Preregistration:
    test_id: str
    question: str
    null_hypothesis: str
    metrics: list[str]
    falseability_condition: str
    kill_switch: str
    control_arms: list[str]
    path: Path
    sha256: str


def preregister(
    *,
    test_id: str,
    question: str,
    null_hypothesis: str,
    metrics: Iterable[str],
    falseability_condition: str,
    kill_switch: str,
    control_arms: Iterable[str],
    verification_dir: Path | None = None,
) -> Preregistration:
    base = verification_dir or VERIFICATION
    prereg_dir = base / "preregistered"
    prereg_dir.mkdir(parents=True, exist_ok=True)
    path = prereg_dir / f"{test_id}.md"
    body = "\n".join(
        [
            f"# {test_id}",
            "",
            f"Question: {question}",
            "",
            f"Null hypothesis: {null_hypothesis}",
            "",
            "Metrics:",
            *[f"- {metric}" for metric in metrics],
            "",
            f"Falseability condition: {falseability_condition}",
            "",
            f"Kill-switch: {kill_switch}",
            "",
            "Control arms:",
            *[f"- {arm}" for arm in control_arms],
            "",
        ]
    )
    if not path.exists() or path.read_text(encoding="utf-8") != body:
        path.write_text(body, encoding="utf-8")
    return Preregistration(
        test_id=test_id,
        question=question,
        null_hypothesis=null_hypothesis,
        metrics=list(metrics),
        falseability_condition=falseability_condition,
        kill_switch=kill_switch,
        control_arms=list(control_arms),
        path=path,
        sha256=sha256_file(path),
    )


def summarize(values: Iterable[float], *, high_variance_cv: float = 0.15) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"mean": 0.0, "stdev": 0.0, "n": 0, "cv": 0.0, "high_variance": False}
    mean = statistics.fmean(data)
    stdev = statistics.pstdev(data) if len(data) > 1 else 0.0
    cv = 0.0 if math.isclose(mean, 0.0) else abs(stdev / mean)
    return {
        "mean": mean,
        "stdev": stdev,
        "n": len(data),
        "cv": cv,
        "high_variance": cv > high_variance_cv,
    }


def confusion_metrics(*, tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    fpr = fp / max(fp + tn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def percentile(values: Iterable[float], q: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    index = min(len(data) - 1, max(0, round((len(data) - 1) * q)))
    return data[index]


def base_artifact(
    *,
    test_id: str,
    run_id: str,
    run_date_utc: str,
    preregistration: Preregistration,
    replica_count: int,
    random_baseline: dict[str, Any],
    no_helix_baseline: dict[str, Any],
    helix_arm: dict[str, Any],
    public_claim_ladder: str,
    claims_allowed: list[str],
    claims_not_allowed: list[str],
    prompt_selection_risk: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if public_claim_ladder not in CLAIM_LADDER:
        raise ValueError(f"invalid public_claim_ladder={public_claim_ladder!r}")
    seed = stable_seed(run_id, test_id, preregistration.sha256)
    payload: dict[str, Any] = {
        "artifact": f"local-v4-{test_id}",
        "test_id": test_id,
        "run_id": run_id,
        "run_date_utc": run_date_utc,
        "generated_ms": int(time.time() * 1000),
        "preregistered_path": str(preregistration.path),
        "preregistered_hash": preregistration.sha256,
        "seed": seed,
        "replica_count": replica_count,
        "null_hypothesis": preregistration.null_hypothesis,
        "falseability_condition": preregistration.falseability_condition,
        "kill_switch": preregistration.kill_switch,
        "control_arms": preregistration.control_arms,
        "random_baseline": random_baseline,
        "no_helix_baseline": no_helix_baseline,
        "helix_arm": helix_arm,
        "public_claim_ladder": public_claim_ladder,
        "claim_boundary": (
            "v4 gauntlet claims are bounded by preregistered controls, baselines, "
            "falseability conditions and the recorded public_claim_ladder."
        ),
        "claims_allowed": claims_allowed,
        "claims_not_allowed": claims_not_allowed,
        "prompt_selection_risk": prompt_selection_risk,
    }
    if extra:
        payload.update(extra)
    return payload


def write_artifact(name: str, payload: dict[str, Any], *, verification_dir: Path | None = None) -> Path:
    base = verification_dir or VERIFICATION
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
