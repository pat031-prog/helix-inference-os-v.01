from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from helix_kv.config import canonical_mode_name


def token_negative_log_likelihood(logits: np.ndarray, token_id: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    clipped = values - np.max(values)
    exp = np.exp(clipped)
    total = float(np.sum(exp))
    if total <= 0.0:
        return float("inf")
    probability = float(exp[int(token_id)] / total)
    probability = min(max(probability, 1e-12), 1.0)
    return float(-math.log(probability))


@dataclass(slots=True)
class AdaptiveKVPolicy:
    ppl_threshold_up: float = 1.15
    ppl_threshold_down: float = 0.95
    warmup_steps: int = 8
    window: int = 8
    check_interval: int = 4
    cooldown_steps: int = 8
    mode_order: tuple[str, ...] = ("turbo-4bit", "turbo-int8-hadamard", "fp32")
    _baseline_losses: list[float] = field(default_factory=list)
    _recent_losses: deque[float] = field(default_factory=deque)
    _step_count: int = 0
    _cooldown_remaining: int = 0
    _mode_trace: list[str] = field(default_factory=list)
    _switch_events: list[dict[str, Any]] = field(default_factory=list)

    def reset_runtime_state(self) -> None:
        self._baseline_losses.clear()
        self._recent_losses.clear()
        self._step_count = 0
        self._cooldown_remaining = 0
        self._mode_trace.clear()
        self._switch_events.clear()

    def record_mode(self, mode: str) -> None:
        self._mode_trace.append(canonical_mode_name(mode))

    def record_switch(self, *, old_mode: str, new_mode: str, reason: str, step_index: int) -> None:
        self._switch_events.append(
            {
                "step_index": int(step_index),
                "old_mode": canonical_mode_name(old_mode),
                "new_mode": canonical_mode_name(new_mode),
                "reason": str(reason),
            }
        )

    def allowed_modes(self, allowed_modes: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        if not allowed_modes:
            return tuple(canonical_mode_name(mode) for mode in self.mode_order)
        return tuple(canonical_mode_name(mode) for mode in allowed_modes)

    def observe(
        self,
        *,
        logits: np.ndarray,
        token_id: int,
        current_mode: str,
        allowed_modes: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        loss = token_negative_log_likelihood(logits, token_id)
        self._step_count += 1
        current = canonical_mode_name(current_mode)
        self.record_mode(current)

        if len(self._baseline_losses) < max(int(self.warmup_steps), 1):
            self._baseline_losses.append(loss)
            return self._snapshot("hold", current, current)

        self._recent_losses.append(loss)
        while len(self._recent_losses) > max(int(self.window), 1):
            self._recent_losses.popleft()

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return self._snapshot("hold", current, current)
        if len(self._recent_losses) < max(int(self.window), 1):
            return self._snapshot("hold", current, current)
        if self._step_count % max(int(self.check_interval), 1) != 0:
            return self._snapshot("hold", current, current)

        baseline = float(np.mean(self._baseline_losses))
        recent = float(np.mean(list(self._recent_losses)))
        ratio = recent / baseline if baseline > 0.0 else 1.0
        allowed = self.allowed_modes(allowed_modes)
        try:
            current_index = allowed.index(current)
        except ValueError:
            current_index = len(allowed) - 1

        if ratio > float(self.ppl_threshold_up) and current_index < len(allowed) - 1:
            target = allowed[current_index + 1]
            self._cooldown_remaining = max(int(self.cooldown_steps), 0)
            self.record_switch(
                old_mode=current,
                new_mode=target,
                reason=f"policy_upgrade ratio={ratio:.4f}",
                step_index=self._step_count,
            )
            return self._snapshot("upgrade", current, target, baseline=baseline, recent=recent)
        if ratio < float(self.ppl_threshold_down) and current_index > 0:
            target = allowed[current_index - 1]
            self._cooldown_remaining = max(int(self.cooldown_steps), 0)
            self.record_switch(
                old_mode=current,
                new_mode=target,
                reason=f"policy_downgrade ratio={ratio:.4f}",
                step_index=self._step_count,
            )
            return self._snapshot("downgrade", current, target, baseline=baseline, recent=recent)
        return self._snapshot("hold", current, current, baseline=baseline, recent=recent)

    def mode_histogram(self) -> dict[str, int]:
        return dict(Counter(self._mode_trace))

    def current_baseline_loss(self) -> float | None:
        if not self._baseline_losses:
            return None
        return float(np.mean(self._baseline_losses))

    def current_recent_loss(self) -> float | None:
        if not self._recent_losses:
            return None
        return float(np.mean(list(self._recent_losses)))

    def to_json(self) -> dict[str, Any]:
        return {
            "ppl_threshold_up": float(self.ppl_threshold_up),
            "ppl_threshold_down": float(self.ppl_threshold_down),
            "warmup_steps": int(self.warmup_steps),
            "window": int(self.window),
            "check_interval": int(self.check_interval),
            "cooldown_steps": int(self.cooldown_steps),
            "mode_order": list(self.mode_order),
            "baseline_losses": list(self._baseline_losses),
            "recent_losses": list(self._recent_losses),
            "step_count": int(self._step_count),
            "cooldown_remaining": int(self._cooldown_remaining),
            "mode_trace": list(self._mode_trace),
            "switch_events": list(self._switch_events),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "AdaptiveKVPolicy":
        policy = cls(
            ppl_threshold_up=float(payload.get("ppl_threshold_up", 1.15)),
            ppl_threshold_down=float(payload.get("ppl_threshold_down", 0.95)),
            warmup_steps=int(payload.get("warmup_steps", 8)),
            window=int(payload.get("window", 8)),
            check_interval=int(payload.get("check_interval", 4)),
            cooldown_steps=int(payload.get("cooldown_steps", 8)),
            mode_order=tuple(payload.get("mode_order", ("turbo-4bit", "turbo-int8-hadamard", "fp32"))),
        )
        policy._baseline_losses = [float(item) for item in payload.get("baseline_losses", [])]
        policy._recent_losses = deque(float(item) for item in payload.get("recent_losses", []))
        policy._step_count = int(payload.get("step_count", 0))
        policy._cooldown_remaining = int(payload.get("cooldown_remaining", 0))
        policy._mode_trace = [canonical_mode_name(item) for item in payload.get("mode_trace", [])]
        policy._switch_events = list(payload.get("switch_events", []))
        return policy

    def _snapshot(
        self,
        action: str,
        current_mode: str,
        target_mode: str,
        *,
        baseline: float | None = None,
        recent: float | None = None,
    ) -> dict[str, Any]:
        return {
            "action": str(action),
            "current_mode": canonical_mode_name(current_mode),
            "target_mode": canonical_mode_name(target_mode),
            "baseline_loss": baseline if baseline is not None else self.current_baseline_loss(),
            "recent_loss": recent if recent is not None else self.current_recent_loss(),
            "step_count": int(self._step_count),
        }
