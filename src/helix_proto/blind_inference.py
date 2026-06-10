from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from helix_kv.memory_catalog import privacy_filter


_DEFAULT_DETECTORS = {
    "email": True,
    "document_id": True,
    "account": True,
    "phone": True,
}

_DETECTOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b"),
    "document_id": re.compile(r"\b\d{7,12}\b"),
    "account": re.compile(r"\b[A-Z]{2,6}[-_]?\d{6,18}\b"),
    "phone": re.compile(r"(?:(?:\+\d{1,3}\s*)?(?:\(?\d{2,4}\)?[\s\-]*)?\d{3,4}[\s\-]?\d{4})"),
}


def _stable_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalize_sensitive_type(value: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "TEXT").strip().upper()).strip("_")
    return cleaned or "TEXT"


@dataclass(frozen=True)
class BlindRule:
    name: str
    sensitive_type: str
    values: tuple[str, ...] = ()
    pattern: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], index: int) -> "BlindRule":
        values = tuple(str(item) for item in (payload.get("values") or []) if str(item))
        pattern = str(payload.get("pattern") or "").strip() or None
        return cls(
            name=str(payload.get("name") or payload.get("label") or f"rule_{index}"),
            sensitive_type=_normalize_sensitive_type(str(payload.get("type") or payload.get("sensitive_type") or "TEXT")),
            values=values,
            pattern=pattern,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.sensitive_type,
            "values": list(self.values),
            "pattern": self.pattern,
        }


@dataclass
class BlindInferencePolicy:
    enabled: bool = False
    scope: str = "cloud_proxy"
    placeholder_stability: str = "per_task"
    rules: tuple[BlindRule, ...] = ()
    detectors: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_DETECTORS))
    policy_id: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "BlindInferencePolicy":
        body = dict(payload or {})
        rules = tuple(
            BlindRule.from_payload(item, index)
            for index, item in enumerate(body.get("rules") or [], start=1)
            if isinstance(item, dict)
        )
        detectors = dict(_DEFAULT_DETECTORS)
        configured = body.get("detectors")
        if isinstance(configured, dict):
            for key, value in configured.items():
                detectors[str(key)] = bool(value)
        scope = str(body.get("scope") or "cloud_proxy").strip().lower() or "cloud_proxy"
        placeholder_stability = str(body.get("placeholder_stability") or "per_task").strip().lower() or "per_task"
        material = {
            "enabled": bool(body.get("enabled", False)),
            "scope": scope,
            "placeholder_stability": placeholder_stability,
            "rules": [rule.to_dict() for rule in rules],
            "detectors": detectors,
        }
        return cls(
            enabled=bool(body.get("enabled", False)),
            scope=scope,
            placeholder_stability=placeholder_stability,
            rules=rules,
            detectors=detectors,
            policy_id=_stable_hash(_canonical_json(material))[:16],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scope": self.scope,
            "placeholder_stability": self.placeholder_stability,
            "rules": [rule.to_dict() for rule in self.rules],
            "detectors": dict(self.detectors),
            "policy_id": self.policy_id,
        }


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    text: str
    sensitive_type: str
    origin_rule: str


class TokenVault:
    def __init__(self, *, task_id: str) -> None:
        self.task_id = str(task_id or "task")
        self.task_tag = _stable_hash(self.task_id)[:4].upper()
        self._by_original: dict[tuple[str, str], str] = {}
        self._by_placeholder: dict[str, str] = {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._counters: dict[str, int] = {}

    def issue_placeholder(self, *, sensitive_type: str, original: str, origin_rule: str) -> str:
        key = (_normalize_sensitive_type(sensitive_type), str(original))
        if key in self._by_original:
            return self._by_original[key]
        kind = key[0]
        next_index = self._counters.get(kind, 0) + 1
        self._counters[kind] = next_index
        placeholder = f"{kind}__T{self.task_tag}__{next_index:03d}"
        self._by_original[key] = placeholder
        self._by_placeholder[placeholder] = key[1]
        self._entries[placeholder] = {
            "sensitive_type": kind,
            "placeholder": placeholder,
            "origin_rule": origin_rule,
            "original_sha256": _stable_hash(key[1]),
        }
        return placeholder

    def rehydrate_text(self, text: str) -> str:
        restored = str(text or "")
        for placeholder in sorted(self._by_placeholder, key=len, reverse=True):
            restored = restored.replace(placeholder, self._by_placeholder[placeholder])
        return restored

    def summary(self) -> dict[str, Any]:
        classes = sorted({entry["sensitive_type"] for entry in self._entries.values()})
        return {
            "task_id": self.task_id,
            "task_tag": self.task_tag,
            "vault_present": bool(self._entries),
            "token_count": len(self._entries),
            "sensitive_classes": classes,
            "entries": [dict(item) for item in self._entries.values()],
        }


def _literal_spans(text: str, literal: str, *, sensitive_type: str, origin_rule: str) -> list[_Span]:
    if not literal:
        return []
    spans: list[_Span] = []
    start = 0
    while True:
        index = text.find(literal, start)
        if index < 0:
            break
        spans.append(
            _Span(
                start=index,
                end=index + len(literal),
                text=literal,
                sensitive_type=sensitive_type,
                origin_rule=origin_rule,
            )
        )
        start = index + len(literal)
    return spans


def _pattern_spans(text: str, pattern: str, *, sensitive_type: str, origin_rule: str) -> list[_Span]:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return []
    return [
        _Span(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            sensitive_type=sensitive_type,
            origin_rule=origin_rule,
        )
        for match in compiled.finditer(text)
        if match.group(0)
    ]


def _select_non_overlapping_spans(spans: list[_Span]) -> list[_Span]:
    selected: list[_Span] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start), item.origin_rule)):
        if any(span.start < other.end and span.end > other.start for other in selected):
            continue
        selected.append(span)
    return selected


def _detect_sensitive_suggestions(text: str, *, detectors: dict[str, bool], covered_spans: list[_Span]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for detector_name, enabled in detectors.items():
        if not enabled:
            continue
        pattern = _DETECTOR_PATTERNS.get(detector_name)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            if not matched_text:
                continue
            overlapped = any(match.start() < span.end and match.end() > span.start for span in covered_spans)
            if overlapped:
                continue
            warnings.append(
                {
                    "detector": detector_name,
                    "match_sha256": _stable_hash(matched_text),
                    "length": len(matched_text),
                    "status": "suggested_not_redacted",
                }
            )
    return warnings


def blind_transform_text(
    text: str,
    *,
    policy: BlindInferencePolicy,
    vault: TokenVault,
) -> dict[str, Any]:
    source = str(text or "")
    candidate_spans: list[_Span] = []
    for rule in policy.rules:
        for literal in rule.values:
            candidate_spans.extend(
                _literal_spans(
                    source,
                    literal,
                    sensitive_type=rule.sensitive_type,
                    origin_rule=rule.name,
                )
            )
        if rule.pattern:
            candidate_spans.extend(
                _pattern_spans(
                    source,
                    rule.pattern,
                    sensitive_type=rule.sensitive_type,
                    origin_rule=rule.name,
                )
            )
    selected_spans = _select_non_overlapping_spans(candidate_spans)
    pieces: list[str] = []
    cursor = 0
    applied: list[dict[str, Any]] = []
    for span in selected_spans:
        pieces.append(source[cursor:span.start])
        placeholder = vault.issue_placeholder(
            sensitive_type=span.sensitive_type,
            original=span.text,
            origin_rule=span.origin_rule,
        )
        pieces.append(placeholder)
        applied.append(
            {
                "sensitive_type": span.sensitive_type,
                "placeholder": placeholder,
                "origin_rule": span.origin_rule,
                "original_sha256": _stable_hash(span.text),
            }
        )
        cursor = span.end
    pieces.append(source[cursor:])
    blinded = "".join(pieces)
    after_privacy_filter = privacy_filter(blinded)
    warnings = _detect_sensitive_suggestions(source, detectors=policy.detectors, covered_spans=selected_spans)
    return {
        "text": after_privacy_filter,
        "applied": applied,
        "warnings": warnings,
        "baseline_redaction_applied": after_privacy_filter != blinded,
    }


def blind_transform_request(
    messages: list[dict[str, str]],
    *,
    policy: BlindInferencePolicy,
    task_id: str,
) -> dict[str, Any]:
    vault = TokenVault(task_id=task_id)
    transformed_messages: list[dict[str, str]] = []
    applied_count = 0
    warnings: list[dict[str, Any]] = []
    baseline_redaction_applied = False
    classes: set[str] = set()
    for message in messages:
        content = str(message.get("content") or "")
        transformed = blind_transform_text(content, policy=policy, vault=vault)
        transformed_messages.append({**message, "content": transformed["text"]})
        applied = list(transformed.get("applied") or [])
        applied_count += len(applied)
        classes.update(str(item.get("sensitive_type") or "") for item in applied if item.get("sensitive_type"))
        warnings.extend(list(transformed.get("warnings") or []))
        baseline_redaction_applied = baseline_redaction_applied or bool(transformed.get("baseline_redaction_applied"))
    return {
        "messages": transformed_messages,
        "vault": vault,
        "task_id": task_id,
        "policy_id": policy.policy_id,
        "span_count": applied_count,
        "sensitive_classes": sorted(classes),
        "warnings": warnings,
        "baseline_redaction_applied": baseline_redaction_applied,
    }


def blind_rehydrate_response(text: str, vault: TokenVault | None) -> str:
    if vault is None:
        return str(text or "")
    return vault.rehydrate_text(str(text or ""))
