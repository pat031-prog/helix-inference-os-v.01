from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from helix_kv.ipc_state_server import StateClient
from helix_kv.memory_catalog import privacy_filter


PROJECT = "osint-zero-day-oracle"
AGENT_ID = "osint-oracle"
ALLOWED_CLAIM_LEVELS = {
    "correlated_osint_signal",
    "pre_advisory_hypothesis",
    "advisory_candidate",
    "needs_human_review",
}
DEFAULT_GITHUB_REPOS = ("v8/v8", "torvalds/linux", "openssl/openssl", "curl/curl")

COMPONENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "v8": ("v8", "garbage collector", "gc", "jit", "wasm", "renderer"),
    "linux-kernel": ("linux", "kernel", "kvm", "ebpf", "netfilter", "io_uring"),
    "openssl": ("openssl", "tls", "ssl", "x509", "certificate"),
    "curl": ("curl", "libcurl", "http2", "http/2", "h2", "proxy"),
}

RISK_TERMS = (
    "crash",
    "uaf",
    "use-after-free",
    "oob",
    "out-of-bounds",
    "overflow",
    "race",
    "revert",
    "security",
    "exploit",
    "sandbox escape",
    "urgent",
    "regression",
    "fix",
    "patch",
    "memory corruption",
)


@dataclass(frozen=True)
class RawFeedItem:
    source_type: str
    source_name: str
    source_url: str
    title: str
    body: str = ""
    published_at: str = ""


@dataclass(frozen=True)
class OsintSignal:
    source_type: str
    source_name: str
    source_url: str
    published_at: str
    observed_at_ms: int
    title: str
    body: str
    source_hash: str
    tags: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    risk_terms: list[str] = field(default_factory=list)

    @property
    def memory_id(self) -> str:
        return f"osint-{self.source_hash[:20]}"

    def to_memory_item(self) -> dict[str, Any]:
        clean_title = privacy_filter(self.title)
        clean_body = privacy_filter(self.body)
        content = (
            f"OSINT signal {self.source_hash}\n"
            f"source_type={self.source_type}\n"
            f"source_name={self.source_name}\n"
            f"source_url={self.source_url}\n"
            f"published_at={self.published_at}\n"
            f"title={clean_title}\n\n{clean_body}"
        )
        tags = [
            "osint",
            f"source:{self.source_type}",
            f"source_hash:{self.source_hash[:12]}",
            *[f"component:{item}" for item in self.components],
            *[f"risk:{item.replace(' ', '-')}" for item in self.risk_terms],
            *self.tags,
        ]
        return {
            "project": PROJECT,
            "agent_id": AGENT_ID,
            "memory_type": "semantic",
            "memory_id": self.memory_id,
            "summary": f"{clean_title[:120]} [{','.join(self.components[:2]) or 'unclassified'}]",
            "content": content[:5000],
            "importance": 8 if self.risk_terms and self.components else 5,
            "tags": tags,
            "decay_score": 1.0,
        }


@dataclass
class OracleAlert:
    alert_id: str
    claim_level: str
    severity: str
    confidence_score: float
    component: str
    matched_terms: list[str]
    source_count: int
    timeline: list[dict[str, Any]]
    memory_ids: list[str]
    chain_receipts: list[dict[str, Any]]
    search_ms_p50: float | None
    llm_latency_ms: float | None
    caveats: list[str]
    synthesis: str
    semantic_router_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["claim_level"] not in ALLOWED_CLAIM_LEVELS:
            payload["claim_level"] = "needs_human_review"
            payload["caveats"].append("claim_level_was_downgraded_to_allowed_set")
        return payload


class SourceFetcher:
    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = float(timeout)

    async def fetch_github_atom(self, repo: str, *, max_items: int = 50) -> tuple[list[RawFeedItem], list[str]]:
        url = f"https://github.com/{repo}/commits.atom"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
            return self.parse_atom(response.text, source_type="github", source_name=repo, fallback_url=url)[:max_items], []
        except Exception as exc:  # noqa: BLE001
            return [], [f"github:{repo}:{type(exc).__name__}:{exc}"]

    async def fetch_rss(self, url: str, *, source_name: str | None = None, max_items: int = 50) -> tuple[list[RawFeedItem], list[str]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
            return self.parse_feed(response.text, source_type="rss", source_name=source_name or url, fallback_url=url)[:max_items], []
        except Exception as exc:  # noqa: BLE001
            return [], [f"rss:{url}:{type(exc).__name__}:{exc}"]

    async def fetch_hackernews(self, *, max_items: int = 50) -> tuple[list[RawFeedItem], list[str]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
                response.raise_for_status()
                ids = response.json()[:max_items]
                responses = await asyncio.gather(
                    *[client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json") for sid in ids],
                    return_exceptions=True,
                )
            items: list[RawFeedItem] = []
            errors: list[str] = []
            for sid, item_response in zip(ids, responses):
                if isinstance(item_response, Exception):
                    errors.append(f"hackernews:{sid}:{type(item_response).__name__}:{item_response}")
                    continue
                data = item_response.json()
                if not isinstance(data, dict) or not data.get("title"):
                    continue
                published = ""
                if data.get("time"):
                    published = datetime.fromtimestamp(float(data["time"]), tz=timezone.utc).isoformat()
                items.append(
                    RawFeedItem(
                        "hackernews",
                        "hackernews-top",
                        str(data.get("url") or f"https://news.ycombinator.com/item?id={sid}"),
                        str(data.get("title") or ""),
                        f"score={data.get('score', 0)} comments={data.get('descendants', 0)} id={sid}",
                        published,
                    )
                )
            return items, errors
        except Exception as exc:  # noqa: BLE001
            return [], [f"hackernews:topstories:{type(exc).__name__}:{exc}"]

    @staticmethod
    def parse_atom(xml_text: str, *, source_type: str, source_name: str, fallback_url: str = "") -> list[RawFeedItem]:
        root = ET.fromstring(xml_text)
        items: list[RawFeedItem] = []
        for entry in _iter_named(root, "entry"):
            title = _xml_text(entry, "title")
            body = _xml_text(entry, "summary") or _xml_text(entry, "content")
            published = _xml_text(entry, "updated") or _xml_text(entry, "published")
            link = fallback_url
            for link_node in _iter_named(entry, "link", direct=True):
                if link_node.attrib.get("href"):
                    link = str(link_node.attrib["href"])
                    break
            if title:
                items.append(RawFeedItem(source_type, source_name, link, _clean_text(title), _clean_text(body), published))
        return items

    @staticmethod
    def parse_feed(xml_text: str, *, source_type: str, source_name: str, fallback_url: str = "") -> list[RawFeedItem]:
        root = ET.fromstring(xml_text)
        if list(_iter_named(root, "entry")):
            return SourceFetcher.parse_atom(xml_text, source_type=source_type, source_name=source_name, fallback_url=fallback_url)
        items: list[RawFeedItem] = []
        for item in _iter_named(root, "item"):
            title = _xml_text(item, "title")
            body = _xml_text(item, "description")
            published = _xml_text(item, "pubDate")
            link = _xml_text(item, "link") or fallback_url
            if title:
                items.append(RawFeedItem(source_type, source_name, link, _clean_text(title), _clean_text(body), published))
        return items


class SignalNormalizer:
    def normalize_many(self, items: list[RawFeedItem], *, observed_at_ms: int | None = None) -> list[OsintSignal]:
        observed = int(observed_at_ms or time.time() * 1000)
        seen: set[str] = set()
        signals: list[OsintSignal] = []
        for item in items:
            signal = self.normalize(item, observed_at_ms=observed)
            if signal.source_hash in seen:
                continue
            seen.add(signal.source_hash)
            signals.append(signal)
        return signals

    def normalize(self, item: RawFeedItem, *, observed_at_ms: int | None = None) -> OsintSignal:
        title = _clean_text(item.title)
        body = _clean_text(item.body)
        published = _normalize_timestamp(item.published_at)
        digest = hashlib.sha256(
            f"{item.source_url}|{title}|{published}|{body[:500]}".encode("utf-8", errors="replace")
        ).hexdigest()
        components, risks = extract_anchors(f"{title}\n{body}")
        return OsintSignal(
            source_type=_safe_token(item.source_type or "unknown"),
            source_name=(item.source_name or "unknown")[:120],
            source_url=item.source_url or "",
            published_at=published,
            observed_at_ms=int(observed_at_ms or time.time() * 1000),
            title=title,
            body=body,
            source_hash=digest,
            tags=["fixture" if item.source_type == "fixture" else "live"],
            components=components,
            risk_terms=risks,
        )


class SignalIngestor:
    def __init__(self, client: StateClient) -> None:
        self.client = client

    async def ingest(self, signals: list[OsintSignal]) -> dict[str, Any]:
        unique: dict[str, OsintSignal] = {}
        for signal in signals:
            unique.setdefault(signal.source_hash, signal)
        items = [signal.to_memory_item() for signal in unique.values()]
        stored = await self.client.bulk_remember(items) if items else []
        return {
            "input_count": len(signals),
            "deduped_count": len(items),
            "stored_count": len(stored),
            "memory_ids": [item["memory_id"] for item in items],
            "source_hashes": [signal.source_hash for signal in unique.values()],
        }


class CorrelationEngine:
    def __init__(self, *, min_independent_sources: int = 2, correlation_window_hours: float = 24.0) -> None:
        self.min_independent_sources = int(min_independent_sources)
        self.correlation_window_ms = float(correlation_window_hours) * 60.0 * 60.0 * 1000.0

    async def correlate(self, client: StateClient, signals: list[OsintSignal]) -> dict[str, Any]:
        by_memory_id = {signal.memory_id: signal for signal in signals}
        components = sorted({component for signal in signals for component in signal.components})
        correlations: list[dict[str, Any]] = []
        noise: list[dict[str, Any]] = []
        for component in components:
            component_signals = [signal for signal in signals if component in signal.components]
            risk_terms = sorted({term for signal in component_signals for term in signal.risk_terms})
            if not risk_terms:
                noise.append({"component": component, "reason": "no_risk_terms", "signal_count": len(component_signals)})
                continue
            query = " ".join([component, *risk_terms[:5]])
            t0 = time.perf_counter()
            hits = await client.search(
                project=PROJECT,
                agent_id=None,
                query=query,
                limit=12,
                memory_types=["semantic"],
                route_query=True,
            )
            search_ms = (time.perf_counter() - t0) * 1000.0
            matched: list[tuple[OsintSignal, dict[str, Any]]] = []
            router_actions: list[str] = []
            for hit in hits:
                signal = by_memory_id.get(str(hit.get("memory_id") or ""))
                if signal is None or component not in signal.components:
                    continue
                router = hit.get("semantic_router") or {}
                if isinstance(router, dict):
                    router_actions.append(str(router.get("action") or "pass_through"))
                matched.append((signal, hit))
            row = self._score(component, risk_terms, query, matched, search_ms, router_actions)
            if row["eligible"]:
                correlations.append(row)
            else:
                noise.append(row)
        return {"correlations": correlations, "noise": noise}

    def _score(
        self,
        component: str,
        risk_terms: list[str],
        query: str,
        matched: list[tuple[OsintSignal, dict[str, Any]]],
        search_ms: float,
        router_actions: list[str],
    ) -> dict[str, Any]:
        rows = list({signal.source_hash: (signal, hit) for signal, hit in matched}.values())
        source_types = sorted({signal.source_type for signal, _ in rows})
        timestamps = [_signal_time_ms(signal) for signal, _ in rows]
        window_ok = True if len(timestamps) < 2 else (max(timestamps) - min(timestamps)) <= self.correlation_window_ms
        has_commit = "github" in source_types
        has_discussion = any(src in {"rss", "hackernews", "forum", "mastodon"} for src in source_types)
        recent_fallback = "recent_fallback" in router_actions
        source_count = len(source_types)
        eligible = source_count >= self.min_independent_sources and window_ok and not recent_fallback
        strong = eligible and has_commit and has_discussion and len(risk_terms) >= 2
        confidence = max(0.0, min(0.2 + min(source_count, 4) * 0.15 + min(len(risk_terms), 5) * 0.05 + (0.15 if has_commit and has_discussion else 0.0) - (0.25 if recent_fallback else 0.0), 0.92))
        claim_level = "needs_human_review"
        if eligible:
            claim_level = "correlated_osint_signal"
        if strong:
            claim_level = "advisory_candidate" if confidence >= 0.85 else "pre_advisory_hypothesis"
        return {
            "component": component,
            "query": query,
            "eligible": eligible,
            "claim_level": claim_level,
            "severity": "high" if strong and {"uaf", "use-after-free", "memory corruption", "sandbox escape"} & set(risk_terms) else ("medium" if eligible else "low"),
            "confidence_score": round(confidence, 3),
            "matched_terms": risk_terms[:8],
            "source_count": source_count,
            "source_types": source_types,
            "timeline": [_timeline_row(signal, hit) for signal, hit in rows],
            "memory_ids": [signal.memory_id for signal, _ in rows],
            "node_hashes": [str(hit.get("node_hash") or "") for _, hit in rows if hit.get("node_hash")],
            "search_ms": search_ms,
            "semantic_router_actions": router_actions or ["pass_through"],
            "caveats": _base_caveats(recent_fallback=recent_fallback, window_ok=window_ok),
        }


class OracleSynthesizer:
    def __init__(self, *, mode: str = "synthetic", model: str = "Qwen/Qwen3.5-122B-A10B") -> None:
        self.mode = mode
        self.model = model

    async def synthesize(self, correlation: dict[str, Any]) -> tuple[str, float | None]:
        if self.mode != "deepinfra" or not os.environ.get("DEEPINFRA_API_TOKEN"):
            return self._synthetic_summary(correlation), None
        prompt = (
            "Produce a conservative OSINT pre-advisory hypothesis. Cite memory IDs. "
            "Do not invent a CVE. Do not provide exploit steps or proof-of-concept code. "
            "Keep under 140 words.\n\n"
            f"{json.dumps(_redacted_correlation_for_prompt(correlation), ensure_ascii=False)}"
        )
        request_json: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a cautious defensive security analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 220,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.deepinfra.com/v1/openai/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['DEEPINFRA_API_TOKEN']}"},
                json=request_json,
            )
            if response.status_code == 400:
                request_json.pop("enable_thinking", None)
                request_json.pop("chat_template_kwargs", None)
                response = await client.post(
                    "https://api.deepinfra.com/v1/openai/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['DEEPINFRA_API_TOKEN']}"},
                    json=request_json,
                )
            response.raise_for_status()
            data = response.json()
        text = str(data["choices"][0]["message"].get("content") or "").strip()
        return _sanitize_output(text)[:1200] or self._synthetic_summary(correlation), (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _synthetic_summary(correlation: dict[str, Any]) -> str:
        ids = ", ".join(correlation.get("memory_ids", [])[:4]) or "no-memory-ids"
        component = correlation.get("component", "unknown")
        terms = ", ".join(correlation.get("matched_terms", [])[:5])
        return (
            f"Conservative OSINT hypothesis: {component} has a correlated public signal involving {terms}. "
            f"This is not a confirmed zero-day or CVE. Review cited HeliX memories: {ids}."
        )


async def build_alerts(
    client: StateClient,
    signals: list[OsintSignal],
    *,
    min_independent_sources: int = 2,
    correlation_window_hours: float = 24.0,
    llm_mode: str = "synthetic",
    model: str = "Qwen/Qwen3.5-122B-A10B",
) -> tuple[list[OracleAlert], dict[str, Any]]:
    correlation_payload = await CorrelationEngine(
        min_independent_sources=min_independent_sources,
        correlation_window_hours=correlation_window_hours,
    ).correlate(client, signals)
    synthesizer = OracleSynthesizer(mode=llm_mode, model=model)
    alerts: list[OracleAlert] = []
    for idx, correlation in enumerate(correlation_payload["correlations"], start=1):
        receipts = []
        for node_hash in correlation.get("node_hashes", []):
            try:
                receipt = await client.verify_chain(node_hash)
            except Exception as exc:  # noqa: BLE001
                receipt = {"status": "verify_error", "node_hash": node_hash, "error": str(exc)}
            receipts.append(receipt)
        synthesis, llm_latency = await synthesizer.synthesize(correlation)
        alerts.append(
            OracleAlert(
                alert_id=f"osint-alert-{idx:03d}-{correlation['component']}",
                claim_level=correlation["claim_level"],
                severity=correlation["severity"],
                confidence_score=correlation["confidence_score"],
                component=correlation["component"],
                matched_terms=correlation["matched_terms"],
                source_count=correlation["source_count"],
                timeline=correlation["timeline"],
                memory_ids=correlation["memory_ids"],
                chain_receipts=receipts,
                search_ms_p50=correlation["search_ms"],
                llm_latency_ms=llm_latency,
                caveats=correlation["caveats"],
                synthesis=synthesis,
                semantic_router_actions=correlation["semantic_router_actions"],
            )
        )
    return alerts, correlation_payload


def build_oracle_artifact(
    *,
    mode: str,
    profile: str,
    source_errors: list[str],
    signals: list[OsintSignal],
    ingest_receipt: dict[str, Any],
    alerts: list[OracleAlert],
    correlation_payload: dict[str, Any],
    duration_s: float,
    llm_mode: str,
) -> dict[str, Any]:
    alert_payloads = [alert.to_dict() for alert in alerts]
    search_values = [alert.search_ms_p50 for alert in alerts if alert.search_ms_p50 is not None]
    payload = {
        "artifact": "local-zero-day-osint-oracle",
        "mode": mode,
        "profile": profile,
        "project": PROJECT,
        "llm_mode": llm_mode,
        "signals_ingested": ingest_receipt.get("stored_count", 0),
        "signals_seen": len(signals),
        "deduped_count": ingest_receipt.get("deduped_count", 0),
        "alerts_count": len(alert_payloads),
        "alerts": alert_payloads,
        "noise_floor_report": correlation_payload.get("noise", [])[:12],
        "source_errors": source_errors,
        "search_ms_p50": statistics.median(search_values) if search_values else None,
        "duration_s": duration_s,
        "claim_boundary": (
            "Conservative OSINT correlation only. This artifact does not confirm a zero-day, "
            "does not assign CVEs, and does not include exploit guidance."
        ),
        "generated_ms": int(time.time() * 1000),
    }
    if _contains_secret(json.dumps(payload, ensure_ascii=False)):
        payload["artifact_safety_error"] = "secret_pattern_detected_after_sanitization"
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fixture_feed_items() -> list[RawFeedItem]:
    base = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc).isoformat()
    later = datetime(2026, 4, 17, 16, 0, tzinfo=timezone.utc).isoformat()
    return [
        RawFeedItem(
            "rss",
            "security-research-public",
            "https://research.example/v8-gc-weirdness",
            "Odd V8 garbage collector crash behavior under renderer pressure",
            "Researcher notes strange garbage collector behavior, crash loops, and possible memory corruption. sk-proj-SECRETSECRETSECRET123456",
            base,
        ),
        RawFeedItem(
            "github",
            "v8/v8",
            "https://github.com/v8/v8/commit/urgent-gc-fix",
            "Reland urgent GC fix for V8 renderer regression",
            "Patch fixes garbage collector race and use-after-free risk in V8 object marking. Security-sensitive fix without advisory.",
            later,
        ),
        RawFeedItem(
            "hackernews",
            "hackernews-top",
            "https://news.ycombinator.com/item?id=999001",
            "Chrome renderer crashes after latest V8 rollout",
            "Operators discuss crash regression in renderer tied to wasm and GC pressure.",
            later,
        ),
        RawFeedItem("rss", "benign-feed", "https://example.com/postgres-release", "PostgreSQL maintenance release improves planner statistics", "A normal release note with no browser exploit signal.", base),
    ]


def extract_anchors(text: str) -> tuple[list[str], list[str]]:
    lower = str(text or "").lower()
    components = [component for component, patterns in COMPONENT_PATTERNS.items() if any(_contains_term(lower, pattern) for pattern in patterns)]
    risks = [term for term in RISK_TERMS if _contains_term(lower, term)]
    return sorted(set(components)), sorted(set(risks))


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _iter_named(node: ET.Element, tag: str, *, direct: bool = False) -> list[ET.Element]:
    children = list(node) if direct else list(node.iter())
    return [child for child in children if _local_name(child.tag) == tag]


def _xml_text(node: ET.Element, tag: str) -> str:
    found = next(iter(_iter_named(node, tag, direct=True)), None)
    return "" if found is None or found.text is None else found.text


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))).strip()


def _normalize_timestamp(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()
    except ValueError:
        try:
            return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            return raw[:80]


def _safe_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(value or "unknown").lower()).strip("-") or "unknown"


def _contains_term(text: str, term: str) -> bool:
    if " " in term or "-" in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _signal_time_ms(signal: OsintSignal) -> int:
    if signal.published_at:
        try:
            return int(datetime.fromisoformat(signal.published_at).timestamp() * 1000)
        except ValueError:
            pass
    return signal.observed_at_ms


def _timeline_row(signal: OsintSignal, hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": signal.memory_id,
        "source_type": signal.source_type,
        "source_name": signal.source_name,
        "published_at": signal.published_at,
        "title": signal.title[:180],
        "source_url": signal.source_url,
        "node_hash": hit.get("node_hash"),
        "score": hit.get("score"),
    }


def _base_caveats(*, recent_fallback: bool, window_ok: bool) -> list[str]:
    caveats = [
        "This is an OSINT correlation, not a confirmed vulnerability.",
        "No CVE or vendor advisory is asserted by HeliX v0.",
        "Human review is required before operational action.",
    ]
    if recent_fallback:
        caveats.append("Semantic Router used recent fallback; alert cannot be elevated.")
    if not window_ok:
        caveats.append("Signals fall outside the configured correlation window.")
    return caveats


def _redacted_correlation_for_prompt(correlation: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_level": correlation.get("claim_level"),
        "component": correlation.get("component"),
        "matched_terms": correlation.get("matched_terms", []),
        "source_count": correlation.get("source_count"),
        "memory_ids": correlation.get("memory_ids", []),
        "timeline": [
            {
                "memory_id": row.get("memory_id"),
                "source_type": row.get("source_type"),
                "published_at": row.get("published_at"),
                "title": privacy_filter(str(row.get("title") or ""))[:180],
            }
            for row in correlation.get("timeline", [])
        ],
        "caveats": correlation.get("caveats", []),
    }


def _sanitize_output(text: str) -> str:
    cleaned = privacy_filter(str(text or ""))
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "", cleaned).replace("```", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_secret(text: str) -> bool:
    return bool(re.search(r"(sk-proj-[A-Za-z0-9_\-]{12,}|gh[pus]_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._\-+/=]{20,}|api_key\s*=)", text))


__all__ = [
    "ALLOWED_CLAIM_LEVELS",
    "CorrelationEngine",
    "DEFAULT_GITHUB_REPOS",
    "OracleAlert",
    "OracleSynthesizer",
    "OsintSignal",
    "PROJECT",
    "RawFeedItem",
    "SignalIngestor",
    "SignalNormalizer",
    "SourceFetcher",
    "build_alerts",
    "build_oracle_artifact",
    "extract_anchors",
    "fixture_feed_items",
    "write_json",
]
