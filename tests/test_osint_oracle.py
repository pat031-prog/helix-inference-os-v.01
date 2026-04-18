from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto.osint_oracle import (
    ALLOWED_CLAIM_LEVELS,
    CorrelationEngine,
    RawFeedItem,
    SignalIngestor,
    SignalNormalizer,
    SourceFetcher,
    build_alerts,
    build_oracle_artifact,
    extract_anchors,
    fixture_feed_items,
)


GITHUB_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Reland urgent GC fix for V8 renderer regression</title>
    <updated>2026-04-17T16:00:00Z</updated>
    <link href="https://github.com/v8/v8/commit/abc123"/>
    <content>Fix garbage collector race and use-after-free risk.</content>
  </entry>
</feed>
"""


RSS_FIXTURE = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>Odd V8 garbage collector crash behavior</title>
    <link>https://research.example/v8-gc</link>
    <pubDate>Fri, 17 Apr 2026 12:00:00 GMT</pubDate>
    <description>Researcher notes crash loops and memory corruption symptoms.</description>
  </item>
</channel></rss>
"""


class FakeStateClient:
    def __init__(self) -> None:
        self.catalog = MemoryCatalog(":memory:")
        self.node_by_memory_id: dict[str, str] = {}

    async def bulk_remember(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stored = self.catalog.bulk_remember(items)
        rows = []
        for item in stored:
            digest = f"node-{item.memory_id}"
            self.node_by_memory_id[item.memory_id] = digest
            row = item.to_dict()
            row["node_hash"] = digest
            rows.append(row)
        return rows

    async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        hits = self.catalog.search(**kwargs)
        for hit in hits:
            hit.setdefault("node_hash", self.node_by_memory_id.get(hit.get("memory_id", ""), ""))
        return hits

    async def verify_chain(self, leaf_hash: str, policy: str | None = None) -> dict[str, Any]:
        return {"status": "verified", "leaf_hash": leaf_hash, "policy": policy, "backend": "fake"}


def test_parse_github_atom_and_rss_fixture() -> None:
    github_items = SourceFetcher.parse_atom(GITHUB_ATOM, source_type="github", source_name="v8/v8")
    rss_items = SourceFetcher.parse_feed(RSS_FIXTURE, source_type="rss", source_name="research")

    assert github_items[0].title.startswith("Reland urgent GC")
    assert github_items[0].source_url.endswith("abc123")
    assert rss_items[0].title.startswith("Odd V8")
    assert "memory corruption" in rss_items[0].body


def test_normalization_hashes_are_deterministic_and_deduped() -> None:
    normalizer = SignalNormalizer()
    item = RawFeedItem("rss", "research", "https://x", "V8 GC crash", "crash fix", "2026-04-17T12:00:00Z")
    one = normalizer.normalize(item, observed_at_ms=1)
    two = normalizer.normalize(item, observed_at_ms=999)
    many = normalizer.normalize_many([item, item], observed_at_ms=1)

    assert one.source_hash == two.source_hash
    assert len(many) == 1
    assert "v8" in one.components
    assert "crash" in one.risk_terms


@pytest.mark.asyncio
async def test_privacy_filter_redacts_before_remember() -> None:
    client = FakeStateClient()
    signals = SignalNormalizer().normalize_many(fixture_feed_items())
    await SignalIngestor(client).ingest(signals)

    stored = client.catalog.get_memory(signals[0].memory_id)
    assert stored is not None
    assert "sk-proj-" not in stored.content
    assert "[REDACTED_SECRET]" in stored.content


@pytest.mark.asyncio
async def test_correlation_finds_v8_gc_commit_and_discussion() -> None:
    client = FakeStateClient()
    signals = SignalNormalizer().normalize_many(fixture_feed_items())
    await SignalIngestor(client).ingest(signals)
    alerts, correlation = await build_alerts(client, signals, llm_mode="synthetic")

    assert correlation["correlations"]
    assert alerts
    alert = alerts[0].to_dict()
    assert alert["component"] == "v8"
    assert alert["claim_level"] in {"advisory_candidate", "pre_advisory_hypothesis", "correlated_osint_signal"}
    assert alert["chain_receipts"]
    assert all(receipt["status"] == "verified" for receipt in alert["chain_receipts"])


@pytest.mark.asyncio
async def test_semantic_router_uses_anchor_queries_not_generic_latest_vulnerability() -> None:
    client = FakeStateClient()
    signals = SignalNormalizer().normalize_many(fixture_feed_items())
    await SignalIngestor(client).ingest(signals)
    payload = await CorrelationEngine().correlate(client, signals)

    queries = [row["query"] for row in payload["correlations"] + payload["noise"] if row.get("query")]
    assert queries
    assert all("latest vulnerability" not in query for query in queries)
    assert any("v8" in query and ("crash" in query or "fix" in query) for query in queries)


def test_alert_scorer_never_emits_forbidden_claim_level() -> None:
    components, risks = extract_anchors("V8 garbage collector urgent use-after-free fix")
    assert components == ["v8"]
    assert "use-after-free" in risks
    assert "confirmed_zero_day" not in ALLOWED_CLAIM_LEVELS
    assert "exploit_available" not in ALLOWED_CLAIM_LEVELS
    assert "cve_confirmed" not in ALLOWED_CLAIM_LEVELS


def test_artifact_does_not_persist_secret_patterns(tmp_path: Path) -> None:
    signals = SignalNormalizer().normalize_many(fixture_feed_items())
    artifact = build_oracle_artifact(
        mode="fixture",
        profile="public-light",
        source_errors=[],
        signals=signals,
        ingest_receipt={"stored_count": len(signals), "deduped_count": len(signals)},
        alerts=[],
        correlation_payload={"noise": []},
        duration_s=0.1,
        llm_mode="synthetic",
    )
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    assert "sk-proj-" not in text
    assert artifact["alerts_count"] == 0
