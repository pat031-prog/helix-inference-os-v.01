from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


GENERIC_QUERY_TERMS = {
    "about",
    "agent",
    "agents",
    "all",
    "anything",
    "context",
    "dag",
    "data",
    "details",
    "find",
    "helix",
    "history",
    "indexed",
    "info",
    "information",
    "latest",
    "memory",
    "memories",
    "notes",
    "previous",
    "recall",
    "records",
    "retrieval",
    "search",
    "semantic",
    "session",
    "sessions",
    "show",
    "stuff",
    "summary",
    "synthetic",
    "things",
}

STOP_TERMS = {
    "and",
    "are",
    "but",
    "can",
    "con",
    "del",
    "for",
    "from",
    "hay",
    "las",
    "los",
    "que",
    "the",
    "una",
    "uno",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_\-]{1,}", str(text or ""))]


class MemoryLike(Protocol):
    memory_id: str
    project: str
    agent_id: str
    memory_type: str
    summary: str
    content: str
    importance: int
    tags: list[str]
    decay_score: float
    created_ms: float
    last_access_ms: float


@dataclass(frozen=True)
class RoutedQuery:
    original_query: str
    routed_query: str
    action: str
    reason: str
    anchor_terms: list[str] = field(default_factory=list)
    retained_terms: list[str] = field(default_factory=list)
    dropped_terms: list[str] = field(default_factory=list)
    generic_terms: list[str] = field(default_factory=list)
    corpus_doc_count: int = 0
    max_doc_frequency_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "original_query": self.original_query,
            "routed_query": self.routed_query,
            "action": self.action,
            "reason": self.reason,
            "anchor_terms": list(self.anchor_terms),
            "retained_terms": list(self.retained_terms),
            "dropped_terms": list(self.dropped_terms),
            "generic_terms": list(self.generic_terms),
            "corpus_doc_count": self.corpus_doc_count,
            "max_doc_frequency_ratio": self.max_doc_frequency_ratio,
        }


class SemanticQueryRouter:
    """Deterministic guardrail that keeps broad LLM queries off the worst WAND path.

    The router does not invent facts. It only replaces broad retrieval prompts with
    high-information lexical anchors already present in the scoped HeliX memory corpus.
    """

    def __init__(
        self,
        *,
        max_anchor_terms: int = 1,
        max_doc_frequency_ratio: float = 0.35,
        min_specific_terms: int = 2,
    ) -> None:
        self.max_anchor_terms = int(max_anchor_terms)
        self.max_doc_frequency_ratio = float(max_doc_frequency_ratio)
        self.min_specific_terms = int(min_specific_terms)

    def route(
        self,
        *,
        query: str,
        memories: Iterable[MemoryLike],
        project: str,
        agent_filter: str | None,
        type_filter: set[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> RoutedQuery:
        query_tokens = list(dict.fromkeys(tokenize(query)))
        generic_terms = [tok for tok in query_tokens if self._is_generic(tok)]
        excluded = exclude_ids or set()
        type_filter = type_filter or set()
        scoped = [
            item
            for item in memories
            if item.project == project
            and (agent_filter is None or item.agent_id == agent_filter)
            and (not type_filter or item.memory_type in type_filter)
            and item.memory_id not in excluded
        ]
        if not query_tokens:
            return RoutedQuery(
                original_query=query,
                routed_query="",
                action="empty",
                reason="query_tokenization_empty",
                corpus_doc_count=len(scoped),
            )

        term_doc_freq = self._document_frequency(scoped)
        doc_count = max(len(scoped), 1)
        retained_terms = [
            tok for tok in query_tokens if self._is_query_term_selective(tok, term_doc_freq.get(tok, 0), doc_count)
        ]
        max_query_df_ratio = max((term_doc_freq.get(tok, 0) / doc_count for tok in query_tokens), default=0.0)

        if generic_terms and retained_terms:
            return RoutedQuery(
                original_query=query,
                routed_query=" ".join(retained_terms),
                action="rewrite",
                reason="generic_terms_dropped_kept_selective_terms",
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=len(scoped),
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        if len(retained_terms) >= self.min_specific_terms or (
            len(retained_terms) == 1 and self._has_structural_signal(retained_terms[0])
        ):
            return RoutedQuery(
                original_query=query,
                routed_query=query,
                action="pass_through",
                reason="query_already_selective",
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=len(scoped),
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        if generic_terms and not retained_terms:
            return RoutedQuery(
                original_query=query,
                routed_query=query,
                action="recent_fallback",
                reason="fully_generic_query_uses_recent_guard",
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=len(scoped),
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        anchors = self._select_anchor_terms(scoped, term_doc_freq, set(query_tokens))
        if anchors:
            routed_terms = list(dict.fromkeys(retained_terms + anchors))
            return RoutedQuery(
                original_query=query,
                routed_query=" ".join(routed_terms),
                action="rewrite",
                reason="generic_query_rewritten_with_corpus_anchors",
                anchor_terms=anchors,
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=len(scoped),
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        return RoutedQuery(
            original_query=query,
            routed_query=query,
            action="recent_fallback",
            reason="generic_query_without_selective_anchors",
            retained_terms=retained_terms,
            dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
            generic_terms=generic_terms,
            corpus_doc_count=len(scoped),
            max_doc_frequency_ratio=max_query_df_ratio,
        )

    def route_from_index(
        self,
        *,
        query: str,
        doc_count: int,
        term_doc_freq: dict[str, int],
        term_anchor_scores: dict[str, float],
    ) -> RoutedQuery:
        query_tokens = list(dict.fromkeys(tokenize(query)))
        generic_terms = [tok for tok in query_tokens if self._is_generic(tok)]
        if not query_tokens:
            return RoutedQuery(
                original_query=query,
                routed_query="",
                action="empty",
                reason="query_tokenization_empty",
                corpus_doc_count=doc_count,
            )

        safe_doc_count = max(int(doc_count), 1)
        retained_terms = [
            tok for tok in query_tokens if self._is_query_term_selective(tok, term_doc_freq.get(tok, 0), safe_doc_count)
        ]
        max_query_df_ratio = max((term_doc_freq.get(tok, 0) / safe_doc_count for tok in query_tokens), default=0.0)
        if generic_terms and retained_terms:
            return RoutedQuery(
                original_query=query,
                routed_query=" ".join(retained_terms),
                action="rewrite",
                reason="generic_terms_dropped_kept_selective_terms",
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=doc_count,
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        if len(retained_terms) >= self.min_specific_terms or (
            len(retained_terms) == 1 and self._has_structural_signal(retained_terms[0])
        ):
            return RoutedQuery(
                original_query=query,
                routed_query=query,
                action="pass_through",
                reason="query_already_selective",
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=doc_count,
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        if generic_terms and not retained_terms:
            return RoutedQuery(
                original_query=query,
                routed_query=query,
                action="recent_fallback",
                reason="fully_generic_query_uses_recent_guard",
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=doc_count,
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        anchors = self._select_anchor_terms_from_index(term_doc_freq, term_anchor_scores, set(query_tokens), safe_doc_count)
        if anchors:
            routed_terms = list(dict.fromkeys(retained_terms + anchors))
            return RoutedQuery(
                original_query=query,
                routed_query=" ".join(routed_terms),
                action="rewrite",
                reason="generic_query_rewritten_with_indexed_anchors",
                anchor_terms=anchors,
                retained_terms=retained_terms,
                dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
                generic_terms=generic_terms,
                corpus_doc_count=doc_count,
                max_doc_frequency_ratio=max_query_df_ratio,
            )

        return RoutedQuery(
            original_query=query,
            routed_query=query,
            action="recent_fallback",
            reason="generic_query_without_selective_anchors",
            retained_terms=retained_terms,
            dropped_terms=[tok for tok in query_tokens if tok not in retained_terms],
            generic_terms=generic_terms,
            corpus_doc_count=doc_count,
            max_doc_frequency_ratio=max_query_df_ratio,
        )

    @staticmethod
    def _document_frequency(memories: list[MemoryLike]) -> dict[str, int]:
        df: dict[str, int] = {}
        for item in memories:
            terms = set(tokenize(item.summary))
            terms.update(tokenize(item.content))
            terms.update(tokenize(" ".join(item.tags)))
            for term in terms:
                df[term] = df.get(term, 0) + 1
        return df

    def _select_anchor_terms(
        self,
        memories: list[MemoryLike],
        term_doc_freq: dict[str, int],
        query_terms: set[str],
    ) -> list[str]:
        doc_count = max(len(memories), 1)
        weighted: dict[str, float] = {}
        for item in memories:
            quality = 1.0 + (max(float(item.importance), 0.0) / 10.0) * 0.4 + max(float(item.decay_score), 0.0) * 0.2
            fields = [
                (item.summary, 3.0),
                (" ".join(item.tags), 2.2),
                (item.content, 1.0),
            ]
            for text, field_weight in fields:
                for term in tokenize(text):
                    if term in query_terms or self._is_generic(term):
                        continue
                    if not self._is_anchor_candidate(term):
                        continue
                    df = max(term_doc_freq.get(term, 1), 1)
                    df_ratio = df / doc_count
                    if doc_count >= 100 and df_ratio > self.max_doc_frequency_ratio:
                        continue
                    idf = math.log((doc_count + 1.0) / (df + 0.5)) + 1.0
                    structural_bonus = 1.6 if self._has_structural_signal(term) else 1.0
                    score = field_weight * quality * idf * structural_bonus
                    weighted[term] = max(weighted.get(term, 0.0), score)

        ranked = sorted(
            weighted.items(),
            key=lambda item: (-item[1], term_doc_freq.get(item[0], 0), item[0]),
        )
        return [term for term, _ in ranked[: self.max_anchor_terms]]

    def _select_anchor_terms_from_index(
        self,
        term_doc_freq: dict[str, int],
        term_anchor_scores: dict[str, float],
        query_terms: set[str],
        doc_count: int,
    ) -> list[str]:
        ranked: list[tuple[str, float]] = []
        for term, base_score in term_anchor_scores.items():
            if term in query_terms or self._is_generic(term):
                continue
            df = max(term_doc_freq.get(term, 1), 1)
            df_ratio = df / max(doc_count, 1)
            if doc_count >= 100 and df_ratio > self.max_doc_frequency_ratio:
                continue
            idf = math.log((doc_count + 1.0) / (df + 0.5)) + 1.0
            ranked.append((term, float(base_score) * idf))
        ranked.sort(key=lambda item: (-item[1], term_doc_freq.get(item[0], 0), item[0]))
        return [term for term, _ in ranked[: self.max_anchor_terms]]

    @staticmethod
    def _has_structural_signal(term: str) -> bool:
        return any(ch.isdigit() for ch in term) or "-" in term or "_" in term

    def _is_query_term_selective(self, term: str, doc_freq: int, doc_count: int) -> bool:
        if self._is_generic(term):
            return False
        if self._has_structural_signal(term):
            return True
        if doc_count < 100:
            return True
        if len(term) >= 8 and doc_freq <= max(1, int(doc_count * 0.50)):
            return True
        return doc_freq > 0 and (doc_freq / max(doc_count, 1)) <= self.max_doc_frequency_ratio

    def _is_anchor_candidate(self, term: str) -> bool:
        if self._is_generic(term):
            return False
        return len(term) >= 4 or self._has_structural_signal(term)

    @staticmethod
    def _is_generic(term: str) -> bool:
        return term in STOP_TERMS or term in GENERIC_QUERY_TERMS


__all__ = ["RoutedQuery", "SemanticQueryRouter", "tokenize"]
