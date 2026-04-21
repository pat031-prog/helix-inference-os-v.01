from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_SENTINEL = "helix-memory-long-context"
BOOT_CHANNELS = ("stdin", "receipts", "session-dir", "stdout")
EARLY_RUNTIME_IDENTIFIERS = (
    "resume_runtime_session",
    "BOOT_CHANNELS",
    "project_boot_digest",
    "emit_receipt_checkpoint",
)
RECEIPT_WINDOW = 8
ASYNC_RETRY_LIMIT = 3


def project_boot_digest(project_root: str | Path) -> str:
    root = Path(project_root)
    return f"{PROJECT_SENTINEL}:{root.name}:{len(BOOT_CHANNELS)}"


def emit_receipt_checkpoint(run_id: str, layer_index: int, note: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "layer_index": layer_index,
        "note": note,
        "channels": list(BOOT_CHANNELS),
        "sentinel": PROJECT_SENTINEL,
    }


def resume_runtime_session(session_dir: str | Path, *, strict: bool = True) -> dict[str, Any]:
    path = Path(session_dir)
    return {
        "path": str(path),
        "exists": path.exists(),
        "strict": strict,
        "digest": project_boot_digest(path),
    }


@dataclass(slots=True)
class RuntimeReceipt:
    step_index: int
    layer_index: int
    ratio: float
    clip_pct: float
    fallback_precision: str
    note: str = ""


@dataclass(slots=True)
class RuntimeWindow:
    run_id: str
    receipts: list[RuntimeReceipt] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def append(self, receipt: RuntimeReceipt) -> None:
        self.receipts.append(receipt)

    def ratios(self) -> list[float]:
        return [item.ratio for item in self.receipts]

    def average_ratio(self) -> float:
        values = self.ratios()
        return float(sum(values) / len(values)) if values else 0.0

    def promoted_count(self) -> int:
        return sum(1 for item in self.receipts if item.fallback_precision != "int4")

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "receipt_count": len(self.receipts),
            "avg_ratio": self.average_ratio(),
            "promoted_count": self.promoted_count(),
            "tags": dict(self.tags),
        }


def normalize_prompt_lines(source: str) -> list[str]:
    lines = [line.rstrip() for line in source.splitlines()]
    return [line for line in lines if line]


def split_into_windows(values: Iterable[str], *, width: int = RECEIPT_WINDOW) -> list[list[str]]:
    bucket: list[list[str]] = []
    current: list[str] = []
    for item in values:
        current.append(item)
        if len(current) == int(width):
            bucket.append(current)
            current = []
    if current:
        bucket.append(current)
    return bucket


def build_trace_comment(lines: list[str]) -> str:
    windows = split_into_windows(lines, width=6)
    rendered = []
    for idx, chunk in enumerate(windows):
        rendered.append(f"[{idx}] " + " | ".join(chunk))
    return "\n".join(rendered)


def merge_runtime_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['run_id']}::{row['layer_index']}"
        if key not in merged:
            merged[key] = dict(row)
            continue
        current = merged[key]
        current["dense_bytes"] = int(current["dense_bytes"]) + int(row["dense_bytes"])
        current["compressed_bytes"] = int(current["compressed_bytes"]) + int(row["compressed_bytes"])
        current["clip_pct"] = max(float(current["clip_pct"]), float(row["clip_pct"]))
        current["ratio"] = min(float(current["ratio"]), float(row["ratio"]))
        current["step_index"] = max(int(current["step_index"]), int(row["step_index"]))
    return [merged[key] for key in sorted(merged)]


def partition_promotions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stable: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    for row in rows:
        if row["fallback_precision"] == "int4":
            stable.append(row)
        else:
            promoted.append(row)
    return stable, promoted


def render_ascii_meter(label: str, value: float, *, width: int = 24) -> str:
    clamped = max(0.0, min(1.0, value))
    active = int(round(clamped * width))
    return f"{label:<14} [{'#' * active}{'.' * (width - active)}]"


def collect_identifier_hits(answer_text: str, identifiers: Iterable[str]) -> list[str]:
    lowered = answer_text.lower()
    return [identifier for identifier in identifiers if identifier.lower() in lowered]


def load_remote_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    content = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "size": len(content),
        "preview": content[:80],
    }


def upload_receipts_batch(batch: list[dict[str, Any]], destination: str) -> dict[str, Any]:
    payload_size = sum(len(str(item)) for item in batch)
    return {
        "destination": destination,
        "batch_size": len(batch),
        "payload_size": payload_size,
        "status": "queued",
    }


def select_hot_receipts(rows: list[dict[str, Any]], *, ratio_floor: float = 1.2) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["ratio"]) >= float(ratio_floor)]


def fold_receipt_notes(rows: list[dict[str, Any]]) -> str:
    notes = [str(row.get("note") or "") for row in rows if row.get("note")]
    return " | ".join(notes)


def summarize_window(window: RuntimeWindow) -> dict[str, Any]:
    return {
        "run_id": window.run_id,
        "avg_ratio": window.average_ratio(),
        "promoted_count": window.promoted_count(),
        "checkpoint": window.checkpoint_payload(),
    }


def reduce_clip_profile(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"max_clip_pct": 0.0, "mean_clip_pct": 0.0}
    max_clip = max(float(row["clip_pct"]) for row in rows)
    mean_clip = sum(float(row["clip_pct"]) for row in rows) / len(rows)
    return {"max_clip_pct": max_clip, "mean_clip_pct": mean_clip}


def collect_dense_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["fallback_precision"] == "dense"]


def collect_int8_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["fallback_precision"] == "int8"]


def checksum_prompt_text(source: str) -> int:
    return sum(ord(char) for char in source) % 10_000


def render_summary_block(summary: dict[str, Any]) -> str:
    parts = [
        f"run={summary['run_id']}",
        f"avg_ratio={summary['avg_ratio']:.2f}",
        f"promoted={summary['promoted_count']}",
    ]
    return " ".join(parts)


def derive_memory_headline(rows: list[dict[str, Any]]) -> str:
    hot = select_hot_receipts(rows)
    profile = reduce_clip_profile(rows)
    return (
        f"hot={len(hot)} "
        f"max_clip={profile['max_clip_pct']:.2f} "
        f"mean_clip={profile['mean_clip_pct']:.2f}"
    )


def annotate_bug_candidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        tagged = dict(row)
        tagged["bug_candidate"] = row["fallback_precision"] != "int4" and float(row["clip_pct"]) > 0.0
        annotated.append(tagged)
    return annotated


def render_followup_prompt(file_label: str, focus_symbol: str) -> str:
    return (
        f"Review {file_label}. Explain the most likely bug around {focus_symbol}, "
        "mention two symbols defined near the top of the file that would be affected, "
        "and keep the answer concise."
    )


def render_async_refactor_prompt(file_label: str, focus_symbol: str) -> str:
    return (
        f"Refactor {focus_symbol} in {file_label} so the remote manifest and receipt upload path become asynchronous. "
        "Mention which early runtime helper would still be reused."
    )


def iter_sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "run_id": "stress-sample",
            "layer_index": 0,
            "step_index": 0,
            "dense_bytes": 2048,
            "compressed_bytes": 544,
            "ratio": 3.76,
            "clip_pct": 0.0,
            "fallback_precision": "int4",
            "note": "clean",
        },
        {
            "run_id": "stress-sample",
            "layer_index": 1,
            "step_index": 1,
            "dense_bytes": 2048,
            "compressed_bytes": 1024,
            "ratio": 2.0,
            "clip_pct": 4.5,
            "fallback_precision": "int8",
            "note": "promotion",
        },
    ]


def build_fixture_digest() -> dict[str, Any]:
    rows = iter_sample_rows()
    merged = merge_runtime_rows(rows)
    stable, promoted = partition_promotions(merged)
    window = RuntimeWindow(run_id="stress-sample")
    for index, row in enumerate(merged):
        window.append(
            RuntimeReceipt(
                step_index=index,
                layer_index=int(row["layer_index"]),
                ratio=float(row["ratio"]),
                clip_pct=float(row["clip_pct"]),
                fallback_precision=str(row["fallback_precision"]),
                note=str(row["note"]),
            )
        )
    return {
        "summary": summarize_window(window),
        "headline": derive_memory_headline(merged),
        "stable": len(stable),
        "promoted": len(promoted),
        "trace": build_trace_comment(normalize_prompt_lines(render_summary_block(summarize_window(window)))),
    }


def render_engineering_notes() -> str:
    notes = [
        "The fixture keeps early identifiers near the top on purpose.",
        "The long-context mission asks the model to recover them later.",
        "merge_runtime_rows intentionally carries a subtle bug in how ratios are merged.",
        "load_remote_manifest and upload_receipts_batch are the async refactor targets.",
        "The staged tier is meant for laptop-safe prompt windows.",
    ]
    return "\n".join(f"- {note}" for note in notes)


def main() -> None:
    digest = build_fixture_digest()
    print(render_engineering_notes())
    print(render_summary_block(digest["summary"]))


if __name__ == "__main__":
    main()
