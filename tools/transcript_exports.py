from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _string_or_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def write_case_transcript_exports(
    *,
    output_dir: Path,
    case_id: str,
    run_id: str,
    prefix: str,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    judge: dict[str, Any],
    auditor: dict[str, Any],
    prompt_contract: dict[str, Any] | None = None,
    extra_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write per-case transcript sidecars in JSONL and Markdown formats."""

    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}-{case_id}-{run_id}-transcript"
    jsonl_path = case_dir / f"{base}.jsonl"
    md_path = case_dir / f"{base}.md"

    literal_events = list(extra_events or [])
    events = [
        {
            "event": "case_context",
            "case_id": case_id,
            "run_id": run_id,
            "evidence": evidence,
            "expected_hidden_ground_truth": expected,
            "prompt_contract": prompt_contract,
        },
        {
            "event": "judge",
            "case_id": case_id,
            "run_id": run_id,
            "requested_model": judge.get("requested_model"),
            "actual_model": judge.get("actual_model"),
            "status": judge.get("status"),
            "finish_reason": judge.get("finish_reason"),
            "tokens_used": judge.get("tokens_used"),
            "latency_ms": judge.get("latency_ms"),
            "text": judge.get("text"),
            "json": judge.get("json"),
        },
        {
            "event": "auditor",
            "case_id": case_id,
            "run_id": run_id,
            "requested_model": auditor.get("requested_model"),
            "actual_model": auditor.get("actual_model"),
            "status": auditor.get("status"),
            "finish_reason": auditor.get("finish_reason"),
            "tokens_used": auditor.get("tokens_used"),
            "latency_ms": auditor.get("latency_ms"),
            "text": auditor.get("text"),
            "json": auditor.get("json"),
        },
        *literal_events,
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )

    md = [
        f"# Transcript: {case_id}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Judge requested: `{judge.get('requested_model')}`",
        f"- Judge actual: `{judge.get('actual_model')}`",
        f"- Auditor requested: `{auditor.get('requested_model')}`",
        f"- Auditor actual: `{auditor.get('actual_model')}`",
        "",
        "## Expected / Ground Truth",
        "",
        "```json",
        json.dumps(expected, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Visible Contract",
        "",
        "```json",
        json.dumps(prompt_contract or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Judge Output",
        "",
        "```json",
        _string_or_json(judge.get("json") if judge.get("json") is not None else judge.get("text")),
        "```",
        "",
        "## Auditor Output",
        "",
        "```json",
        _string_or_json(auditor.get("json") if auditor.get("json") is not None else auditor.get("text")),
        "```",
        "",
    ]
    if literal_events:
        md.extend(["## Literal Model Transcript", ""])
        for event in literal_events:
            label = event.get("role") or event.get("event") or "model"
            md.extend([
                f"### {label}",
                "",
                f"- Requested model: `{event.get('requested_model')}`",
                f"- Actual model: `{event.get('actual_model')}`",
                f"- Status: `{event.get('status')}`",
                f"- Text SHA256: `{event.get('text_sha256')}`",
                f"- Text chars: `{event.get('text_chars')}`",
                f"- Error: `{event.get('error')}`",
                "",
                "#### System Prompt",
                "",
                "```text",
                str(event.get("system_prompt") or ""),
                "```",
                "",
                "#### User Prompt",
                "",
                "```text",
                str(event.get("user_prompt") or ""),
                "```",
                "",
                "#### Model Output",
                "",
                "```text",
                str(event.get("text") or ""),
                "```",
                "",
            ])
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "event_count": len(events),
    }


def write_suite_transcript_exports(
    *,
    output_dir: Path,
    run_id: str,
    prefix: str,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write suite-level transcript indexes that point at every case sidecar."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}-{run_id}-transcripts.jsonl"
    md_path = output_dir / f"{prefix}-{run_id}-transcripts.md"

    events = []
    md = [
        f"# Transcript Index: {run_id}",
        "",
        f"- Cases: `{len(artifacts)}`",
        "",
    ]
    for artifact in artifacts:
        exports = artifact.get("transcript_exports") or {}
        event = {
            "event": "case_transcript",
            "run_id": run_id,
            "case_id": artifact.get("case_id"),
            "status": artifact.get("status"),
            "score": (artifact.get("score") or {}).get("score"),
            "artifact_path": artifact.get("artifact_path"),
            "jsonl_path": exports.get("jsonl_path"),
            "md_path": exports.get("md_path"),
        }
        events.append(event)
        md.extend([
            f"## {artifact.get('case_id')}",
            "",
            f"- Status: `{artifact.get('status')}`",
            f"- Score: `{(artifact.get('score') or {}).get('score')}`",
            f"- Artifact: `{artifact.get('artifact_path')}`",
            f"- JSONL transcript: `{exports.get('jsonl_path')}`",
            f"- Markdown transcript: `{exports.get('md_path')}`",
            "",
        ])

    jsonl_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "case_count": len(artifacts),
    }
