"""
Export readable transcripts from emergent behavior observatory artifacts.

The transcript contains visible model outputs, parsed JSON when available,
memory IDs, node hashes, finish reasons, token/latency metadata, analyst output,
and auditor output. It does not reconstruct hidden chain-of-thought. If a
provider reported a reasoning side channel that the runner did not store, the
transcript records only the omitted character count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_TURN_JSON_FIELDS = (
    "turn_label",
    "field_note",
    "memory_use",
    "response_to_previous",
    "noteworthy_observed_pattern",
    "surprise_or_tension",
    "next_prompt_to_next_model",
)


def _json_ready(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated {len(text) - max_chars} chars]"


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _turn_output_classification(turn: dict[str, Any]) -> dict[str, Any]:
    existing = turn.get("output_classification")
    if isinstance(existing, dict) and existing:
        return existing
    call = turn.get("call") or {}
    output = turn.get("output") or {}
    text = str(output.get("text") or "")
    parsed = output.get("json")
    finish_reason = call.get("finish_reason")
    finish_is_length = (finish_reason or "") in {"length", "max_tokens"}
    is_dict = isinstance(parsed, dict)
    missing_fields = [field for field in REQUIRED_TURN_JSON_FIELDS if not (is_dict and field in parsed)]
    memory_use = parsed.get("memory_use") if is_dict else None
    if not isinstance(memory_use, dict) and is_dict and {
        "cited_memory_ids",
        "used_parent_chain_or_signature",
    }.issubset(set(parsed.keys())):
        memory_use = parsed
    memory_use_schema_complete = isinstance(memory_use, dict) and isinstance(memory_use.get("cited_memory_ids"), list)
    top_level_schema_complete = is_dict and not missing_fields
    if call.get("status") != "ok":
        output_class = "call_error"
    elif not text.strip():
        output_class = "empty_output"
    elif not is_dict:
        output_class = "unparseable_truncated" if finish_is_length else "unparseable"
    elif top_level_schema_complete and finish_is_length:
        output_class = "schema_complete_length_finish"
    elif top_level_schema_complete:
        output_class = "schema_complete"
    elif memory_use_schema_complete and finish_is_length:
        output_class = "json_fragment_from_truncation"
    elif memory_use_schema_complete:
        output_class = "json_fragment"
    else:
        output_class = "schema_deviant_json"
    return {
        "output_class": output_class,
        "call_status": call.get("status"),
        "finish_reason": finish_reason,
        "finish_is_length": finish_is_length,
        "json_parseable": is_dict,
        "top_level_schema_complete": top_level_schema_complete,
        "missing_top_level_fields": missing_fields,
        "memory_use_schema_complete": memory_use_schema_complete,
        "visible_output_chars": len(text),
        "provider_reasoning_side_channel_omitted": int(call.get("omitted_reasoning_chars") or 0) > 0,
        "omitted_reasoning_chars": int(call.get("omitted_reasoning_chars") or 0),
    }


def _derive_output_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    turns = artifact.get("turns") or []
    by_model: dict[str, dict[str, Any]] = {}
    output_classes: dict[str, int] = {}
    finish_reasons: dict[str, int] = {}
    for turn in turns:
        classification = _turn_output_classification(turn)
        model = str(turn.get("model") or "unknown")
        output_class = str(classification.get("output_class") or "unknown")
        finish_reason = str((turn.get("call") or {}).get("finish_reason") or "none")
        stats = by_model.setdefault(
            model,
            {
                "turn_count": 0,
                "ok_call_count": 0,
                "schema_complete_count": 0,
                "json_parseable_count": 0,
                "length_finish_count": 0,
                "visible_output_chars": 0,
                "finish_reasons": {},
                "output_classes": {},
            },
        )
        stats["turn_count"] += 1
        if (turn.get("call") or {}).get("status") == "ok":
            stats["ok_call_count"] += 1
        if classification.get("top_level_schema_complete"):
            stats["schema_complete_count"] += 1
        if classification.get("json_parseable"):
            stats["json_parseable_count"] += 1
        if classification.get("finish_is_length"):
            stats["length_finish_count"] += 1
        stats["visible_output_chars"] += int(classification.get("visible_output_chars") or 0)
        stats["finish_reasons"][finish_reason] = stats["finish_reasons"].get(finish_reason, 0) + 1
        stats["output_classes"][output_class] = stats["output_classes"].get(output_class, 0) + 1
        output_classes[output_class] = output_classes.get(output_class, 0) + 1
        finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
    return {
        "turn_count": len(turns),
        "schema_complete_count": sum(1 for turn in turns if _turn_output_classification(turn).get("top_level_schema_complete")),
        "json_parseable_count": sum(1 for turn in turns if _turn_output_classification(turn).get("json_parseable")),
        "length_finish_count": sum(1 for turn in turns if _turn_output_classification(turn).get("finish_is_length")),
        "output_classes": dict(sorted(output_classes.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "by_model": dict(sorted(by_model.items())),
    }


def _turn_record(turn: dict[str, Any], *, max_output_chars: int = 0) -> dict[str, Any]:
    call = dict(turn.get("call") or {})
    output = dict(turn.get("output") or {})
    text = str(output.get("text") or "")
    classification = _turn_output_classification(turn)
    return {
        "kind": "turn",
        "turn_id": turn.get("turn_id"),
        "round": turn.get("round"),
        "model": turn.get("model"),
        "memory_id": (turn.get("memory") or {}).get("memory_id"),
        "node_hash": (turn.get("memory") or {}).get("node_hash"),
        "parent_hash": (turn.get("memory") or {}).get("parent_hash"),
        "strict_context_memory_ids": turn.get("strict_context_memory_ids") or [],
        "call": {
            "status": call.get("status"),
            "requested_model": call.get("requested_model"),
            "actual_model": call.get("actual_model"),
            "finish_reason": call.get("finish_reason"),
            "tokens_used": call.get("tokens_used"),
            "latency_ms": call.get("latency_ms"),
            "retry_count": call.get("retry_count"),
            "omitted_reasoning_chars": call.get("omitted_reasoning_chars", 0),
            "raw_message_keys": call.get("raw_message_keys") or [],
            "error": call.get("error"),
        },
        "json_parseable": output.get("json") is not None,
        "output_classification": classification,
        "parsed_json": output.get("json"),
        "visible_output": _clip(text, max_output_chars),
        "visible_output_chars": len(text),
        "reasoning_note": (
            "Visible output is preserved as returned. Hidden reasoning is not reconstructed. "
            "omitted_reasoning_chars records provider side-channel content that was not stored."
        ),
    }


def build_transcript_records(artifact: dict[str, Any], *, max_output_chars: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "kind": "run",
            "artifact": artifact.get("artifact"),
            "run_id": artifact.get("run_id"),
            "status": artifact.get("status"),
            "score": (artifact.get("observatory_score") or {}).get("score"),
            "behavior_count": (artifact.get("observatory_score") or {}).get("behavior_count"),
            "models": (artifact.get("models") or {}).get("round_robin") or [],
            "analyst_actual": (artifact.get("models") or {}).get("analyst_actual"),
            "auditor_actual": (artifact.get("models") or {}).get("auditor_actual"),
            "claim_boundary": artifact.get("claim_boundary"),
        },
        {
            "kind": "score_gates",
            "gates": (artifact.get("observatory_score") or {}).get("gates") or {},
        },
        {
            "kind": "evidence_id_registry",
            "registry": artifact.get("evidence_id_registry") or {},
        },
    ]
    for key in ("root_memory", "method_memory", "signed_poison_lure", "unsigned_lure"):
        value = artifact.get(key)
        if isinstance(value, dict):
            records.append(
                {
                    "kind": "memory_seed",
                    "role": key,
                    "memory_id": value.get("memory_id"),
                    "summary": value.get("summary"),
                    "node_hash": value.get("node_hash"),
                    "parent_hash": value.get("parent_hash"),
                    "signature_verified": value.get("signature_verified"),
                    "content": value.get("content"),
                }
            )
    for turn in artifact.get("turns") or []:
        records.append(_turn_record(turn, max_output_chars=max_output_chars))
    records.append(
        {
            "kind": "analyst",
            "call": artifact.get("analyst_call") or {},
            "parsed_json": (artifact.get("analyst_output") or {}).get("json"),
            "visible_output": _clip(str((artifact.get("analyst_output") or {}).get("text") or ""), max_output_chars),
        }
    )
    records.append(
        {
            "kind": "auditor",
            "call": artifact.get("auditor_call") or {},
            "parsed_json": (artifact.get("auditor_output") or {}).get("json"),
            "visible_output": _clip(str((artifact.get("auditor_output") or {}).get("text") or ""), max_output_chars),
        }
    )
    return records


def render_markdown_transcript(artifact: dict[str, Any], *, max_output_chars: int = 0) -> str:
    score = artifact.get("observatory_score") or {}
    gates = score.get("gates") or {}
    models = artifact.get("models") or {}
    lines = [
        f"# Emergent Behavior Observatory Transcript: {artifact.get('run_id')}",
        "",
        "## Run Summary",
        "",
        f"- Status: `{artifact.get('status')}`",
        f"- Score: `{score.get('score')}`",
        f"- Behavior count: `{score.get('behavior_count')}`",
        f"- Analyst actual: `{models.get('analyst_actual')}`",
        f"- Auditor actual: `{models.get('auditor_actual')}`",
        f"- Artifact SHA256: `{artifact.get('artifact_sha256')}`",
        "",
        "## Claim Boundary",
        "",
        str(artifact.get("claim_boundary") or ""),
        "",
        "## Models",
        "",
    ]
    for model in models.get("round_robin") or []:
        lines.append(f"- `{model}`")
    lines.extend(["", "## Score Gates", ""])
    for key, value in gates.items():
        lines.append(f"- `{key}`: `{value}`")

    registry = artifact.get("evidence_id_registry") or {}
    if registry:
        lines.extend(["", "## Evidence ID Registry", "", "````json", _json_ready(registry), "````", ""])

    lines.extend(["", "## Seed Memories", ""])
    for key in ("root_memory", "method_memory", "signed_poison_lure", "unsigned_lure"):
        value = artifact.get(key)
        if not isinstance(value, dict):
            continue
        lines.extend(
            [
                f"### {key}",
                "",
                f"- Memory ID: `{value.get('memory_id')}`",
                f"- Signature verified: `{value.get('signature_verified')}`",
                f"- Node hash: `{value.get('node_hash')}`",
                "",
                "````text",
                str(value.get("content") or ""),
                "````",
                "",
            ]
        )

    lines.extend(["## Turn Transcript", ""])
    for turn in artifact.get("turns") or []:
        call = turn.get("call") or {}
        output = turn.get("output") or {}
        memory = turn.get("memory") or {}
        text = str(output.get("text") or "")
        classification = _turn_output_classification(turn)
        lines.extend(
            [
                f"### {turn.get('turn_id')} - {turn.get('model')}",
                "",
                f"- Round: `{turn.get('round')}`",
                f"- Memory ID: `{memory.get('memory_id')}`",
                f"- Node hash: `{memory.get('node_hash')}`",
                f"- Parent hash: `{memory.get('parent_hash')}`",
                f"- Status: `{call.get('status')}`",
                f"- Finish reason: `{call.get('finish_reason')}`",
                f"- Output class: `{classification.get('output_class')}`",
                f"- Top-level schema complete: `{classification.get('top_level_schema_complete')}`",
                f"- Tokens used: `{call.get('tokens_used')}`",
                f"- Latency ms: `{call.get('latency_ms')}`",
                f"- JSON parseable: `{output.get('json') is not None}`",
                f"- Omitted reasoning chars: `{call.get('omitted_reasoning_chars', 0)}`",
                "",
                "Strict context memory IDs:",
                "",
            ]
        )
        for memory_id in turn.get("strict_context_memory_ids") or []:
            lines.append(f"- `{memory_id}`")
        lines.extend(["", "Parsed JSON:", "", "````json", _json_ready(output.get("json")), "````", ""])
        lines.extend(["Visible output:", "", "````text", _clip(text, max_output_chars), "````", ""])

    lines.extend(["## Analyst Output", "", "````json", _json_ready((artifact.get("analyst_output") or {}).get("json")), "````", ""])
    analyst_text = str((artifact.get("analyst_output") or {}).get("text") or "")
    lines.extend(["Analyst visible output:", "", "````text", _clip(analyst_text, max_output_chars), "````", ""])
    lines.extend(["## Auditor Output", "", "````json", _json_ready((artifact.get("auditor_output") or {}).get("json")), "````", ""])
    auditor_text = str((artifact.get("auditor_output") or {}).get("text") or "")
    lines.extend(["Auditor visible output:", "", "````text", _clip(auditor_text, max_output_chars), "````", ""])
    lines.extend(
        [
            "## Reasoning Boundary",
            "",
            (
                "This transcript preserves visible model text and parsed JSON. It does not "
                "reconstruct hidden chain-of-thought. If provider metadata reported omitted "
                "reasoning side-channel characters, only the count is shown."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_extract_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    score = artifact.get("observatory_score") or {}
    gates = score.get("gates") or {}
    analyst_json = (artifact.get("analyst_output") or {}).get("json") or {}
    behaviors = analyst_json.get("noteworthy_behaviors") if isinstance(analyst_json, dict) else []
    if not isinstance(behaviors, list):
        behaviors = []
    diagnostics = artifact.get("output_diagnostics") or _derive_output_diagnostics(artifact)
    return {
        "run_id": artifact.get("run_id"),
        "status": artifact.get("status"),
        "score": score.get("score"),
        "behavior_count": score.get("behavior_count"),
        "failing_gates": {key: value for key, value in gates.items() if not value},
        "models": artifact.get("models") or {},
        "parameters": artifact.get("parameters") or {},
        "output_diagnostics": diagnostics,
        "behaviors": behaviors,
        "negative_findings": analyst_json.get("negative_findings") if isinstance(analyst_json, dict) else [],
        "method_caveats": analyst_json.get("method_caveats") if isinstance(analyst_json, dict) else [],
        "auditor": {
            "call": artifact.get("auditor_call") or {},
            "json": (artifact.get("auditor_output") or {}).get("json"),
        },
        "transcript_artifacts": artifact.get("transcript_artifacts") or {},
        "claim_boundary": artifact.get("claim_boundary"),
    }


def render_extract_brief_markdown(artifact: dict[str, Any]) -> str:
    summary = build_extract_summary(artifact)
    failing_gates = summary["failing_gates"]
    diagnostics = summary.get("output_diagnostics") or {}
    models = summary.get("models") or {}
    parameters = summary.get("parameters") or {}
    lines = [
        f"# Emergent Behavior Observatory Extract: {summary.get('run_id')}",
        "",
        "## Verdict",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Score: `{summary.get('score')}`",
        f"- Behavior count: `{summary.get('behavior_count')}`",
        f"- Analyst actual: `{models.get('analyst_actual')}`",
        f"- Auditor actual: `{models.get('auditor_actual')}`",
        f"- Rounds: `{parameters.get('rounds')}`",
        f"- Context limit: `{parameters.get('context_limit')}`",
        "",
        "## Failing Gates",
        "",
    ]
    if failing_gates:
        for key, value in failing_gates.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Output Health",
            "",
            f"- Turns: `{diagnostics.get('turn_count')}`",
            f"- Top-level schema complete: `{diagnostics.get('schema_complete_count')}`",
            f"- JSON parseable: `{diagnostics.get('json_parseable_count')}`",
            f"- Length finishes: `{diagnostics.get('length_finish_count')}`",
            f"- Output classes: `{json.dumps(diagnostics.get('output_classes') or {}, sort_keys=True)}`",
            f"- Finish reasons: `{json.dumps(diagnostics.get('finish_reasons') or {}, sort_keys=True)}`",
            "",
            "## Model Matrix",
            "",
            "| Model | Turns | OK | Schema | Length | Classes |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for model, stats in (diagnostics.get("by_model") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(model),
                    _md_cell(stats.get("turn_count")),
                    _md_cell(stats.get("ok_call_count")),
                    _md_cell(stats.get("schema_complete_count")),
                    _md_cell(stats.get("length_finish_count")),
                    _md_cell(json.dumps(stats.get("output_classes") or {}, sort_keys=True)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Behaviors", ""])
    behaviors = summary.get("behaviors") or []
    if not behaviors:
        lines.append("- No analyst behaviors parsed.")
    for index, behavior in enumerate(behaviors, start=1):
        if not isinstance(behavior, dict):
            continue
        lines.extend(
            [
                f"### {index}. {behavior.get('label')}",
                "",
                f"- Type: `{behavior.get('behavior_type')}`",
                f"- Claim strength: `{behavior.get('claim_strength')}`",
                f"- Evidence turns: `{', '.join(str(item) for item in behavior.get('evidence_turns') or [])}`",
                f"- Evidence memories: `{', '.join(str(item) for item in behavior.get('evidence_memory_ids') or [])}`",
                f"- Short quote: {behavior.get('short_quote')}",
                f"- Why noteworthy: {behavior.get('why_noteworthy')}",
                "",
            ]
        )

    lines.extend(["## Negative Findings", ""])
    negative_findings = summary.get("negative_findings") or []
    if negative_findings:
        for item in negative_findings:
            lines.append(f"- {item}")
    else:
        lines.append("- None parsed.")

    lines.extend(["", "## Method Caveats", ""])
    method_caveats = summary.get("method_caveats") or []
    if method_caveats:
        for item in method_caveats:
            lines.append(f"- {item}")
    else:
        lines.append("- None parsed.")

    lines.extend(["", "## Auditor", "", "````json", _json_ready(summary.get("auditor")), "````", ""])
    artifacts = summary.get("transcript_artifacts") or {}
    if artifacts:
        lines.extend(["## Files", ""])
        for key, value in artifacts.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Reasoning Boundary",
            "",
            (
                "This extract summarizes visible model outputs, parsed JSON, finish reasons, "
                "and provider metadata. It does not reconstruct hidden chain-of-thought."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def export_transcript_from_artifact(
    artifact: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    max_output_chars: int = 0,
) -> dict[str, Any]:
    run_id = str(artifact.get("run_id") or "unknown-run")
    target_dir = Path(output_dir) if output_dir is not None else Path(str(artifact.get("output_scope") or "."))
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = target_dir / f"local-emergent-behavior-observatory-{run_id}-transcript.md"
    jsonl_path = target_dir / f"local-emergent-behavior-observatory-{run_id}-transcript.jsonl"
    extract_md_path = target_dir / f"local-emergent-behavior-observatory-{run_id}-extract.md"
    extract_json_path = target_dir / f"local-emergent-behavior-observatory-{run_id}-extract.json"
    markdown = render_markdown_transcript(artifact, max_output_chars=max_output_chars)
    records = build_transcript_records(artifact, max_output_chars=max_output_chars)
    transcript_artifacts = {
        "markdown_path": str(md_path),
        "jsonl_path": str(jsonl_path),
        "extract_markdown_path": str(extract_md_path),
        "extract_json_path": str(extract_json_path),
        "record_count": len(records),
        "max_output_chars": max_output_chars,
    }
    artifact_for_extract = dict(artifact)
    artifact_for_extract["transcript_artifacts"] = transcript_artifacts
    extract_summary = build_extract_summary(artifact_for_extract)
    extract_markdown = render_extract_brief_markdown(artifact_for_extract)
    md_path.write_text(markdown, encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    extract_md_path.write_text(extract_markdown, encoding="utf-8")
    extract_json_path.write_text(_json_ready(extract_summary), encoding="utf-8")
    return transcript_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an emergent observatory transcript")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-output-chars", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_path = Path(args.artifact)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir is not None else artifact_path.parent
    result = export_transcript_from_artifact(
        artifact,
        output_dir=output_dir,
        max_output_chars=args.max_output_chars,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
