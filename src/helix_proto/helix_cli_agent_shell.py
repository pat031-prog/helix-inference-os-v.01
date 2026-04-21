from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL_BLOCK_RE = re.compile(r"(?is)<tool_call>(.*?)</tool_call>")
_FENCED_JSON_RE = re.compile(r"(?is)```(?:json)?\s*([\s\S]*?)\s*```")
_PATCH_BLOCK_RE = re.compile(r"(?is)```(?:diff|patch)\s*(.*?)```")


def _coerce_tool_call(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    name = payload.get("tool") or payload.get("name")
    if not name:
        return None
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("args")
    if arguments is None:
        arguments = payload.get("input")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {"tool": str(name), "arguments": arguments, "id": payload.get("id")}


def _json_candidates(text: str) -> list[str]:
    raw = str(text or "")
    candidates = list(_TOOL_CALL_BLOCK_RE.findall(raw))
    for candidate in _FENCED_JSON_RE.findall(raw):
        stripped = candidate.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            candidates.append(stripped)
    return candidates


def parse_agent_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for candidate in _json_candidates(str(text or "")):
        try:
            parsed = json.loads(candidate.strip())
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            for item in parsed["tool_calls"]:
                call = _coerce_tool_call(item)
                if call:
                    calls.append(call)
            continue
        if isinstance(parsed, list):
            for item in parsed:
                call = _coerce_tool_call(item)
                if call:
                    calls.append(call)
            continue
        call = _coerce_tool_call(parsed)
        if call:
            calls.append(call)
    return calls


def extract_patch(text: str) -> str | None:
    raw = str(text or "")
    matches = _PATCH_BLOCK_RE.findall(raw)
    if matches:
        return matches[-1].strip()
    marker = raw.find("diff --git ")
    if marker >= 0:
        return raw[marker:].strip()
    return None


def normalize_tool_event(event: Any) -> dict[str, Any]:
    event_payload = event if isinstance(event, dict) else {}
    payload = event_payload.get("result")
    payload_dict = payload if isinstance(payload, dict) else {}
    nested_result = payload_dict.get("result")
    if isinstance(nested_result, dict):
        result_payload = nested_result
    elif payload_dict:
        result_payload = payload_dict
    else:
        result_payload = {}

    arguments = event_payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = payload_dict.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    tool_name = event_payload.get("tool") or payload_dict.get("tool") or "tool"
    return {
        "tool": str(tool_name),
        "arguments": arguments,
        "result": result_payload,
    }


def tool_event_detail(event: Any) -> str:
    normalized = normalize_tool_event(event)
    result_payload = normalized["result"]
    if not isinstance(result_payload, dict):
        return ""
    if result_payload.get("path"):
        return str(result_payload.get("path"))
    if result_payload.get("query"):
        return str(result_payload.get("query"))
    if result_payload.get("command"):
        return " ".join(str(item) for item in result_payload.get("command") or [])
    if result_payload.get("record_count") is not None:
        return f"{result_payload.get('record_count')} records"
    return ""
