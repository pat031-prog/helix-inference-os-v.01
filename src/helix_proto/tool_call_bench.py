from __future__ import annotations

import gc
import json
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix_proto.cdna import generate_text_with_target, load_generation_target
from helix_proto.hf import _process_rss_mb
from helix_proto.model_bench import _cache_size_bytes
from helix_proto.text import render_messages_prompt


PRESET_TOOLSETS: dict[str, list[dict[str, Any]]] = {
    "workspace_basic": [
        {
            "name": "workspace.list_models",
            "description": "List prepared model workspaces and sessions.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "workspace.model_info",
            "description": "Inspect one prepared model alias.",
            "input_schema": {
                "type": "object",
                "properties": {"alias": {"type": "string"}},
                "required": ["alias"],
            },
        },
        {
            "name": "rag.search",
            "description": "Search the persistent knowledge base.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
        },
        {
            "name": "memory.search",
            "description": "Search episodic memory from previous runs.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
        },
        {
            "name": "gpt.generate_text",
            "description": "Generate text with a prepared local model alias.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alias": {"type": "string"},
                    "prompt": {"type": "string"},
                    "max_new_tokens": {"type": "integer", "default": 96},
                },
                "required": ["alias", "prompt"],
            },
        },
    ],
    "ops_readonly": [
        {
            "name": "ops.read_syslog",
            "description": "Read recent syslog lines, optionally filtered by a string.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "grep": {"type": "string"},
                    "lines": {"type": "integer", "default": 200},
                },
            },
        },
        {
            "name": "ops.read_journald",
            "description": "Read journald logs for one systemd service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "lines": {"type": "integer", "default": 200},
                },
                "required": ["service"],
            },
        },
        {
            "name": "ops.read_process_list",
            "description": "List current processes sorted by cpu or memory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sort_by": {"type": "string", "enum": ["cpu", "memory"]},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        },
        {
            "name": "ops.read_disk_usage",
            "description": "Read disk usage for the server or one specific path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
            },
        },
        {
            "name": "ops.read_network_stats",
            "description": "Inspect active connections and network counters.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ],
    "agent_mixed": [],
}
PRESET_TOOLSETS["agent_mixed"] = PRESET_TOOLSETS["workspace_basic"] + PRESET_TOOLSETS["ops_readonly"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tool_call_suite_path() -> Path:
    return repo_root() / "benchmarks" / "tool_call_cases.json"


def _strip_think_blocks(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _tool_manifest_to_llama_tools(tool_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": str(item["name"]),
                "description": str(item.get("description", "")),
                "parameters": dict(item.get("input_schema", {"type": "object", "properties": {}})),
            },
        }
        for item in tool_manifest
    ]


def _extract_tool_call_tag_payload(text: str) -> dict[str, Any] | None:
    cleaned = _strip_think_blocks(text)

    json_tag = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", cleaned, flags=re.DOTALL)
    if json_tag:
        try:
            payload = json.loads(json_tag.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("tool_name")
            arguments = payload.get("arguments", {})
            if isinstance(name, str) and isinstance(arguments, dict):
                return {
                    "kind": "tool",
                    "thought": "",
                    "tool_name": name,
                    "arguments": arguments,
                }

    xml_tag = re.search(
        r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        cleaned,
        flags=re.DOTALL,
    )
    if not xml_tag:
        return None
    tool_name = xml_tag.group(1).strip()
    body = xml_tag.group(2)
    arguments: dict[str, Any] = {}
    for match in re.finditer(
        r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
        body,
        flags=re.DOTALL,
    ):
        key = match.group(1).strip()
        raw_value = match.group(2).strip()
        if not key:
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        arguments[key] = value
    return {
        "kind": "tool",
        "thought": "",
        "tool_name": tool_name,
        "arguments": arguments,
    }


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    def _as_object(value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    text = _strip_think_blocks(text)
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            return _as_object(json.loads(text[start:]))
        except json.JSONDecodeError:
            pass

        stack = [char]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            token = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif token == "\\":
                    escaped = True
                elif token == '"':
                    in_string = False
                continue
            if token == '"':
                in_string = True
                continue
            if token in "{[":
                stack.append(token)
                continue
            if token in "}]":
                if not stack:
                    break
                opener = stack[-1]
                if (opener == "{" and token == "}") or (opener == "[" and token == "]"):
                    stack.pop()
                else:
                    break
                if not stack:
                    candidate = text[start : index + 1]
                    try:
                        return _as_object(json.loads(candidate))
                    except json.JSONDecodeError:
                        break
    return None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        left = _normalize_value(expected)
        right = _normalize_value(actual)
        return left == right or left in right or right in left
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except Exception:  # noqa: BLE001
            return False
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except Exception:  # noqa: BLE001
            return False
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _value_matches(left, right) for left, right in zip(expected, actual)
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _value_matches(val, actual[key]) for key, val in expected.items()
        )
    return expected == actual


@dataclass(slots=True)
class ToolCallStep:
    goal: str
    expected_kind: str
    expected_tool_name: str | None
    expected_arguments: dict[str, Any]
    expected_final_contains: list[str]
    observations: list[dict[str, Any]]
    default_model_alias: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallStep":
        return cls(
            goal=str(data["goal"]),
            expected_kind=str(data["expected"]["kind"]),
            expected_tool_name=data["expected"].get("tool_name"),
            expected_arguments=dict(data["expected"].get("arguments", {}) or {}),
            expected_final_contains=[str(item) for item in data["expected"].get("final_contains", [])],
            observations=list(data.get("observations", [])),
            default_model_alias=data.get("default_model_alias"),
        )


@dataclass(slots=True)
class ToolCallCase:
    id: str
    domain: str
    toolset: str
    steps: list[ToolCallStep]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallCase":
        return cls(
            id=str(data["id"]),
            domain=str(data["domain"]),
            toolset=str(data["toolset"]),
            steps=[ToolCallStep.from_dict(item) for item in data["steps"]],
        )


def load_tool_call_cases(path: str | Path | None = None) -> list[ToolCallCase]:
    source = Path(path) if path is not None else default_tool_call_suite_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    return [ToolCallCase.from_dict(item) for item in data["cases"]]


def _render_case_prompt(case: ToolCallCase, step: ToolCallStep, *, tool_manifest: list[dict[str, Any]]) -> str:
    observations = json.dumps(step.observations, indent=2, ensure_ascii=False)
    tools = json.dumps(tool_manifest, indent=2, ensure_ascii=False)
    return (
        "Sos HelixPlanner. Debes decidir exactamente el siguiente paso del agente.\n"
        "Respondé solo con un objeto JSON.\n"
        'Formato válido para tool:\n'
        '{"kind":"tool","thought":"...","tool_name":"...","arguments":{}}\n'
        'Formato válido para final:\n'
        '{"kind":"final","thought":"...","final":"..."}\n'
        "Reglas:\n"
        "- Elegí una sola acción.\n"
        "- Si falta contexto, usá una tool.\n"
        "- Si ya alcanza para responder, usá kind=final.\n"
        "- No agregues texto fuera del JSON.\n"
        "- No muestres thinking ni reasoning. /no_think\n"
        f"- Alias por defecto disponible: {step.default_model_alias!r}\n\n"
        f"Caso: {case.id}\n"
        f"Dominio: {case.domain}\n\n"
        f"Goal:\n{step.goal}\n\n"
        f"Observaciones previas:\n{observations}\n\n"
        f"Tools disponibles:\n{tools}\n\n"
        "/no_think\n"
    )


def _build_generation_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Sos un planner estricto de Helix. Respondé únicamente JSON válido. /no_think",
        },
        {"role": "user", "content": f"{prompt}\n\n/no_think"},
    ]
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:  # noqa: BLE001
            pass
    return render_messages_prompt(messages, assistant_prefix=True)


def _score_decision(step: ToolCallStep, parsed: dict[str, Any] | None) -> dict[str, Any]:
    if parsed is None:
        return {
            "json_valid": False,
            "kind_match": False,
            "tool_match": False,
            "arguments_match": False,
            "final_match": False,
            "step_success": False,
        }

    kind = str(parsed.get("kind", "")).strip().lower()
    kind_match = kind == step.expected_kind

    tool_match = True
    arguments_match = True
    final_match = True

    if step.expected_kind == "tool":
        tool_match = str(parsed.get("tool_name")) == str(step.expected_tool_name)
        actual_arguments = dict(parsed.get("arguments", {}) or {})
        arguments_match = all(
            key in actual_arguments and _value_matches(expected, actual_arguments[key])
            for key, expected in step.expected_arguments.items()
        )
        final_match = False
    else:
        final_text = str(parsed.get("final", "") or "")
        final_match = all(token.lower() in final_text.lower() for token in step.expected_final_contains)
        tool_match = False
        arguments_match = False

    step_success = kind_match and (tool_match and arguments_match if step.expected_kind == "tool" else final_match)
    return {
        "json_valid": True,
        "kind_match": kind_match,
        "tool_match": tool_match,
        "arguments_match": arguments_match,
        "final_match": final_match,
        "step_success": step_success,
    }


def benchmark_tool_calling_one_model(
    model_ref: str,
    *,
    cases: list[ToolCallCase],
    workspace_root: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    max_new_tokens: int = 192,
    max_input_tokens: int = 2048,
    torch_dtype: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("tool-calling benchmark requires transformers and torch") from exc

    before_load_rss = _process_rss_mb()
    load_started = time.perf_counter()
    target = load_generation_target(
        model_ref,
        workspace_root=workspace_root,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    tokenizer = target.tokenizer
    model = target.model

    if target.backend == "transformers" and tokenizer is not None:
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token_id is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                model.resize_token_embeddings(len(tokenizer))

    load_time_s = time.perf_counter() - load_started
    after_load_rss = _process_rss_mb()
    peak_rss_mb = max(before_load_rss, after_load_rss)

    case_results: list[dict[str, Any]] = []
    total_steps = 0
    total_step_success = 0
    total_json_valid = 0
    total_generation_time_s = 0.0
    total_generated_tokens = 0

    context = torch.inference_mode() if target.backend == "transformers" else nullcontext()
    with context:
        for case in cases:
            tool_manifest = PRESET_TOOLSETS[case.toolset]
            llama_tools = _tool_manifest_to_llama_tools(tool_manifest)
            step_results: list[dict[str, Any]] = []
            case_success = True
            for step_index, step in enumerate(case.steps):
                prompt = _render_case_prompt(case, step, tool_manifest=tool_manifest)
                generation_started = time.perf_counter()
                if target.backend == "llama-cpp-python":
                    raw_response = target.model.create_chat_completion(
                        messages=[
                            {
                                "role": "system",
                            "content": (
                                "Sos un planner estricto de Helix. "
                                "Si necesitás usar una tool, usá tool calling nativo. "
                                "Si ya podés responder directamente sin usar tools, respondé directo. "
                                "Solo usá tools cuando necesites información o ejecutar acciones que no podés hacer solo. "
                                "Si ya alcanza para responder, devolvé solo un objeto JSON válido. /no_think"
                            ),
                        },
                        {"role": "user", "content": f"{prompt}\n\n/no_think"},
                    ],
                    tools=llama_tools,
                    tool_choice="auto",
                    max_tokens=max(max_new_tokens, 512),
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    )
                    completion = str(
                        dict(raw_response["choices"][0].get("message", {}) or {}).get("content", "") or ""
                    ).strip()
                    generated_tokens = int(raw_response.get("usage", {}).get("completion_tokens", 0) or 0)
                else:
                    generated = generate_text_with_target(
                        target,
                        messages=[
                            {
                                "role": "system",
                                "content": "Sos un planner estricto de Helix. Respondé únicamente JSON válido. /no_think",
                            },
                            {"role": "user", "content": f"{prompt}\n\n/no_think"},
                        ],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        max_input_tokens=max_input_tokens,
                    )
                    raw_response = generated.get("raw_response")
                    completion = str(generated["text"]).strip()
                    generated_tokens = int(generated["generated_tokens"])
                generation_time_s = time.perf_counter() - generation_started
                parsed = _extract_tool_call_tag_payload(completion) or _extract_first_json_object(completion)
                score = _score_decision(step, parsed)

                total_steps += 1
                total_generation_time_s += generation_time_s
                total_generated_tokens += int(generated_tokens)
                total_step_success += int(bool(score["step_success"]))
                total_json_valid += int(bool(score["json_valid"]))
                case_success = case_success and bool(score["step_success"])
                peak_rss_mb = max(peak_rss_mb, _process_rss_mb())

                step_results.append(
                    {
                        "step_index": step_index,
                        "goal": step.goal,
                        "raw_completion": completion,
                        "parsed": parsed,
                        "generated_tokens": int(generated_tokens),
                        "generation_time_s": round(generation_time_s, 4),
                        "tokens_per_second": round(int(generated_tokens) / generation_time_s, 4)
                        if generation_time_s
                        else 0.0,
                        "score": score,
                        "expected_kind": step.expected_kind,
                        "expected_tool_name": step.expected_tool_name,
                        "expected_arguments": step.expected_arguments,
                        "expected_final_contains": step.expected_final_contains,
                    }
                )

            case_results.append(
                {
                    "id": case.id,
                    "domain": case.domain,
                    "toolset": case.toolset,
                    "case_success": case_success,
                    "steps": step_results,
                }
            )

    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - GPU envs vary
        torch.cuda.empty_cache()

    case_successes = sum(1 for item in case_results if item["case_success"])
    return {
        "model_ref": target.model_ref,
        "requested_ref": model_ref,
        "model_label": target.model_label,
        "status": "ok",
        "load_time_s": round(load_time_s, 4),
        "rss_before_load_mb": round(before_load_rss, 2),
        "rss_after_load_mb": round(after_load_rss, 2),
        "rss_peak_mb": round(peak_rss_mb, 2),
        "cache_size_bytes": target.storage_bytes if target.storage_bytes is not None else _cache_size_bytes(target.model_ref),
        "load_mode": target.load_mode,
        "total_steps": total_steps,
        "step_success_rate": round(total_step_success / total_steps, 4) if total_steps else 0.0,
        "json_valid_rate": round(total_json_valid / total_steps, 4) if total_steps else 0.0,
        "case_success_rate": round(case_successes / len(case_results), 4) if case_results else 0.0,
        "total_generation_time_s": round(total_generation_time_s, 4),
        "total_generated_tokens": total_generated_tokens,
        "tokens_per_second": round(total_generated_tokens / total_generation_time_s, 4)
        if total_generation_time_s
        else 0.0,
        "cases": case_results,
    }


def benchmark_tool_calling(
    model_refs: list[str],
    *,
    case_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    max_new_tokens: int = 192,
    max_input_tokens: int = 2048,
    torch_dtype: str = "auto",
    limit_cases: int | None = None,
) -> dict[str, Any]:
    cases = load_tool_call_cases(case_path)
    if limit_cases is not None:
        cases = cases[:limit_cases]
    results: list[dict[str, Any]] = []
    for model_ref in model_refs:
        started = time.perf_counter()
        try:
            result = benchmark_tool_calling_one_model(
                model_ref,
                cases=cases,
                workspace_root=workspace_root,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "model_ref": model_ref,
                "status": "error",
                "error": str(exc),
            }
        result["wall_time_s"] = round(time.perf_counter() - started, 4)
        results.append(result)

    recommendations = recommend_tool_call_models(results)
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_suite_path": str(Path(case_path) if case_path else default_tool_call_suite_path()),
        "case_count": len(cases),
        "local_files_only": local_files_only,
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "models": results,
        "recommendation": recommendations,
    }


def recommend_tool_call_models(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in results if item.get("status") == "ok"]
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda item: (
            item.get("case_success_rate", 0.0),
            item.get("step_success_rate", 0.0),
            item.get("json_valid_rate", 0.0),
            item.get("tokens_per_second", 0.0),
        ),
        reverse=True,
    )[0]
    return {
        "model_ref": best["model_ref"],
        "case_success_rate": best["case_success_rate"],
        "step_success_rate": best["step_success_rate"],
        "json_valid_rate": best["json_valid_rate"],
        "tokens_per_second": best["tokens_per_second"],
    }


def render_tool_call_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tool Calling Benchmark",
        "",
        f"- Generated at: `{report['generated_at_utc']}`",
        f"- Case suite: `{report['case_suite_path']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Local files only: `{report['local_files_only']}`",
        "",
        "## Recommendation",
    ]
    recommendation = report.get("recommendation")
    if recommendation is None:
        lines.append("- No successful model run.")
    else:
        lines.append(
            f"- `{recommendation['model_ref']}` | case_success={recommendation['case_success_rate']:.4f} | "
            f"step_success={recommendation['step_success_rate']:.4f} | json_valid={recommendation['json_valid_rate']:.4f} | "
            f"tok/s={recommendation['tokens_per_second']:.2f}"
        )
    lines.extend(["", "## Models"])
    for model in report.get("models", []):
        if model.get("status") != "ok":
            lines.append(f"- `{model['model_ref']}`: error `{model.get('error', 'unknown')}`")
            continue
        lines.append(
            f"- `{model['model_ref']}`: case_success={model['case_success_rate']:.4f}, "
            f"step_success={model['step_success_rate']:.4f}, json_valid={model['json_valid_rate']:.4f}, "
            f"tok/s={model['tokens_per_second']:.2f}, peak_rss_mb={model['rss_peak_mb']:.2f}"
        )
    return "\n".join(lines) + "\n"


def save_tool_call_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "tool_call_report.json"
    md_path = root / "tool_call_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_tool_call_markdown(report), encoding="utf-8")
    return json_path, md_path
