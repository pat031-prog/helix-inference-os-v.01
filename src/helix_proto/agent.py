from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix_kv.memory_catalog import privacy_filter
from helix_kv.policy import AdaptiveKVPolicy
from helix_proto import hmem
from helix_proto.memory import (
    add_knowledge_file,
    add_knowledge_text,
    append_memory_event,
    save_run_trace,
    search_knowledge,
)
from helix_proto.tools import ToolRegistry, ToolSpec, build_runtime_tool_registry
from helix_proto.workspace import slugify, workspace_root


_REMOTE_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "have",
    "this",
    "your",
    "what",
    "about",
    "into",
    "they",
    "them",
    "then",
    "when",
    "where",
    "were",
    "will",
    "would",
    "there",
    "their",
    "para",
    "como",
    "esto",
    "that",
    "edge",
}
_AGENT_PHASE_MODES = {
    "plan": {"default_mode": "turbo-int8-hadamard", "allowed_modes": ("turbo-int8-hadamard", "fp32")},
    "tool_call": {"default_mode": "turbo-4bit", "allowed_modes": ("turbo-4bit", "turbo-int8-hadamard")},
    "observation": {"default_mode": "turbo-int8-hadamard", "allowed_modes": ("turbo-int8-hadamard", "fp32")},
    "final": {"default_mode": "fp32", "allowed_modes": ("fp32",)},
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


def _shorten(text: str, *, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _synthesize_hits(goal: str, hits: list[dict[str, Any]], *, max_sentences: int = 4) -> str:
    if not hits:
        return "No relevant context was found."
    query_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", goal.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }
    candidates: list[tuple[float, str]] = []
    for item in hits[:4]:
        text = str(item.get("text", ""))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cleaned = _shorten(sentence, limit=220)
            if not cleaned:
                continue
            sentence_terms = set(re.findall(r"[A-Za-z0-9_]+", cleaned.lower()))
            overlap = len(query_terms & sentence_terms)
            source_bonus = float(item.get("score", 0.0))
            score = overlap * 10.0 + source_bonus
            if score > 0:
                candidates.append((score, cleaned))
    if not candidates:
        return "\n\n".join(_shorten(item["text"]) for item in hits[:3])
    candidates.sort(key=lambda item: item[0], reverse=True)
    summary: list[str] = []
    for _, sentence in candidates:
        if sentence in summary:
            continue
        summary.append(sentence)
        if len(summary) >= max_sentences:
            break
    return " ".join(summary)


def _active_memory_query(
    goal: str,
    tool_name: str,
    arguments: dict[str, Any],
    scratchpad: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    parts = [f"Goal: {goal}", f"Tool: {tool_name}"]
    prompt = arguments.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        parts.append(f"Prompt: {_shorten(prompt, limit=600)}")
    else:
        parts.append(f"Arguments: {_shorten(json.dumps(arguments, ensure_ascii=True, sort_keys=True), limit=600)}")
    for item in scratchpad[-2:]:
        parts.append(
            "Step: "
            + _shorten(
                json.dumps(
                    {
                        "tool_name": item.get("tool_name"),
                        "thought": item.get("thought"),
                        "observation_preview": item.get("observation_preview"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                limit=360,
            )
        )
    for item in observations[-1:]:
        parts.append(
            "Previous observation: "
            + _shorten(json.dumps(item.get("observation", {}), ensure_ascii=True, sort_keys=True), limit=360)
        )
    return _shorten("\n".join(parts), limit=1400)


def _inject_active_context(prompt: str, context: dict[str, Any]) -> str:
    context_text = str(context.get("context") or "").strip()
    if not context_text:
        return prompt
    return (
        "<helix-active-context>\n"
        f"{context_text}\n"
        "</helix-active-context>\n\n"
        "User/task prompt:\n"
        f"{prompt}"
    )


def _strip_active_context_blocks(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"<helix-active-context>.*?</helix-active-context>\s*",
            "[HELIX_ACTIVE_CONTEXT_REDACTED]\n\n",
            value,
            flags=re.DOTALL,
        )
    if isinstance(value, dict):
        return {key: _strip_active_context_blocks(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_active_context_blocks(item) for item in value]
    return value


def _active_memory_event(
    *,
    step_index: int,
    tool_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    items = context.get("items") if isinstance(context.get("items"), list) else []
    router_metadata = [
        item.get("semantic_router")
        for item in items
        if isinstance(item, dict) and item.get("semantic_router")
    ]
    memory_ids = [str(item) for item in context.get("memory_ids", [])]
    return {
        "step_index": step_index,
        "tool_name": tool_name,
        "active_memory_used": bool(context.get("context")),
        "memory_ids": memory_ids,
        "memory_context_tokens": int(context.get("tokens") or 0),
        "query": privacy_filter(str(context.get("query") or "")),
        "mode": context.get("mode"),
        "source": "hmem",
        "router_metadata": router_metadata,
    }


def _planner_prompt(
    *,
    goal: str,
    tools: list[dict[str, Any]],
    scratchpad: list[dict[str, Any]],
    memory_hits: list[dict[str, Any]],
    knowledge_hits: list[dict[str, Any]],
    memory_context: dict[str, Any] | None,
    default_model_alias: str | None,
) -> str:
    examples = {
        "tool": {
            "kind": "tool",
            "thought": "Need more knowledge before answering.",
            "tool_name": "helix.search",
            "arguments": {"query": "user topic", "top_k": 3},
        },
        "final": {
            "kind": "final",
            "thought": "Enough context gathered.",
            "final": "Concise answer to the user.",
        },
    }
    return (
        "You are HelixAgent. Decide exactly one next action.\n"
        "Return only one JSON object.\n"
        'Valid shapes:\n'
        f"{json.dumps(examples['tool'])}\n"
        f"{json.dumps(examples['final'])}\n"
        "Rules:\n"
        "- Use at most one tool per response.\n"
        "- If you need context, use helix.search first; it combines agent memory and knowledge.\n"
        "- If a default model alias exists, you may use gpt.generate_text without asking the user.\n"
        "- If enough context already exists, return kind=final.\n"
        "- If you can answer directly without tools, answer directly.\n"
        "- Only use tools when you need information or actions you cannot do alone.\n"
        "- Do not output thinking or reasoning. /no_think\n"
        f"- Default model alias: {default_model_alias!r}\n\n"
        f"Goal:\n{goal}\n\n"
        f"Tools:\n{json.dumps(tools, indent=2)}\n\n"
        f"HeliX memory context:\n{json.dumps(memory_context or {}, indent=2)}\n\n"
        f"Relevant memory:\n{json.dumps(memory_hits, indent=2)}\n\n"
        f"Relevant knowledge:\n{json.dumps(knowledge_hits, indent=2)}\n\n"
        f"Scratchpad:\n{json.dumps(scratchpad, indent=2)}\n\n"
        "/no_think\n"
    )


@dataclass(slots=True)
class PlannerDecision:
    kind: str
    thought: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    final: str | None = None
    planner: str | None = None
    raw_text: str | None = None


class _BasePlanner:
    name = "base"

    def decide(self, state: dict[str, Any]) -> PlannerDecision:  # pragma: no cover - interface only
        raise NotImplementedError


class RuntimePlanner(_BasePlanner):
    name = "local"

    def __init__(self, runtime: Any, alias: str) -> None:
        self.runtime = runtime
        self.alias = alias

    def decide(self, state: dict[str, Any]) -> PlannerDecision:
        prompt = _planner_prompt(
            goal=state["goal"],
            tools=state["tool_manifest"],
            scratchpad=state["scratchpad"],
            memory_hits=state["memory_hits"],
            knowledge_hits=state["knowledge_hits"],
            memory_context=state.get("memory_context"),
            default_model_alias=state.get("default_model_alias"),
        )
        if hasattr(self.runtime, "_uses_llama_cpp") and self.runtime._uses_llama_cpp(self.alias):
            engine = self.runtime._engine(self.alias, cache_mode="session")
            response = engine.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are HelixAgent. Use native tool calling when a tool is needed. "
                            "If you can answer directly without tools, answer directly. "
                            "Only use tools when you need information or actions you cannot do alone. "
                            "If enough context already exists, return only one JSON object. /no_think"
                        ),
                    },
                    {"role": "user", "content": f"{prompt}\n\n/no_think"},
                ],
                tools=_tool_manifest_to_llama_tools(state["tool_manifest"]),
                tool_choice="auto",
                max_tokens=512,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            generated = str(
                dict(response["choices"][0].get("message", {}) or {}).get("content", "") or ""
            )
        else:
            result = self.runtime.generate_text(
                alias=self.alias,
                prompt=prompt,
                max_new_tokens=192,
                do_sample=False,
                cache_mode="session",
            )
            generated = result.get("completion_text") or result.get("generated_text") or ""
        generated = _strip_think_blocks(generated)
        parsed = _extract_tool_call_tag_payload(generated) or _extract_first_json_object(generated)
        if parsed is None:
            raise ValueError("local planner did not return valid JSON")
        return PlannerDecision(
            kind=str(parsed.get("kind", "")).strip().lower(),
            thought=str(parsed.get("thought", "")).strip(),
            tool_name=parsed.get("tool_name"),
            arguments=dict(parsed.get("arguments", {}) or {}),
            final=parsed.get("final"),
            planner=self.name,
            raw_text=generated,
        )


class RemotePlanner(_BasePlanner):
    name = "remote"

    def __init__(self, model_ref: str, *, trust_remote_code: bool = False) -> None:
        self.model_ref = model_ref
        self.trust_remote_code = trust_remote_code

    def _model(self):
        cached = _REMOTE_MODEL_CACHE.get(self.model_ref)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("remote planner needs optional HF dependencies") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_ref,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_ref,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        _REMOTE_MODEL_CACHE[self.model_ref] = (tokenizer, model)
        return tokenizer, model

    def decide(self, state: dict[str, Any]) -> PlannerDecision:
        import torch

        tokenizer, model = self._model()
        prompt = _planner_prompt(
            goal=state["goal"],
            tools=state["tool_manifest"],
            scratchpad=state["scratchpad"],
            memory_hits=state["memory_hits"],
            knowledge_hits=state["knowledge_hits"],
            memory_context=state.get("memory_context"),
            default_model_alias=state.get("default_model_alias"),
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = _strip_think_blocks(
            tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        parsed = _extract_tool_call_tag_payload(completion) or _extract_first_json_object(completion)
        if parsed is None:
            raise ValueError("remote planner did not return valid JSON")
        return PlannerDecision(
            kind=str(parsed.get("kind", "")).strip().lower(),
            thought=str(parsed.get("thought", "")).strip(),
            tool_name=parsed.get("tool_name"),
            arguments=dict(parsed.get("arguments", {}) or {}),
            final=parsed.get("final"),
            planner=self.name,
            raw_text=completion,
        )


class HeuristicPlanner(_BasePlanner):
    name = "heuristic"

    def decide(self, state: dict[str, Any]) -> PlannerDecision:
        goal = state["goal"]
        goal_lower = goal.lower()
        default_model_alias = state.get("default_model_alias")
        observations = state["observations"]
        knowledge_hits = state["knowledge_hits"]
        memory_hits = state["memory_hits"]
        hybrid_hits = state.get("hybrid_hits", [])

        if not observations:
            if any(term in goal_lower for term in ["previous", "before", "memoria", "remember", "earlier"]):
                return PlannerDecision(
                    kind="tool",
                    thought="Need memory recall before answering.",
                    tool_name="memory.search",
                    arguments={"query": goal, "top_k": 4},
                    planner=self.name,
                )
            if hybrid_hits or knowledge_hits or memory_hits:
                return PlannerDecision(
                    kind="tool",
                    thought="HeliX memory/knowledge has relevant context worth checking explicitly.",
                    tool_name="helix.search",
                    arguments={"query": goal, "top_k": 4},
                    planner=self.name,
                )
            if any(term in goal_lower for term in ["list models", "list-models", "available models", "modelos"]):
                return PlannerDecision(
                    kind="tool",
                    thought="Need the registry listing.",
                    tool_name="workspace.list_models",
                    arguments={},
                    planner=self.name,
                )
            if default_model_alias:
                return PlannerDecision(
                    kind="tool",
                    thought="Use the default worker model to draft the answer.",
                    tool_name="gpt.generate_text",
                    arguments={
                        "alias": default_model_alias,
                        "prompt": goal,
                        "max_new_tokens": state.get("generation_max_new_tokens", 96),
                    },
                    planner=self.name,
                )
            if knowledge_hits or memory_hits:
                context = knowledge_hits or memory_hits
                return PlannerDecision(
                    kind="final",
                    thought="Can answer directly from retrieved context.",
                    final=_synthesize_hits(goal, context),
                    planner=self.name,
                )
            return PlannerDecision(
                kind="final",
                thought="No tool or model is available, so answer conservatively.",
                final="No model or knowledge source is configured strongly enough to answer this yet.",
                planner=self.name,
            )

        last = observations[-1]
        tool_name = last["tool_name"]
        tool_result = last["observation"]

        if tool_name in {"rag.search", "memory.search", "helix.search"}:
            results = tool_result.get("result", {}).get("results", [])
            if results and default_model_alias:
                context_text = "\n\n".join(
                    f"Source: {item.get('source', item.get('kind', 'memory'))}\nText: {item['text']}"
                    for item in results[:3]
                )
                return PlannerDecision(
                    kind="tool",
                    thought="Use retrieved context to draft a final answer.",
                    tool_name="gpt.generate_text",
                    arguments={
                        "alias": default_model_alias,
                        "prompt": (
                            "Answer the user using this context when helpful.\n\n"
                            f"Context:\n{context_text}\n\nUser request:\n{goal}"
                        ),
                        "max_new_tokens": state.get("generation_max_new_tokens", 128),
                    },
                    planner=self.name,
                )
            if results:
                return PlannerDecision(
                    kind="final",
                    thought="Retrieved context is enough for a direct answer.",
                    final=_synthesize_hits(goal, results),
                    planner=self.name,
                )
            return PlannerDecision(
                kind="final",
                thought="No relevant retrieval hits were found.",
                final="I searched the current knowledge and memory but did not find relevant matches.",
                planner=self.name,
            )

        if tool_name.startswith("workspace."):
            payload = tool_result.get("result", {})
            return PlannerDecision(
                kind="final",
                thought="Workspace inspection returned the needed answer.",
                final=json.dumps(payload, indent=2),
                planner=self.name,
            )

        if tool_name.startswith("gpt."):
            payload = tool_result.get("result", {})
            final_text = payload.get("completion_text") or payload.get("generated_text") or json.dumps(payload, indent=2)
            return PlannerDecision(
                kind="final",
                thought="Worker model produced the answer.",
                final=final_text,
                planner=self.name,
            )

        return PlannerDecision(
            kind="final",
            thought="Unknown tool result, returning raw observation.",
            final=json.dumps(tool_result, indent=2),
            planner=self.name,
        )


class AgentRoutingPolicy:
    def __init__(
        self,
        *,
        local_planner_alias: str | None = None,
        remote_model_ref: str | None = None,
        prefer_remote: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.local_planner_alias = local_planner_alias
        self.remote_model_ref = remote_model_ref
        self.prefer_remote = prefer_remote
        self.trust_remote_code = trust_remote_code

    def planners(self, runtime: Any) -> list[_BasePlanner]:
        planners: list[_BasePlanner] = []
        local = RuntimePlanner(runtime, self.local_planner_alias) if self.local_planner_alias else None
        remote = (
            RemotePlanner(self.remote_model_ref, trust_remote_code=self.trust_remote_code)
            if self.remote_model_ref
            else None
        )
        if self.prefer_remote:
            if remote is not None:
                planners.append(remote)
            if local is not None:
                planners.append(local)
        else:
            if local is not None:
                planners.append(local)
            if remote is not None:
                planners.append(remote)
        planners.append(HeuristicPlanner())
        return planners


class AgentRunner:
    def __init__(self, runtime: Any, *, root: str | Path | None = None) -> None:
        self.runtime = runtime
        self.root = workspace_root(root)

    def add_knowledge_text(
        self,
        agent_name: str,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return add_knowledge_text(agent_name, text, source=source, root=self.root, metadata=metadata)

    def add_knowledge_file(
        self,
        agent_name: str,
        file_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return add_knowledge_file(agent_name, file_path, root=self.root, metadata=metadata)

    def search_knowledge(self, agent_name: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return search_knowledge(agent_name, query, top_k=top_k, root=self.root)

    def search_memory(self, agent_name: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return hmem.search(root=self.root, agent_id=agent_name, query=query, top_k=top_k)

    def search_hybrid(self, agent_name: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return hmem.hybrid_search(root=self.root, agent_id=agent_name, query=query, top_k=top_k)

    def _agent_tool_registry(self, *, agent_name: str) -> ToolRegistry:
        return ToolRegistry(
            [
                ToolSpec(
                    name="helix.search",
                    description="Search HeliX memory and the agent knowledge base together.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 4},
                        },
                        "required": ["query"],
                    },
                    handler=lambda args: self.search_hybrid(
                        agent_name,
                        str(args["query"]),
                        top_k=int(args.get("top_k", 4)),
                    ),
                ),
                ToolSpec(
                    name="rag.search",
                    description="Search the persistent knowledge base for this agent.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 4},
                        },
                        "required": ["query"],
                    },
                    handler=lambda args: self.search_knowledge(
                        agent_name,
                        str(args["query"]),
                        top_k=int(args.get("top_k", 4)),
                    ),
                ),
                ToolSpec(
                    name="memory.search",
                    description="Search episodic memory from previous agent runs.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 4},
                        },
                        "required": ["query"],
                    },
                    handler=lambda args: self.search_memory(
                        agent_name,
                        str(args["query"]),
                        top_k=int(args.get("top_k", 4)),
                    ),
                ),
            ]
        )

    def _configure_local_kv_phase(
        self,
        *,
        alias: str | None,
        phase: str,
        step_index: int,
        phase_trace: list[dict[str, Any]],
        reason: str,
    ) -> None:
        if not alias or not hasattr(self.runtime, "_engine") or not hasattr(self.runtime, "_uses_llama_cpp"):
            return
        if self.runtime._uses_llama_cpp(alias):
            phase_trace.append(
                {
                    "step_index": step_index,
                    "phase": phase,
                    "alias": alias,
                    "reason": reason,
                    "backend": "llama-cpp-python",
                    "configured": False,
                }
            )
            return
        engine = self.runtime._engine(alias, cache_mode="session")
        phase_config = _AGENT_PHASE_MODES[phase]
        current_mode_before = getattr(engine, "current_kv_mode", None)
        policy = getattr(engine, "_kv_policy", None)
        if not isinstance(policy, AdaptiveKVPolicy):
            policy = AdaptiveKVPolicy()
        engine.set_kv_policy(
            policy,
            phase=phase,
            allowed_modes=phase_config["allowed_modes"],
        )
        engine.switch_kv_mode(str(phase_config["default_mode"]), reason=f"phase_default:{phase}")
        phase_trace.append(
            {
                "step_index": step_index,
                "phase": phase,
                "alias": alias,
                "reason": reason,
                "backend": "helix-local",
                "configured": True,
                "current_mode_before": current_mode_before,
                "current_mode_after": getattr(engine, "current_kv_mode", None),
                "allowed_modes": list(phase_config["allowed_modes"]),
            }
        )

    def _build_step_memory_context(
        self,
        *,
        goal: str,
        agent_name: str,
        memory_project: str,
        memory_mode: str,
        memory_budget_tokens: int,
        tool_name: str,
        arguments: dict[str, Any],
        scratchpad: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        query = _active_memory_query(goal, tool_name, arguments, scratchpad, observations)
        if str(memory_mode or "off") == "off":
            return {
                "mode": "off",
                "context": "",
                "tokens": 0,
                "memory_ids": [],
                "items": [],
                "query": query,
                "source": "hmem",
            }
        try:
            context = hmem.build_context(
                root=self.root,
                project=memory_project,
                agent_id=agent_name,
                query=query,
                budget_tokens=memory_budget_tokens,
                mode=memory_mode,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "mode": "error",
                "context": "",
                "tokens": 0,
                "memory_ids": [],
                "items": [],
                "query": query,
                "source": "hmem",
                "error": str(exc),
            }
        context["query"] = query
        context["source"] = "hmem"
        return context

    def _prepare_active_tool_arguments(
        self,
        *,
        goal: str,
        agent_name: str,
        memory_project: str,
        memory_mode: str,
        memory_budget_tokens: int,
        step_index: int,
        tool_name: str,
        arguments: dict[str, Any],
        scratchpad: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        if tool_name != "gpt.generate_text":
            return dict(arguments), None, None
        context = self._build_step_memory_context(
            goal=goal,
            agent_name=agent_name,
            memory_project=memory_project,
            memory_mode=memory_mode,
            memory_budget_tokens=memory_budget_tokens,
            tool_name=tool_name,
            arguments=arguments,
            scratchpad=scratchpad,
            observations=observations,
        )
        event = _active_memory_event(step_index=step_index, tool_name=tool_name, context=context)
        enriched = dict(arguments)
        prompt = enriched.get("prompt")
        if isinstance(prompt, str) and context.get("context") and "<helix-active-context>" not in prompt:
            enriched["prompt"] = _inject_active_context(prompt, context)
        return enriched, event, context

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        runtime_tools: ToolRegistry,
        agent_tools: ToolRegistry,
        default_model_alias: str | None,
        step_index: int = 0,
        phase_trace: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if phase_trace is None:
            phase_trace = []
        if observations is None:
            observations = []
        enriched = dict(arguments)
        if tool_name in {"gpt.generate_text", "gpt.resume_text"} and default_model_alias and "alias" not in enriched:
            enriched["alias"] = default_model_alias
        if tool_name in {"gpt.generate_text", "gpt.resume_text"}:
            phase = "observation" if observations else "tool_call"
            self._configure_local_kv_phase(
                alias=str(enriched.get("alias") or default_model_alias or ""),
                phase=phase,
                step_index=step_index,
                phase_trace=phase_trace,
                reason=f"tool:{tool_name}",
            )
        try:
            return runtime_tools.call(tool_name, enriched)
        except KeyError:
            return agent_tools.call(tool_name, enriched)

    def run(
        self,
        *,
        goal: str,
        agent_name: str = "default-agent",
        default_model_alias: str | None = None,
        local_planner_alias: str | None = None,
        remote_model_ref: str | None = None,
        prefer_remote: bool = False,
        trust_remote_code: bool = False,
        max_steps: int = 4,
        generation_max_new_tokens: int = 128,
        memory_project: str = hmem.DEFAULT_PROJECT,
        memory_mode: str = "search",
        memory_budget_tokens: int = 800,
        memory_compressor_alias: str | None = None,
    ) -> dict[str, Any]:
        run_id = f"{slugify(agent_name)}-{int(time.time())}"
        runtime_tools = build_runtime_tool_registry(self.runtime)
        agent_tools = self._agent_tool_registry(agent_name=agent_name)
        tool_manifest = runtime_tools.manifest() + agent_tools.manifest()
        memory_compress_fn = hmem.runtime_compress_fn(self.runtime, memory_compressor_alias)
        initial_memory_context = hmem.build_context(
            root=self.root,
            project=memory_project,
            agent_id=agent_name,
            query=goal,
            budget_tokens=memory_budget_tokens,
            mode=memory_mode,
        )

        append_memory_event(
            agent_name,
            kind="goal",
            text=goal,
            root=self.root,
            metadata={"run_id": run_id},
        )
        hmem.observe_event(
            root=self.root,
            project=memory_project,
            agent_id=agent_name,
            session_id=run_id,
            event_type="goal",
            content=goal,
            summary=f"Goal: {goal}",
            tags=["goal", f"run:{run_id}"],
            importance=5,
            promote=False,
        )

        scratchpad: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        planner_attempts: list[dict[str, Any]] = []
        kv_phase_trace: list[dict[str, Any]] = []
        active_memory_events: list[dict[str, Any]] = []
        memory_context = initial_memory_context
        policy = AgentRoutingPolicy(
            local_planner_alias=local_planner_alias,
            remote_model_ref=remote_model_ref,
            prefer_remote=prefer_remote,
            trust_remote_code=trust_remote_code,
        )

        final_answer: str | None = None
        final_planner = None

        for step_index in range(max_steps):
            memory_hits = self.search_memory(
                agent_name,
                goal,
                top_k=4,
            )["results"]
            knowledge_hits = self.search_knowledge(agent_name, goal, top_k=4)["results"]
            hybrid_hits = self.search_hybrid(agent_name, goal, top_k=4)["results"]
            memory_context = self._build_step_memory_context(
                goal=goal,
                agent_name=agent_name,
                memory_project=memory_project,
                memory_mode=memory_mode,
                memory_budget_tokens=memory_budget_tokens,
                tool_name="__planner__",
                arguments={},
                scratchpad=scratchpad,
                observations=observations,
            )
            active_memory_events.append(
                _active_memory_event(step_index=step_index, tool_name="__planner__", context=memory_context)
            )
            state = {
                "goal": goal,
                "tool_manifest": tool_manifest,
                "scratchpad": scratchpad,
                "memory_hits": memory_hits,
                "knowledge_hits": knowledge_hits,
                "hybrid_hits": hybrid_hits,
                "memory_context": memory_context,
                "observations": observations,
                "default_model_alias": default_model_alias,
                "generation_max_new_tokens": generation_max_new_tokens,
            }

            self._configure_local_kv_phase(
                alias=local_planner_alias,
                phase="plan",
                step_index=step_index,
                phase_trace=kv_phase_trace,
                reason="planner",
            )
            decision: PlannerDecision | None = None
            errors: list[str] = []
            for planner in policy.planners(self.runtime):
                try:
                    candidate = planner.decide(state)
                    candidate.planner = planner.name
                    if candidate.kind not in {"tool", "final"}:
                        raise ValueError(f"invalid decision kind: {candidate.kind!r}")
                    decision = candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{planner.name}: {exc}")
            planner_attempts.append({"step_index": step_index, "errors": errors})
            if decision is None:
                final_answer = "The agent could not produce a valid plan."
                final_planner = "none"
                break

            step_record = {
                "step_index": step_index,
                "planner": decision.planner,
                "thought": decision.thought,
                "kind": decision.kind,
            }

            if decision.kind == "final":
                self._configure_local_kv_phase(
                    alias=default_model_alias,
                    phase="final",
                    step_index=step_index,
                    phase_trace=kv_phase_trace,
                    reason="final",
                )
                final_answer = decision.final or ""
                final_planner = decision.planner
                scratchpad.append(step_record | {"final": final_answer})
                break

            tool_name = decision.tool_name or ""
            original_tool_arguments = decision.arguments or {}
            tool_arguments, active_tool_event, _active_tool_context = self._prepare_active_tool_arguments(
                goal=goal,
                agent_name=agent_name,
                memory_project=memory_project,
                memory_mode=memory_mode,
                memory_budget_tokens=memory_budget_tokens,
                step_index=step_index,
                tool_name=tool_name,
                arguments=original_tool_arguments,
                scratchpad=scratchpad,
                observations=observations,
            )
            if active_tool_event is not None:
                active_memory_events.append(active_tool_event)
            recorded_tool_arguments = dict(original_tool_arguments)
            if active_tool_event is not None:
                recorded_tool_arguments["active_memory"] = active_tool_event
            tool_result = self._call_tool(
                tool_name,
                tool_arguments,
                runtime_tools=runtime_tools,
                agent_tools=agent_tools,
                default_model_alias=default_model_alias,
                step_index=step_index,
                phase_trace=kv_phase_trace,
                observations=observations,
            )
            recorded_tool_result = _strip_active_context_blocks(tool_result)
            observation = {
                "tool_name": tool_name,
                "arguments": recorded_tool_arguments,
                "observation": recorded_tool_result,
            }
            hmem_record = hmem.observe_tool_call(
                root=self.root,
                project=memory_project,
                agent_id=agent_name,
                run_id=run_id,
                step_index=step_index,
                tool_name=tool_name,
                arguments=recorded_tool_arguments,
                result=recorded_tool_result,
                planner=decision.planner,
                compress_fn=memory_compress_fn,
            )
            observation["hmem"] = hmem_record
            observations.append(observation)
            scratchpad.append(
                step_record
                | {
                    "tool_name": tool_name,
                    "arguments": recorded_tool_arguments,
                    "active_memory": active_tool_event,
                    "observation_preview": _shorten(json.dumps(recorded_tool_result, ensure_ascii=True)),
                    "hmem_memory_id": (hmem_record.get("memory") or {}).get("memory_id"),
                }
            )
            append_memory_event(
                agent_name,
                kind="tool_observation",
                text=(
                    f"Goal: {goal}\nTool: {tool_name}\n"
                    f"Arguments: {json.dumps(recorded_tool_arguments)}\n"
                    f"Observation: {_shorten(json.dumps(recorded_tool_result, ensure_ascii=True), limit=600)}"
                ),
                root=self.root,
                metadata={"run_id": run_id, "step_index": step_index, "planner": decision.planner},
            )

        if final_answer is None:
            if observations:
                final_answer = json.dumps(observations[-1]["observation"], indent=2)
                final_planner = "fallback-summary"
            else:
                final_answer = "The agent stopped before producing a final answer."
                final_planner = "fallback-empty"

        append_memory_event(
            agent_name,
            kind="final_answer",
            text=final_answer,
            root=self.root,
            metadata={"run_id": run_id, "planner": final_planner},
        )
        hmem.observe_event(
            root=self.root,
            project=memory_project,
            agent_id=agent_name,
            session_id=run_id,
            event_type="final_answer",
            content=final_answer,
            summary=f"Final answer: {final_answer}",
            tags=["final_answer", f"run:{run_id}"],
            importance=6,
            promote=True,
            memory_type="episodic",
            compress_fn=memory_compress_fn,
        )

        trace = {
            "agent_name": agent_name,
            "run_id": run_id,
            "goal": goal,
            "default_model_alias": default_model_alias,
            "local_planner_alias": local_planner_alias,
            "remote_model_ref": remote_model_ref,
            "prefer_remote": prefer_remote,
            "max_steps": max_steps,
            "planner_attempts": planner_attempts,
            "kv_phase_trace": kv_phase_trace,
            "steps": scratchpad,
            "observations": observations,
            "final_answer": final_answer,
            "final_planner": final_planner,
            "memory_context": memory_context,
            "initial_memory_context": initial_memory_context,
            "active_memory_events": active_memory_events,
            "memory_hits": memory_hits if "memory_hits" in locals() else [],
            "knowledge_hits": knowledge_hits if "knowledge_hits" in locals() else [],
            "hybrid_hits": hybrid_hits if "hybrid_hits" in locals() else [],
        }
        trace_path = save_run_trace(agent_name, run_id, trace, root=self.root)
        trace["trace_path"] = str(trace_path)
        return trace

    def run_stream(
        self,
        *,
        goal: str,
        agent_name: str = "default-agent",
        default_model_alias: str | None = None,
        local_planner_alias: str | None = None,
        remote_model_ref: str | None = None,
        prefer_remote: bool = False,
        trust_remote_code: bool = False,
        max_steps: int = 4,
        generation_max_new_tokens: int = 128,
        memory_project: str = hmem.DEFAULT_PROJECT,
        memory_mode: str = "search",
        memory_budget_tokens: int = 800,
        memory_compressor_alias: str | None = None,
    ):
        run_id = f"{slugify(agent_name)}-{int(time.time())}"
        runtime_tools = build_runtime_tool_registry(self.runtime)
        agent_tools = self._agent_tool_registry(agent_name=agent_name)
        tool_manifest = runtime_tools.manifest() + agent_tools.manifest()
        memory_compress_fn = hmem.runtime_compress_fn(self.runtime, memory_compressor_alias)
        initial_memory_context = hmem.build_context(
            root=self.root,
            project=memory_project,
            agent_id=agent_name,
            query=goal,
            budget_tokens=memory_budget_tokens,
            mode=memory_mode,
        )

        append_memory_event(
            agent_name,
            kind="goal",
            text=goal,
            root=self.root,
            metadata={"run_id": run_id},
        )
        hmem.observe_event(
            root=self.root,
            project=memory_project,
            agent_id=agent_name,
            session_id=run_id,
            event_type="goal",
            content=goal,
            summary=f"Goal: {goal}",
            tags=["goal", f"run:{run_id}"],
            importance=5,
            promote=False,
        )

        scratchpad: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        planner_attempts: list[dict[str, Any]] = []
        kv_phase_trace: list[dict[str, Any]] = []
        active_memory_events: list[dict[str, Any]] = []
        memory_context = initial_memory_context
        policy = AgentRoutingPolicy(
            local_planner_alias=local_planner_alias,
            remote_model_ref=remote_model_ref,
            prefer_remote=prefer_remote,
            trust_remote_code=trust_remote_code,
        )

        yield {
            "event": "start",
            "agent_name": agent_name,
            "run_id": run_id,
            "goal": goal,
            "default_model_alias": default_model_alias,
            "memory_context": initial_memory_context,
        }

        final_answer: str | None = None
        final_planner = None
        memory_hits: list[dict[str, Any]] = []
        knowledge_hits: list[dict[str, Any]] = []
        hybrid_hits: list[dict[str, Any]] = []

        for step_index in range(max_steps):
            memory_hits = self.search_memory(
                agent_name,
                goal,
                top_k=4,
            )["results"]
            knowledge_hits = self.search_knowledge(agent_name, goal, top_k=4)["results"]
            hybrid_hits = self.search_hybrid(agent_name, goal, top_k=4)["results"]
            memory_context = self._build_step_memory_context(
                goal=goal,
                agent_name=agent_name,
                memory_project=memory_project,
                memory_mode=memory_mode,
                memory_budget_tokens=memory_budget_tokens,
                tool_name="__planner__",
                arguments={},
                scratchpad=scratchpad,
                observations=observations,
            )
            active_memory_events.append(
                _active_memory_event(step_index=step_index, tool_name="__planner__", context=memory_context)
            )
            state = {
                "goal": goal,
                "tool_manifest": tool_manifest,
                "scratchpad": scratchpad,
                "memory_hits": memory_hits,
                "knowledge_hits": knowledge_hits,
                "hybrid_hits": hybrid_hits,
                "memory_context": memory_context,
                "observations": observations,
                "default_model_alias": default_model_alias,
                "generation_max_new_tokens": generation_max_new_tokens,
            }

            decision: PlannerDecision | None = None
            errors: list[str] = []
            for planner in policy.planners(self.runtime):
                try:
                    candidate = planner.decide(state)
                    candidate.planner = planner.name
                    if candidate.kind not in {"tool", "final"}:
                        raise ValueError(f"invalid decision kind: {candidate.kind!r}")
                    decision = candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{planner.name}: {exc}")
            planner_attempts.append({"step_index": step_index, "errors": errors})
            if decision is None:
                final_answer = "The agent could not produce a valid plan."
                final_planner = "none"
                break

            yield {
                "event": "plan",
                "step_index": step_index,
                "planner": decision.planner,
                "thought": decision.thought,
                "kind": decision.kind,
                "errors": errors,
            }

            step_record = {
                "step_index": step_index,
                "planner": decision.planner,
                "thought": decision.thought,
                "kind": decision.kind,
            }

            if decision.kind == "final":
                final_answer = decision.final or ""
                final_planner = decision.planner
                scratchpad.append(step_record | {"final": final_answer})
                yield {
                    "event": "final",
                    "step_index": step_index,
                    "planner": final_planner,
                    "final_answer": final_answer,
                }
                break

            tool_name = decision.tool_name or ""
            original_tool_arguments = decision.arguments or {}
            tool_arguments, active_tool_event, _active_tool_context = self._prepare_active_tool_arguments(
                goal=goal,
                agent_name=agent_name,
                memory_project=memory_project,
                memory_mode=memory_mode,
                memory_budget_tokens=memory_budget_tokens,
                step_index=step_index,
                tool_name=tool_name,
                arguments=original_tool_arguments,
                scratchpad=scratchpad,
                observations=observations,
            )
            if active_tool_event is not None:
                active_memory_events.append(active_tool_event)
                yield {
                    "event": "active_memory_context",
                    **active_tool_event,
                }
            recorded_tool_arguments = dict(original_tool_arguments)
            if active_tool_event is not None:
                recorded_tool_arguments["active_memory"] = active_tool_event
            yield {
                "event": "tool_call",
                "step_index": step_index,
                "tool_name": tool_name,
                "arguments": recorded_tool_arguments,
            }

            if (
                tool_name == "gpt.generate_text"
                and hasattr(self.runtime, "stream_text")
            ):
                streamed_args = dict(tool_arguments)
                if default_model_alias and "alias" not in streamed_args:
                    streamed_args["alias"] = default_model_alias
                tool_result = None
                for stream_event in self.runtime.stream_text(**streamed_args):
                    yield {
                        "event": "tool_stream",
                        "step_index": step_index,
                        "tool_name": tool_name,
                        "payload": stream_event,
                    }
                    if stream_event.get("event") == "done":
                        tool_result = {
                            "tool": tool_name,
                            "arguments": recorded_tool_arguments,
                            "result": stream_event,
                        }
                if tool_result is None:
                    raise RuntimeError("streamed tool call ended without a done event")
            else:
                tool_result = self._call_tool(
                    tool_name,
                    tool_arguments,
                    runtime_tools=runtime_tools,
                    agent_tools=agent_tools,
                    default_model_alias=default_model_alias,
                    step_index=step_index,
                    phase_trace=kv_phase_trace,
                    observations=observations,
                )
            recorded_tool_result = _strip_active_context_blocks(tool_result)

            observation = {
                "tool_name": tool_name,
                "arguments": recorded_tool_arguments,
                "observation": recorded_tool_result,
            }
            hmem_record = hmem.observe_tool_call(
                root=self.root,
                project=memory_project,
                agent_id=agent_name,
                run_id=run_id,
                step_index=step_index,
                tool_name=tool_name,
                arguments=recorded_tool_arguments,
                result=recorded_tool_result,
                planner=decision.planner,
                compress_fn=memory_compress_fn,
            )
            observation["hmem"] = hmem_record
            observations.append(observation)
            scratchpad.append(
                step_record
                | {
                    "tool_name": tool_name,
                    "arguments": recorded_tool_arguments,
                    "active_memory": active_tool_event,
                    "observation_preview": _shorten(json.dumps(recorded_tool_result, ensure_ascii=True)),
                    "hmem_memory_id": (hmem_record.get("memory") or {}).get("memory_id"),
                }
            )
            append_memory_event(
                agent_name,
                kind="tool_observation",
                text=(
                    f"Goal: {goal}\nTool: {tool_name}\n"
                    f"Arguments: {json.dumps(recorded_tool_arguments)}\n"
                    f"Observation: {_shorten(json.dumps(recorded_tool_result, ensure_ascii=True), limit=600)}"
                ),
                root=self.root,
                metadata={"run_id": run_id, "step_index": step_index, "planner": decision.planner},
            )
            yield {
                "event": "tool_result",
                "step_index": step_index,
                "tool_name": tool_name,
                "result": recorded_tool_result,
            }

        if final_answer is None:
            if observations:
                final_answer = json.dumps(observations[-1]["observation"], indent=2)
                final_planner = "fallback-summary"
            else:
                final_answer = "The agent stopped before producing a final answer."
                final_planner = "fallback-empty"
            yield {
                "event": "final",
                "planner": final_planner,
                "final_answer": final_answer,
            }

        append_memory_event(
            agent_name,
            kind="final_answer",
            text=final_answer,
            root=self.root,
            metadata={"run_id": run_id, "planner": final_planner},
        )
        hmem.observe_event(
            root=self.root,
            project=memory_project,
            agent_id=agent_name,
            session_id=run_id,
            event_type="final_answer",
            content=final_answer,
            summary=f"Final answer: {final_answer}",
            tags=["final_answer", f"run:{run_id}"],
            importance=6,
            promote=True,
            memory_type="episodic",
            compress_fn=memory_compress_fn,
        )

        trace = {
            "agent_name": agent_name,
            "run_id": run_id,
            "goal": goal,
            "default_model_alias": default_model_alias,
            "local_planner_alias": local_planner_alias,
            "remote_model_ref": remote_model_ref,
            "prefer_remote": prefer_remote,
            "max_steps": max_steps,
            "planner_attempts": planner_attempts,
            "kv_phase_trace": kv_phase_trace,
            "steps": scratchpad,
            "observations": observations,
            "final_answer": final_answer,
            "final_planner": final_planner,
            "memory_context": memory_context,
            "initial_memory_context": initial_memory_context,
            "active_memory_events": active_memory_events,
            "memory_hits": memory_hits,
            "knowledge_hits": knowledge_hits,
            "hybrid_hits": hybrid_hits,
        }
        trace_path = save_run_trace(agent_name, run_id, trace, root=self.root)
        yield {
            "event": "done",
            "trace_path": str(trace_path),
            "final_answer": final_answer,
            "run_id": run_id,
        }
