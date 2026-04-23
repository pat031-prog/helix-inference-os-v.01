from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def manifest(self) -> list[dict[str, Any]]:
        return [self._tools[name].manifest() for name in sorted(self._tools)]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        required = list(tool.input_schema.get("required", []))
        missing = [
            field
            for field in required
            if field not in arguments
            or arguments.get(field) is None
            or (isinstance(arguments.get(field), str) and not str(arguments.get(field)).strip())
        ]
        if missing:
            raise ValueError(f"missing tool arguments: {', '.join(missing)}")
        return {
            "tool": name,
            "arguments": arguments,
            "result": tool.handler(arguments),
        }


def build_runtime_tool_registry(runtime: Any) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="workspace.list_models",
                description="List prepared model workspaces and any available sessions.",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: {"models": runtime.list_models()},
            ),
            ToolSpec(
                name="workspace.model_info",
                description="Inspect metadata for one prepared model alias.",
                input_schema={
                    "type": "object",
                    "properties": {"alias": {"type": "string"}},
                    "required": ["alias"],
                },
                handler=lambda args: runtime.model_info(args["alias"]),
            ),
            ToolSpec(
                name="gpt.generate_text",
                description="Generate text from a prepared GPT model alias.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "prompt": {"type": "string"},
                        "max_new_tokens": {"type": "integer", "default": 4},
                        "do_sample": {"type": "boolean", "default": False},
                        "temperature": {"type": "number", "default": 0.9},
                        "top_k": {"type": "integer", "default": 10},
                        "top_p": {"type": "number", "default": 0.9},
                        "seed": {"type": "integer"},
                        "cache_mode": {"type": "string", "default": "session"},
                    },
                    "required": ["alias", "prompt"],
                },
                handler=lambda args: runtime.generate_text(
                    alias=args["alias"],
                    prompt=args["prompt"],
                    max_new_tokens=int(args.get("max_new_tokens", 4)),
                    do_sample=bool(args.get("do_sample", False)),
                    temperature=float(args.get("temperature", 0.9)),
                    top_k=int(args.get("top_k", 10)),
                    top_p=float(args.get("top_p", 0.9)),
                    seed=args.get("seed"),
                    cache_mode=str(args.get("cache_mode", "session")),
                    session_id=args.get("session_id"),
                ),
            ),
            ToolSpec(
                name="gpt.resume_text",
                description="Resume a saved text generation session for a prepared GPT model alias.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "session_id": {"type": "string"},
                        "max_new_tokens": {"type": "integer", "default": 4},
                        "do_sample": {"type": "boolean", "default": False},
                        "temperature": {"type": "number", "default": 0.9},
                        "top_k": {"type": "integer", "default": 10},
                        "top_p": {"type": "number", "default": 0.9},
                        "seed": {"type": "integer"},
                        "cache_mode": {"type": "string", "default": "session"},
                    },
                    "required": ["alias", "session_id"],
                },
                handler=lambda args: runtime.resume_text(
                    alias=args["alias"],
                    session_id=args["session_id"],
                    max_new_tokens=int(args.get("max_new_tokens", 4)),
                    do_sample=bool(args.get("do_sample", False)),
                    temperature=float(args.get("temperature", 0.9)),
                    top_k=int(args.get("top_k", 10)),
                    top_p=float(args.get("top_p", 0.9)),
                    seed=args.get("seed"),
                    cache_mode=str(args.get("cache_mode", "session")),
                ),
            ),
        ]
    )
