from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from helix_proto import hmem
from helix_proto.agent import AgentRunner
from helix_proto.assistants import (
    build_assistant_messages,
    configure_assistant,
    list_assistants,
    resolve_assistant,
)
from helix_proto.cdna import load_generation_target
from helix_proto.hf import GPT2StreamingEngine
from helix_proto.research_artifacts import (
    artifact_title as _artifact_title,
    load_research_artifact as _load_research_artifact,
    research_artifact_manifest as _research_artifact_manifest,
)
from helix_proto.text import decode_tokens, encode_text, render_messages_prompt
from helix_proto.tools import build_runtime_tool_registry
from helix_proto.workspace import (
    list_model_workspaces,
    model_session_dir,
    prepare_model_workspace,
    resolve_export_dir,
    resolve_model_info,
    resolve_tokenizer_dir,
    slugify,
    workspace_root,
)


def _session_id(prefix: str = "session") -> str:
    return f"{slugify(prefix)}-{int(time.time())}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _web_root() -> Path:
    return _repo_root() / "web"


def _web_static_root() -> Path:
    return _web_root() / "static"


def _legacy_frontend_root() -> Path:
    return _repo_root() / "frontend"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = dict(value)
        last_logits = cleaned.pop("last_logits", None)
        if isinstance(last_logits, np.ndarray):
            cleaned["last_logits_shape"] = list(last_logits.shape)
        return {key: _json_ready(item) for key, item in cleaned.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cors_origins() -> list[str]:
    raw = os.environ.get("HELIX_CORS_ORIGINS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _allowed_origin(request_origin: str | None) -> str | None:
    if not request_origin:
        return None
    configured = _cors_origins()
    if not configured:
        return request_origin
    if "*" in configured:
        return "*"
    if request_origin in configured:
        return request_origin
    return None


def _sanitize_assistant_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\n\n/no_think", "")
    text = text.replace("\n/no_think", "")
    text = text.replace("/no_think", "")
    return text.strip()


def _extract_chat_chunk_text(chunk: dict[str, Any]) -> str:
    choice = dict(chunk.get("choices", [{}])[0] or {})
    delta = choice.get("delta")
    if isinstance(delta, dict):
        message = delta.get("message")
        if isinstance(message, dict):
            text = str(message.get("content", "") or "")
            if text:
                return text
        text = str(delta.get("content", "") or "")
        if text:
            return text
    message = choice.get("message")
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(choice.get("text", "") or "")


class _AssistantTokenCleaner:
    _PLAIN_THINK_PREFIXES = (
        "thinking process:",
        "thinking:",
        "razonamiento:",
        "pensamiento:",
    )
    _REASONING_MARKERS = (
        "analyze the request",
        "determine the response",
        "drafting the response",
        "internal reasoning",
        "response plan:",
        "reasoning:",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._visible = False
        self._in_think_tag = False
        self._dropping_plain_reasoning = False

    def push(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += text
        emitted = ""

        while True:
            if self._in_think_tag:
                end = self._buffer.find("</think>")
                if end == -1:
                    self._buffer = self._buffer[-64:]
                    return emitted
                self._buffer = self._buffer[end + len("</think>") :]
                self._in_think_tag = False
                continue

            if not self._visible:
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = stripped
                    return emitted

                if stripped.startswith("<think>"):
                    self._buffer = stripped[len("<think>") :]
                    self._in_think_tag = True
                    continue

                lowered = stripped.lower()
                if self._dropping_plain_reasoning:
                    if not self._looks_like_reasoning_scaffold(stripped) and not re.match(
                        r"^\d+\.\s+", lowered
                    ):
                        self._buffer = stripped
                        self._dropping_plain_reasoning = False
                        self._visible = True
                        continue
                    match = re.search(r"\n\s*\n", stripped)
                    if match is None:
                        if len(stripped) > 4096:
                            self._buffer = stripped[-1024:]
                        else:
                            self._buffer = stripped
                        return emitted
                    candidate = stripped[match.end() :].lstrip()
                    if not candidate:
                        self._buffer = candidate
                        return emitted
                    self._buffer = candidate
                    self._dropping_plain_reasoning = bool(
                        self._looks_like_reasoning_scaffold(candidate)
                        or re.match(r"^\d+\.\s+", candidate.lower())
                    )
                    continue

                if any(prefix.startswith(lowered) for prefix in self._PLAIN_THINK_PREFIXES):
                    self._buffer = stripped
                    return emitted
                if self._looks_like_reasoning_scaffold(stripped):
                    self._buffer = stripped
                    self._dropping_plain_reasoning = True
                    continue

                self._buffer = stripped
                self._visible = True
                continue

            start = self._buffer.find("<think>")
            if start != -1:
                emitted += self._buffer[:start]
                self._buffer = self._buffer[start + len("<think>") :]
                self._in_think_tag = True
                continue

            emitted += self._buffer
            self._buffer = ""
            return emitted

    def finish(self) -> str:
        if self._in_think_tag:
            return ""
        if self._visible:
            tail = self._buffer
            self._buffer = ""
            return tail
        return ""

    def _looks_like_reasoning_scaffold(self, text: str) -> bool:
        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in self._PLAIN_THINK_PREFIXES):
            return True
        if lowered.startswith("<think>"):
            return True
        if any(marker in lowered[:240] for marker in self._REASONING_MARKERS):
            return True
        if re.match(r"^\d+\.\s*(?:<think>|\*\*|analy[sz]e|determin|draft|reason|thought)", lowered):
            return True
        if re.match(r"^\d+\.\s*$", lowered):
            return True
        return False


class HelixRuntime:
    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root = workspace_root(root)
        self._engines: dict[tuple[str, str], Any] = {}
        self._tools = None

    def list_models(self) -> list[dict[str, Any]]:
        return list_model_workspaces(self.root)

    def model_info(self, alias: str) -> dict[str, Any]:
        return resolve_model_info(alias, self.root)

    def prepare_model(
        self,
        *,
        model_ref: str,
        alias: str | None = None,
        block_rows: int = 256,
        compress: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        force: bool = False,
        chat_format: str | None = None,
        n_ctx: int = 4096,
    ) -> dict[str, Any]:
        return prepare_model_workspace(
            model_ref=model_ref,
            alias=alias,
            root=self.root,
            block_rows=block_rows,
            compress=compress,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            force=force,
            chat_format=chat_format,
            n_ctx=n_ctx,
        )

    def list_assistants(self) -> list[dict[str, Any]]:
        return list_assistants(self.root)

    def configure_assistant(self, assistant_id: str, **kwargs: Any) -> dict[str, Any]:
        return configure_assistant(assistant_id, root=self.root, **kwargs)

    def _assistant_request(
        self,
        assistant_id: str,
        *,
        message: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], str | None, list[dict[str, str]] | None]:
        assistant = resolve_assistant(assistant_id, self.root)
        alias = str(assistant.get("alias") or "").strip()
        if not alias or not bool(assistant.get("available")):
            raise FileNotFoundError(f"assistant '{assistant_id}' has no available configured model")
        prompt, chat_messages = build_assistant_messages(
            assistant,
            prompt=message,
            messages=messages,
        )
        return assistant, prompt, chat_messages

    def _assistant_completion_prompt(
        self,
        assistant: dict[str, Any],
        *,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        assistant_id = str(assistant.get("assistant_id") or assistant.get("id") or "general")
        concise_role_prompt = {
            "general": "Sos Helix General. Respondé en español rioplatense, breve, útil y directo.",
            "code": "Sos Helix Código. Respondé en español claro, con foco en solución concreta y correcta.",
            "legal": (
                "Sos Helix Jurídico. Respondé en español claro para Argentina, en términos generales y con cautela."
            ),
        }.get(assistant_id, str(assistant.get("system_prompt") or "").strip())
        lines: list[str] = []
        if concise_role_prompt:
            lines.append(concise_role_prompt)
        lines.append("Responde en texto plano y solo con la respuesta final.")
        lines.append("No uses thinking, analisis, encabezados, markdown ni listas numeradas salvo que el usuario lo pida.")
        lines.append("Si alcanza con una frase corta, usa una frase corta.")
        lines.append("Si el usuario saluda, responde en una sola linea amable.")
        lines.append("Respondé solo con la respuesta final para el usuario.")
        lines.append("No muestres razonamiento, pasos, análisis, traducciones ni thinking.")
        lines.append("No repitas instrucciones ni describas tu proceso.")
        lines.append("")

        if messages:
            for item in messages:
                role = str(item.get("role", "user")).strip().lower() or "user"
                if role == "system":
                    continue
                label = {
                    "user": "Usuario",
                    "assistant": "Asistente",
                    "tool": "Herramienta",
                }.get(role, role.title())
                content = _sanitize_assistant_text(str(item.get("content", "")))
                if content:
                    lines.append(f"{label}: {content}")
        elif prompt is not None:
            user_prompt = _sanitize_assistant_text(prompt)
            if user_prompt:
                lines.append(f"Usuario: {user_prompt}")
        else:
            raise ValueError("prompt or messages is required")

        lines.append("Asistente:")
        return "\n".join(lines)

    def _assistant_llama_messages(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]] | None:
        if messages:
            return messages
        if prompt is None:
            return None
        content = _sanitize_assistant_text(prompt)
        if not content:
            return None
        return [{"role": "user", "content": content}]

    def _llama_qwen_nonthinking_chat_handler(self, engine: Any):
        cached = getattr(engine, "_helix_qwen_nonthinking_chat_handler", None)
        if cached is not None:
            return cached
        metadata = getattr(engine, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        if str(metadata.get("general.architecture") or "") != "qwen35":
            return None
        template = str(metadata.get("tokenizer.chat_template") or "")
        if not template or "enable_thinking" not in template:
            return None
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from llama_cpp.llama_chat_format import (
                ChatFormatterResponse,
                chat_formatter_to_chat_completion_handler,
            )
        except Exception:  # noqa: BLE001
            return None

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(template)

        def raise_exception(message: str) -> None:
            raise ValueError(message)

        def formatter(
            *,
            messages: list[dict[str, Any]],
            functions: list[dict[str, Any]] | None = None,
            function_call: dict[str, Any] | str | None = None,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: dict[str, Any] | str | None = None,
            **_: Any,
        ) -> Any:
            prompt = environment.render(
                messages=messages,
                eos_token="<|im_end|>",
                bos_token="<|im_start|>",
                raise_exception=raise_exception,
                add_generation_prompt=True,
                add_vision_id=False,
                enable_thinking=False,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                strftime_now=lambda fmt: datetime.now().strftime(fmt),
            )
            return ChatFormatterResponse(
                prompt=prompt,
                stop=["<|im_end|>"],
                added_special=True,
            )

        handler = chat_formatter_to_chat_completion_handler(formatter)
        setattr(engine, "_helix_qwen_nonthinking_chat_handler", handler)
        return handler

    def _assistant_llama_chat_completion(
        self,
        engine: Any,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        stream: bool = False,
    ) -> Any | None:
        handler = self._llama_qwen_nonthinking_chat_handler(engine)
        if handler is None:
            return None
        return handler(
            llama=engine,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            presence_penalty=0.0,
            repeat_penalty=1.0,
            stream=stream,
            seed=seed,
        )

    def generate_assistant_text(
        self,
        *,
        assistant_id: str,
        message: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        skip_special_tokens: bool = True,
    ) -> dict[str, Any]:
        assistant, prompt, chat_messages = self._assistant_request(
            assistant_id,
            message=message,
            messages=messages,
        )
        alias = str(assistant["alias"])
        if self._uses_llama_cpp(alias):
            engine = self._engine(alias, cache_mode=cache_mode)
            assistant_messages = self._assistant_llama_messages(prompt=prompt, messages=chat_messages)
            if assistant_messages:
                response = self._assistant_llama_chat_completion(
                    engine,
                    messages=assistant_messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    stream=False,
                )
                if response is not None:
                    completion_text = str(
                        dict(response["choices"][0].get("message", {}) or {}).get("content", "") or ""
                    ).strip()
                    return {
                        "assistant_id": assistant["assistant_id"],
                        "alias": alias,
                        "backend": "llama-cpp-python",
                        "prompt_text": render_messages_prompt(assistant_messages),
                        "completion_text": completion_text,
                        "generated_text": completion_text,
                        "generated_ids": [],
                        "new_ids": [],
                        "prompt_ids": [],
                        "session_id": None,
                        "messages": assistant_messages,
                    }
            completion_prompt = self._assistant_completion_prompt(
                assistant,
                prompt=prompt,
                messages=chat_messages,
            )
            response = engine.create_completion(
                prompt=completion_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                repeat_penalty=1.05,
                stop=["\nUsuario:", "\nAsistente:", "\nuser:", "\nassistant:"],
            )
            completion_text = str(response["choices"][0].get("text", "") or "").strip()
            return {
                "assistant_id": assistant["assistant_id"],
                "alias": alias,
                "backend": "llama-cpp-python",
                "prompt_text": completion_prompt,
                "completion_text": completion_text,
                "generated_text": completion_text,
                "generated_ids": [],
                "new_ids": [],
                "prompt_ids": [],
                "session_id": None,
            }
        result = self.generate_text(
            alias=alias,
            prompt=prompt,
            messages=chat_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            cache_mode=cache_mode,
            skip_special_tokens=skip_special_tokens,
        )
        result["assistant_id"] = assistant["assistant_id"]
        return result

    def stream_assistant_text(
        self,
        *,
        assistant_id: str,
        message: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        skip_special_tokens: bool = True,
    ):
        assistant, prompt, chat_messages = self._assistant_request(
            assistant_id,
            message=message,
            messages=messages,
        )
        alias = str(assistant["alias"])
        if self._uses_llama_cpp(alias):
            engine = self._engine(alias, cache_mode=cache_mode)
            assistant_messages = self._assistant_llama_messages(prompt=prompt, messages=chat_messages)
            if assistant_messages:
                stream = self._assistant_llama_chat_completion(
                    engine,
                    messages=assistant_messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    stream=True,
                )
                if stream is not None:
                    prompt_text = render_messages_prompt(assistant_messages)
                    yield {
                        "event": "start",
                        "assistant_id": assistant["assistant_id"],
                        "alias": alias,
                        "prompt_text": prompt_text,
                        "backend": "llama-cpp-python",
                    }
                    generated_text = ""
                    for chunk in stream:
                        text = _extract_chat_chunk_text(dict(chunk))
                        if not text:
                            continue
                        generated_text += text
                        yield {
                            "event": "token",
                            "assistant_id": assistant["assistant_id"],
                            "alias": alias,
                            "token_id": None,
                            "token_text": text,
                            "completion_text": generated_text,
                            "generated_text": generated_text,
                            "backend": "llama-cpp-python",
                        }
                    yield {
                        "event": "done",
                        "assistant_id": assistant["assistant_id"],
                        "alias": alias,
                        "completion_text": generated_text.strip(),
                        "generated_text": generated_text.strip(),
                        "backend": "llama-cpp-python",
                    }
                    return
            completion_prompt = self._assistant_completion_prompt(
                assistant,
                prompt=prompt,
                messages=chat_messages,
            )
            yield {
                "event": "start",
                "assistant_id": assistant["assistant_id"],
                "alias": alias,
                "prompt_text": completion_prompt,
                "backend": "llama-cpp-python",
            }
            stream = engine.create_completion(
                prompt=completion_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                repeat_penalty=1.05,
                stop=["\nUsuario:", "\nAsistente:", "\nuser:", "\nassistant:"],
                stream=True,
            )
            generated_text = ""
            for chunk in stream:
                text = str(chunk["choices"][0].get("text", "") or "")
                if not text:
                    continue
                generated_text += text
                yield {
                    "event": "token",
                    "assistant_id": assistant["assistant_id"],
                    "alias": alias,
                    "token_id": None,
                    "token_text": text,
                    "completion_text": generated_text,
                    "generated_text": generated_text,
                    "backend": "llama-cpp-python",
                }
            yield {
                "event": "done",
                "assistant_id": assistant["assistant_id"],
                "alias": alias,
                "completion_text": generated_text.strip(),
                "generated_text": generated_text.strip(),
                "backend": "llama-cpp-python",
            }
            return
        for item in self.stream_text(
            alias=alias,
            prompt=prompt,
            messages=chat_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            cache_mode=cache_mode,
            skip_special_tokens=skip_special_tokens,
        ):
            enriched = dict(item)
            enriched["assistant_id"] = assistant["assistant_id"]
            yield enriched

    def _model_info(self, alias: str) -> dict[str, Any]:
        return resolve_model_info(alias, self.root)

    def _uses_llama_cpp(self, alias: str) -> bool:
        info = self._model_info(alias)
        return (
            info.get("source_format") == "gguf"
            or info.get("inference_backend") == "llama-cpp-python"
        )

    def _engine(self, alias: str, *, cache_mode: str) -> GPT2StreamingEngine:
        key = (slugify(alias), cache_mode)
        engine = self._engines.get(key)
        if engine is None:
            if self._uses_llama_cpp(alias):
                target = load_generation_target(alias, workspace_root=self.root)
                engine = target.model
            else:
                export_dir = resolve_export_dir(alias, self.root)
                engine = GPT2StreamingEngine(export_dir, cache_mode=cache_mode)
            self._engines[key] = engine
        return engine

    def _tokenizer_dir(self, alias: str) -> Path:
        return resolve_tokenizer_dir(alias, self.root)

    def generate(
        self,
        *,
        alias: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if self._uses_llama_cpp(alias):
            raise NotImplementedError("token-id generation is not supported for GGUF llama.cpp aliases")
        engine = self._engine(alias, cache_mode=cache_mode)
        result = engine.generate_advanced(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            seed=seed,
        )
        if session_id is not None:
            session_dir = model_session_dir(alias, session_id, self.root)
            engine.save_session(
                session_dir,
                generated_ids=result["generated_ids"],
                last_logits=result.get("last_logits"),
            )
            result["session_id"] = slugify(session_id)
            result["session_dir"] = str(session_dir)
        return result

    def generate_text(
        self,
        *,
        alias: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        session_id: str | None = None,
        add_special_tokens: bool = False,
        skip_special_tokens: bool = True,
    ) -> dict[str, Any]:
        if self._uses_llama_cpp(alias):
            engine = self._engine(alias, cache_mode=cache_mode)
            chat_messages = messages
            if chat_messages is None and prompt is not None:
                chat_messages = [{"role": "user", "content": prompt}]
            if chat_messages:
                response = engine.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_k=top_k if do_sample else 0,
                    top_p=top_p if do_sample else 1.0,
                )
                completion_text = str(
                    dict(response["choices"][0].get("message", {}) or {}).get("content", "") or ""
                )
                prompt_text = render_messages_prompt(chat_messages)
            else:
                raise ValueError("prompt or messages is required")
            completion_text = completion_text.strip()
            result = {
                "alias": alias,
                "backend": "llama-cpp-python",
                "prompt_text": prompt_text,
                "prompt_ids": [],
                "new_ids": [],
                "generated_ids": [],
                "completion_text": completion_text,
                "generated_text": completion_text,
                "session_id": None,
            }
            if chat_messages:
                result["messages"] = chat_messages
            return result

        if messages:
            prompt_text = render_messages_prompt(messages)
        elif prompt is not None:
            prompt_text = prompt
        else:
            raise ValueError("prompt or messages is required")

        tokenizer_dir = self._tokenizer_dir(alias)
        prompt_ids = encode_text(
            tokenizer_dir,
            prompt_text,
            add_special_tokens=add_special_tokens,
        )
        if not prompt_ids:
            raise ValueError("tokenizer produced no prompt tokens")
        result = self.generate(
            alias=alias,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            cache_mode=cache_mode,
            session_id=session_id,
        )
        new_ids = result["new_ids"]
        result["prompt_text"] = prompt_text
        result["prompt_ids"] = prompt_ids
        result["completion_text"] = decode_tokens(
            tokenizer_dir,
            new_ids,
            skip_special_tokens=skip_special_tokens,
        )
        result["generated_text"] = decode_tokens(
            tokenizer_dir,
            result["generated_ids"],
            skip_special_tokens=skip_special_tokens,
        )
        if messages:
            result["messages"] = messages
        return result

    def openai_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        model = str(body.get("model") or body.get("alias") or "")
        if not model:
            raise ValueError("model is required")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
        helix_session_id = body.get("helix_session_id") or extra.get("helix_session_id")
        agent_id = body.get("agent_id") or extra.get("agent_id")
        session_id = str(helix_session_id or _session_id(f"{model}-{agent_id or 'chat'}"))
        memory_mode = str(extra.get("helix_memory_mode") or body.get("helix_memory_mode") or "off")
        memory_context: dict[str, Any] = {"mode": "off", "context": "", "tokens": 0, "memory_ids": [], "items": []}
        call_messages = list(messages)
        if memory_mode != "off":
            query = extra.get("helix_recall_query") or body.get("helix_recall_query")
            if not query:
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "user":
                        query = message.get("content")
                        break
            budget_tokens = int(extra.get("helix_memory_budget_tokens") or body.get("helix_memory_budget_tokens") or 800)
            memory_project = str(extra.get("helix_project") or body.get("helix_project") or "default")
            memory_agent = str(agent_id or extra.get("helix_agent_id") or body.get("helix_agent_id") or "chat")
            memory_context = hmem.build_context(
                root=self.root,
                project=memory_project,
                agent_id=memory_agent,
                query=None if query is None else str(query),
                budget_tokens=budget_tokens,
                mode=memory_mode,
            )
            if memory_context.get("context"):
                call_messages = [{"role": "system", "content": str(memory_context["context"])}, *call_messages]
        result = self.generate_text(
            alias=model,
            messages=call_messages,
            max_new_tokens=int(body.get("max_tokens", body.get("max_completion_tokens", 1))),
            do_sample=bool(body.get("temperature", 0.0) and float(body.get("temperature", 0.0)) > 0.0),
            temperature=float(body.get("temperature", 1.0)),
            top_p=float(body.get("top_p", 1.0)),
            seed=None if body.get("seed") is None else int(body["seed"]),
            cache_mode=str(extra.get("restore_policy") or body.get("restore_policy") or "session"),
            session_id=session_id,
            skip_special_tokens=True,
        )
        completion_text = str(result.get("completion_text") or "")
        prompt_tokens = len(result.get("prompt_ids") or [])
        completion_tokens = len(result.get("new_ids") or [])
        return {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": "length" if completion_tokens >= int(body.get("max_tokens", body.get("max_completion_tokens", 1))) else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "helix": {
                "session_id": result.get("session_id") or session_id,
                "agent_id": agent_id,
                "audit_policy": extra.get("audit_policy") or body.get("audit_policy"),
                "compression_mode": extra.get("compression_mode") or body.get("compression_mode"),
                "restore_policy": extra.get("restore_policy") or body.get("restore_policy") or "session",
                "prefix_reuse_status": result.get("prefix_reuse_status"),
                "session_dir": result.get("session_dir"),
                "memory_mode": memory_mode,
                "memory_context_tokens": memory_context.get("tokens", 0),
                "memory_ids": memory_context.get("memory_ids", []),
                "memory_item_count": len(memory_context.get("items", [])),
            },
        }

    def stream_text(
        self,
        *,
        alias: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        session_id: str | None = None,
        add_special_tokens: bool = False,
        skip_special_tokens: bool = True,
    ):
        if self._uses_llama_cpp(alias):
            engine = self._engine(alias, cache_mode=cache_mode)
            chat_messages = messages
            if chat_messages is None and prompt is not None:
                chat_messages = [{"role": "user", "content": prompt}]
            if chat_messages:
                prompt_text = render_messages_prompt(chat_messages)
                stream = engine.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_k=top_k if do_sample else 0,
                    top_p=top_p if do_sample else 1.0,
                    stream=True,
                )
                extractor = lambda chunk: str(  # noqa: E731
                    dict(
                        dict(chunk["choices"][0].get("delta", {}) or {}).get("message", {})
                        if isinstance(chunk["choices"][0].get("delta"), dict)
                        else {}
                    ).get("content", "")
                    or dict(chunk["choices"][0].get("delta", {}) or {}).get("content", "")
                    or ""
                )
            else:
                raise ValueError("prompt or messages is required")

            generated_text = ""
            yield {
                "event": "start",
                "alias": alias,
                "prompt_text": prompt_text,
                "prompt_ids": [],
                "session_id": None,
                "backend": "llama-cpp-python",
            }
            for chunk in stream:
                text = extractor(chunk)
                if not text:
                    continue
                generated_text += text
                yield {
                    "event": "token",
                    "token_id": None,
                    "token_text": text,
                    "generated_ids": [],
                    "new_ids": [],
                    "completion_text": generated_text,
                    "generated_text": generated_text,
                    "backend": "llama-cpp-python",
                }
            yield {
                "event": "done",
                "alias": alias,
                "prompt_text": prompt_text,
                "prompt_ids": [],
                "generated_ids": [],
                "new_ids": [],
                "completion_text": generated_text,
                "generated_text": generated_text,
                "session_id": None,
                "backend": "llama-cpp-python",
            }
            return

        if messages:
            prompt_text = render_messages_prompt(messages)
        elif prompt is not None:
            prompt_text = prompt
        else:
            raise ValueError("prompt or messages is required")

        tokenizer_dir = self._tokenizer_dir(alias)
        prompt_ids = encode_text(
            tokenizer_dir,
            prompt_text,
            add_special_tokens=add_special_tokens,
        )
        if not prompt_ids:
            raise ValueError("tokenizer produced no prompt tokens")

        engine = self._engine(alias, cache_mode=cache_mode)
        prompt_len = len(prompt_ids)
        last_logits = None
        current_generated = list(prompt_ids)

        yield {
            "event": "start",
            "alias": alias,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "session_id": slugify(session_id) if session_id else None,
        }

        for event in engine.stream_generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            seed=seed,
        ):
            last_logits = event["last_logits"]
            if event["phase"] != "generated":
                continue
            generated_ids = event["generated_ids"]
            current_generated = list(generated_ids)
            new_ids = generated_ids[prompt_len:]
            token_id = int(event["token_id"])
            token_text = decode_tokens(
                tokenizer_dir,
                [token_id],
                skip_special_tokens=skip_special_tokens,
            )
            yield {
                "event": "token",
                "token_id": token_id,
                "token_text": token_text,
                "generated_ids": generated_ids,
                "new_ids": new_ids,
                "completion_text": decode_tokens(
                    tokenizer_dir,
                    new_ids,
                    skip_special_tokens=skip_special_tokens,
                ),
                "generated_text": decode_tokens(
                    tokenizer_dir,
                    generated_ids,
                    skip_special_tokens=skip_special_tokens,
                ),
            }

        final_generated = current_generated
        final_new_ids = final_generated[prompt_len:]
        if session_id is not None:
            session_dir = model_session_dir(alias, session_id, self.root)
            engine.save_session(
                session_dir,
                generated_ids=final_generated,
                last_logits=last_logits,
            )
        yield {
            "event": "done",
            "alias": alias,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "generated_ids": final_generated,
            "new_ids": final_new_ids,
            "completion_text": decode_tokens(
                tokenizer_dir,
                final_new_ids,
                skip_special_tokens=skip_special_tokens,
            ),
            "generated_text": decode_tokens(
                tokenizer_dir,
                final_generated,
                skip_special_tokens=skip_special_tokens,
            ),
            "session_id": slugify(session_id) if session_id else None,
        }

    def resume(
        self,
        *,
        alias: str,
        session_id: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        save_session: bool = True,
    ) -> dict[str, Any]:
        if self._uses_llama_cpp(alias):
            raise NotImplementedError("resume is not supported for GGUF llama.cpp aliases")
        session_dir = model_session_dir(alias, session_id, self.root)
        engine = self._engine(alias, cache_mode=cache_mode)
        result = engine.resume_advanced(
            session_dir,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            seed=seed,
        )
        if save_session:
            engine.save_session(
                session_dir,
                generated_ids=result["generated_ids"],
                last_logits=result.get("last_logits"),
            )
        result["session_id"] = slugify(session_id)
        result["session_dir"] = str(session_dir)
        return result

    def resume_text(
        self,
        *,
        alias: str,
        session_id: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        save_session: bool = True,
        skip_special_tokens: bool = True,
    ) -> dict[str, Any]:
        tokenizer_dir = self._tokenizer_dir(alias)
        result = self.resume(
            alias=alias,
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            cache_mode=cache_mode,
            save_session=save_session,
        )
        result["completion_text"] = decode_tokens(
            tokenizer_dir,
            result["new_ids"],
            skip_special_tokens=skip_special_tokens,
        )
        result["generated_text"] = decode_tokens(
            tokenizer_dir,
            result["generated_ids"],
            skip_special_tokens=skip_special_tokens,
        )
        return result

    def tool_manifest(self) -> list[dict[str, Any]]:
        if self._tools is None:
            self._tools = build_runtime_tool_registry(self)
        return self._tools.manifest()

    def research_artifact_manifest(self) -> list[dict[str, Any]]:
        return _research_artifact_manifest()

    def research_artifact(self, name: str) -> dict[str, Any]:
        payload = _load_research_artifact(name)
        return {
            "name": str(name),
            "title": _artifact_title(str(name)),
            "payload": payload,
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._tools is None:
            self._tools = build_runtime_tool_registry(self)
        return self._tools.call(name, arguments)

    def agent_runner(self) -> AgentRunner:
        return AgentRunner(self, root=self.root)

    def memory_observe(self, body: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(body.get("agent_id") or body.get("agent_name") or "default-agent")
        compressor = hmem.runtime_compress_fn(self, body.get("memory_compressor_alias"))
        return hmem.observe_event(
            root=self.root,
            project=str(body.get("project") or hmem.DEFAULT_PROJECT),
            agent_id=agent_id,
            session_id=None if body.get("session_id") is None else str(body.get("session_id")),
            event_type=str(body.get("event_type") or body.get("observation_type") or "manual"),
            content=str(body.get("content") or body.get("text") or ""),
            summary=None if body.get("summary") is None else str(body.get("summary")),
            tags=[str(item) for item in body.get("tags", [])] if isinstance(body.get("tags"), list) else None,
            importance=int(body.get("importance", 5)),
            promote=bool(body.get("promote", True)),
            memory_type=str(body.get("memory_type") or "episodic"),
            compress_fn=compressor,
        )

    def memory_search(self, body: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(body.get("agent_id") or body.get("agent_name") or "default-agent")
        query = str(body["query"])
        top_k = int(body.get("top_k", body.get("limit", 5)))
        project = str(body.get("project") or hmem.DEFAULT_PROJECT)
        session_id = None if body.get("session_id") is None else str(body.get("session_id"))
        retrieval_scope = str(body.get("retrieval_scope") or "workspace")
        if bool(body.get("hybrid", False)) or str(body.get("mode", "")).lower() == "hybrid":
            return hmem.hybrid_search(
                root=self.root,
                project=project,
                agent_id=agent_id,
                session_id=session_id,
                query=query,
                top_k=top_k,
                retrieval_scope=retrieval_scope,
            )
        return hmem.search(
            root=self.root,
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            query=query,
            top_k=top_k,
            retrieval_scope=retrieval_scope,
        )

    def memory_context(self, body: dict[str, Any]) -> dict[str, Any]:
        return hmem.build_context(
            root=self.root,
            project=str(body.get("project") or hmem.DEFAULT_PROJECT),
            agent_id=str(body.get("agent_id") or body.get("agent_name") or "default-agent"),
            session_id=None if body.get("session_id") is None else str(body.get("session_id")),
            query=None if body.get("query") is None else str(body.get("query")),
            budget_tokens=int(body.get("budget_tokens", body.get("helix_memory_budget_tokens", 800))),
            mode=str(body.get("mode") or body.get("helix_memory_mode") or "search"),
            limit=int(body.get("limit", body.get("top_k", 5))),
            retrieval_scope=str(body.get("retrieval_scope") or "workspace"),
        )

    def memory_graph(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = body or {}
        return hmem.graph(
            root=self.root,
            project=None if payload.get("project") is None else str(payload.get("project")),
            agent_id=None if payload.get("agent_id") is None and payload.get("agent_name") is None else str(payload.get("agent_id") or payload.get("agent_name")),
            limit=int(payload.get("limit", 50)),
        )

    def memory_stats(self) -> dict[str, Any]:
        return hmem.stats(root=self.root)

    def prewarm_assistants(self) -> list[dict[str, Any]]:
        warmed: list[dict[str, Any]] = []
        seen_aliases: set[str] = set()
        for assistant in self.list_assistants():
            alias = str(assistant.get("alias") or "").strip()
            if not alias or alias in seen_aliases or not bool(assistant.get("available")):
                continue
            seen_aliases.add(alias)
            started = time.perf_counter()
            engine = self._engine(alias, cache_mode="session")
            if self._uses_llama_cpp(alias):
                try:
                    warm_messages = [
                        {"role": "system", "content": "Sos un asistente breve."},
                        {"role": "user", "content": "hola"},
                    ]
                    response = self._assistant_llama_chat_completion(
                        engine,
                        messages=warm_messages,
                        max_new_tokens=1,
                        stream=False,
                    )
                    if response is None:
                        engine.create_completion(
                            prompt="Sos un asistente breve.\nUsuario: hola\nAsistente:",
                            max_tokens=1,
                            temperature=0.0,
                            top_k=0,
                            top_p=1.0,
                            stop=["\nUsuario:", "\nAsistente:"],
                        )
                except Exception:  # noqa: BLE001
                    pass
            warmed.append(
                {
                    "assistant_id": str(assistant["assistant_id"]),
                    "alias": alias,
                    "backend": "llama-cpp-python" if self._uses_llama_cpp(alias) else "transformers",
                    "prewarm_time_s": round(time.perf_counter() - started, 3),
                }
            )
        return warmed


class _HelixHandler(BaseHTTPRequestHandler):
    runtime: HelixRuntime

    def _apply_cors_headers(self) -> None:
        allowed_origin = _allowed_origin(self.headers.get("Origin"))
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_json_ready(payload), indent=2).encode("utf-8")
        self.send_response(status)
        self._apply_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body: bytes, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self._apply_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._apply_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_sse_event(self, name: str, payload: Any) -> None:
        body = f"event: {name}\ndata: {json.dumps(_json_ready(payload), ensure_ascii=True)}\n\n".encode(
            "utf-8"
        )
        self.wfile.write(body)
        self.wfile.flush()

    def _write_sse_data(self, payload: Any) -> None:
        if isinstance(payload, str):
            body = f"data: {payload}\n\n".encode("utf-8")
        else:
            body = f"data: {json.dumps(_json_ready(payload), ensure_ascii=True)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _send_error(self, message: str, *, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message}, status=status)

    def _serve_path(self, *, root: Path, relative_path: str) -> None:
        path = (root / relative_path).resolve()
        if not str(path).startswith(str(root.resolve())) or not path.exists():
            self._send_error("route not found", status=HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(path.name)
        self._send_text(
            path.read_bytes(),
            content_type=content_type or "application/octet-stream",
        )

    def _serve_static(self, relative_path: str) -> None:
        self._serve_path(root=_web_root(), relative_path=relative_path)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._apply_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._serve_static("index.html")
                return
            if path == "/app":
                self._serve_path(root=_legacy_frontend_root(), relative_path="index.html")
                return
            if path in {"/research", "/frontier"}:
                self._serve_static("research.html")
                return
            if path == "/meta-demo":
                self._serve_static("meta-demo.html")
                return
            if path == "/meta-demo-real-cached":
                self._serve_static("meta-demo-real-cached.html")
                return
            if path.startswith("/static/"):
                self._serve_path(root=_web_static_root(), relative_path=path.removeprefix("/static/"))
                return
            if path == "/health":
                self._send_json({"status": "ok", "workspace_root": str(self.runtime.root)})
                return
            if path == "/research/artifacts":
                self._send_json({"artifacts": self.runtime.research_artifact_manifest()})
                return
            if path.startswith("/research/artifacts/"):
                artifact_name = path.removeprefix("/research/artifacts/")
                self._send_json(self.runtime.research_artifact(artifact_name))
                return
            if path == "/models":
                self._send_json({"models": self.runtime.list_models()})
                return
            if path == "/assistants":
                self._send_json({"assistants": self.runtime.list_assistants()})
                return
            if path == "/tools":
                self._send_json({"tools": self.runtime.tool_manifest()})
                return
            if path == "/agent/knowledge/search":
                params = parse_qs(urlparse(self.path).query)
                agent_name = params.get("agent_name", ["default-agent"])[0]
                goal = params.get("query", [""])[0]
                top_k = int(params.get("top_k", ["5"])[0])
                self._send_json(self.runtime.agent_runner().search_knowledge(agent_name, goal, top_k=top_k))
                return
            if path == "/memory/graph":
                params = parse_qs(urlparse(self.path).query)
                self._send_json(
                    self.runtime.memory_graph(
                        {
                            "project": params.get("project", [hmem.DEFAULT_PROJECT])[0],
                            "agent_id": params.get("agent_id", [None])[0],
                            "limit": int(params.get("limit", ["50"])[0]),
                        }
                    )
                )
                return
            if path == "/memory/stats":
                self._send_json(self.runtime.memory_stats())
                return
            if path.startswith("/models/"):
                alias = path.removeprefix("/models/")
                self._send_json(self.runtime.model_info(alias))
                return
            self._send_error("route not found", status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_error(str(exc), status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_error(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json()
        try:
            if path == "/v1/chat/completions":
                self._send_json(self.runtime.openai_chat_completion(body))
                return
            if path == "/memory/observe":
                self._send_json(self.runtime.memory_observe(body), status=HTTPStatus.CREATED)
                return
            if path == "/memory/search":
                self._send_json(self.runtime.memory_search(body))
                return
            if path == "/memory/context":
                self._send_json(self.runtime.memory_context(body))
                return
            if path == "/memory/graph":
                self._send_json(self.runtime.memory_graph(body))
                return
            if path == "/prepare":
                self._send_json(
                    self.runtime.prepare_model(
                        model_ref=body["model_ref"],
                        alias=body.get("alias"),
                        block_rows=int(body.get("block_rows", 256)),
                        compress=body.get("compress"),
                        local_files_only=bool(body.get("local_files_only", False)),
                        trust_remote_code=bool(body.get("trust_remote_code", False)),
                        force=bool(body.get("force", False)),
                        chat_format=body.get("chat_format"),
                        n_ctx=int(body.get("n_ctx", 4096)),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/generate":
                save_session = bool(body.get("save_session", False))
                session_id = body.get("session_id")
                if save_session and not session_id:
                    session_id = _session_id(body["alias"])
                self._send_json(
                    self.runtime.generate(
                        alias=body["alias"],
                        prompt_ids=[int(item) for item in body["prompt_ids"]],
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        session_id=session_id,
                    )
                )
                return
            if path == "/chat":
                save_session = bool(body.get("save_session", False))
                session_id = body.get("session_id")
                if save_session and not session_id:
                    session_id = _session_id(body["alias"])
                self._send_json(
                    self.runtime.generate_text(
                        alias=body["alias"],
                        prompt=body.get("prompt"),
                        messages=body.get("messages"),
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        session_id=session_id,
                        add_special_tokens=bool(body.get("add_special_tokens", False)),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    )
                )
                return
            if path == "/chat/stream":
                save_session = bool(body.get("save_session", False))
                session_id = body.get("session_id")
                if save_session and not session_id:
                    session_id = _session_id(body["alias"])
                self._send_sse_headers()
                try:
                    for item in self.runtime.stream_text(
                        alias=body["alias"],
                        prompt=body.get("prompt"),
                        messages=body.get("messages"),
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        session_id=session_id,
                        add_special_tokens=bool(body.get("add_special_tokens", False)),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    ):
                        event_name = str(item.get("event", "message"))
                        self._write_sse_event(event_name, item)
                except Exception as exc:  # noqa: BLE001
                    self._write_sse_event("error", {"error": str(exc)})
                self.close_connection = True
                return
            if path == "/assistants/chat/stream":
                self._send_sse_headers()
                try:
                    cleaner = _AssistantTokenCleaner()
                    for item in self.runtime.stream_assistant_text(
                        assistant_id=body["assistant_id"],
                        message=body.get("message"),
                        messages=body.get("messages"),
                        max_new_tokens=int(body.get("max_new_tokens", 64)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    ):
                        if str(item.get("event")) == "token":
                            clean_text = cleaner.push(str(item.get("token_text", "")))
                            if clean_text:
                                self._write_sse_data({"token": clean_text})
                    tail = cleaner.finish()
                    if tail:
                        self._write_sse_data({"token": tail})
                    self._write_sse_data("[DONE]")
                except Exception as exc:  # noqa: BLE001
                    self._write_sse_data({"error": str(exc)})
                self.close_connection = True
                return
            if path == "/resume":
                self._send_json(
                    self.runtime.resume(
                        alias=body["alias"],
                        session_id=body["session_id"],
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        save_session=bool(body.get("save_session", True)),
                    )
                )
                return
            if path == "/chat/resume":
                self._send_json(
                    self.runtime.resume_text(
                        alias=body["alias"],
                        session_id=body["session_id"],
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        save_session=bool(body.get("save_session", True)),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    )
                )
                return
            if path == "/agent/knowledge/add-text":
                self._send_json(
                    self.runtime.agent_runner().add_knowledge_text(
                        body.get("agent_name", "default-agent"),
                        body["text"],
                        source=body.get("source", "inline-text"),
                        metadata=body.get("metadata"),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/agent/knowledge/add-file":
                self._send_json(
                    self.runtime.agent_runner().add_knowledge_file(
                        body.get("agent_name", "default-agent"),
                        body["file_path"],
                        metadata=body.get("metadata"),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/agent/memory/search":
                self._send_json(self.runtime.memory_search(body | {"agent_id": body.get("agent_name", "default-agent")}))
                return
            if path == "/agent/run":
                self._send_json(
                    self.runtime.agent_runner().run(
                        goal=body["goal"],
                        agent_name=body.get("agent_name", "default-agent"),
                        default_model_alias=body.get("default_model_alias"),
                        local_planner_alias=body.get("local_planner_alias"),
                        remote_model_ref=body.get("remote_model_ref"),
                        prefer_remote=bool(body.get("prefer_remote", False)),
                        trust_remote_code=bool(body.get("trust_remote_code", False)),
                        max_steps=int(body.get("max_steps", 4)),
                        generation_max_new_tokens=int(body.get("generation_max_new_tokens", 128)),
                        memory_project=str(body.get("memory_project", hmem.DEFAULT_PROJECT)),
                        memory_mode=str(body.get("memory_mode", "search")),
                        memory_budget_tokens=int(body.get("memory_budget_tokens", 800)),
                        memory_compressor_alias=body.get("memory_compressor_alias"),
                    )
                )
                return
            if path == "/agent/run/stream":
                self._send_sse_headers()
                try:
                    for item in self.runtime.agent_runner().run_stream(
                        goal=body["goal"],
                        agent_name=body.get("agent_name", "default-agent"),
                        default_model_alias=body.get("default_model_alias"),
                        local_planner_alias=body.get("local_planner_alias"),
                        remote_model_ref=body.get("remote_model_ref"),
                        prefer_remote=bool(body.get("prefer_remote", False)),
                        trust_remote_code=bool(body.get("trust_remote_code", False)),
                        max_steps=int(body.get("max_steps", 4)),
                        generation_max_new_tokens=int(body.get("generation_max_new_tokens", 128)),
                        memory_project=str(body.get("memory_project", hmem.DEFAULT_PROJECT)),
                        memory_mode=str(body.get("memory_mode", "search")),
                        memory_budget_tokens=int(body.get("memory_budget_tokens", 800)),
                        memory_compressor_alias=body.get("memory_compressor_alias"),
                    ):
                        event_name = str(item.get("event", "message"))
                        self._write_sse_event(event_name, item)
                except Exception as exc:  # noqa: BLE001
                    self._write_sse_event("error", {"error": str(exc)})
                self.close_connection = True
                return
            if path.startswith("/tools/"):
                tool_name = path.removeprefix("/tools/")
                self._send_json(self.runtime.call_tool(tool_name, body))
                return
            self._send_error("route not found", status=HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_error(f"missing field: {exc.args[0]}")
        except FileNotFoundError as exc:
            self._send_error(str(exc), status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_error(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def create_api_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    root: str | Path | None = None,
) -> ThreadingHTTPServer:
    runtime = HelixRuntime(root=root)
    if os.environ.get("HELIX_PREWARM_ASSISTANTS", "1") not in {"0", "false", "False"}:
        runtime.prewarm_assistants()
    handler = type("HelixHandler", (_HelixHandler,), {"runtime": runtime})
    return ThreadingHTTPServer((host, port), handler)


def serve_api(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    root: str | Path | None = None,
) -> None:
    server = create_api_server(host=host, port=port, root=root)
    try:
        server.serve_forever()
    finally:
        server.server_close()
