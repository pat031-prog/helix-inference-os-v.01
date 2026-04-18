from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.run_local_hybrid_stress import _json_ready, _write_json


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
        agent_id = extra.get("agent_id") or body.get("agent_id") or "agent"
        session_id = extra.get("helix_session_id") or body.get("helix_session_id") or f"{agent_id}-session"
        payload = {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "gpt2"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Observation: {agent_id} used an OpenAI-compatible HeliX session endpoint.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 16, "completion_tokens": 10, "total_tokens": 26},
        "helix": {
            "session_id": session_id,
            "agent_id": agent_id,
            "audit_policy": extra.get("audit_policy") or body.get("audit_policy"),
            "restore_policy": extra.get("restore_policy") or body.get("restore_policy"),
            "prefix_reuse_status": "mock_contract",
        },
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def _start_mock_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


def _run_openai_client(*, base_url: str, model: str, api_key: str) -> list[dict[str, Any]]:
    agents = [
        {
            "agent_id": "code_reviewer",
            "session_id": "framework-code-reviewer",
            "content": "Review the HeliX prefix reuse claim and keep it precise.",
        },
        {
            "agent_id": "release_writer",
            "session_id": "framework-release-writer",
            "content": "Turn the prior review into a short release note.",
        },
    ]
    events: list[dict[str, Any]] = []
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
    except Exception:  # noqa: BLE001
        client = None

    for agent in agents:
        extra_body = {
                "agent_id": agent["agent_id"],
                "helix_session_id": agent["session_id"],
                "audit_policy": "deferred",
                "restore_policy": "session",
                "compression_mode": "turbo-int8-hadamard",
        }
        if client is not None:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": agent["content"]}],
                max_tokens=24,
                temperature=0,
                extra_body=extra_body,
            )
            message = response.choices[0].message.content or ""
            raw = response.model_dump() if hasattr(response, "model_dump") else json.loads(response.json())
        else:
            request_body = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": agent["content"]}],
                    "max_tokens": 24,
                    "temperature": 0,
                    "extra_body": extra_body,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=request_body,
                headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                raw = json.loads(resp.read().decode("utf-8"))
            message = raw["choices"][0]["message"]["content"]
        events.append(
            {
                "agent_id": agent["agent_id"],
                "session_id": agent["session_id"],
                "answer_text": message,
                "answer_preview": " ".join(message.split())[:240],
                "helix": raw.get("helix"),
                "object": raw.get("object"),
            }
        )
    return events


def _run_langchain_client(*, base_url: str, model: str, api_key: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        from langchain_openai import ChatOpenAI
    except Exception:  # noqa: BLE001
        return "skipped_dependency_missing", []
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0, max_tokens=24)
    response = llm.invoke("Write one safe HeliX Session OS claim.")
    return "completed", [{"agent_id": "langchain_smoke", "answer_text": str(response.content), "answer_preview": str(response.content)[:240]}]


def _run_crewai_client(*, base_url: str, model: str, api_key: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        from crewai import Agent, Crew, LLM, Task
    except Exception:  # noqa: BLE001
        return "skipped_dependency_missing", []

    llm = LLM(
        model=f"openai/{model}",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=24,
    )
    reviewer = Agent(
        role="HeliX code reviewer",
        goal="Find the safest public claim for a local inference-session OS demo.",
        backstory="You protect evidence wording and avoid overclaiming.",
        llm=llm,
        verbose=False,
    )
    task = Task(
        description="Write one careful sentence about HeliX using OpenAI-compatible session metadata.",
        expected_output="One precise sentence.",
        agent=reviewer,
    )
    crew = Crew(agents=[reviewer], tasks=[task], verbose=False)
    result = crew.kickoff()
    text = str(result)
    return "completed", [{"agent_id": "crewai_smoke", "answer_text": text, "answer_preview": text[:240]}]


def run_agent_framework_showcase(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server: ThreadingHTTPServer | None = None
    base_url = str(args.base_url)
    server_mode = "external-localhost"
    if bool(args.mock_server):
        server, base_url = _start_mock_server()
        server_mode = "mock-openai-compatible-contract"
    try:
        if str(args.client) == "langchain":
            status, events = _run_langchain_client(base_url=base_url, model=str(args.model), api_key=str(args.api_key))
            skip_reason = None if status == "completed" else "langchain_openai_not_installed"
        elif str(args.client) == "crewai":
            status, events = _run_crewai_client(base_url=base_url, model=str(args.model), api_key=str(args.api_key))
            skip_reason = None if status == "completed" else "crewai_not_installed"
        else:
            try:
                events = _run_openai_client(base_url=base_url, model=str(args.model), api_key=str(args.api_key))
                status = "completed"
                skip_reason = None
            except Exception as exc:  # noqa: BLE001
                events = []
                status = "skipped_server_unavailable"
                skip_reason = str(exc)
        payload = {
            "title": "HeliX Agent Framework Showcase",
            "benchmark_kind": "session-os-agent-framework-showcase-v1",
            "status": status,
            "client": str(args.client),
            "model": str(args.model),
            "base_url": base_url,
            "server_mode": server_mode,
            "skip_reason": skip_reason,
            "events": events,
            "agent_count": len(events),
            "supported_clients": ["openai", "langchain_optional", "crewai_optional"],
            "claim_boundary": "This showcase validates OpenAI-compatible client plumbing; it is not a model-quality benchmark.",
        }
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    _write_json(output_dir / "local-agent-framework-showcase.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small external-agent-framework showcase against HeliX's OpenAI-compatible endpoint.")
    parser.add_argument("--client", default="openai", choices=["openai", "langchain", "crewai"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="helix-local")
    parser.add_argument("--mock-server", action="store_true")
    parser.add_argument("--output-dir", default="verification")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_agent_framework_showcase(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
