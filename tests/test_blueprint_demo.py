from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix_proto.blueprints import load_blueprint, quality_check_html, render_meta_microsite, sanitize_model_text
from tools import run_local_blueprint_demo


def test_blueprint_parser_loads_meta_microsite() -> None:
    blueprint = load_blueprint("blueprints/meta-microsite.json")

    assert blueprint.blueprint_id == "meta-microsite"
    assert len(blueprint.payload["tasks"]) == 4
    assert blueprint.payload["outputs"]["artifact"] == "local-blueprint-meta-microsite-demo.json"


def test_blueprint_parser_rejects_unknown_agent(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "bad",
                "title": "Bad",
                "models": {"m": {"model_id": "m"}},
                "agents": {"a": {"model": "m"}},
                "tasks": [{"task_id": "x", "agent": "missing", "slot": "copy"}],
                "memory_policy": {},
                "session_policy": {},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_blueprint(path)


def test_renderer_sanitizes_model_text_and_rejects_bad_content() -> None:
    assert "<script" not in sanitize_model_text("<script>alert(1)</script>Safe copy")
    assert "```" not in sanitize_model_text("```html\n<div>bad</div>\n```")


def test_meta_renderer_quality_checks_pass_for_complete_artifact() -> None:
    artifact = {
        "mode": "mock-only",
        "public_claim_level": "orchestration-and-renderer",
        "content_slots": {
            "architecture_plan": "HeliX is an Inference OS with private state and shared memory.",
            "editorial_copy": "The scheduler governs models while hmem carries portable meaning.",
            "layout_notes": "Render a compact editorial page with SVG, timeline and telemetry.",
            "editorial_review": "Approved with caveats about fallback content.",
        },
        "agents": [{"agent_id": "architect"}, {"agent_id": "developer"}],
        "task_timeline": [{"agent_id": "architect", "task_id": "plan", "handoff_summary": "Plan complete.", "model_id": "mock"}],
        "private_state_events": [{"event": "session_saved"}],
        "hmem_events": [{"memory_id": "mem-1"}],
        "memory_graph": {"node_count": 3, "edge_count": 2},
        "scheduler_decisions": [{"task_id": "plan", "selected_model_id": "mock", "session_restored": False, "estimated_cost_ms": 1}],
        "final_audit_status": "verified",
    }

    html = render_meta_microsite(artifact)
    quality = quality_check_html(html)

    assert quality["status"] == "passed"
    assert "HeliX is an Inference OS" in html


def test_meta_renderer_quality_rejects_visible_slot_markers() -> None:
    quality = quality_check_html(
        "HeliX is an Inference OS Four layers Meta Build Timeline Shared memory graph "
        "Claims & caveats Footer Log Four HeliX Inference OS layers [layout-slots]"
    )

    assert quality["status"] == "failed"
    assert quality["contains_visible_slot_marker"] is True


def test_blueprint_runner_mock_generates_artifact_and_html(tmp_path: Path) -> None:
    site_output = tmp_path / "site-dist" / "meta-demo.html"
    payload = run_local_blueprint_demo.run_blueprint_demo(
        run_local_blueprint_demo.build_parser().parse_args(
            [
                "--mode",
                "mock-only",
                "--output-dir",
                str(tmp_path),
                "--site-output",
                str(site_output),
            ]
        )
    )

    assert payload["status"] == "completed"
    assert payload["quality_checks"]["status"] == "passed"
    assert payload["fallback_content_used"] is True
    assert payload["final_audit_status"] == "verified"
    assert payload["memory_graph"]["node_count"] > 0
    assert (tmp_path / "local-blueprint-meta-microsite-demo.json").exists()
    assert site_output.exists()
    assert "Meta Build Timeline" in site_output.read_text(encoding="utf-8")


def test_blueprint_runner_real_only_skips_without_prepared_aliases(tmp_path: Path) -> None:
    payload = run_local_blueprint_demo.run_blueprint_demo(
        run_local_blueprint_demo.build_parser().parse_args(
            [
                "--mode",
                "real-only",
                "--output-dir",
                str(tmp_path),
                "--site-output",
                str(tmp_path / "meta-demo.html"),
            ]
        )
    )

    assert payload["status"] in {"skipped_model_not_cached", "completed"}
