from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_claim_lint_module():
    module_path = REPO_ROOT / "tools" / "helix_claim_lint.py"
    spec = importlib.util.spec_from_file_location("helix_claim_lint", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_evidence_index_is_consistent() -> None:
    claim_lint = _load_claim_lint_module()
    issues = claim_lint.validate_public_evidence_index(REPO_ROOT / "evidence" / "index.json")
    assert issues == []


def test_public_docs_are_complete_for_curated_layer() -> None:
    claim_lint = _load_claim_lint_module()
    issues = claim_lint.validate_public_docs(REPO_ROOT)
    assert issues == []


def test_verification_public_index_is_bridge_to_curated_layer() -> None:
    payload = json.loads((REPO_ROOT / "verification" / "public-evidence-index.json").read_text(encoding="utf-8-sig"))
    assert payload["status"] == "deprecated_bridge"
    assert payload["canonical_public_index"] == "evidence/index.json"
    assert "CLAIMS.md" in payload["public_docs"]
