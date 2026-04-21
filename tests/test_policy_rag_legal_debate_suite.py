from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

import tools.run_policy_rag_legal_debate_suite_v1 as suite


def _requires_rust_extension() -> None:
    try:
        cls = suite._rust_indexed_dag_class()
        if not hasattr(cls(), "build_context_fast"):
            pytest.skip("Rust extension was not rebuilt with build_context_fast")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _workspace() -> Path:
    workspace = (
        Path.cwd()
        / "verification"
        / "nuclear-methodology"
        / "_pytest"
        / "policy-rag-legal-debate"
        / uuid.uuid4().hex
    )
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _insert_meta(cur: sqlite3.Cursor, row_id: int, values: dict[str, str]) -> None:
    for key, value in values.items():
        cur.execute(
            "insert into embedding_metadata(id, key, string_value, int_value, float_value, bool_value) values (?, ?, ?, null, null, null)",
            (row_id, key, value),
        )


def _fixture_repo(workspace: Path) -> Path:
    source = workspace / "rag_polizas"
    (source / "data" / "pdfs").mkdir(parents=True)
    (source / "data" / "data").mkdir(parents=True)
    (source / "vectorstore").mkdir(parents=True)
    (source / "data" / "pdfs" / "Constancia_de_Poliza_10985500_0_15-04-2026.pdf").write_bytes(b"%PDF fixture latest")
    (source / "data" / "pdfs" / "Poliza_10985500_0_13-04-2026.pdf").write_bytes(b"%PDF fixture stale")
    (source / "data" / "pdfs" / "VisualStudioSetup.exe").write_bytes(b"not policy evidence")
    (source / "data" / "data" / "policy_metadata.json").write_text(
        json.dumps(
            {
                "poliza_nro": "17418",
                "dominio": "AG141EZ",
                "asegurado": "si no reclama dentro de",
                "confidence_extraction": "baja",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    con = sqlite3.connect(source / "vectorstore" / "chroma.sqlite3")
    try:
        cur = con.cursor()
        cur.execute(
            "create table embedding_metadata(id integer, key text, string_value text, int_value integer, float_value real, bool_value integer)"
        )
        rows = [
            {
                "source": "Constancia_de_Poliza_10985500_0_15-04-2026.pdf",
                "page_number": "1",
                "section_type": "administrativo",
                "heading_path": "Ref: 10985500 Poliza nro. 13434982",
                "policy_id": "17418",
                "chroma:document": "Asegurado Certificado de cobertura de poliza VALBUSA PATRICIO DANIEL Ref: 10985500 Poliza nro. 13434982 VIGENCIA",
            },
            {
                "source": "Constancia_de_Poliza_10985500_0_15-04-2026.pdf",
                "page_number": "1",
                "section_type": "vigencia",
                "heading_path": "Desde las 12 hs 3/4/2026",
                "policy_id": "17418",
                "chroma:document": "Desde las 12 hs 3/4/2026 Hasta las 12 hs 3/5/2026 (*)",
            },
            {
                "source": "Constancia_de_Poliza_10985500_0_15-04-2026.pdf",
                "page_number": "1",
                "section_type": "general",
                "heading_path": "CITROEN BERLINGO FURGON 1.6I BUSINESS",
                "policy_id": "17418",
                "chroma:document": "CITROEN BERLINGO FURGON 1.6I BUSINESS Dominio: AG141EZ Modelo: 2023",
            },
            {
                "source": "Constancia_de_Poliza_10985500_0_15-04-2026.pdf",
                "page_number": "2",
                "section_type": "cobertura",
                "heading_path": "CA-RH 5.4 Cobertura de las ruedas",
                "policy_id": "17418",
                "chroma:document": "Cobertura de Ruedas: Reposicion limitada a una rueda y un evento por vigencia de contrato (s/CA-RH 5.4).",
            },
            {
                "source": "Poliza_10985500_0_13-04-2026.pdf",
                "page_number": "5",
                "section_type": "exclusion",
                "heading_path": "CG-DA 2.1 Exclusiones a la cobertura para Danos",
                "policy_id": "17418",
                "chroma:document": "El Asegurador no indemnizara los siniestros originados o derivados de la energia nuclear.",
            },
            {
                "source": "Poliza_10985500_0_13-04-2026.pdf",
                "page_number": "1",
                "section_type": "vigencia",
                "heading_path": "Asegurado VALBUSA PATRICIO DANIEL",
                "policy_id": "17418",
                "chroma:document": "Asegurado VALBUSA PATRICIO DANIEL Desde las 12 hs 3/7/2025 Hasta las 12 hs 3/8/2025.",
            },
        ]
        for index, row in enumerate(rows, start=1):
            _insert_meta(cur, index, row)
        con.commit()
    finally:
        con.close()
    return source


def _args(source: Path, workspace: Path, *, case: str = "all") -> Namespace:
    return Namespace(
        case=case,
        source_repo=str(source),
        output_dir=str(workspace / "out"),
        run_id="pytest-policy-rag-legal-debate",
        repeats=1,
        min_chunks=5,
        min_pdf_count=2,
        use_deepinfra=False,
        client_model="qwen/test",
        insurer_model="llama/test",
        auditor_model="claude/test",
        tokens=512,
    )


def test_corpus_chain_flags_bad_metadata_and_binary_noise() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        source = _fixture_repo(workspace)
        args = _args(source, workspace)
        artifact = suite._case_corpus_chain(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["binary_noise_detected_and_not_authoritative"] is True
        assert gates["metadata_asegurado_conflict_flagged"] is True
        assert gates["metadata_policy_number_conflict_flagged"] is True
        assert gates["helix_identity_lane_native_verified"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_recency_identity_uses_latest_cited_evidence_not_bad_metadata() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        source = _fixture_repo(workspace)
        args = _args(source, workspace)
        artifact = suite._case_recency_identity(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["latest_constancia_selected"] is True
        assert gates["answer_uses_corpus_asegurado"] is True
        assert gates["answer_rejects_bad_metadata_asegurado"] is True
        assert gates["answer_rejects_law_number_as_policy_number"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_wheels_debate_preserves_coverage_and_limit() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        source = _fixture_repo(workspace)
        args = _args(source, workspace)
        artifact = suite._case_wheels(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["coverage_detected"] is True
        assert gates["limit_detected"] is True
        assert gates["clause_detected"] is True
        assert gates["auditor_rejects_unconditional_yes"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_nuclear_exclusion_debate_rejects_coverage_overclaim() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        source = _fixture_repo(workspace)
        args = _args(source, workspace)
        artifact = suite._case_nuclear(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["exclusion_chunk_selected"] is True
        assert gates["nuclear_term_present"] is True
        assert gates["client_does_not_overclaim"] is True
        assert gates["auditor_rejects_coverage"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_policy_legal_debate_suite_writes_transcripts() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        source = _fixture_repo(workspace)
        args = _args(source, workspace)
        artifact = suite.run_suite(args)

        assert artifact["status"] == "completed"
        assert artifact["case_count"] == 6
        assert Path(artifact["artifact_path"]).exists()
        assert Path(artifact["transcript_exports"]["jsonl_path"]).exists()
        assert Path(artifact["transcript_exports"]["md_path"]).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
