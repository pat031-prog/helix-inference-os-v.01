"""CLI entrypoint to export a HeliX session proof bundle.

Usage:
  python -m helix_kv.export_proof <db_path> <session_id> \\
      [--out file.json] [--ref REF] \\
      [--include-quarantined | --exclude-quarantined]

The bundle is one self-contained JSON file: session_proof + DAG chain +
checkpoints + journal slice for that session + journal hash-chain integrity
report + Python<->Rust DAG coverage report + signing public key. Nothing in
the local catalog (sqlite, journal, transcripts) is modified.

Designed so an auditor can rebuild the canonical_head locally by replaying
the embedded `journal_entries` against an empty MemoryCatalog and verify each
receipt's Ed25519 signature against the embedded `signer.public_key`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .memory_catalog import MemoryCatalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m helix_kv.export_proof",
        description="Export a self-contained HeliX session proof bundle.",
    )
    parser.add_argument("db_path", help="Path to the catalog .sqlite file (the journal lives next to it).")
    parser.add_argument("session_id", help="thread_id / session_id to export.")
    parser.add_argument("--out", default=None, help="Write the bundle to this file. Default: stdout.")
    parser.add_argument("--ref", default=None, help="Specific memory_id or node_hash prefix to anchor the proof.")
    quar = parser.add_mutually_exclusive_group()
    quar.add_argument("--include-quarantined", dest="include_quarantined", action="store_true")
    quar.add_argument("--exclude-quarantined", dest="include_quarantined", action="store_false")
    parser.set_defaults(include_quarantined=True)
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit non-zero if journal hash-chain or DAG coverage check fails.",
    )
    args = parser.parse_args(argv)

    catalog = MemoryCatalog.open(args.db_path)
    try:
        bundle = catalog.export_full_session_bundle(
            args.session_id,
            ref=args.ref,
            include_quarantined=args.include_quarantined,
        )
    finally:
        catalog.close()

    text = json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"[export_proof] wrote {len(text):,} bytes -> {out_path}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")

    journal_status = bundle.get("journal_integrity", {}).get("status")
    coverage_status = bundle.get("dag_coverage", {}).get("status")
    proof_status = bundle.get("session_proof", {}).get("status")
    print(
        f"[export_proof] proof_status={proof_status} "
        f"journal_integrity={journal_status} "
        f"dag_coverage={coverage_status}",
        file=sys.stderr,
    )
    if args.fail_on_drift:
        if journal_status not in {"verified", "no_chain", "missing"}:
            return 2
        if coverage_status not in {"verified", "no_journal"}:
            return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
