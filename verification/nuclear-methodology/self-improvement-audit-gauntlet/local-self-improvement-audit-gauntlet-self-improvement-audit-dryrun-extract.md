# HeliX Self-Improvement Audit Gauntlet: self-improvement-audit-dryrun

## Verdict

- Status: `dry_run`
- Score: `0.4286`
- Reviewer proposal count: `0`
- Ranked backlog count: `0`
- Analyst actual: `None`
- Auditor actual: `None`

## Failing Gates

- `all_reviewer_calls_ok`: `False`
- `all_reviewer_json_parseable`: `False`
- `reviewer_proposals_present`: `False`
- `reviewer_proposals_have_valid_evidence`: `False`
- `reviewer_proposals_include_tests`: `False`
- `analyst_json_parseable`: `False`
- `analyst_ranked_backlog_present`: `False`
- `analyst_items_have_valid_evidence`: `False`
- `analyst_items_include_tests`: `False`
- `auditor_json_parseable`: `False`
- `auditor_verdict_pass`: `False`
- `auditor_gate_failures_empty`: `False`

## Evidence Pack

- Evidence count: `16`
- Layers: `python-agent-memory, python-memory, python-provider-audit, python-receipts, rust-cli-core, rust-merkle-dag, rust-state-core`
- Secret hits: `0`

| ID | Layer | Path | Lines | Topic |
| --- | --- | --- | --- | --- |
| `E001` | `python-memory` | `helix_kv/memory_catalog.py` | `286-366` | memory catalog trust root and local signing key |
| `E002` | `python-memory` | `helix_kv/memory_catalog.py` | `1661-1727` | memory catalog signed receipt payload |
| `E003` | `python-memory` | `helix_kv/memory_catalog.py` | `1542-1636` | memory catalog strict signature search |
| `E004` | `python-memory` | `helix_kv/memory_catalog.py` | `2964-3042` | memory catalog context assembly |
| `E005` | `python-memory` | `helix_kv/memory_catalog.py` | `2354-2440` | memory catalog session lineage verification |
| `E006` | `python-agent-memory` | `src/helix_proto/hmem.py` | `156-234` | hmem rollback fence primitive |
| `E007` | `python-receipts` | `src/helix_proto/signed_receipts.py` | `196-272` | signed receipt verifier |
| `E008` | `python-receipts` | `src/helix_proto/signed_receipts.py` | `278-305` | retrieval signature enforcement |
| `E009` | `python-provider-audit` | `src/helix_proto/provider_audit.py` | `13-93` | provider audit configuration and fingerprinting |
| `E010` | `rust-state-core` | `crates/helix-state-core/src/lib.rs` | `429-527` | rust state pending receipt fast path |
| `E011` | `rust-state-core` | `crates/helix-state-core/src/lib.rs` | `565-661` | rust state session verifier |
| `E012` | `rust-state-core` | `crates/helix-state-core/src/lib.rs` | `519-579` | rust state manifest reader |
| `E013` | `rust-merkle-dag` | `crates/helix-merkle-dag/src/lib.rs` | `146-194` | rust merkle hash construction |
| `E014` | `rust-merkle-dag` | `crates/helix-merkle-dag/src/lib.rs` | `251-329` | rust merkle receipt verifier |
| `E015` | `rust-merkle-dag` | `crates/helix-merkle-dag/src/lib.rs` | `795-851` | rust merkle audit chain depth guard |
| `E016` | `rust-cli-core` | `crates/helix-cli-core/src/main.rs` | `758-836` | rust cli capsule verifier |

## Ranked Backlog

- No analyst backlog parsed.
## Cross-Cutting Themes


## Auditor

````json
null
````

## Claim Boundary

This artifact is an evidence-cited candidate backlog for improving HeliX. It does not apply patches and does not prove semantic correctness until changes are implemented and tested.
