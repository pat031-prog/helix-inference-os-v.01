# HeliX Self-Improvement Audit Gauntlet: self-improvement-audit-20260605-185538

## Verdict

- Status: `partial`
- Score: `0.9524`
- Reviewer proposal count: `15`
- Ranked backlog count: `8`
- Analyst actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor actual: `anthropic/claude-sonnet-4-6`

## Failing Gates

- `reviewer_finish_reasons_not_length`: `False`

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

### P1. Harden Local Signing Key Storage and Permissioning

- Severity: `critical`
- Target layer: `python-memory`
- Evidence: `E001`
- Why now: Multiple reviewers identified that silent failures in `os.chmod` allow private keys to be world-readable, compromising the entire trust root. This is a foundational security vulnerability.
- Patch size: `small`
- Risk/tradeoff: May cause startup failures on non-POSIX filesystems (e.g., FAT32) where chmod is unsupported.
- Implementation plan:
  - Remove bare `try...except OSError: pass` blocks in `_local_signing_key_unlocked`.
  - Implement strict permission checks that raise `PermissionError` if `chmod` fails.
  - Ensure directory creation uses 0o700 permissions before file write.
  - Add unit tests mocking `os.chmod` to verify exception raising.
- Tests:
  - Mock `os.chmod` to raise `OSError` and verify critical exception is raised.
  - Verify file permissions of generated key file on Linux/macOS using `os.stat`.
- Acceptance criteria:
  - Key files are guaranteed to be 0o600 or the system fails to start.
  - No silent failure of security-critical filesystem operations.

### P2. Enforce Canonicalization Consistency Between Rust Merkle and Python Receipts

- Severity: `critical`
- Target layer: `cross-layer`
- Evidence: `E013, E007, E002`
- Why now: Serialization drift (e.g., float representation, key ordering) between Python and Rust causes signature verification failures, breaking audit chain integrity.
- Patch size: `medium`
- Risk/tradeoff: Requires refactoring shared logic; potential performance overhead if canonicalization is not cached.
- Implementation plan:
  - Extract `canonical_json_value` logic from Rust (`crates/helix-merkle-dag/src/lib.rs`) to a shared module.
  - Enforce strict canonical JSON serialization in Python `helix_kv/memory_catalog.py` for all receipt payloads.
  - Ensure `verify_signed_receipt` uses the same canonicalization function.
  - Add cross-layer unit tests for receipt generation and verification.
- Tests:
  - Generate a receipt with float values in Python, sign it, and verify in Rust.
  - Generate a receipt with nested dicts in Rust, sign it, and verify in Python.
  - Verify that `canonical_json_value` rejects floats in both layers.
- Acceptance criteria:
  - Receipts signed in Python verify successfully in Rust.
  - Receipts signed in Rust verify successfully in Python.
  - Floats are explicitly rejected or coerced to strings in canonicalization.

### P3. Strengthen Local Signing Key Provenance and Rotation

- Severity: `high`
- Target layer: `python-memory`
- Evidence: `E001, E002`
- Why now: Lack of key rotation and expiration mechanisms prevents revocation of compromised keys and complicates machine replacement scenarios.
- Patch size: `medium`
- Risk/tradeoff: Increased complexity in key management; requires careful handling of historical data.
- Implementation plan:
  - Introduce a `key_ring` structure in `MemoryCatalog` to store multiple keys with `active` status.
  - Add `rotate_key()` method to generate new keys and mark old ones as `expired`.
  - Update `_local_signing_key_unlocked` to check for key expiration.
  - Include `key_id` in all receipts and verify against the active key ring.
- Tests:
  - Verify that a receipt signed with an expired key is rejected or flagged.
  - Verify that a new key can sign and verify new receipts.
  - Verify that historical receipts signed with the old key can still be verified if the old key is retained.
- Acceptance criteria:
  - Key rotation is supported and documented.
  - Receipts include `key_id`.
  - Expired keys are not used for new signatures.

### P4. Enforce Strict Signature Verification in Memory Catalog Search

- Severity: `high`
- Target layer: `python-memory`
- Evidence: `E003, E008`
- Why now: Permissive signature enforcement in search allows unverified receipts to be returned, compromising data integrity.
- Patch size: `small`
- Risk/tradeoff: Potential performance impact due to additional verification checks.
- Implementation plan:
  - Modify `search` function in `helix_kv/memory_catalog.py` to enforce strict signature verification by default.
  - Remove or deprecate 'permissive' and 'warn' modes if they pose security risks.
  - Add logging for any fallback to permissive modes if retained for debugging.
  - Add unit tests to verify filtering of unsigned receipts.
- Tests:
  - Add unit tests to verify that strict signature enforcement filters out unsigned receipts.
  - Add integration tests to ensure that permissive mode logs warnings appropriately.
- Acceptance criteria:
  - All search results in strict mode must have verified signatures.
  - Permissive mode must log warnings for unsigned receipts.

### P5. Enforce Rust Merkle Chain Depth Limits in Python API

- Severity: `medium`
- Target layer: `rust-merkle-dag`
- Evidence: `E015, E005`
- Why now: Python layer relies on Rust defaults for chain depth, risking DoS via long chain traversal if limits are not explicitly enforced and configured.
- Patch size: `small`
- Risk/tradeoff: May reject valid but very long chains; requires tuning the limit based on expected usage.
- Implementation plan:
  - Define a constant `MAX_AUDIT_DEPTH` in the Rust crate (e.g., 1000).
  - Update Python `verify_session_lineage` to pass this limit to Rust `audit_chain`.
  - Add configuration option in `MemoryCatalog` to override limit.
  - Add tests for chain lengths exceeding the limit.
- Tests:
  - Verify that a chain of length > MAX_AUDIT_DEPTH raises an error.
  - Verify that a chain of length <= MAX_AUDIT_DEPTH is verified successfully.
- Acceptance criteria:
  - Chain traversal respects the configured max depth.
  - Error message clearly indicates depth limit exceeded.

### P6. Harden Rust State Core Bundle Packing

- Severity: `high`
- Target layer: `rust-state-core`
- Evidence: `E010, E011`
- Why now: Pending bundles may contain empty hashes, allowing corrupted data to bypass strict provenance checks if the verifier is not perfectly aligned.
- Patch size: `small`
- Risk/tradeoff: Slight increase in packing latency due to immediate hashing.
- Implementation plan:
  - Modify `pack_hlx_buffers_pending_bundle` to compute SHA256 of array bytes immediately during packing.
  - Update `HlxArrayEntry` to make `sha256` a required non-empty field.
  - Add tests to reject bundles with empty hashes.
- Tests:
  - Attempt to verify a bundle where `sha256` is an empty string and ensure it is rejected explicitly.
- Acceptance criteria:
  - All `.hlx` bundles produced by the Rust layer contain valid, pre-computed hashes for all arrays.

### P7. Eliminate Hash Collision Risk in Merkle DAG Construction

- Severity: `medium`
- Target layer: `rust-merkle-dag`
- Evidence: `E013`
- Why now: Direct concatenation of content and parent_hash in `compute_hash` is vulnerable to length-extension-style collisions.
- Patch size: `medium`
- Risk/tradeoff: Breaks backward compatibility with existing hashes in the DAG; requires a migration or version bump of the hashing scheme.
- Implementation plan:
  - Introduce domain separators or length-prefixing for `content` and `parent_hash` fields.
  - Change update sequence to include explicit markers (e.g., `b"content:"`, `b"parent:"`).
  - Test with crafted collision inputs.
- Tests:
  - Create two different pairs of (content, parent_hash) that concatenate to the same string and verify they now produce different hashes.
- Acceptance criteria:
  - Hash output is unique to the structured input, not just the concatenated byte stream.

### P8. Improve Rust State Session Verifier Error Handling

- Severity: `medium`
- Target layer: `rust-state-core`
- Evidence: `E011`
- Why now: Abrupt failures on hash mismatches can lead to data loss or corruption without adequate logging or graceful degradation.
- Patch size: `small`
- Risk/tradeoff: Potential increase in complexity due to additional error handling and logging.
- Implementation plan:
  - Modify `verify_hlx_session` to handle hash mismatches gracefully.
  - Add detailed logging for hash mismatch debugging.
  - Implement retry mechanism or fallback strategy if applicable.
- Tests:
  - Add unit tests to simulate hash mismatches and verify graceful handling.
  - Add integration tests to ensure system behavior during mismatches.
- Acceptance criteria:
  - The verifier should handle hash mismatches gracefully without abrupt failures.
  - The system should log the details of hash mismatches for debugging purposes.

## Cross-Cutting Themes

- Cryptographic Integrity: Ensuring consistent canonicalization and robust hashing across Python/Rust boundaries.
- Key Management: Strengthening local key storage, permissions, and rotation mechanisms.
- Defense in Depth: Enforcing strict signature verification and depth limits to prevent DoS and data tampering.

## Auditor

````json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "All backlog items cite valid evidence IDs from the registry, include tests and acceptance criteria, contain no claims of already-modified code, and make no overclaims of formal correctness."
}
````

## Claim Boundary

This artifact is an evidence-cited candidate backlog for improving HeliX. It does not apply patches and does not prove semantic correctness until changes are implemented and tested.
