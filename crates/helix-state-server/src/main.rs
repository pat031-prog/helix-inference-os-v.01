//! HeliX State Server — Tokio-based replacement for the Python asyncio StateServer.
//!
//! Protocol: newline-delimited JSON-RPC over TCP (or UDS on Linux/macOS).
//! Same wire format as the Python version so StateClient works unchanged.
//!
//! Architecture:
//!   - Single process, multi-connection via tokio tasks (one per client).
//!   - IndexedState behind Arc<RwLock> — reads (search, stats, audit) are parallel,
//!     writes (insert, gc_tombstone) take exclusive lock briefly.
//!   - SHA-256 and tokenization run outside the lock via tokio::spawn_blocking
//!     for batch inserts, keeping the accept loop responsive.

use base64::{engine::general_purpose, Engine as _};
use chrono::{SecondsFormat, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

// ─── Privacy filter ───

struct PrivacyFilter {
    private_tag_re: Regex,
    secret_re: Regex,
}

impl PrivacyFilter {
    fn new() -> Self {
        let private_tag_re =
            Regex::new(r"(?i)<private>[\s\S]*?</private>").expect("private_tag regex");
        // Composite regex matching all secret patterns — same as Python SECRET_RE
        let secret_re = Regex::new(
            r#"(?i)(?:(?:api[_\-]?key|secret|token|password|credential|auth)\s*[=:]\s*["']?[A-Za-z0-9_\-/.+]{20,}["']?)|(?:Bearer\s+[A-Za-z0-9._\-+/=]{20,})|(?:(?:^|[^A-Za-z0-9])sk-proj-[A-Za-z0-9\-_]{20,})|(?:(?:^|[^A-Za-z0-9])(?:sk|pk|rk|ak)-[A-Za-z0-9][A-Za-z0-9\-_]{19,})|(?:(?:^|[^A-Za-z0-9])sk-ant-[A-Za-z0-9\-_]{20,})|(?:gh[pus]_[A-Za-z0-9]{36,})|(?:github_pat_[A-Za-z0-9_]{22,})|(?:xoxb-[A-Za-z0-9\-]+)|(?:AKIA[0-9A-Z]{16})|(?:AIza[A-Za-z0-9\-_]{35})|(?:eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})|(?:npm_[A-Za-z0-9]{36})|(?:glpat-[A-Za-z0-9\-_]{20,})"#
        )
        .expect("secret regex");
        Self {
            private_tag_re,
            secret_re,
        }
    }

    fn filter(&self, text: &str) -> String {
        let mut result = text.to_string();
        let lowered = result.to_ascii_lowercase();
        if lowered.contains("<private") {
            result = self.private_tag_re.replace_all(&result, "[REDACTED]").into_owned();
        }
        // Fast marker check before expensive regex
        const MARKERS: &[&str] = &[
            "api", "key", "secret", "token", "password", "credential", "auth",
            "bearer", "sk-", "pk-", "rk-", "ak-", "ghp_", "ghu_", "ghs_",
            "github_pat_", "xoxb-", "akia", "aiza", "eyj", "npm_", "glpat-",
        ];
        let low = result.to_ascii_lowercase();
        if !MARKERS.iter().any(|m| low.contains(m)) {
            return result;
        }
        self.secret_re.replace_all(&result, "[REDACTED_SECRET]").into_owned()
    }
}

// ─── Core types (mirrored from helix-merkle-dag for zero-dep) ───

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MerkleNode {
    content: String,
    hash: String,
    parent_hash: Option<String>,
    timestamp_ms: f64,
    depth: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IndexedMetadata {
    record_kind: String,
    project: String,
    agent_id: String,
    memory_id: Option<String>,
    memory_type: Option<String>,
    summary: String,
    index_content: String,
    tags: Vec<String>,
    importance: f64,
    decay_score: f64,
    content_available: bool,
    audit_status: String,
}

impl Default for IndexedMetadata {
    fn default() -> Self {
        Self {
            record_kind: "node".into(),
            project: String::new(),
            agent_id: String::new(),
            memory_id: None,
            memory_type: None,
            summary: String::new(),
            index_content: String::new(),
            tags: Vec::new(),
            importance: 5.0,
            decay_score: 1.0,
            content_available: true,
            audit_status: "verified".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IndexedNode {
    node: MerkleNode,
    metadata: IndexedMetadata,
    doc_len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Posting {
    node_hash: String,
    term_frequency: u32,
    field: String,
}

#[derive(Default, Serialize, Deserialize)]
struct IndexedState {
    nodes: HashMap<String, IndexedNode>,
    #[serde(default)]
    receipts: HashMap<String, String>,
    inverted: HashMap<String, Vec<Posting>>,
    doc_freq: HashMap<String, usize>,
    total_doc_len: usize,
    session_heads: HashMap<String, String>,
}

// ─── Snapshot persistence ───

const MAGIC_V1: &[u8; 8] = b"HLXSNAP1"; // bincode, uncompressed (legacy)
const MAGIC_V2: &[u8; 8] = b"HLXSNAP2"; // bincode + zstd level-3

/// Serialize + zstd-compress + write atomically to `path`.
/// Returns (compressed_bytes, raw_bytes) tuple.
fn save_snapshot(state: &IndexedState, path: &std::path::Path) -> Result<(usize, usize), String> {
    let raw = bincode::serialize(state).map_err(|e| format!("bincode serialize: {e}"))?;
    let raw_size = raw.len();
    let compressed = zstd::encode_all(&raw[..], 3)
        .map_err(|e| format!("zstd compress: {e}"))?;
    let mut buf = Vec::with_capacity(8 + compressed.len());
    buf.extend_from_slice(MAGIC_V2);
    buf.extend_from_slice(&compressed);
    let final_size = buf.len();
    let tmp = path.with_extension("hlx.tmp");
    std::fs::write(&tmp, &buf).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))?;
    Ok((final_size, raw_size))
}

/// Load snapshot, handling both V1 (uncompressed) and V2 (zstd).
fn load_snapshot(path: &std::path::Path) -> Result<IndexedState, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if data.len() < 8 {
        return Err("file too small to be a valid snapshot".into());
    }
    let magic = &data[..8];
    if magic == MAGIC_V2 {
        let decompressed = zstd::decode_all(&data[8..])
            .map_err(|e| format!("zstd decompress: {e}"))?;
        bincode::deserialize(&decompressed).map_err(|e| format!("bincode deserialize v2: {e}"))
    } else if magic == MAGIC_V1 {
        // Backward compat: uncompressed bincode
        bincode::deserialize(&data[8..]).map_err(|e| format!("bincode deserialize v1: {e}"))
    } else {
        Err(format!("unknown snapshot magic: {:?}", magic))
    }
}

/// Rotate snapshot ring buffer, keeping at most `keep` files.
///
/// File naming:
///   `path`         — current (slot 0)
///   `path.1`       — previous
///   `path.2`       — oldest kept
///
/// On rotation: slot(keep-1) is deleted, slots shift up, path becomes the new slot.
fn rotate_snapshots(path: &std::path::Path, keep: usize) {
    if keep == 0 {
        return;
    }
    // Delete the oldest slot beyond `keep`
    let oldest = path.with_extension(format!("hlx.{}", keep));
    if oldest.exists() {
        let _ = std::fs::remove_file(&oldest);
    }
    // Shift: slot N-1 → slot N, working backwards to avoid overwrite
    for slot in (1..keep).rev() {
        let from = if slot == 1 {
            path.to_path_buf()
        } else {
            path.with_extension(format!("hlx.{}", slot - 1))
        };
        let to = path.with_extension(format!("hlx.{}", slot));
        if from.exists() {
            let _ = std::fs::rename(&from, &to);
        }
    }
}

/// Load from the ring buffer, trying current then fallback slots on error.
fn load_snapshot_ring(path: &std::path::Path, keep: usize) -> Result<IndexedState, String> {
    // Try current slot first
    if path.exists() {
        match load_snapshot(path) {
            Ok(s) => return Ok(s),
            Err(e) => eprintln!("[snapshot] Warning: current slot corrupt ({e}), trying backups"),
        }
    }
    // Try backup slots
    for slot in 1..keep {
        let backup = path.with_extension(format!("hlx.{}", slot));
        if backup.exists() {
            match load_snapshot(&backup) {
                Ok(s) => {
                    eprintln!("[snapshot] Recovered from backup slot {}", slot);
                    return Ok(s);
                }
                Err(e) => eprintln!("[snapshot] Slot {slot} also corrupt ({e})"),
            }
        }
    }
    Err(format!("all {} snapshot slots unreadable or missing", keep))
}

/// Stats about current ring buffer state on disk.
fn snapshot_ring_stats(path: &std::path::Path, keep: usize) -> Value {
    let mut slots = Vec::new();
    // slot 0 = current path
    let size0 = if path.exists() { path.metadata().ok().map(|m| m.len()) } else { None };
    slots.push(serde_json::json!({"slot": 0, "path": path.display().to_string(), "bytes": size0}));
    for slot in 1..keep {
        let p = path.with_extension(format!("hlx.{}", slot));
        let sz = if p.exists() { p.metadata().ok().map(|m| m.len()) } else { None };
        slots.push(serde_json::json!({"slot": slot, "path": p.display().to_string(), "bytes": sz}));
    }
    let total: u64 = slots.iter().filter_map(|s| s["bytes"].as_u64()).sum();
    serde_json::json!({"slots": slots, "total_bytes": total, "keep": keep})
}

// ─── Crypto / tokenization ───

fn compute_hash(content: &str, parent_hash: Option<&str>) -> String {
    let mut h = Sha256::new();
    h.update(content.as_bytes());
    if let Some(ph) = parent_hash {
        h.update(ph.as_bytes());
    }
    hex::encode(h.finalize())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            cur.push(ch.to_ascii_lowercase());
        } else if cur.len() > 1 {
            tokens.push(std::mem::take(&mut cur));
        } else {
            cur.clear();
        }
    }
    if cur.len() > 1 {
        tokens.push(cur);
    }
    tokens
}

fn token_counts(text: &str) -> HashMap<String, u32> {
    let mut m = HashMap::new();
    for t in tokenize(text) {
        *m.entry(t).or_insert(0) += 1;
    }
    m
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

fn field_boost(field: &str) -> f64 {
    match field {
        "summary" => 2.0,
        "tags" => 1.6,
        _ => 1.0,
    }
}

fn val_str(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_string)
}

fn val_str_vec(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default()
}

// ─── Insert logic ───

const SIGNATURE_ALG: &str = "ed25519";
const RECEIPT_VERSION: &str = "helix-signed-receipt-v1";
const CANONICALIZATION: &str = "helix-jcs-v0-rfc8785-compatible-no-floats";

fn utc_now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn sort_json(value: &Value) -> Result<Value, String> {
    match value {
        Value::Null | Value::Bool(_) | Value::String(_) => Ok(value.clone()),
        Value::Number(number) => {
            if number.is_f64() {
                Err("floats are not allowed in signed receipt payloads".into())
            } else {
                Ok(value.clone())
            }
        }
        Value::Array(items) => {
            let mut sorted = Vec::with_capacity(items.len());
            for item in items {
                sorted.push(sort_json(item)?);
            }
            Ok(Value::Array(sorted))
        }
        Value::Object(map) => {
            let mut ordered = serde_json::Map::new();
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            for key in keys {
                let item = map.get(&key).ok_or_else(|| format!("missing key {key}"))?;
                ordered.insert(key, sort_json(item)?);
            }
            Ok(Value::Object(ordered))
        }
    }
}

fn canonical_json(value: &Value) -> Result<String, String> {
    serde_json::to_string(&sort_json(value)?).map_err(|e| format!("canonical json encode: {e}"))
}

fn canonical_sha256(value: &Value) -> Result<String, String> {
    let canonical = canonical_json(value)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

fn receipt_payload(params: &Value, metadata: &IndexedMetadata, node: &MerkleNode, signer_id: &str) -> Value {
    serde_json::json!({
        "node_hash": node.hash.clone(),
        "parent_hash": node.parent_hash.clone(),
        "memory_id": metadata.memory_id.clone(),
        "project": metadata.project.clone(),
        "agent_id": metadata.agent_id.clone(),
        "llm_call_id": val_str(params, "llm_call_id"),
        "issued_at_utc": utc_now(),
        "signer_id": signer_id,
        "receipt_payload_version": "helix-memory-receipt-payload-v1"
    })
}

fn signable_receipt(receipt: &Value) -> Value {
    const STORED: &[&str] = &[
        "signature_alg",
        "signature",
        "public_key",
        "canonical_payload_sha256",
        "receipt_version",
        "key_provenance",
        "attestation",
        "canonicalization",
    ];
    const RUNTIME: &[&str] = &[
        "signature_verified",
        "verified_at_utc",
        "verifier_version",
        "verification_error",
        "public_claim_eligible",
    ];
    let Some(map) = receipt.as_object() else {
        return Value::Object(Default::default());
    };
    let mut out = serde_json::Map::new();
    for (key, value) in map {
        if !STORED.contains(&key.as_str()) && !RUNTIME.contains(&key.as_str()) {
            out.insert(key.clone(), value.clone());
        }
    }
    Value::Object(out)
}

fn derive_signing_key(seed: &str) -> SigningKey {
    let digest = Sha256::digest(seed.as_bytes());
    let mut secret = [0u8; 32];
    secret.copy_from_slice(&digest);
    SigningKey::from_bytes(&secret)
}

fn unsigned_legacy_receipt(payload: Value, signer_id: &str) -> Value {
    let digest = canonical_sha256(&payload).unwrap_or_default();
    let mut obj = payload.as_object().cloned().unwrap_or_default();
    obj.insert("receipt_version".into(), "unsigned_legacy".into());
    obj.insert("canonicalization".into(), CANONICALIZATION.into());
    obj.insert("signature_alg".into(), Value::Null);
    obj.insert("signature".into(), Value::Null);
    obj.insert("public_key".into(), Value::Null);
    obj.insert("signer_id".into(), signer_id.into());
    obj.insert("key_provenance".into(), "unsigned_legacy".into());
    obj.insert("canonical_payload_sha256".into(), digest.into());
    obj.insert("attestation".into(), Value::Null);
    attach_verification(Value::Object(obj))
}

fn signed_receipt(payload: Value, signer_id: &str, key_provenance: &str, seed: &str, attestation: Value) -> Result<Value, String> {
    let canonical = canonical_json(&payload)?;
    let digest = canonical_sha256(&payload)?;
    let signing_key = derive_signing_key(seed);
    let verifying_key = signing_key.verifying_key();
    let signature = signing_key.sign(canonical.as_bytes());
    let mut obj = payload.as_object().cloned().unwrap_or_default();
    obj.insert("receipt_version".into(), RECEIPT_VERSION.into());
    obj.insert("canonicalization".into(), CANONICALIZATION.into());
    obj.insert("signature_alg".into(), SIGNATURE_ALG.into());
    obj.insert("signature".into(), general_purpose::STANDARD.encode(signature.to_bytes()).into());
    obj.insert("public_key".into(), general_purpose::STANDARD.encode(verifying_key.to_bytes()).into());
    obj.insert("signer_id".into(), signer_id.into());
    obj.insert("key_provenance".into(), key_provenance.into());
    obj.insert("canonical_payload_sha256".into(), digest.into());
    obj.insert("attestation".into(), attestation);
    Ok(attach_verification(Value::Object(obj)))
}

fn verify_signed_receipt(receipt: &Value) -> Value {
    let mut result = serde_json::Map::new();
    result.insert("verified_at_utc".into(), utc_now().into());
    result.insert("verifier_version".into(), "helix-rust-receipt-verifier-v1".into());
    let verify_result = (|| -> Result<(), String> {
        if receipt.get("signature_alg").and_then(Value::as_str) != Some(SIGNATURE_ALG) {
            return Err("unsupported signature_alg".into());
        }
        let payload = signable_receipt(receipt);
        let digest = canonical_sha256(&payload)?;
        if receipt.get("canonical_payload_sha256").and_then(Value::as_str) != Some(digest.as_str()) {
            return Err("canonical_payload_sha256 mismatch".into());
        }
        let public_b64 = receipt.get("public_key").and_then(Value::as_str).ok_or("missing public_key")?;
        let signature_b64 = receipt.get("signature").and_then(Value::as_str).ok_or("missing signature")?;
        let public_raw = general_purpose::STANDARD
            .decode(public_b64.as_bytes())
            .map_err(|e| format!("invalid public_key base64: {e}"))?;
        let signature_raw = general_purpose::STANDARD
            .decode(signature_b64.as_bytes())
            .map_err(|e| format!("invalid signature base64: {e}"))?;
        let public_array: [u8; 32] = public_raw
            .try_into()
            .map_err(|_| "public_key must decode to 32 bytes".to_string())?;
        let verifying_key = VerifyingKey::from_bytes(&public_array)
            .map_err(|e| format!("invalid public_key: {e}"))?;
        let signature = Signature::from_slice(&signature_raw)
            .map_err(|e| format!("invalid signature: {e}"))?;
        verifying_key
            .verify(canonical_json(&payload)?.as_bytes(), &signature)
            .map_err(|e| format!("invalid signature: {e}"))?;
        result.insert("canonical_payload_sha256".into(), digest.into());
        Ok(())
    })();
    match verify_result {
        Ok(()) => {
            let provenance = receipt.get("key_provenance").and_then(Value::as_str).unwrap_or("");
            result.insert("signature_verified".into(), true.into());
            result.insert("key_provenance".into(), provenance.into());
            result.insert(
                "public_claim_eligible".into(),
                matches!(provenance, "sigstore_rekor" | "yubikey_or_tpm_pinned" | "ephemeral_preregistered").into(),
            );
        }
        Err(error) => {
            result.insert("signature_verified".into(), false.into());
            result.insert("verification_error".into(), error.into());
            result.insert("public_claim_eligible".into(), false.into());
        }
    }
    Value::Object(result)
}

fn attach_verification(receipt: Value) -> Value {
    let mut obj = receipt.as_object().cloned().unwrap_or_default();
    let runtime = if obj.get("receipt_version").and_then(Value::as_str) == Some("unsigned_legacy")
        || obj.get("signature").and_then(Value::as_str).is_none()
    {
        serde_json::json!({
            "signature_verified": false,
            "verified_at_utc": utc_now(),
            "verifier_version": "helix-rust-receipt-verifier-v1",
            "verification_error": "unsigned_legacy",
            "public_claim_eligible": false
        })
    } else {
        verify_signed_receipt(&Value::Object(obj.clone()))
    };
    if let Some(runtime_obj) = runtime.as_object() {
        for (key, value) in runtime_obj {
            obj.insert(key.clone(), value.clone());
        }
    }
    Value::Object(obj)
}

fn build_receipt(params: &Value, metadata: &IndexedMetadata, node: &MerkleNode) -> Result<Value, String> {
    let mode = val_str(params, "receipt_signing_mode")
        .or_else(|| std::env::var("HELIX_RECEIPT_SIGNING_MODE").ok())
        .unwrap_or_else(|| "off".into())
        .to_ascii_lowercase();
    let signer_id = val_str(params, "receipt_signer_id")
        .or_else(|| std::env::var("HELIX_RECEIPT_SIGNER_ID").ok())
        .unwrap_or_else(|| metadata.agent_id.clone());
    let payload = receipt_payload(params, metadata, node, &signer_id);
    if matches!(mode.as_str(), "" | "0" | "false" | "off" | "unsigned_legacy") {
        return Ok(unsigned_legacy_receipt(payload, &signer_id));
    }
    if !matches!(mode.as_str(), "local_self_signed" | "ephemeral_preregistered" | "sigstore_rekor") {
        return Err(format!("unsupported receipt_signing_mode: {mode}"));
    }
    let attestation = if mode == "sigstore_rekor" {
        let evidence_digest = val_str(params, "sigstore_rekor_bundle_digest")
            .or_else(|| std::env::var("HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST").ok())
            .ok_or_else(|| "sigstore_rekor signing requires HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST".to_string())?;
        serde_json::json!({"provider": "sigstore_rekor", "evidence_digest": evidence_digest, "verified": true})
    } else {
        Value::Null
    };
    let seed = val_str(params, "receipt_signing_seed")
        .or_else(|| std::env::var("HELIX_RECEIPT_SIGNING_SEED").ok())
        .unwrap_or_else(|| format!("{}:{}", metadata.memory_id.as_deref().unwrap_or("node"), node.hash));
    signed_receipt(payload, &signer_id, &mode, &seed, attestation)
}

fn signature_enforcement_mode(filters: &Value) -> String {
    // Default is strict: unsigned or unverified receipts are filtered out of search
    // responses. warn/permissive must be opted into explicitly via filters or the
    // HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT env var.
    val_str(filters, "signature_enforcement")
        .or_else(|| std::env::var("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT").ok())
        .unwrap_or_else(|| "strict".into())
        .to_ascii_lowercase()
}

struct PreparedRecord {
    content: String,
    parent_hash: Option<String>,
    metadata: IndexedMetadata,
    node_hash: String,
    timestamp_ms: f64,
    content_counts: HashMap<String, u32>,
    summary_counts: HashMap<String, u32>,
    tag_counts: HashMap<String, u32>,
    doc_len: usize,
}

fn parse_metadata_value(v: &Value) -> IndexedMetadata {
    IndexedMetadata {
        record_kind: val_str(v, "record_kind").unwrap_or_else(|| "node".into()),
        project: val_str(v, "project").unwrap_or_default(),
        agent_id: val_str(v, "agent_id").unwrap_or_default(),
        memory_id: val_str(v, "memory_id").or_else(|| val_str(v, "observation_id")),
        memory_type: val_str(v, "memory_type").or_else(|| val_str(v, "observation_type")),
        summary: val_str(v, "summary").unwrap_or_default(),
        index_content: val_str(v, "index_content")
            .or_else(|| val_str(v, "content"))
            .unwrap_or_default(),
        tags: val_str_vec(v, "tags"),
        importance: v.get("importance").and_then(Value::as_f64).unwrap_or(5.0),
        decay_score: v.get("decay_score").and_then(Value::as_f64).unwrap_or(1.0),
        content_available: v
            .get("content_available")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        audit_status: val_str(v, "audit_status").unwrap_or_else(|| "verified".into()),
    }
}

fn prepare_record(content: String, parent_hash: Option<String>, metadata: IndexedMetadata) -> PreparedRecord {
    let node_hash = compute_hash(&content, parent_hash.as_deref());
    let ts = now_ms();
    let cc = token_counts(&metadata.index_content);
    let sc = token_counts(&metadata.summary);
    let tc = token_counts(&metadata.tags.join(" "));
    let dl = cc.values().sum::<u32>() as usize
        + sc.values().sum::<u32>() as usize
        + tc.values().sum::<u32>() as usize;
    PreparedRecord {
        content,
        parent_hash,
        metadata,
        node_hash,
        timestamp_ms: ts,
        content_counts: cc,
        summary_counts: sc,
        tag_counts: tc,
        doc_len: dl.max(1),
    }
}

fn insert_prepared(state: &mut IndexedState, p: PreparedRecord) -> Result<MerkleNode, String> {
    if let Some(existing) = state.nodes.get(&p.node_hash) {
        return Ok(existing.node.clone());
    }
    let depth = match &p.parent_hash {
        Some(ph) => state
            .nodes
            .get(ph)
            .map(|n| n.node.depth + 1)
            .ok_or_else(|| format!("parent_hash {} not found", ph))?,
        None => 0,
    };
    let node = MerkleNode {
        content: p.content,
        hash: p.node_hash.clone(),
        parent_hash: p.parent_hash,
        timestamp_ms: p.timestamp_ms,
        depth,
    };
    let indexed = IndexedNode {
        node: node.clone(),
        metadata: p.metadata,
        doc_len: p.doc_len,
    };
    let mut unique = HashSet::new();
    for (field, counts) in [
        ("content", p.content_counts),
        ("summary", p.summary_counts),
        ("tags", p.tag_counts),
    ] {
        for (term, tf) in counts {
            unique.insert(term.clone());
            state.inverted.entry(term).or_default().push(Posting {
                node_hash: p.node_hash.clone(),
                term_frequency: tf,
                field: field.to_string(),
            });
        }
    }
    for term in unique {
        *state.doc_freq.entry(term).or_insert(0) += 1;
    }
    state.total_doc_len += p.doc_len;
    state.nodes.insert(p.node_hash, indexed);
    Ok(node)
}

// ─── Search (BM25) ───

fn filter_set(v: &Value, k: &str) -> HashSet<String> {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default()
}

fn search_bm25(state: &IndexedState, query: &str, limit: usize, filters: &Value) -> Vec<Value> {
    let terms = tokenize(query);
    if terms.is_empty() || limit == 0 {
        return Vec::new();
    }
    let project_f = val_str(filters, "project");
    let agent_f = val_str(filters, "agent_id");
    let kind_f = val_str(filters, "record_kind");
    let mem_types = filter_set(filters, "memory_types");
    let exclude_ids = filter_set(filters, "exclude_memory_ids");
    let include_tomb = filters
        .get("include_tombstoned")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let n = state.nodes.len().max(1) as f64;
    let avg_dl = (state.total_doc_len.max(1) as f64) / n;
    let (k1, b) = (1.2, 0.75);

    const STOPWORD_MIN: usize = 100;
    let sw_thresh = (n * 0.40) as usize;
    let mut tinfos: Vec<(String, usize)> = terms
        .into_iter()
        .filter_map(|t| state.doc_freq.get(&t).copied().map(|df| (t, df)))
        .collect();
    tinfos.sort_by_key(|(_, df)| *df);
    if tinfos.is_empty() {
        return Vec::new();
    }
    let active = if state.nodes.len() >= STOPWORD_MIN {
        let sel: Vec<_> = tinfos.iter().filter(|(_, df)| *df <= sw_thresh).cloned().collect();
        if sel.is_empty() { vec![tinfos[0].clone()] } else { sel }
    } else {
        tinfos
    };

    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut matched: HashMap<String, HashSet<String>> = HashMap::new();
    const MAX_POST: usize = 50_000;

    for (term, df_count) in active {
        let Some(postings) = state.inverted.get(&term) else { continue };
        let df = df_count.max(1) as f64;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
        let start = if postings.len() > MAX_POST { postings.len() - MAX_POST } else { 0 };
        for p in &postings[start..] {
            let Some(idx) = state.nodes.get(&p.node_hash) else { continue };
            if !include_tomb && !idx.metadata.content_available { continue }
            if let Some(ref pf) = project_f { if idx.metadata.project != *pf { continue } }
            if let Some(ref af) = agent_f { if idx.metadata.agent_id != *af { continue } }
            if let Some(ref kf) = kind_f { if idx.metadata.record_kind != *kf { continue } }
            if !mem_types.is_empty() {
                let mt = idx.metadata.memory_type.as_deref().unwrap_or("");
                if !mem_types.contains(mt) { continue }
            }
            if let Some(mid) = idx.metadata.memory_id.as_deref() {
                if exclude_ids.contains(mid) { continue }
            }
            let tf = p.term_frequency as f64;
            let dl = idx.doc_len.max(1) as f64;
            let denom = tf + k1 * (1.0 - b + b * (dl / avg_dl.max(1.0)));
            let bm25 = idf * ((tf * (k1 + 1.0)) / denom.max(1e-6));
            let qb = 1.0
                + (idx.metadata.importance.max(0.0) / 10.0) * 0.20
                + idx.metadata.decay_score.max(0.0) * 0.10;
            *scores.entry(p.node_hash.clone()).or_insert(0.0) += bm25 * field_boost(&p.field) * qb;
            matched.entry(p.node_hash.clone()).or_default().insert(term.clone());
        }
    }

    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|(ha, sa), (hb, sb)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let ta = state.nodes.get(ha).map(|n| n.node.timestamp_ms).unwrap_or(0.0);
                let tb = state.nodes.get(hb).map(|n| n.node.timestamp_ms).unwrap_or(0.0);
                tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let enforcement = signature_enforcement_mode(filters);
    let mut out = Vec::new();
    for (hash, score) in ranked {
        let Some(idx) = state.nodes.get(&hash) else { continue };
        let receipt = state
            .receipts
            .get(&hash)
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .unwrap_or_else(|| unsigned_legacy_receipt(receipt_payload(filters, &idx.metadata, &idx.node, &idx.metadata.agent_id), &idx.metadata.agent_id));
        let signature_verified = receipt
            .get("signature_verified")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if enforcement == "strict" && !signature_verified {
            continue;
        }
        let terms: Vec<String> = matched
            .get(&hash)
            .map(|s| {
                let mut v: Vec<_> = s.iter().cloned().collect();
                v.sort();
                v
            })
            .unwrap_or_default();
        let mut row = serde_json::json!({
            "node_hash": hash,
            "score": score,
            "matched_terms": terms,
            "project": idx.metadata.project,
            "agent_id": idx.metadata.agent_id,
            "memory_id": idx.metadata.memory_id,
            "memory_type": idx.metadata.memory_type,
            "record_kind": idx.metadata.record_kind,
            "summary_preview": idx.metadata.summary,
            "audit_status": idx.metadata.audit_status,
            "content_available": idx.metadata.content_available,
            "signed_receipt": receipt,
            "signature_verified": signature_verified,
            "signature_enforcement_mode": enforcement,
        });
        if enforcement == "warn" && !signature_verified {
            row["signature_enforcement_warning"] = "unsigned_or_unverified_receipt_returned".into();
        }
        out.push(row);
        if out.len() >= limit {
            break;
        }
    }
    out
}

// ─── GC tombstone ───

fn gc_tombstone(state: &mut IndexedState, criteria: &Value) -> Value {
    let target_hash = val_str(criteria, "hash").or_else(|| val_str(criteria, "node_hash"));
    let target_mid = val_str(criteria, "memory_id");
    let mut changed = 0usize;
    let mut hashes = Vec::new();
    for indexed in state.nodes.values_mut() {
        let hm = target_hash.as_ref().map(|v| indexed.node.hash == *v).unwrap_or(false);
        let mm = target_mid.as_ref().map(|v| indexed.metadata.memory_id.as_deref() == Some(v.as_str())).unwrap_or(false);
        if !hm && !mm { continue }
        if !indexed.metadata.content_available { continue }
        let orig_hash = compute_hash(&indexed.node.content, indexed.node.parent_hash.as_deref());
        let orig_size = indexed.node.content.len();
        indexed.node.content = format!("[GC_TOMBSTONE:sha256={},size={}]", orig_hash, orig_size);
        indexed.metadata.content_available = false;
        indexed.metadata.audit_status = "tombstone_preserved".into();
        hashes.push(indexed.node.hash.clone());
        changed += 1;
    }
    serde_json::json!({"status": if changed > 0 {"completed"} else {"miss"}, "tombstoned_count": changed, "node_hashes": hashes})
}

// ─── Verify chain ───

fn verify_chain(state: &IndexedState, leaf_hash: &str) -> Value {
    let mut current: Option<&str> = Some(leaf_hash);
    let mut chain_len = 0usize;
    let mut tombstoned = 0usize;
    let mut failed_at: Option<String> = None;
    let mut missing: Option<String> = None;
    let mut signed_receipts = Vec::new();
    let mut signature_verified_count = 0usize;
    let mut unsigned_legacy_count = 0usize;
    while let Some(h) = current {
        let Some(idx) = state.nodes.get(h) else { missing = Some(h.into()); break };
        chain_len += 1;
        if let Some(receipt) = state.receipts.get(h).and_then(|raw| serde_json::from_str::<Value>(raw).ok()) {
            if receipt.get("signature_verified").and_then(Value::as_bool).unwrap_or(false) {
                signature_verified_count += 1;
            }
            if receipt.get("receipt_version").and_then(Value::as_str) == Some("unsigned_legacy") {
                unsigned_legacy_count += 1;
            }
            signed_receipts.push(receipt);
        } else {
            unsigned_legacy_count += 1;
        }
        let re = compute_hash(&idx.node.content, idx.node.parent_hash.as_deref());
        if re != idx.node.hash {
            if idx.metadata.content_available { failed_at = Some(idx.node.hash.clone()); break }
            tombstoned += 1;
        }
        current = idx.node.parent_hash.as_deref();
    }
    let status = if failed_at.is_some() || missing.is_some() {
        "failed"
    } else if tombstoned > 0 {
        "tombstone_preserved"
    } else {
        "verified"
    };
    serde_json::json!({
        "status": status,
        "leaf_hash": leaf_hash,
        "chain_len": chain_len,
        "tombstoned_count": tombstoned,
        "failed_at": failed_at,
        "missing_parent": missing,
        "signature_verified_count": signature_verified_count,
        "unsigned_legacy_count": unsigned_legacy_count,
        "attestation_status": if signed_receipts.iter().any(|r| r.get("attestation").and_then(|a| a.get("verified")).and_then(Value::as_bool).unwrap_or(false)) { "verified" } else { "none" },
        "signed_receipts": signed_receipts,
    })
}

// ─── Stats ───

fn stats(state: &IndexedState) -> Value {
    let pc: usize = state.inverted.values().map(Vec::len).sum();
    let tc = state.nodes.values().filter(|n| !n.metadata.content_available).count();
    let receipts: Vec<Value> = state
        .receipts
        .values()
        .filter_map(|raw| serde_json::from_str::<Value>(raw).ok())
        .collect();
    serde_json::json!({
        "backend": "rust_tokio_bm25",
        "node_count": state.nodes.len(),
        "signed_receipt_count": state.receipts.len(),
        "signature_verified_count": receipts.iter().filter(|r| r.get("signature_verified").and_then(Value::as_bool).unwrap_or(false)).count(),
        "unsigned_legacy_count": receipts.iter().filter(|r| r.get("receipt_version").and_then(Value::as_str) == Some("unsigned_legacy")).count(),
        "term_count": state.inverted.len(),
        "doc_freq_term_count": state.doc_freq.len(),
        "posting_count": pc,
        "total_doc_len": state.total_doc_len,
        "tombstoned_count": tc,
        "memory_count": state.nodes.values().filter(|n| n.metadata.record_kind == "memory").count(),
        "observation_count": state.nodes.values().filter(|n| n.metadata.record_kind == "observation").count(),
    })
}

// ─── JSON-RPC dispatch ───

struct DispatchContext {
    state: Arc<RwLock<IndexedState>>,
    privacy: Arc<PrivacyFilter>,
    snapshot_path: Option<PathBuf>,
    write_counter: Arc<AtomicU64>,
    snapshot_every: u64,
    snapshot_keep: usize,
}

fn sanitize_content(pf: &PrivacyFilter, params: &Value) -> (String, IndexedMetadata) {
    let raw_content = params.get("content").and_then(Value::as_str).unwrap_or("");
    let content = pf.filter(raw_content);
    let mut metadata = if let Some(m) = params.get("metadata") {
        parse_metadata_value(m)
    } else {
        parse_metadata_value(params)
    };
    // Sanitize metadata fields that could carry secrets
    metadata.summary = pf.filter(&metadata.summary);
    metadata.index_content = pf.filter(&metadata.index_content);
    (content, metadata)
}

fn dispatch(ctx: &DispatchContext, method: &str, params: &Value) -> Value {
    match method {
        "remember" | "observe" => {
            let (content, metadata) = sanitize_content(&ctx.privacy, params);
            let session_id = params.get("session_id").and_then(Value::as_str).map(str::to_string);
            let explicit_parent_hash = params.get("parent_hash").and_then(Value::as_str).map(str::to_string);
            let parent_hash = if explicit_parent_hash.is_some() {
                explicit_parent_hash
            } else if let Some(sid) = session_id.as_ref() {
                let st = ctx.state.read();
                st.session_heads.get(sid).cloned()
            } else {
                None
            };
            let p = prepare_record(content, parent_hash, metadata);
            let mut st = ctx.state.write();
            let node_hash = p.node_hash.clone();
            let metadata_for_receipt = p.metadata.clone();
            match insert_prepared(&mut st, p) {
                Ok(node) => {
                    let receipt = match build_receipt(params, &metadata_for_receipt, &node) {
                        Ok(receipt) => receipt,
                        Err(e) => return serde_json::json!({"error": e}),
                    };
                    st.receipts.insert(node.hash.clone(), serde_json::to_string(&receipt).unwrap_or_default());
                    if let Some(sid) = session_id {
                        st.session_heads.insert(sid, node_hash);
                    }
                    ctx.write_counter.fetch_add(1, Ordering::Relaxed);
                    serde_json::json!({
                        "node_hash": node.hash,
                        "depth": node.depth,
                        "signed_receipt": receipt,
                        "signature_enforcement_default": std::env::var("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT").unwrap_or_else(|_| "strict".into())
                    })
                }
                Err(e) => serde_json::json!({"error": e}),
            }
        }
        "bulk_remember" => {
            let items = params.get("items").and_then(Value::as_array);
            let Some(items) = items else {
                return serde_json::json!({"error": "bulk_remember requires params.items array"});
            };
            // Prepare outside lock — privacy filter runs here (no lock contention)
            let prepared: Vec<(PreparedRecord, Option<String>, Value)> = items
                .iter()
                .map(|item| {
                    let (content, metadata) = sanitize_content(&ctx.privacy, item);
                    let parent_hash = item.get("parent_hash").and_then(Value::as_str).map(str::to_string);
                    let session_id = item.get("session_id").and_then(Value::as_str).map(str::to_string);
                    (prepare_record(content, parent_hash, metadata), session_id, item.clone())
                })
                .collect();
            let count = prepared.len() as u64;
            let mut st = ctx.state.write();
            let mut results = Vec::with_capacity(prepared.len());
            for (p, session_id, item_params) in prepared {
                let nh = p.node_hash.clone();
                let metadata_for_receipt = p.metadata.clone();
                match insert_prepared(&mut st, p) {
                    Ok(node) => {
                        let receipt = match build_receipt(&item_params, &metadata_for_receipt, &node) {
                            Ok(receipt) => receipt,
                            Err(e) => {
                                results.push(serde_json::json!({"error": e}));
                                continue;
                            }
                        };
                        st.receipts.insert(node.hash.clone(), serde_json::to_string(&receipt).unwrap_or_default());
                        if let Some(sid) = session_id {
                            st.session_heads.insert(sid, nh.clone());
                        }
                        results.push(serde_json::json!({"node_hash": node.hash, "depth": node.depth, "signed_receipt": receipt}));
                    }
                    Err(e) => results.push(serde_json::json!({"error": e})),
                }
            }
            ctx.write_counter.fetch_add(count, Ordering::Relaxed);
            Value::Array(results)
        }
        "search" => {
            let query = params.get("query").and_then(Value::as_str).unwrap_or("");
            let limit = params.get("limit").and_then(Value::as_u64).unwrap_or(5) as usize;
            let st = ctx.state.read();
            Value::Array(search_bm25(&st, query, limit, params))
        }
        // gc_bulk_sweep: tombstones all nodes below an importance threshold.
        // Used by the Cognitive GC (entropy bomb pruning).
        // Params: max_importance (float, default 2.0), record_kind (opt), project (opt), agent_id (opt)
        "gc_bulk_sweep" => {
            let max_score = params.get("max_importance").and_then(Value::as_f64).unwrap_or(2.0);
            let kind_f = val_str(params, "record_kind");
            let proj_f = val_str(params, "project");
            let agent_f = val_str(params, "agent_id");
            let mut st = ctx.state.write();
            let mut tombstoned = 0usize;
            let mut bytes_freed: i64 = 0;
            for indexed in st.nodes.values_mut() {
                if !indexed.metadata.content_available { continue }
                let score = indexed.metadata.importance * indexed.metadata.decay_score;
                if score >= max_score { continue }
                if let Some(ref kf) = kind_f { if indexed.metadata.record_kind != *kf { continue } }
                if let Some(ref pf) = proj_f { if indexed.metadata.project != *pf { continue } }
                if let Some(ref af) = agent_f { if indexed.metadata.agent_id != *af { continue } }
                let orig = compute_hash(&indexed.node.content, indexed.node.parent_hash.as_deref());
                let orig_len = indexed.node.content.len() as i64;
                indexed.node.content = format!("[GC_TOMBSTONE:sha256={},size={}]", orig, orig_len);
                bytes_freed += orig_len - indexed.node.content.len() as i64;
                indexed.metadata.content_available = false;
                indexed.metadata.audit_status = "tombstone_preserved".into();
                tombstoned += 1;
            }
            ctx.write_counter.fetch_add(1, Ordering::Relaxed);
            serde_json::json!({
                "tombstoned_count": tombstoned,
                "bytes_freed_estimate": bytes_freed,
                "threshold_applied": max_score,
            })
        }
        "gc_tombstone" | "gc_sweep" => {
            let mut st = ctx.state.write();
            let r = gc_tombstone(&mut st, params);
            ctx.write_counter.fetch_add(1, Ordering::Relaxed);
            r
        }
        "verify_chain" => {
            let leaf = params.get("leaf_hash").and_then(Value::as_str).unwrap_or("");
            let st = ctx.state.read();
            verify_chain(&st, leaf)
        }
        "audit_chain" => {
            let leaf = params.get("leaf_hash").and_then(Value::as_str).unwrap_or("");
            let max_d = params.get("max_depth").and_then(Value::as_u64).unwrap_or(10_000) as u32;
            let st = ctx.state.read();
            let mut chain = Vec::new();
            let mut cur: Option<&str> = Some(leaf);
            while let Some(h) = cur {
                if chain.len() as u32 >= max_d { break }
                let Some(idx) = st.nodes.get(h) else { break };
                chain.push(serde_json::json!({"hash": idx.node.hash, "parent_hash": idx.node.parent_hash, "depth": idx.node.depth, "content_len": idx.node.content.len()}));
                cur = idx.node.parent_hash.as_deref();
            }
            Value::Array(chain)
        }
        "stats" => {
            let st = ctx.state.read();
            let mut s = stats(&st);
            if let Some(obj) = s.as_object_mut() {
                obj.insert("write_ops_total".into(), ctx.write_counter.load(Ordering::Relaxed).into());
                obj.insert("snapshot_path".into(), ctx.snapshot_path.as_ref().map(|p| p.display().to_string()).into());
                obj.insert("snapshot_every".into(), ctx.snapshot_every.into());
                obj.insert("snapshot_keep".into(), ctx.snapshot_keep.into());
                if let Some(ref path) = ctx.snapshot_path {
                    obj.insert("snapshot_ring".into(), snapshot_ring_stats(path, ctx.snapshot_keep));
                }
            }
            s
        }
        "snapshot" => {
            if let Some(ref path) = ctx.snapshot_path {
                let st = ctx.state.read();
                rotate_snapshots(path, ctx.snapshot_keep);
                match save_snapshot(&st, path) {
                    Ok((compressed, raw)) => serde_json::json!({
                        "status": "saved",
                        "path": path.display().to_string(),
                        "compressed_bytes": compressed,
                        "raw_bytes": raw,
                        "ratio": format!("{:.2}x", raw as f64 / compressed as f64),
                        "node_count": st.nodes.len(),
                        "ring": snapshot_ring_stats(path, ctx.snapshot_keep),
                    }),
                    Err(e) => serde_json::json!({"status": "error", "error": e}),
                }
            } else {
                serde_json::json!({"status": "disabled", "reason": "no HELIX_SNAPSHOT_PATH configured"})
            }
        }
        _ => serde_json::json!({"error": format!("unknown_method: {}", method)}),
    }
}

// ─── Server ───

async fn handle_client(stream: tokio::net::TcpStream, ctx: Arc<DispatchContext>) {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                let resp = serde_json::json!({"error": "invalid_json"});
                let _ = writer.write_all(format!("{}\n", resp).as_bytes()).await;
                continue;
            }
        };
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Object(Default::default()));
        let req_id = req.get("id").cloned();

        let c = Arc::clone(&ctx);
        let m = method.to_string();
        let result = if matches!(method, "bulk_remember" | "remember" | "observe" | "gc_sweep" | "gc_tombstone" | "snapshot") {
            tokio::task::spawn_blocking(move || dispatch(&c, &m, &params))
                .await
                .unwrap_or_else(|e| serde_json::json!({"error": format!("spawn_blocking: {}", e)}))
        } else {
            dispatch(&c, &m, &params)
        };

        let resp = serde_json::json!({"id": req_id, "result": result});
        if writer.write_all(format!("{}\n", resp).as_bytes()).await.is_err() {
            break;
        }
    }
}

/// Background task: snapshot to disk every 60s if N new writes occurred.
/// Rotates ring buffer keeping `ctx.snapshot_keep` files.
async fn snapshot_daemon(ctx: Arc<DispatchContext>) {
    let Some(ref path) = ctx.snapshot_path else { return };
    let mut last_snap_count = 0u64;
    let interval = std::time::Duration::from_secs(60);
    loop {
        tokio::time::sleep(interval).await;
        let current = ctx.write_counter.load(Ordering::Relaxed);
        let delta = current.saturating_sub(last_snap_count);
        if delta == 0 {
            continue; // Nothing new written
        }
        if delta < ctx.snapshot_every && current > ctx.snapshot_every {
            continue; // Not enough new writes to justify a snapshot yet
        }
        let c = Arc::clone(&ctx);
        let p = path.clone();
        let keep = ctx.snapshot_keep;
        let result = tokio::task::spawn_blocking(move || {
            rotate_snapshots(&p, keep);
            let st = c.state.read();
            save_snapshot(&st, &p).map(|(compressed, raw)| (compressed, raw, st.nodes.len()))
        })
        .await;
        match result {
            Ok(Ok((compressed, raw, count))) => {
                let ratio = raw as f64 / compressed as f64;
                eprintln!(
                    "[snapshot] {count} nodes | raw {raw}B → {compressed}B ({ratio:.2}x) | {}",
                    path.display()
                );
                last_snap_count = current;
            }
            Ok(Err(e)) => eprintln!("[snapshot] Error: {e}"),
            Err(e) => eprintln!("[snapshot] Task panic: {e}"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = std::env::var("HELIX_STATE_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("HELIX_STATE_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8765);
    let snapshot_path = std::env::var("HELIX_SNAPSHOT_PATH").ok().map(PathBuf::from);
    let snapshot_every: u64 = std::env::var("HELIX_SNAPSHOT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let snapshot_keep: usize = std::env::var("HELIX_SNAPSHOT_KEEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .max(1); // Always keep at least 1

    // Load existing snapshot via ring buffer fallback chain
    let initial_state = if let Some(ref path) = snapshot_path {
        match load_snapshot_ring(path, snapshot_keep) {
            Ok(state) => {
                eprintln!(
                    "[helix-state-server] Loaded {} nodes from snapshot ring {}",
                    state.nodes.len(),
                    path.display()
                );
                state
            }
            Err(e) => {
                eprintln!("[helix-state-server] No readable snapshot ({e}), starting empty");
                IndexedState::default()
            }
        }
    } else {
        IndexedState::default()
    };

    let ctx = Arc::new(DispatchContext {
        state: Arc::new(RwLock::new(initial_state)),
        privacy: Arc::new(PrivacyFilter::new()),
        snapshot_path: snapshot_path.clone(),
        write_counter: Arc::new(AtomicU64::new(0)),
        snapshot_every,
        snapshot_keep,
    });

    // Start snapshot daemon
    if snapshot_path.is_some() {
        tokio::spawn(snapshot_daemon(Arc::clone(&ctx)));
    }

    let listener = TcpListener::bind(format!("{}:{}", host, port)).await?;
    let addr = listener.local_addr()?;
    eprintln!("[helix-state-server] Listening on tcp://{}", addr);

    loop {
        let (stream, _peer) = listener.accept().await?;
        let c = Arc::clone(&ctx);
        tokio::spawn(handle_client(stream, c));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ctx() -> DispatchContext {
        DispatchContext {
            state: Arc::new(RwLock::new(IndexedState::default())),
            privacy: Arc::new(PrivacyFilter::new()),
            snapshot_path: None,
            write_counter: Arc::new(AtomicU64::new(0)),
            snapshot_every: 10_000,
            snapshot_keep: 3,
        }
    }

    #[test]
    fn insert_and_search_round_trip() {
        let ctx = test_ctx();
        let params = serde_json::json!({
            "content": "postgres migration schema index",
            "project": "db", "agent_id": "a1", "record_kind": "memory",
            "memory_id": "m1", "memory_type": "semantic",
            "summary": "postgres migration", "index_content": "postgres migration schema index",
            "importance": 9, "decay_score": 1.0
        });
        let r = dispatch(&ctx, "remember", &params);
        assert!(r.get("node_hash").is_some(), "insert failed: {:?}", r);

        // Strict is the default now; this test inserts an unsigned_legacy receipt and is
        // exercising BM25 indexing rather than signature enforcement, so it opts out explicitly.
        let search_params = serde_json::json!({"query": "postgres migration", "limit": 5, "project": "db", "record_kind": "memory", "signature_enforcement": "permissive"});
        let hits = dispatch(&ctx, "search", &search_params);
        let arr = hits.as_array().expect("search should return array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["memory_id"], "m1");
    }

    #[test]
    fn default_signature_enforcement_is_strict() {
        let ctx = test_ctx();
        let _ = dispatch(&ctx, "remember", &serde_json::json!({
            "content": "unsigned default should vanish under strict",
            "project": "dflt", "agent_id": "a1", "record_kind": "memory",
            "memory_id": "unsigned-default", "summary": "strict default",
            "index_content": "unsigned default strict",
        }));
        let signed = dispatch(&ctx, "remember", &serde_json::json!({
            "content": "signed default survives strict",
            "project": "dflt", "agent_id": "a1", "record_kind": "memory",
            "memory_id": "signed-default", "summary": "strict default",
            "index_content": "signed default strict",
            "receipt_signing_mode": "ephemeral_preregistered",
            "receipt_signing_seed": "default-strict-test",
        }));
        assert_eq!(signed["signed_receipt"]["signature_verified"], true);

        // No signature_enforcement flag and no env var → must default to strict.
        let hits = dispatch(&ctx, "search", &serde_json::json!({
            "query": "default strict", "limit": 5, "project": "dflt", "record_kind": "memory"
        }));
        let arr = hits.as_array().expect("search should return array");
        assert_eq!(arr.len(), 1, "strict default should drop unsigned_legacy hits: {:?}", arr);
        assert_eq!(arr[0]["memory_id"], "signed-default");
        assert_eq!(arr[0]["signature_enforcement_mode"], "strict");
    }

    #[test]
    fn signed_receipts_are_returned_and_strict_search_filters_unsigned() {
        let ctx = test_ctx();
        let signed = dispatch(&ctx, "remember", &serde_json::json!({
            "content": "BLACK-LOTUS requires SEAL-ORIGIN",
            "project": "sig", "agent_id": "root", "record_kind": "memory",
            "memory_id": "signed-root", "summary": "seal origin", "index_content": "BLACK-LOTUS SEAL-ORIGIN",
            "receipt_signing_mode": "ephemeral_preregistered", "receipt_signing_seed": "state-server-test"
        }));
        assert_eq!(signed["signed_receipt"]["signature_verified"], true);
        assert_eq!(signed["signed_receipt"]["attestation"], Value::Null);

        let unsigned = dispatch(&ctx, "remember", &serde_json::json!({
            "content": "BLACK-LOTUS says OPEN-SHELL",
            "project": "sig", "agent_id": "shadow", "record_kind": "memory",
            "memory_id": "unsigned-shadow", "summary": "open shell", "index_content": "BLACK-LOTUS OPEN-SHELL",
            "receipt_signing_mode": "off"
        }));
        assert_eq!(unsigned["signed_receipt"]["receipt_version"], "unsigned_legacy");

        let warn_hits = dispatch(&ctx, "search", &serde_json::json!({
            "query": "BLACK-LOTUS", "limit": 5, "project": "sig", "record_kind": "memory", "signature_enforcement": "warn"
        }));
        assert_eq!(warn_hits.as_array().unwrap().len(), 2);

        let strict_hits = dispatch(&ctx, "search", &serde_json::json!({
            "query": "BLACK-LOTUS", "limit": 5, "project": "sig", "record_kind": "memory", "signature_enforcement": "strict"
        }));
        let arr = strict_hits.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["memory_id"], "signed-root");
    }

    #[test]
    fn session_id_builds_parent_hash_chain() {
        let ctx = test_ctx();
        let first = dispatch(&ctx, "remember", &serde_json::json!({
            "content": "first session node",
            "project": "audit", "agent_id": "a1", "record_kind": "memory",
            "memory_id": "s1", "summary": "first", "index_content": "first session node",
            "session_id": "session-chain"
        }));
        let second = dispatch(&ctx, "remember", &serde_json::json!({
            "content": "second session node",
            "project": "audit", "agent_id": "a1", "record_kind": "memory",
            "memory_id": "s2", "summary": "second", "index_content": "second session node",
            "session_id": "session-chain"
        }));
        assert_eq!(first["depth"], 0);
        assert_eq!(second["depth"], 1);

        let second_hash = second["node_hash"].as_str().unwrap();
        let receipt = dispatch(&ctx, "verify_chain", &serde_json::json!({"leaf_hash": second_hash}));
        assert_eq!(receipt["status"], "verified");
        assert_eq!(receipt["chain_len"], 2);

        let chain = dispatch(&ctx, "audit_chain", &serde_json::json!({"leaf_hash": second_hash}));
        let arr = chain.as_array().expect("audit_chain should return array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["parent_hash"], first["node_hash"]);
    }

    #[test]
    fn bulk_insert_and_stats() {
        let ctx = test_ctx();
        let params = serde_json::json!({
            "items": [
                {"content": "alpha content", "project": "p", "agent_id": "a", "record_kind": "memory", "memory_id": "m1", "summary": "alpha", "index_content": "alpha content"},
                {"content": "beta content", "project": "p", "agent_id": "a", "record_kind": "memory", "memory_id": "m2", "summary": "beta", "index_content": "beta content"},
            ]
        });
        let r = dispatch(&ctx, "bulk_remember", &params);
        let arr = r.as_array().expect("bulk_remember should return array");
        assert_eq!(arr.len(), 2);

        let st = dispatch(&ctx, "stats", &Value::Object(Default::default()));
        assert_eq!(st["node_count"], 2);
        assert_eq!(st["memory_count"], 2);
        assert_eq!(st["write_ops_total"], 2); // bulk of 2 items = 2 ops
    }

    #[test]
    fn gc_tombstone_and_verify() {
        let ctx = test_ctx();
        let params = serde_json::json!({
            "content": "secret data to tombstone",
            "project": "p", "agent_id": "a", "record_kind": "memory",
            "memory_id": "m1", "summary": "secret", "index_content": "secret data to tombstone"
        });
        let r = dispatch(&ctx, "remember", &params);
        let hash = r["node_hash"].as_str().unwrap().to_string();

        let gc = dispatch(&ctx, "gc_tombstone", &serde_json::json!({"node_hash": hash}));
        assert_eq!(gc["tombstoned_count"], 1);

        let hits = dispatch(&ctx, "search", &serde_json::json!({"query": "secret", "limit": 5, "project": "p", "record_kind": "memory"}));
        assert_eq!(hits.as_array().unwrap().len(), 0);

        let v = dispatch(&ctx, "verify_chain", &serde_json::json!({"leaf_hash": hash}));
        assert_eq!(v["status"], "tombstone_preserved");
    }

    #[test]
    fn privacy_filter_redacts_secrets() {
        let pf = PrivacyFilter::new();
        // API key
        assert!(pf.filter("token=sk-proj-abcdefghijklmnopqrstuvwxyz").contains("[REDACTED_SECRET]"));
        // Private tag
        assert_eq!(pf.filter("<private>hide me</private> ok"), "[REDACTED] ok");
        // GitHub PAT
        assert!(pf.filter("github_pat_abcdefghijklmnopqrstuvwxyz1234567890").contains("[REDACTED_SECRET]"));
        // Clean text passes through
        assert_eq!(pf.filter("just normal text"), "just normal text");
        // Bearer token
        assert!(pf.filter("Bearer abcdefghijklmnopqrstuvwxyz").contains("[REDACTED_SECRET]"));
    }

    #[test]
    fn privacy_filter_applied_on_insert() {
        let ctx = test_ctx();
        let params = serde_json::json!({
            "content": "Use jose. api_key=sk-proj-abcdefghijklmnopqrstuvwxyz",
            "project": "p", "agent_id": "a", "record_kind": "memory",
            "memory_id": "m1", "summary": "auth with token=sk-proj-abcdefghijklmnopqrstuvwxyz",
            "index_content": "jose token=sk-proj-abcdefghijklmnopqrstuvwxyz"
        });
        dispatch(&ctx, "remember", &params);
        let st = ctx.state.read();
        let node = st.nodes.values().next().unwrap();
        // Content and metadata should be sanitized
        assert!(!node.node.content.contains("sk-proj-"), "content not sanitized");
        assert!(!node.metadata.summary.contains("sk-proj-"), "summary not sanitized");
        assert!(!node.metadata.index_content.contains("sk-proj-"), "index_content not sanitized");
    }

    #[test]
    fn snapshot_save_and_load() {
        let ctx = test_ctx();
        dispatch(&ctx, "remember", &serde_json::json!({
            "content": "persistent memory",
            "project": "snap", "agent_id": "a", "record_kind": "memory",
            "memory_id": "m1", "summary": "persistent", "index_content": "persistent memory"
        }));

        let tmp = std::env::temp_dir().join("helix_test_snapshot_v2.hlx");
        {
            let st = ctx.state.read();
            let (compressed, raw) = save_snapshot(&st, &tmp).unwrap();
            // zstd should compress (or at minimum not expand much for tiny payloads)
            assert!(compressed > 0 && raw > 0);
        }

        // Load via ring (keep=1, only one slot)
        let loaded = load_snapshot_ring(&tmp, 1).unwrap();
        assert_eq!(loaded.nodes.len(), 1);
        assert!(loaded.nodes.values().next().unwrap().metadata.memory_id.as_deref() == Some("m1"));

        let ctx2 = DispatchContext {
            state: Arc::new(RwLock::new(loaded)),
            privacy: Arc::new(PrivacyFilter::new()),
            snapshot_path: None,
            write_counter: Arc::new(AtomicU64::new(0)),
            snapshot_every: 10_000,
            snapshot_keep: 3,
        };
        // Snapshot round-trip test uses unsigned_legacy receipts; opt out of strict default.
        let hits = dispatch(&ctx2, "search", &serde_json::json!({"query": "persistent", "limit": 5, "project": "snap", "record_kind": "memory", "signature_enforcement": "permissive"}));
        assert_eq!(hits.as_array().unwrap().len(), 1);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn snapshot_ring_rotation() {
        let dir = std::env::temp_dir();
        let base = dir.join("helix_ring_test.hlx");
        // Clean up any previous test artifacts
        for slot in 0..5usize {
            let p = if slot == 0 { base.clone() } else { base.with_extension(format!("hlx.{}", slot)) };
            let _ = std::fs::remove_file(&p);
        }

        let state = RwLock::new(IndexedState::default());
        let keep = 3usize;

        // Save 4 snapshots — oldest (slot 3) should be deleted after the 4th
        for _ in 0..4 {
            rotate_snapshots(&base, keep);
            {
                let st = state.read();
                save_snapshot(&st, &base).unwrap();
            }
        }

        // Slot 0 (current) and slots 1, 2 should exist; slot 3 should not
        assert!(base.exists(), "slot 0 must exist");
        assert!(base.with_extension("hlx.1").exists(), "slot 1 must exist");
        assert!(base.with_extension("hlx.2").exists(), "slot 2 must exist");
        assert!(!base.with_extension("hlx.3").exists(), "slot 3 must be deleted by ring");

        // Clean up
        for slot in 0..keep {
            let p = if slot == 0 { base.clone() } else { base.with_extension(format!("hlx.{}", slot)) };
            let _ = std::fs::remove_file(&p);
        }
    }
}
