//! HeliX MerkleDAG — Rust core with PyO3 bindings.
//!
//! Why this exists:
//!   Python's GIL serializes all threads through MerkleDAG.insert().
//!   SHA-256 computation + HashMap lookup under threading.Lock means
//!   500 concurrent threads get ~1x throughput of a single thread.
//!
//!   This Rust implementation:
//!   - Computes SHA-256 outside any lock (truly parallel across OS threads)
//!   - Uses parking_lot::RwLock so concurrent reads (lookup/audit) don't block
//!   - Only takes a write lock for the brief HashMap insert
//!   - Releases the GIL during all CPU-bound work via py.allow_threads()

use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// A single node in the MerkleDAG. Immutable once created.
#[derive(Clone, Debug)]
struct MerkleNodeInner {
    content: String,
    hash: String,
    parent_hash: Option<String>,
    timestamp_ms: f64,
    depth: u32,
}

#[derive(Clone, Debug)]
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
        IndexedMetadata {
            record_kind: "node".to_string(),
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
            audit_status: "verified".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct IndexedNodeInner {
    node: MerkleNodeInner,
    metadata: IndexedMetadata,
    doc_len: usize,
}

#[derive(Clone, Debug)]
struct Posting {
    node_hash: String,
    doc_id: u64,
    content_tf: u32,
    summary_tf: u32,
    tags_tf: u32,
}

#[derive(Clone, Debug, Default)]
struct TermBounds {
    max_content_tf: u32,
    max_summary_tf: u32,
    max_tags_tf: u32,
    min_doc_len: usize,
    max_quality_boost: f64,
}

#[derive(Clone, Debug, Default)]
struct TermPostingDraft {
    content_tf: u32,
    summary_tf: u32,
    tags_tf: u32,
}

#[derive(Default)]
struct IndexedState {
    nodes: HashMap<String, IndexedNodeInner>,
    inverted: HashMap<String, Vec<Posting>>,
    doc_freq: HashMap<String, usize>,
    term_bounds: HashMap<String, TermBounds>,
    total_doc_len: usize,
    next_doc_id: u64,
}

struct PreparedIndexedRecord {
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

/// Python-visible MerkleNode (read-only).
#[pyclass(frozen, name = "RustMerkleNode")]
#[derive(Clone)]
struct PyMerkleNode {
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    hash: String,
    #[pyo3(get)]
    parent_hash: Option<String>,
    #[pyo3(get)]
    timestamp: f64,
    #[pyo3(get)]
    depth: u32,
}

impl From<&MerkleNodeInner> for PyMerkleNode {
    fn from(n: &MerkleNodeInner) -> Self {
        PyMerkleNode {
            content: n.content.clone(),
            hash: n.hash.clone(),
            parent_hash: n.parent_hash.clone(),
            timestamp: n.timestamp_ms,
            depth: n.depth,
        }
    }
}

fn compute_hash(content: &str, parent_hash: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    if let Some(ph) = parent_hash {
        hasher.update(ph.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            current.push(ch.to_ascii_lowercase());
        } else if current.len() > 1 {
            tokens.push(std::mem::take(&mut current));
        } else {
            current.clear();
        }
    }
    if current.len() > 1 {
        tokens.push(current);
    }
    tokens
}

fn token_counts(text: &str) -> HashMap<String, u32> {
    let mut counts = HashMap::new();
    for token in tokenize(text) {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

fn as_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn as_string_vec(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_metadata(metadata_json: Option<String>) -> PyResult<IndexedMetadata> {
    let Some(raw) = metadata_json else {
        return Ok(IndexedMetadata::default());
    };
    let value: Value = serde_json::from_str(&raw)
        .map_err(|exc| PyValueError::new_err(format!("invalid metadata_json: {exc}")))?;
    Ok(IndexedMetadata {
        record_kind: as_string(&value, "record_kind").unwrap_or_else(|| "node".to_string()),
        project: as_string(&value, "project").unwrap_or_default(),
        agent_id: as_string(&value, "agent_id").unwrap_or_default(),
        memory_id: as_string(&value, "memory_id").or_else(|| as_string(&value, "observation_id")),
        memory_type: as_string(&value, "memory_type").or_else(|| as_string(&value, "observation_type")),
        summary: as_string(&value, "summary").unwrap_or_default(),
        index_content: as_string(&value, "index_content").or_else(|| as_string(&value, "content")).unwrap_or_default(),
        tags: as_string_vec(&value, "tags"),
        importance: value.get("importance").and_then(Value::as_f64).unwrap_or(5.0),
        decay_score: value.get("decay_score").and_then(Value::as_f64).unwrap_or(1.0),
        content_available: value
            .get("content_available")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        audit_status: as_string(&value, "audit_status").unwrap_or_else(|| "verified".to_string()),
    })
}

fn parse_filters(filters_json: Option<String>) -> PyResult<Value> {
    let Some(raw) = filters_json else {
        return Ok(Value::Object(Default::default()));
    };
    serde_json::from_str(&raw)
        .map_err(|exc| PyValueError::new_err(format!("invalid filters_json: {exc}")))
}

fn parse_metadata_value(value: &Value) -> IndexedMetadata {
    IndexedMetadata {
        record_kind: as_string(value, "record_kind").unwrap_or_else(|| "node".to_string()),
        project: as_string(value, "project").unwrap_or_default(),
        agent_id: as_string(value, "agent_id").unwrap_or_default(),
        memory_id: as_string(value, "memory_id").or_else(|| as_string(value, "observation_id")),
        memory_type: as_string(value, "memory_type").or_else(|| as_string(value, "observation_type")),
        summary: as_string(value, "summary").unwrap_or_default(),
        index_content: as_string(value, "index_content").or_else(|| as_string(value, "content")).unwrap_or_default(),
        tags: as_string_vec(value, "tags"),
        importance: value.get("importance").and_then(Value::as_f64).unwrap_or(5.0),
        decay_score: value.get("decay_score").and_then(Value::as_f64).unwrap_or(1.0),
        content_available: value
            .get("content_available")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        audit_status: as_string(value, "audit_status").unwrap_or_else(|| "verified".to_string()),
    }
}

fn parse_batch_records(records_json: String) -> PyResult<Vec<(String, Option<String>, IndexedMetadata)>> {
    let value: Value = serde_json::from_str(&records_json)
        .map_err(|exc| PyValueError::new_err(format!("invalid records_json: {exc}")))?;
    let records = value
        .as_array()
        .ok_or_else(|| PyValueError::new_err("records_json must be a JSON array"))?;
    let mut parsed = Vec::with_capacity(records.len());
    for (index, item) in records.iter().enumerate() {
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| PyValueError::new_err(format!("record {index} is missing string content")))?
            .to_string();
        let parent_hash = item.get("parent_hash").and_then(Value::as_str).map(str::to_string);
        let metadata = if let Some(raw) = item.get("metadata_json").and_then(Value::as_str) {
            parse_metadata(Some(raw.to_string()))?
        } else if let Some(meta_value) = item.get("metadata") {
            parse_metadata_value(meta_value)
        } else {
            IndexedMetadata::default()
        };
        parsed.push((content, parent_hash, metadata));
    }
    Ok(parsed)
}

fn prepare_indexed_record(
    content: String,
    parent_hash: Option<String>,
    metadata: IndexedMetadata,
) -> PreparedIndexedRecord {
    let node_hash = compute_hash(&content, parent_hash.as_deref());
    let timestamp_ms = now_ms();
    let content_counts = token_counts(&metadata.index_content);
    let summary_counts = token_counts(&metadata.summary);
    let tag_counts = token_counts(&metadata.tags.join(" "));
    let doc_len = content_counts.values().sum::<u32>() as usize
        + summary_counts.values().sum::<u32>() as usize
        + tag_counts.values().sum::<u32>() as usize;
    PreparedIndexedRecord {
        content,
        parent_hash,
        metadata,
        node_hash,
        timestamp_ms,
        content_counts,
        summary_counts,
        tag_counts,
        doc_len: doc_len.max(1),
    }
}

fn insert_prepared_locked(
    state: &mut IndexedState,
    prepared: PreparedIndexedRecord,
) -> PyResult<PyMerkleNode> {
    if let Some(existing) = state.nodes.get(&prepared.node_hash) {
        return Ok(PyMerkleNode::from(&existing.node));
    }

    let depth = match &prepared.parent_hash {
        Some(ph) => {
            let parent = state.nodes.get(ph).ok_or_else(|| {
                PyValueError::new_err(format!("parent_hash {} not found in RustIndexedMerkleDAG", ph))
            })?;
            parent.node.depth + 1
        }
        None => 0,
    };

    let doc_id = state.next_doc_id;
    state.next_doc_id += 1;

    let node = MerkleNodeInner {
        content: prepared.content,
        hash: prepared.node_hash.clone(),
        parent_hash: prepared.parent_hash,
        timestamp_ms: prepared.timestamp_ms,
        depth,
    };
    let indexed = IndexedNodeInner {
        node,
        metadata: prepared.metadata,
        doc_len: prepared.doc_len,
    };

    let mut unique_terms = HashSet::new();
    let mut term_postings: HashMap<String, TermPostingDraft> = HashMap::new();
    for (term, tf) in prepared.content_counts {
        unique_terms.insert(term.clone());
        term_postings.entry(term).or_default().content_tf = tf;
    }
    for (term, tf) in prepared.summary_counts {
        unique_terms.insert(term.clone());
        term_postings.entry(term).or_default().summary_tf = tf;
    }
    for (term, tf) in prepared.tag_counts {
        unique_terms.insert(term.clone());
        term_postings.entry(term).or_default().tags_tf = tf;
    }

    let doc_len = prepared.doc_len;
    let doc_quality_boost = quality_boost(&indexed.metadata);
    for (term, draft) in term_postings {
        state
            .inverted
            .entry(term.clone())
            .or_default()
            .push(Posting {
                node_hash: prepared.node_hash.clone(),
                doc_id,
                content_tf: draft.content_tf,
                summary_tf: draft.summary_tf,
                tags_tf: draft.tags_tf,
            });
        let bounds = state.term_bounds.entry(term).or_insert_with(|| TermBounds {
            min_doc_len: doc_len.max(1),
            max_quality_boost: doc_quality_boost,
            ..TermBounds::default()
        });
        bounds.max_content_tf = bounds.max_content_tf.max(draft.content_tf);
        bounds.max_summary_tf = bounds.max_summary_tf.max(draft.summary_tf);
        bounds.max_tags_tf = bounds.max_tags_tf.max(draft.tags_tf);
        bounds.min_doc_len = bounds.min_doc_len.min(doc_len.max(1));
        bounds.max_quality_boost = bounds.max_quality_boost.max(doc_quality_boost);
    }

    for term in unique_terms {
        *state.doc_freq.entry(term).or_insert(0) += 1;
    }
    state.total_doc_len += prepared.doc_len;
    let py_node = PyMerkleNode::from(&indexed.node);
    state.nodes.insert(prepared.node_hash, indexed);
    Ok(py_node)
}

fn filter_string_set(filters: &Value, key: &str) -> HashSet<String> {
    filters
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn filter_string(filters: &Value, key: &str) -> Option<String> {
    filters.get(key).and_then(Value::as_str).map(str::to_string)
}

fn field_boost(field: &str) -> f64 {
    match field {
        "summary" => 2.0,
        "tags" => 1.6,
        "content" => 1.0,
        _ => 1.0,
    }
}

fn quality_boost(metadata: &IndexedMetadata) -> f64 {
    1.0
        + (metadata.importance.max(0.0) / 10.0) * 0.20
        + metadata.decay_score.max(0.0) * 0.10
}

fn bm25_component(tf: u32, idf: f64, doc_len: f64, avg_doc_len: f64, k1: f64, b: f64) -> f64 {
    if tf == 0 {
        return 0.0;
    }
    let tf = tf as f64;
    let denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len.max(1.0)));
    idf * ((tf * (k1 + 1.0)) / denom.max(0.000_001))
}

fn posting_score(
    posting: &Posting,
    indexed: &IndexedNodeInner,
    idf: f64,
    avg_doc_len: f64,
    k1: f64,
    b: f64,
) -> f64 {
    let doc_len = indexed.doc_len.max(1) as f64;
    let bm25 = bm25_component(posting.content_tf, idf, doc_len, avg_doc_len, k1, b) * field_boost("content")
        + bm25_component(posting.summary_tf, idf, doc_len, avg_doc_len, k1, b) * field_boost("summary")
        + bm25_component(posting.tags_tf, idf, doc_len, avg_doc_len, k1, b) * field_boost("tags");
    bm25 * quality_boost(&indexed.metadata)
}

fn term_upper_bound(bounds: &TermBounds, idf: f64, avg_doc_len: f64, k1: f64, b: f64) -> f64 {
    let doc_len = bounds.min_doc_len.max(1) as f64;
    let bm25 = bm25_component(bounds.max_content_tf, idf, doc_len, avg_doc_len, k1, b) * field_boost("content")
        + bm25_component(bounds.max_summary_tf, idf, doc_len, avg_doc_len, k1, b) * field_boost("summary")
        + bm25_component(bounds.max_tags_tf, idf, doc_len, avg_doc_len, k1, b) * field_boost("tags");
    bm25 * bounds.max_quality_boost.max(1.0)
}

struct WandCursor<'a> {
    term: String,
    postings: &'a [Posting],
    cursor: usize,
    idf: f64,
    upper_bound: f64,
}

impl<'a> WandCursor<'a> {
    fn current(&self) -> Option<&'a Posting> {
        self.postings.get(self.cursor)
    }

    fn current_doc(&self) -> Option<u64> {
        self.current().map(|posting| posting.doc_id)
    }

    fn advance_to(&mut self, target_doc: u64) {
        while let Some(doc_id) = self.current_doc() {
            if doc_id >= target_doc {
                break;
            }
            self.cursor += 1;
        }
    }

    fn advance_past(&mut self, doc_id: u64) {
        while self.current_doc() == Some(doc_id) {
            self.cursor += 1;
        }
    }
}

fn indexed_matches_filters(
    indexed: &IndexedNodeInner,
    project_filter: &Option<String>,
    agent_filter: &Option<String>,
    record_kind_filter: &Option<String>,
    memory_types: &HashSet<String>,
    exclude_memory_ids: &HashSet<String>,
    include_tombstoned: bool,
) -> bool {
    if !include_tombstoned && !indexed.metadata.content_available {
        return false;
    }
    if let Some(project) = project_filter {
        if indexed.metadata.project != *project {
            return false;
        }
    }
    if let Some(agent_id) = agent_filter {
        if indexed.metadata.agent_id != *agent_id {
            return false;
        }
    }
    if let Some(kind) = record_kind_filter {
        if indexed.metadata.record_kind != *kind {
            return false;
        }
    }
    if !memory_types.is_empty() {
        let memory_type = indexed.metadata.memory_type.as_deref().unwrap_or("");
        if !memory_types.contains(memory_type) {
            return false;
        }
    }
    if let Some(memory_id) = indexed.metadata.memory_id.as_deref() {
        if exclude_memory_ids.contains(memory_id) {
            return false;
        }
    }
    true
}

fn topk_threshold(top: &[(String, f64, f64)], limit: usize) -> f64 {
    if top.len() < limit {
        0.0
    } else {
        top.iter().map(|(_, score, _)| *score).fold(f64::INFINITY, f64::min)
    }
}

fn maybe_insert_top(top: &mut Vec<(String, f64, f64)>, hash: String, score: f64, timestamp_ms: f64, limit: usize) {
    if limit == 0 {
        return;
    }
    if top.len() < limit {
        top.push((hash, score, timestamp_ms));
        return;
    }
    let mut worst_idx = 0usize;
    let mut worst_score = f64::INFINITY;
    let mut worst_timestamp = f64::INFINITY;
    for (idx, (_, item_score, item_timestamp)) in top.iter().enumerate() {
        if *item_score < worst_score
            || ((*item_score - worst_score).abs() <= f64::EPSILON && *item_timestamp < worst_timestamp)
        {
            worst_score = *item_score;
            worst_timestamp = *item_timestamp;
            worst_idx = idx;
        }
    }
    if score > worst_score || ((score - worst_score).abs() <= f64::EPSILON && timestamp_ms > worst_timestamp) {
        top[worst_idx] = (hash, score, timestamp_ms);
    }
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

/// The core DAG. RwLock allows concurrent reads, exclusive writes.
#[pyclass(name = "RustMerkleDAG")]
struct PyMerkleDAG {
    nodes: RwLock<HashMap<String, MerkleNodeInner>>,
}

#[pymethods]
impl PyMerkleDAG {
    #[new]
    fn new() -> Self {
        PyMerkleDAG {
            nodes: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a node. SHA-256 is computed WITHOUT the GIL.
    #[pyo3(signature = (content, parent_hash=None))]
    fn insert(
        &self,
        py: Python<'_>,
        content: String,
        parent_hash: Option<String>,
    ) -> PyResult<PyMerkleNode> {
        // Release GIL for the expensive SHA-256 computation
        let (node_hash, ts) = py.allow_threads(|| {
            let h = compute_hash(&content, parent_hash.as_deref());
            let t = now_ms();
            (h, t)
        });

        // Brief write lock for the HashMap mutation
        let mut nodes = self.nodes.write();

        // Idempotent return
        if let Some(existing) = nodes.get(&node_hash) {
            return Ok(PyMerkleNode::from(existing));
        }

        // Validate parent
        let depth = match &parent_hash {
            Some(ph) => {
                let parent = nodes.get(ph).ok_or_else(|| {
                    PyValueError::new_err(format!("parent_hash {} not found in MerkleDAG", ph))
                })?;
                parent.depth + 1
            }
            None => 0,
        };

        let inner = MerkleNodeInner {
            content,
            hash: node_hash.clone(),
            parent_hash,
            timestamp_ms: ts,
            depth,
        };

        let py_node = PyMerkleNode::from(&inner);
        nodes.insert(node_hash, inner);
        Ok(py_node)
    }

    /// Lookup by hash. Takes a read lock only — fully parallel.
    fn lookup(&self, hash_hex: &str) -> Option<PyMerkleNode> {
        let nodes = self.nodes.read();
        nodes.get(hash_hex).map(PyMerkleNode::from)
    }

    /// Audit chain traversal. Read lock — parallel with other reads.
    #[pyo3(signature = (leaf_hash, max_depth=None))]
    fn audit_chain(
        &self,
        leaf_hash: &str,
        max_depth: Option<u32>,
    ) -> PyResult<Vec<PyMerkleNode>> {
        let max_d = max_depth.unwrap_or(10_000);
        let nodes = self.nodes.read();
        let mut chain = Vec::new();
        let mut current: Option<&str> = Some(leaf_hash);

        while let Some(h) = current {
            if chain.len() as u32 >= max_d {
                return Err(PyRuntimeError::new_err(format!(
                    "audit_chain exceeded max_depth={}. Possible cycle.",
                    max_d
                )));
            }
            match nodes.get(h) {
                Some(node) => {
                    chain.push(PyMerkleNode::from(node));
                    current = node.parent_hash.as_deref();
                }
                None => break,
            }
        }
        Ok(chain)
    }

    /// Number of nodes. Read lock.
    fn __len__(&self) -> usize {
        self.nodes.read().len()
    }

    /// Bulk insert for benchmarks. Releases GIL for SHA-256 batch.
    #[pyo3(signature = (contents, parent_hash=None))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        contents: Vec<String>,
        parent_hash: Option<String>,
    ) -> PyResult<Vec<PyMerkleNode>> {
        // Pre-compute all hashes without GIL
        let ph_ref = parent_hash.as_deref();
        let hashes: Vec<(String, f64)> = py.allow_threads(|| {
            contents
                .iter()
                .map(|c| {
                    let h = compute_hash(c, ph_ref);
                    let t = now_ms();
                    (h, t)
                })
                .collect()
        });

        let mut nodes = self.nodes.write();
        let mut results = Vec::with_capacity(contents.len());

        // First insert must chain from parent_hash, rest are siblings (same parent)
        for (i, (content, (node_hash, ts))) in contents.into_iter().zip(hashes).enumerate() {
            if let Some(existing) = nodes.get(&node_hash) {
                results.push(PyMerkleNode::from(existing));
                continue;
            }

            let depth = if i == 0 {
                match &parent_hash {
                    Some(ph) => {
                        let parent = nodes.get(ph).ok_or_else(|| {
                            PyValueError::new_err(format!("parent_hash {} not found", ph))
                        })?;
                        parent.depth + 1
                    }
                    None => 0,
                }
            } else {
                // Siblings share parent depth
                match &parent_hash {
                    Some(ph) => nodes.get(ph).map(|p| p.depth + 1).unwrap_or(0),
                    None => 0,
                }
            };

            let inner = MerkleNodeInner {
                content,
                hash: node_hash.clone(),
                parent_hash: parent_hash.clone(),
                timestamp_ms: ts,
                depth,
            };
            let py_node = PyMerkleNode::from(&inner);
            nodes.insert(node_hash, inner);
            results.push(py_node);
        }

        Ok(results)
    }
}

#[pyclass(name = "RustIndexedMerkleDAG")]
struct PyIndexedMerkleDAG {
    state: RwLock<IndexedState>,
}

#[pymethods]
impl PyIndexedMerkleDAG {
    #[new]
    fn new() -> Self {
        PyIndexedMerkleDAG {
            state: RwLock::new(IndexedState::default()),
        }
    }

    #[pyo3(signature = (content, parent_hash=None, metadata_json=None))]
    fn insert_indexed(
        &self,
        py: Python<'_>,
        content: String,
        parent_hash: Option<String>,
        metadata_json: Option<String>,
    ) -> PyResult<PyMerkleNode> {
        let metadata = parse_metadata(metadata_json)?;
        let prepared = py.allow_threads(|| prepare_indexed_record(content, parent_hash, metadata));
        let mut state = self.state.write();
        insert_prepared_locked(&mut state, prepared)
    }

    #[pyo3(signature = (records_json))]
    fn insert_indexed_batch(&self, py: Python<'_>, records_json: String) -> PyResult<Vec<PyMerkleNode>> {
        let records = parse_batch_records(records_json)?;
        let prepared = py.allow_threads(|| {
            records
                .into_iter()
                .map(|(content, parent_hash, metadata)| {
                    prepare_indexed_record(content, parent_hash, metadata)
                })
                .collect::<Vec<_>>()
        });
        let mut state = self.state.write();
        let mut out = Vec::with_capacity(prepared.len());
        for item in prepared {
            out.push(insert_prepared_locked(&mut state, item)?);
        }
        Ok(out)
    }

    #[pyo3(signature = (query, limit, filters_json=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: String,
        limit: usize,
        filters_json: Option<String>,
    ) -> PyResult<Vec<PyObject>> {
        let filters = parse_filters(filters_json)?;
        let query_terms = tokenize(&query);
        if query_terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let project_filter = filter_string(&filters, "project");
        let agent_filter = filter_string(&filters, "agent_id");
        let record_kind_filter = filter_string(&filters, "record_kind");
        let memory_types = filter_string_set(&filters, "memory_types");
        let exclude_memory_ids = filter_string_set(&filters, "exclude_memory_ids");
        let include_tombstoned = filters
            .get("include_tombstoned")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let state = self.state.read();
        let doc_count = state.nodes.len().max(1) as f64;
        let avg_doc_len = (state.total_doc_len.max(1) as f64) / doc_count;
        let k1 = 1.2;
        let b = 0.75;

        // --- Dynamic stopword filtering ---
        // Terms appearing in >40% of docs have near-zero IDF and dominate posting scans
        // for no ranking benefit. Skip them unless they are the ONLY query terms.
        // Only activate when corpus is large enough for DF ratios to be meaningful.
        const STOPWORD_MIN_DOCS: usize = 100;
        let stopword_threshold = (doc_count * 0.40) as usize;
        let mut term_infos: Vec<(String, usize)> = query_terms
            .into_iter()
            .filter_map(|term| state.doc_freq.get(&term).copied().map(|df| (term, df)))
            .collect();
        // Sort by ascending DF — rarest terms first (best IDF, smallest posting lists)
        term_infos.sort_by(|(_, left), (_, right)| left.cmp(right));
        if term_infos.is_empty() {
            return Ok(Vec::new());
        }
        let active_terms = if state.nodes.len() >= STOPWORD_MIN_DOCS {
            // Filter stopwords, but keep at least one term
            let selective: Vec<(String, usize)> = term_infos
                .iter()
                .filter(|(_, df)| *df <= stopword_threshold)
                .cloned()
                .collect();
            if selective.is_empty() {
                // All terms are stopwords — keep only the rarest one
                vec![term_infos[0].clone()]
            } else {
                selective
            }
        } else {
            // Small corpus: use all terms, stopword filtering would be pathological
            term_infos
        };

        let mut top: Vec<(String, f64, f64)> = Vec::with_capacity(limit);
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut matched: HashMap<String, HashSet<String>> = HashMap::new();

        // --- Dynamic WAND ---
        // Posting lists are ordered by monotonically increasing doc_id. Each term carries
        // a conservative upper-bound score, so once the sum of remaining bounds cannot
        // beat the current top-k threshold we stop without dropping old postings.
        let mut cursors: Vec<WandCursor<'_>> = Vec::with_capacity(active_terms.len());
        for (term, df_count) in active_terms {
            let Some(postings) = state.inverted.get(&term) else {
                continue;
            };
            let df = df_count.max(1) as f64;
            let idf = ((doc_count - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
            let upper_bound = state
                .term_bounds
                .get(&term)
                .map(|bounds| term_upper_bound(bounds, idf, avg_doc_len, k1, b))
                .unwrap_or(0.0);
            if upper_bound <= 0.0 || postings.is_empty() {
                continue;
            }
            cursors.push(WandCursor {
                term,
                postings,
                cursor: 0,
                idf,
                upper_bound,
            });
        }

        loop {
            cursors.retain(|cursor| cursor.current_doc().is_some());
            if cursors.is_empty() {
                break;
            }
            cursors.sort_by_key(|cursor| cursor.current_doc().unwrap_or(u64::MAX));

            let threshold = topk_threshold(&top, limit);
            let mut bound_sum = 0.0;
            let mut pivot_idx = None;
            for (idx, cursor) in cursors.iter().enumerate() {
                bound_sum += cursor.upper_bound;
                if bound_sum >= threshold {
                    pivot_idx = Some(idx);
                    break;
                }
            }
            let Some(pivot_idx) = pivot_idx else {
                break;
            };
            let Some(pivot_doc) = cursors[pivot_idx].current_doc() else {
                break;
            };

            let all_prefix_at_pivot = cursors[..=pivot_idx]
                .iter()
                .all(|cursor| cursor.current_doc() == Some(pivot_doc));

            if all_prefix_at_pivot {
                let mut candidate_hash: Option<String> = None;
                let mut candidate_indexed: Option<&IndexedNodeInner> = None;
                for cursor in cursors.iter() {
                    if cursor.current_doc() == Some(pivot_doc) {
                        if let Some(posting) = cursor.current() {
                            if let Some(indexed) = state.nodes.get(&posting.node_hash) {
                                candidate_hash = Some(posting.node_hash.clone());
                                candidate_indexed = Some(indexed);
                                break;
                            }
                        }
                    }
                }

                if let (Some(hash), Some(indexed)) = (candidate_hash, candidate_indexed) {
                    if indexed_matches_filters(
                        indexed,
                        &project_filter,
                        &agent_filter,
                        &record_kind_filter,
                        &memory_types,
                        &exclude_memory_ids,
                        include_tombstoned,
                    ) {
                        let mut score = 0.0;
                        let mut terms = HashSet::new();
                        for cursor in cursors.iter() {
                            if cursor.current_doc() == Some(pivot_doc) {
                                if let Some(posting) = cursor.current() {
                                    score += posting_score(posting, indexed, cursor.idf, avg_doc_len, k1, b);
                                    terms.insert(cursor.term.clone());
                                }
                            }
                        }
                        if score > 0.0 {
                            scores.insert(hash.clone(), score);
                            matched.insert(hash.clone(), terms);
                            maybe_insert_top(&mut top, hash, score, indexed.node.timestamp_ms, limit);
                        }
                    }
                }

                for cursor in cursors.iter_mut() {
                    if cursor.current_doc() == Some(pivot_doc) {
                        cursor.advance_past(pivot_doc);
                    }
                }
            } else {
                for cursor in cursors.iter_mut().take(pivot_idx) {
                    cursor.advance_to(pivot_doc);
                }
            }
        }

        let mut ranked: Vec<(String, f64)> = top
            .into_iter()
            .filter_map(|(hash, _, _)| scores.get(&hash).copied().map(|score| (hash, score)))
            .collect();
        ranked.sort_by(|(hash_a, score_a), (hash_b, score_b)| {
            score_b
                .partial_cmp(score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ta = state.nodes.get(hash_a).map(|n| n.node.timestamp_ms).unwrap_or(0.0);
                    let tb = state.nodes.get(hash_b).map(|n| n.node.timestamp_ms).unwrap_or(0.0);
                    tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| hash_a.cmp(hash_b))
        });

        let mut out = Vec::with_capacity(limit.min(ranked.len()));
        for (hash, score) in ranked.into_iter().take(limit) {
            let Some(indexed) = state.nodes.get(&hash) else {
                continue;
            };
            let dict = PyDict::new_bound(py);
            dict.set_item("node_hash", &indexed.node.hash)?;
            dict.set_item("score", score)?;
            let terms = matched
                .get(&hash)
                .map(|items| {
                    let mut values: Vec<String> = items.iter().cloned().collect();
                    values.sort();
                    values
                })
                .unwrap_or_default();
            dict.set_item("matched_terms", terms)?;
            dict.set_item("project", &indexed.metadata.project)?;
            dict.set_item("agent_id", &indexed.metadata.agent_id)?;
            dict.set_item("memory_id", indexed.metadata.memory_id.clone())?;
            dict.set_item("memory_type", indexed.metadata.memory_type.clone())?;
            dict.set_item("record_kind", &indexed.metadata.record_kind)?;
            dict.set_item("summary_preview", &indexed.metadata.summary)?;
            dict.set_item("audit_status", &indexed.metadata.audit_status)?;
            dict.set_item("content_available", indexed.metadata.content_available)?;
            dict.set_item("search_algorithm", "wand_dynamic_bm25")?;
            out.push(dict.into_py(py));
        }
        Ok(out)
    }

    fn lookup(&self, hash_hex: &str) -> Option<PyMerkleNode> {
        let state = self.state.read();
        state.nodes.get(hash_hex).map(|indexed| PyMerkleNode::from(&indexed.node))
    }

    #[pyo3(signature = (leaf_hash, max_depth=None))]
    fn audit_chain(&self, leaf_hash: &str, max_depth: Option<u32>) -> PyResult<Vec<PyMerkleNode>> {
        let max_d = max_depth.unwrap_or(10_000);
        let state = self.state.read();
        let mut chain = Vec::new();
        let mut current: Option<&str> = Some(leaf_hash);
        while let Some(h) = current {
            if chain.len() as u32 >= max_d {
                return Err(PyRuntimeError::new_err(format!(
                    "audit_chain exceeded max_depth={}. Possible cycle.",
                    max_d
                )));
            }
            match state.nodes.get(h) {
                Some(indexed) => {
                    chain.push(PyMerkleNode::from(&indexed.node));
                    current = indexed.node.parent_hash.as_deref();
                }
                None => break,
            }
        }
        Ok(chain)
    }

    #[pyo3(signature = (leaf_hash, _policy=None))]
    fn verify_chain(&self, py: Python<'_>, leaf_hash: &str, _policy: Option<String>) -> PyResult<PyObject> {
        let state = self.state.read();
        let mut current: Option<&str> = Some(leaf_hash);
        let mut chain_len = 0usize;
        let mut tombstoned = 0usize;
        let mut failed_at: Option<String> = None;
        let mut missing_parent: Option<String> = None;

        while let Some(hash) = current {
            let Some(indexed) = state.nodes.get(hash) else {
                missing_parent = Some(hash.to_string());
                break;
            };
            chain_len += 1;
            let recomputed = compute_hash(&indexed.node.content, indexed.node.parent_hash.as_deref());
            if recomputed != indexed.node.hash {
                if indexed.metadata.content_available {
                    failed_at = Some(indexed.node.hash.clone());
                    break;
                }
                tombstoned += 1;
            }
            current = indexed.node.parent_hash.as_deref();
        }

        let status = if failed_at.is_some() || missing_parent.is_some() {
            "failed"
        } else if tombstoned > 0 {
            "tombstone_preserved"
        } else {
            "verified"
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("status", status)?;
        dict.set_item("leaf_hash", leaf_hash)?;
        dict.set_item("chain_len", chain_len)?;
        dict.set_item("tombstoned_count", tombstoned)?;
        dict.set_item("failed_at", failed_at)?;
        dict.set_item("missing_parent", missing_parent)?;
        Ok(dict.into_py(py))
    }

    #[pyo3(signature = (criteria_json=None))]
    fn gc_tombstone(&self, py: Python<'_>, criteria_json: Option<String>) -> PyResult<PyObject> {
        let criteria = parse_filters(criteria_json)?;
        let target_hash = filter_string(&criteria, "hash").or_else(|| filter_string(&criteria, "node_hash"));
        let target_memory_id = filter_string(&criteria, "memory_id");
        let mut state = self.state.write();
        let mut changed = 0usize;
        let mut hashes = Vec::new();

        for indexed in state.nodes.values_mut() {
            let hash_matches = target_hash
                .as_ref()
                .map(|value| indexed.node.hash == *value)
                .unwrap_or(false);
            let memory_matches = target_memory_id
                .as_ref()
                .map(|value| indexed.metadata.memory_id.as_deref() == Some(value.as_str()))
                .unwrap_or(false);
            if !hash_matches && !memory_matches {
                continue;
            }
            if !indexed.metadata.content_available {
                continue;
            }
            let original_hash = compute_hash(&indexed.node.content, indexed.node.parent_hash.as_deref());
            let original_size = indexed.node.content.len();
            indexed.node.content = format!(
                "[GC_TOMBSTONE:sha256={},size={}]",
                original_hash, original_size
            );
            indexed.metadata.content_available = false;
            indexed.metadata.audit_status = "tombstone_preserved".to_string();
            hashes.push(indexed.node.hash.clone());
            changed += 1;
        }

        let dict = PyDict::new_bound(py);
        dict.set_item("status", if changed > 0 { "completed" } else { "miss" })?;
        dict.set_item("tombstoned_count", changed)?;
        dict.set_item("node_hashes", hashes)?;
        Ok(dict.into_py(py))
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.state.read();
        let posting_count: usize = state.inverted.values().map(Vec::len).sum();
        let tombstoned_count = state
            .nodes
            .values()
            .filter(|indexed| !indexed.metadata.content_available)
            .count();
        let dict = PyDict::new_bound(py);
        dict.set_item("backend", "rust_bm25")?;
        dict.set_item("search_algorithm", "wand_dynamic_bm25")?;
        dict.set_item("node_count", state.nodes.len())?;
        dict.set_item("term_count", state.inverted.len())?;
        dict.set_item("doc_freq_term_count", state.doc_freq.len())?;
        dict.set_item("term_bound_count", state.term_bounds.len())?;
        dict.set_item("posting_count", posting_count)?;
        dict.set_item("total_doc_len", state.total_doc_len)?;
        dict.set_item("tombstoned_count", tombstoned_count)?;
        Ok(dict.into_py(py))
    }

    fn __len__(&self) -> usize {
        self.state.read().nodes.len()
    }
}

#[pymodule]
fn _helix_merkle_dag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMerkleNode>()?;
    m.add_class::<PyMerkleDAG>()?;
    m.add_class::<PyIndexedMerkleDAG>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenization_matches_python_shape() {
        assert_eq!(tokenize("The quick_brown-fox! a"), vec!["the", "quick_brown-fox"]);
    }

    #[test]
    fn indexed_insert_preserves_parent_depth() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dag = PyIndexedMerkleDAG::new();
            let root = dag
                .insert_indexed(
                    py,
                    "root content".to_string(),
                    None,
                    Some(r#"{"project":"p","agent_id":"a","record_kind":"memory","memory_id":"m0"}"#.to_string()),
                )
                .unwrap();
            let child = dag
                .insert_indexed(
                    py,
                    "child content".to_string(),
                    Some(root.hash.clone()),
                    Some(r#"{"project":"p","agent_id":"a","record_kind":"memory","memory_id":"m1"}"#.to_string()),
                )
                .unwrap();
            assert_eq!(child.depth, 1);
            assert_eq!(child.parent_hash, Some(root.hash));
        });
    }

    #[test]
    fn indexed_search_filters_project_and_memory_type() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dag = PyIndexedMerkleDAG::new();
            dag.insert_indexed(
                py,
                "postgres schema migration index".to_string(),
                None,
                Some(r#"{"project":"db","agent_id":"a","record_kind":"memory","memory_id":"m1","memory_type":"semantic","summary":"postgres migration","importance":9,"decay_score":1.0}"#.to_string()),
            )
            .unwrap();
            dag.insert_indexed(
                py,
                "frontend button animation".to_string(),
                None,
                Some(r#"{"project":"ui","agent_id":"a","record_kind":"memory","memory_id":"m2","memory_type":"semantic","summary":"frontend animation"}"#.to_string()),
            )
            .unwrap();
            let hits = dag
                .search(
                    py,
                    "postgres migration".to_string(),
                    5,
                    Some(r#"{"project":"db","agent_id":"a","record_kind":"memory","memory_types":["semantic"]}"#.to_string()),
                )
                .unwrap();
            assert_eq!(hits.len(), 1);
        });
    }

    #[test]
    fn stopword_filter_keeps_small_corpus_recall() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dag = PyIndexedMerkleDAG::new();
            for i in 0..20 {
                dag.insert_indexed(
                    py,
                    format!("common-token document {}", i),
                    None,
                    Some(format!(
                        r#"{{"project":"small","agent_id":"a","record_kind":"memory","memory_id":"m{}","summary":"common-token item {}","index_content":"common-token item {}"}}"#,
                        i, i, i
                    )),
                )
                .unwrap();
            }
            let hits = dag
                .search(
                    py,
                    "common-token".to_string(),
                    25,
                    Some(r#"{"project":"small","record_kind":"memory"}"#.to_string()),
                )
                .unwrap();
            assert_eq!(hits.len(), 20);
        });
    }

    #[test]
    fn indexed_batch_insert_searches_all_records() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dag = PyIndexedMerkleDAG::new();
            let records = r#"[
                {"content":"alpha batch content","metadata":{"project":"batch","agent_id":"a","record_kind":"memory","memory_id":"m-alpha","summary":"alpha batch","index_content":"alpha batch content"}},
                {"content":"beta batch content","metadata":{"project":"batch","agent_id":"a","record_kind":"memory","memory_id":"m-beta","summary":"beta batch","index_content":"beta batch content"}}
            ]"#;
            let nodes = dag.insert_indexed_batch(py, records.to_string()).unwrap();
            assert_eq!(nodes.len(), 2);
            let hits = dag
                .search(
                    py,
                    "beta".to_string(),
                    5,
                    Some(r#"{"project":"batch","record_kind":"memory"}"#.to_string()),
                )
                .unwrap();
            assert_eq!(hits.len(), 1);
        });
    }

    #[test]
    fn wand_keeps_high_score_historical_posting_beyond_old_cap() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dag = PyIndexedMerkleDAG::new();
            dag.insert_indexed(
                py,
                "common historical best".to_string(),
                None,
                Some(r#"{"project":"wand","agent_id":"a","record_kind":"memory","memory_id":"m-historical","summary":"common historical best","index_content":"common historical best","importance":100,"decay_score":1.0}"#.to_string()),
            )
            .unwrap();
            for i in 0..50_100 {
                dag.insert_indexed(
                    py,
                    format!("common recent {}", i),
                    None,
                    Some(format!(
                        r#"{{"project":"wand","agent_id":"a","record_kind":"memory","memory_id":"m-recent-{}","summary":"common recent","index_content":"common recent","importance":1,"decay_score":1.0}}"#,
                        i
                    )),
                )
                .unwrap();
            }
            let hits = dag
                .search(
                    py,
                    "common".to_string(),
                    1,
                    Some(r#"{"project":"wand","record_kind":"memory"}"#.to_string()),
                )
                .unwrap();
            assert_eq!(hits.len(), 1);
            let dict = hits[0].downcast_bound::<PyDict>(py).unwrap();
            assert_eq!(
                dict.get_item("memory_id").unwrap().unwrap().extract::<String>().unwrap(),
                "m-historical"
            );
            assert_eq!(
                dict.get_item("search_algorithm").unwrap().unwrap().extract::<String>().unwrap(),
                "wand_dynamic_bm25"
            );
        });
    }

    #[test]
    fn tombstone_blocks_default_search_but_verifies_as_preserved() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dag = PyIndexedMerkleDAG::new();
            let node = dag
                .insert_indexed(
                    py,
                    "secret indexed content".to_string(),
                    None,
                    Some(r#"{"project":"p","agent_id":"a","record_kind":"memory","memory_id":"m1","summary":"secret"}"#.to_string()),
                )
                .unwrap();
            dag.gc_tombstone(py, Some(format!(r#"{{"node_hash":"{}"}}"#, node.hash)))
                .unwrap();
            let hits = dag
                .search(
                    py,
                    "secret".to_string(),
                    5,
                    Some(r#"{"project":"p","record_kind":"memory"}"#.to_string()),
                )
                .unwrap();
            assert_eq!(hits.len(), 0);
            let receipt = dag.verify_chain(py, &node.hash, None).unwrap();
            let dict = receipt.downcast_bound::<PyDict>(py).unwrap();
            assert_eq!(dict.get_item("status").unwrap().unwrap().extract::<String>().unwrap(), "tombstone_preserved");
        });
    }
}
