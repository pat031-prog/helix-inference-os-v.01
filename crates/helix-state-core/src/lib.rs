use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

const MAGIC: &[u8] = b"HLXSTATE1\n";
pub const CHUNK_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct StagingArray {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct StagingManifest {
    pub arrays: Vec<StagingArray>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlxArrayEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub byte_length: u64,
    pub sha256: String,
    pub chunk_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlxManifest {
    pub format: String,
    pub chunk_size: usize,
    pub total_data_bytes: u64,
    pub arrays: Vec<HlxArrayEntry>,
    pub merkle_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlxReceipt {
    pub format: String,
    pub session_codec: String,
    pub session_hash: String,
    pub session_meta_hash: String,
    pub hlx_file_hash: String,
    pub merkle_root: String,
    pub array_count: usize,
    pub total_data_bytes: u64,
    pub session_total_bytes: u64,
    pub kv_cache_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReceipt {
    pub ok: bool,
    pub receipt: HlxReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlxBufferedReceipt {
    #[serde(flatten)]
    pub receipt: HlxReceipt,
    pub buffered_array_count: usize,
    pub copied_array_count: usize,
    pub hash_time_ms: f64,
    pub write_time_ms: f64,
    pub total_pack_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlxPendingReceipt {
    pub format: String,
    pub session_codec: String,
    pub audit_policy: String,
    pub audit_status: String,
    pub kv_cache_file: String,
    pub array_count: usize,
    pub total_data_bytes: u64,
    pub session_total_bytes: u64,
    pub fast_payload_checksum: String,
    pub write_time_ms: f64,
    pub total_pack_time_ms: f64,
}

#[derive(Debug)]
struct PreparedArray {
    staging: StagingArray,
    byte_length: u64,
    sha256: String,
    chunk_hashes: Vec<String>,
}

pub struct HlxBufferArray<'a> {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
    pub copied: bool,
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; CHUNK_SIZE];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn hash_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().to_vec()
}

fn merkle_root_hex(hex_hashes: &[String]) -> Result<String, Box<dyn std::error::Error>> {
    if hex_hashes.is_empty() {
        return Ok(sha256_bytes(&[]));
    }
    let mut level = hex_hashes
        .iter()
        .map(|value| hex::decode(value))
        .collect::<Result<Vec<_>, _>>()?;
    while level.len() > 1 {
        let mut next = Vec::new();
        for pair in level.chunks(2) {
            let left = &pair[0];
            let right = if pair.len() == 2 { &pair[1] } else { &pair[0] };
            next.push(hash_pair(left, right));
        }
        level = next;
    }
    Ok(hex::encode(&level[0]))
}

fn file_hashes(path: &Path) -> Result<(u64, String, Vec<String>), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut total = 0_u64;
    let mut full_hasher = Sha256::new();
    let mut chunk_hashes = Vec::new();
    let mut buffer = vec![0_u8; CHUNK_SIZE];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        full_hasher.update(&buffer[..count]);
        chunk_hashes.push(sha256_bytes(&buffer[..count]));
    }
    if chunk_hashes.is_empty() {
        chunk_hashes.push(sha256_bytes(&[]));
    }
    Ok((total, hex::encode(full_hasher.finalize()), chunk_hashes))
}

fn canonical_receipt_hash_payload(receipt: &HlxReceipt) -> serde_json::Value {
    serde_json::json!({
        "session_meta_hash": receipt.session_meta_hash,
        "merkle_root": receipt.merkle_root,
        "array_count": receipt.array_count,
        "total_data_bytes": receipt.total_data_bytes,
        "hlx_file_hash": receipt.hlx_file_hash,
    })
}

fn session_hash(receipt: &HlxReceipt) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec(&canonical_receipt_hash_payload(receipt))?;
    Ok(sha256_bytes(&bytes))
}

fn read_staging_manifest(staging_dir: &Path) -> Result<StagingManifest, Box<dyn std::error::Error>> {
    let payload = fs::read_to_string(staging_dir.join("arrays.json"))?;
    Ok(serde_json::from_str(&payload)?)
}

fn validate_staging_array(array: &StagingArray) -> Result<(), Box<dyn std::error::Error>> {
    validate_array_name_and_dtype(&array.name, &array.dtype)?;
    Ok(())
}

fn validate_array_name_and_dtype(name: &str, dtype: &str) -> Result<(), Box<dyn std::error::Error>> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!("invalid array name: {name}").into());
    }
    if dtype == "object" || dtype.contains('O') {
        return Err(format!("object arrays are not supported: {name}").into());
    }
    Ok(())
}

fn bytes_hashes(bytes: &[u8]) -> (String, Vec<String>) {
    let mut full_hasher = Sha256::new();
    full_hasher.update(bytes);
    let mut chunk_hashes = Vec::new();
    for chunk in bytes.chunks(CHUNK_SIZE) {
        chunk_hashes.push(sha256_bytes(chunk));
    }
    if chunk_hashes.is_empty() {
        chunk_hashes.push(sha256_bytes(&[]));
    }
    (hex::encode(full_hasher.finalize()), chunk_hashes)
}

fn fnv1a_update(mut state: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 1099511628211;
    for byte in bytes {
        state ^= *byte as u64;
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

fn fnv1a_hex(state: u64) -> String {
    format!("{state:016x}")
}

fn prepare_arrays(staging_dir: &Path) -> Result<Vec<PreparedArray>, Box<dyn std::error::Error>> {
    let manifest = read_staging_manifest(staging_dir)?;
    let mut prepared = Vec::new();
    for staging in manifest.arrays {
        validate_staging_array(&staging)?;
        let path = staging_dir.join(&staging.path);
        let (byte_length, sha256, chunk_hashes) = file_hashes(&path)?;
        prepared.push(PreparedArray {
            staging,
            byte_length,
            sha256,
            chunk_hashes,
        });
    }
    Ok(prepared)
}

pub fn pack_hlx_bundle(
    staging_dir: &Path,
    session_json: &Path,
    output_dir: &Path,
) -> Result<HlxReceipt, Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let prepared = prepare_arrays(staging_dir)?;
    let mut offset = 0_u64;
    let mut entries = Vec::new();
    let mut all_chunk_hashes = Vec::new();
    for item in &prepared {
        entries.push(HlxArrayEntry {
            name: item.staging.name.clone(),
            dtype: item.staging.dtype.clone(),
            shape: item.staging.shape.clone(),
            offset,
            byte_length: item.byte_length,
            sha256: item.sha256.clone(),
            chunk_hashes: item.chunk_hashes.clone(),
        });
        offset += item.byte_length;
        all_chunk_hashes.extend(item.chunk_hashes.clone());
    }
    let merkle_root = merkle_root_hex(&all_chunk_hashes)?;
    let manifest = HlxManifest {
        format: "helix-hlx-v0".to_string(),
        chunk_size: CHUNK_SIZE,
        total_data_bytes: offset,
        arrays: entries,
        merkle_root,
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let temp_hlx = output_dir.join("kv_cache.hlx.tmp");
    {
        let mut output = File::create(&temp_hlx)?;
        output.write_all(MAGIC)?;
        output.write_all(&(manifest_bytes.len() as u64).to_le_bytes())?;
        output.write_all(&manifest_bytes)?;
        for item in &prepared {
            let mut input = File::open(staging_dir.join(&item.staging.path))?;
            std::io::copy(&mut input, &mut output)?;
        }
        output.flush()?;
    }
    let hlx_path = output_dir.join("kv_cache.hlx");
    if hlx_path.exists() {
        fs::remove_file(&hlx_path)?;
    }
    fs::rename(&temp_hlx, &hlx_path)?;

    let mut meta_value: serde_json::Value = serde_json::from_str(&fs::read_to_string(session_json)?)?;
    if let Some(object) = meta_value.as_object_mut() {
        object.insert("session_codec".to_string(), serde_json::json!("rust-hlx"));
        object.insert("kv_cache_file".to_string(), serde_json::json!("kv_cache.hlx"));
    }
    let output_session_json = output_dir.join("session.json");
    fs::write(&output_session_json, serde_json::to_vec_pretty(&meta_value)?)?;

    let session_meta_hash = sha256_file(&output_session_json)?;
    let hlx_file_hash = sha256_file(&hlx_path)?;
    let session_total_bytes = fs::metadata(&output_session_json)?.len() + fs::metadata(&hlx_path)?.len();
    let mut receipt = HlxReceipt {
        format: "helix-session-receipt-v0".to_string(),
        session_codec: "rust-hlx".to_string(),
        session_hash: String::new(),
        session_meta_hash,
        hlx_file_hash,
        merkle_root: manifest.merkle_root,
        array_count: manifest.arrays.len(),
        total_data_bytes: manifest.total_data_bytes,
        session_total_bytes,
        kv_cache_file: "kv_cache.hlx".to_string(),
    };
    receipt.session_hash = session_hash(&receipt)?;
    fs::write(
        output_dir.join("session-hlx-receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(receipt)
}

pub fn pack_hlx_buffers_bundle(
    session_meta_json: &str,
    output_dir: &Path,
    arrays: &[HlxBufferArray<'_>],
) -> Result<HlxBufferedReceipt, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    fs::create_dir_all(output_dir)?;

    let hash_start = Instant::now();
    let mut offset = 0_u64;
    let mut entries = Vec::new();
    let mut all_chunk_hashes = Vec::new();
    let mut copied_array_count = 0_usize;
    for array in arrays {
        validate_array_name_and_dtype(&array.name, &array.dtype)?;
        if array.copied {
            copied_array_count += 1;
        }
        let (sha256, chunk_hashes) = bytes_hashes(array.bytes);
        entries.push(HlxArrayEntry {
            name: array.name.clone(),
            dtype: array.dtype.clone(),
            shape: array.shape.clone(),
            offset,
            byte_length: array.bytes.len() as u64,
            sha256,
            chunk_hashes: chunk_hashes.clone(),
        });
        offset += array.bytes.len() as u64;
        all_chunk_hashes.extend(chunk_hashes);
    }
    let merkle_root = merkle_root_hex(&all_chunk_hashes)?;
    let hash_time_ms = hash_start.elapsed().as_secs_f64() * 1000.0;

    let manifest = HlxManifest {
        format: "helix-hlx-v0".to_string(),
        chunk_size: CHUNK_SIZE,
        total_data_bytes: offset,
        arrays: entries,
        merkle_root,
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let write_start = Instant::now();
    let temp_hlx = output_dir.join("kv_cache.hlx.tmp");
    {
        let mut output = File::create(&temp_hlx)?;
        output.write_all(MAGIC)?;
        output.write_all(&(manifest_bytes.len() as u64).to_le_bytes())?;
        output.write_all(&manifest_bytes)?;
        for array in arrays {
            output.write_all(array.bytes)?;
        }
        output.flush()?;
    }
    let hlx_path = output_dir.join("kv_cache.hlx");
    if hlx_path.exists() {
        fs::remove_file(&hlx_path)?;
    }
    fs::rename(&temp_hlx, &hlx_path)?;

    let mut meta_value: serde_json::Value = serde_json::from_str(session_meta_json)?;
    let requested_session_codec = meta_value
        .get("session_codec")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("rust-hlx-buffered")
        .to_string();
    if let Some(object) = meta_value.as_object_mut() {
        object.insert("session_codec".to_string(), serde_json::json!(requested_session_codec.clone()));
        object.insert("kv_cache_file".to_string(), serde_json::json!("kv_cache.hlx"));
    }
    let output_session_json = output_dir.join("session.json");
    fs::write(&output_session_json, serde_json::to_vec_pretty(&meta_value)?)?;
    let write_time_ms = write_start.elapsed().as_secs_f64() * 1000.0;

    let session_meta_hash = sha256_file(&output_session_json)?;
    let hlx_file_hash = sha256_file(&hlx_path)?;
    let session_total_bytes = fs::metadata(&output_session_json)?.len() + fs::metadata(&hlx_path)?.len();
    let mut receipt = HlxReceipt {
        format: "helix-session-receipt-v0".to_string(),
        session_codec: requested_session_codec,
        session_hash: String::new(),
        session_meta_hash,
        hlx_file_hash,
        merkle_root: manifest.merkle_root,
        array_count: manifest.arrays.len(),
        total_data_bytes: manifest.total_data_bytes,
        session_total_bytes,
        kv_cache_file: "kv_cache.hlx".to_string(),
    };
    receipt.session_hash = session_hash(&receipt)?;
    let buffered = HlxBufferedReceipt {
        receipt,
        buffered_array_count: arrays.len(),
        copied_array_count,
        hash_time_ms,
        write_time_ms,
        total_pack_time_ms: total_start.elapsed().as_secs_f64() * 1000.0,
    };
    fs::write(
        output_dir.join("session-hlx-receipt.json"),
        serde_json::to_vec_pretty(&buffered)?,
    )?;
    Ok(buffered)
}

pub fn pack_hlx_buffers_pending_bundle(
    session_meta_json: &str,
    output_dir: &Path,
    arrays: &[HlxBufferArray<'_>],
) -> Result<HlxPendingReceipt, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    fs::create_dir_all(output_dir)?;

    let mut offset = 0_u64;
    let mut entries = Vec::new();
    for array in arrays {
        validate_array_name_and_dtype(&array.name, &array.dtype)?;
        entries.push(HlxArrayEntry {
            name: array.name.clone(),
            dtype: array.dtype.clone(),
            shape: array.shape.clone(),
            offset,
            byte_length: array.bytes.len() as u64,
            sha256: String::new(),
            chunk_hashes: Vec::new(),
        });
        offset += array.bytes.len() as u64;
    }

    let manifest = HlxManifest {
        format: "helix-hlx-v0".to_string(),
        chunk_size: CHUNK_SIZE,
        total_data_bytes: offset,
        arrays: entries,
        merkle_root: "pending".to_string(),
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let write_start = Instant::now();
    let temp_hlx = output_dir.join("kv_cache.hlx.tmp");
    let mut checksum = 14695981039346656037_u64;
    {
        let mut output = File::create(&temp_hlx)?;
        output.write_all(MAGIC)?;
        output.write_all(&(manifest_bytes.len() as u64).to_le_bytes())?;
        output.write_all(&manifest_bytes)?;
        for array in arrays {
            output.write_all(array.bytes)?;
            checksum = fnv1a_update(checksum, array.bytes);
        }
        output.flush()?;
    }
    let hlx_path = output_dir.join("kv_cache.hlx");
    if hlx_path.exists() {
        fs::remove_file(&hlx_path)?;
    }
    fs::rename(&temp_hlx, &hlx_path)?;

    let mut meta_value: serde_json::Value = serde_json::from_str(session_meta_json)?;
    let requested_session_codec = meta_value
        .get("session_codec")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("rust-hlx-buffered")
        .to_string();
    if let Some(object) = meta_value.as_object_mut() {
        object.insert("session_codec".to_string(), serde_json::json!(requested_session_codec.clone()));
        object.insert("kv_cache_file".to_string(), serde_json::json!("kv_cache.hlx"));
        object.insert("audit_policy".to_string(), serde_json::json!("deferred"));
        object.insert("audit_status".to_string(), serde_json::json!("pending"));
    }
    let output_session_json = output_dir.join("session.json");
    fs::write(&output_session_json, serde_json::to_vec_pretty(&meta_value)?)?;
    let write_time_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    let session_total_bytes = fs::metadata(&output_session_json)?.len() + fs::metadata(&hlx_path)?.len();
    let pending = HlxPendingReceipt {
        format: "helix-session-pending-v0".to_string(),
        session_codec: requested_session_codec,
        audit_policy: "deferred".to_string(),
        audit_status: "pending".to_string(),
        kv_cache_file: "kv_cache.hlx".to_string(),
        array_count: arrays.len(),
        total_data_bytes: manifest.total_data_bytes,
        session_total_bytes,
        fast_payload_checksum: fnv1a_hex(checksum),
        write_time_ms,
        total_pack_time_ms: total_start.elapsed().as_secs_f64() * 1000.0,
    };
    fs::write(
        output_dir.join("session-hlx-receipt.json"),
        serde_json::to_vec_pretty(&pending)?,
    )?;
    Ok(pending)
}

fn read_hlx_manifest(file: &mut File) -> Result<(HlxManifest, u64), Box<dyn std::error::Error>> {
    let mut magic = vec![0_u8; MAGIC.len()];
    file.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err("invalid HeliX .hlx magic".into());
    }
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)?;
    let manifest_len = u64::from_le_bytes(len_bytes);
    let mut manifest_bytes = vec![0_u8; manifest_len as usize];
    file.read_exact(&mut manifest_bytes)?;
    let manifest: HlxManifest = serde_json::from_slice(&manifest_bytes)?;
    Ok((manifest, MAGIC.len() as u64 + 8 + manifest_len))
}

pub fn unpack_hlx_bundle(bundle_path: &Path, output_staging: &Path) -> Result<HlxManifest, Box<dyn std::error::Error>> {
    fs::create_dir_all(output_staging)?;
    let mut file = File::open(bundle_path)?;
    let (manifest, data_start) = read_hlx_manifest(&mut file)?;
    let arrays_dir = output_staging.join("arrays");
    fs::create_dir_all(&arrays_dir)?;
    let mut staging_arrays = Vec::new();
    for entry in &manifest.arrays {
        let mut bytes = vec![0_u8; entry.byte_length as usize];
        file.seek(SeekFrom::Start(data_start + entry.offset))?;
        file.read_exact(&mut bytes)?;
        let hash = sha256_bytes(&bytes);
        if hash != entry.sha256 {
            return Err(format!("array hash mismatch: {}", entry.name).into());
        }
        let file_name = format!("{}.raw", entry.name);
        fs::write(arrays_dir.join(&file_name), &bytes)?;
        staging_arrays.push(serde_json::json!({
            "name": entry.name,
            "dtype": entry.dtype,
            "shape": entry.shape,
            "path": format!("arrays/{file_name}"),
        }));
    }
    fs::write(
        output_staging.join("arrays.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "arrays": staging_arrays }))?,
    )?;
    Ok(manifest)
}

pub fn verify_hlx_session(session_dir: &Path) -> Result<VerifyReceipt, Box<dyn std::error::Error>> {
    let session_json = session_dir.join("session.json");
    let hlx_path = session_dir.join("kv_cache.hlx");
    let meta_value: serde_json::Value = serde_json::from_str(&fs::read_to_string(&session_json)?)?;
    let session_codec = meta_value
        .get("session_codec")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("rust-hlx")
        .to_string();
    let mut file = File::open(&hlx_path)?;
    let (manifest, data_start) = read_hlx_manifest(&mut file)?;
    let mut all_chunk_hashes = Vec::new();
    for entry in &manifest.arrays {
        let mut bytes = vec![0_u8; entry.byte_length as usize];
        file.seek(SeekFrom::Start(data_start + entry.offset))?;
        file.read_exact(&mut bytes)?;
        if sha256_bytes(&bytes) != entry.sha256 {
            return Err(format!("array hash mismatch: {}", entry.name).into());
        }
        let mut chunk_hashes = Vec::new();
        for chunk in bytes.chunks(CHUNK_SIZE) {
            chunk_hashes.push(sha256_bytes(chunk));
        }
        if chunk_hashes.is_empty() {
            chunk_hashes.push(sha256_bytes(&[]));
        }
        if chunk_hashes != entry.chunk_hashes {
            return Err(format!("chunk hash mismatch: {}", entry.name).into());
        }
        all_chunk_hashes.extend(chunk_hashes);
    }
    let computed_root = merkle_root_hex(&all_chunk_hashes)?;
    if computed_root != manifest.merkle_root {
        return Err("merkle root mismatch".into());
    }
    let session_meta_hash = sha256_file(&session_json)?;
    let hlx_file_hash = sha256_file(&hlx_path)?;
    let session_total_bytes = fs::metadata(&session_json)?.len() + fs::metadata(&hlx_path)?.len();
    let mut receipt = HlxReceipt {
        format: "helix-session-receipt-v0".to_string(),
        session_codec,
        session_hash: String::new(),
        session_meta_hash,
        hlx_file_hash,
        merkle_root: manifest.merkle_root,
        array_count: manifest.arrays.len(),
        total_data_bytes: manifest.total_data_bytes,
        session_total_bytes,
        kv_cache_file: "kv_cache.hlx".to_string(),
    };
    receipt.session_hash = session_hash(&receipt)?;
    Ok(VerifyReceipt { ok: true, receipt })
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::buffer::PyBuffer;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "python")]
#[pyfunction]
fn pack_hlx(staging_dir: String, session_json: String, output_dir: String) -> PyResult<String> {
    pack_hlx_bundle(Path::new(&staging_dir), Path::new(&session_json), Path::new(&output_dir))
        .and_then(|receipt| Ok(serde_json::to_string(&receipt)?))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn pack_hlx_buffers(_py: Python<'_>, session_meta_json: String, output_dir: String, array_specs: &Bound<'_, PyList>) -> PyResult<String> {
    struct PyBufferSpec {
        name: String,
        dtype: String,
        shape: Vec<usize>,
        copied: bool,
        buffer: PyBuffer<u8>,
    }

    let mut buffer_specs = Vec::new();
    for item in array_specs.iter() {
        let spec = item.downcast::<PyDict>()?;
        let name = spec
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing name"))?
            .extract::<String>()?;
        let dtype = spec
            .get_item("dtype")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing dtype"))?
            .extract::<String>()?;
        let shape = spec
            .get_item("shape")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing shape"))?
            .extract::<Vec<usize>>()?;
        let byte_length = spec
            .get_item("byte_length")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing byte_length"))?
            .extract::<usize>()?;
        let copied = spec
            .get_item("copied")?
            .and_then(|value| value.extract::<bool>().ok())
            .unwrap_or(false);
        let buffer_obj = spec
            .get_item("buffer")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing buffer"))?;
        let buffer = PyBuffer::<u8>::get_bound(&buffer_obj)?;
        if !buffer.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array buffer is not C-contiguous: {name}"
            )));
        }
        if buffer.len_bytes() != byte_length {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array byte_length mismatch for {name}: declared {byte_length}, actual {}",
                buffer.len_bytes()
            )));
        }
        buffer_specs.push(PyBufferSpec {
            name,
            dtype,
            shape,
            copied,
            buffer,
        });
    }
    let borrowed = buffer_specs
        .iter()
        .map(|spec| HlxBufferArray {
            name: spec.name.clone(),
            dtype: spec.dtype.clone(),
            shape: spec.shape.clone(),
            bytes: unsafe {
                std::slice::from_raw_parts(spec.buffer.buf_ptr() as *const u8, spec.buffer.len_bytes())
            },
            copied: spec.copied,
        })
        .collect::<Vec<_>>();
    pack_hlx_buffers_bundle(&session_meta_json, Path::new(&output_dir), &borrowed)
        .and_then(|receipt| Ok(serde_json::to_string(&receipt)?))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
fn py_buffer_specs(array_specs: &Bound<'_, PyList>) -> PyResult<Vec<(String, String, Vec<usize>, bool, PyBuffer<u8>)>> {
    let mut buffer_specs = Vec::new();
    for item in array_specs.iter() {
        let spec = item.downcast::<PyDict>()?;
        let name = spec
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing name"))?
            .extract::<String>()?;
        let dtype = spec
            .get_item("dtype")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing dtype"))?
            .extract::<String>()?;
        let shape = spec
            .get_item("shape")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing shape"))?
            .extract::<Vec<usize>>()?;
        let byte_length = spec
            .get_item("byte_length")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing byte_length"))?
            .extract::<usize>()?;
        let copied = spec
            .get_item("copied")?
            .and_then(|value| value.extract::<bool>().ok())
            .unwrap_or(false);
        let buffer_obj = spec
            .get_item("buffer")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array spec missing buffer"))?;
        let buffer = PyBuffer::<u8>::get_bound(&buffer_obj)?;
        if !buffer.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array buffer is not C-contiguous: {name}"
            )));
        }
        if buffer.len_bytes() != byte_length {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array byte_length mismatch for {name}: declared {byte_length}, actual {}",
                buffer.len_bytes()
            )));
        }
        buffer_specs.push((name, dtype, shape, copied, buffer));
    }
    Ok(buffer_specs)
}

#[cfg(feature = "python")]
#[pyfunction]
fn pack_hlx_buffers_pending(_py: Python<'_>, session_meta_json: String, output_dir: String, array_specs: &Bound<'_, PyList>) -> PyResult<String> {
    let buffer_specs = py_buffer_specs(array_specs)?;
    let borrowed = buffer_specs
        .iter()
        .map(|spec| HlxBufferArray {
            name: spec.0.clone(),
            dtype: spec.1.clone(),
            shape: spec.2.clone(),
            bytes: unsafe {
                std::slice::from_raw_parts(spec.4.buf_ptr() as *const u8, spec.4.len_bytes())
            },
            copied: spec.3,
        })
        .collect::<Vec<_>>();
    pack_hlx_buffers_pending_bundle(&session_meta_json, Path::new(&output_dir), &borrowed)
        .and_then(|receipt| Ok(serde_json::to_string(&receipt)?))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn unpack_hlx(bundle_path: String, output_staging: String) -> PyResult<String> {
    unpack_hlx_bundle(Path::new(&bundle_path), Path::new(&output_staging))
        .and_then(|manifest| Ok(serde_json::to_string(&manifest)?))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn verify_hlx(session_dir: String) -> PyResult<String> {
    verify_hlx_session(Path::new(&session_dir))
        .and_then(|receipt| Ok(serde_json::to_string(&receipt)?))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pymodule]
fn _helix_state_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pack_hlx, module)?)?;
    module.add_function(wrap_pyfunction!(pack_hlx_buffers, module)?)?;
    module.add_function(wrap_pyfunction!(pack_hlx_buffers_pending, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_hlx, module)?)?;
    module.add_function(wrap_pyfunction!(verify_hlx, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_array(staging: &Path, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) -> serde_json::Value {
        let arrays_dir = staging.join("arrays");
        fs::create_dir_all(&arrays_dir).unwrap();
        let path = format!("arrays/{name}.raw");
        fs::write(staging.join(&path), bytes).unwrap();
        serde_json::json!({"name": name, "dtype": dtype, "shape": shape, "path": path})
    }

    #[test]
    fn buffered_pack_preserves_arrays_and_reports_counts() {
        let temp = tempdir().unwrap();
        let output = temp.path().join("buffered-session");
        let float_bytes = [0_u8, 0, 0, 0, 1, 2, 3, 4];
        let int_bytes = [1_u8, 2, 3, 255];
        let arrays = vec![
            HlxBufferArray {
                name: "float32_values".to_string(),
                dtype: "float32".to_string(),
                shape: vec![2, 2],
                bytes: &float_bytes,
                copied: false,
            },
            HlxBufferArray {
                name: "int8_values".to_string(),
                dtype: "int8".to_string(),
                shape: vec![4],
                bytes: &int_bytes,
                copied: true,
            },
        ];
        let receipt = pack_hlx_buffers_bundle(r#"{"model_ref":"test"}"#, &output, &arrays).unwrap();
        assert_eq!(receipt.receipt.array_count, 2);
        assert_eq!(receipt.buffered_array_count, 2);
        assert_eq!(receipt.copied_array_count, 1);
        assert_eq!(receipt.receipt.session_codec, "rust-hlx-buffered");

        let verify = verify_hlx_session(&output).unwrap();
        assert!(verify.ok);
        assert_eq!(verify.receipt.session_hash, receipt.receipt.session_hash);
        assert_eq!(verify.receipt.session_codec, "rust-hlx-buffered");
    }

    #[test]
    fn buffered_pack_rejects_object_dtype() {
        let temp = tempdir().unwrap();
        let bytes = [1_u8, 2, 3, 4];
        let arrays = vec![HlxBufferArray {
            name: "bad".to_string(),
            dtype: "object".to_string(),
            shape: vec![4],
            bytes: &bytes,
            copied: false,
        }];
        let err = pack_hlx_buffers_bundle(r#"{"model_ref":"test"}"#, temp.path(), &arrays).unwrap_err();
        assert!(err.to_string().contains("object arrays"));
    }

    #[test]
    fn pending_pack_writes_unverified_receipt_without_hashing_manifest() {
        let temp = tempdir().unwrap();
        let output = temp.path().join("pending-session");
        let bytes = [1_u8, 2, 3, 4];
        let arrays = vec![HlxBufferArray {
            name: "values".to_string(),
            dtype: "uint8".to_string(),
            shape: vec![4],
            bytes: &bytes,
            copied: false,
        }];

        let receipt = pack_hlx_buffers_pending_bundle(
            r#"{"model_ref":"test","session_codec":"rust-hlx-buffered-flat"}"#,
            &output,
            &arrays,
        )
        .unwrap();
        assert_eq!(receipt.audit_status, "pending");
        assert_eq!(receipt.session_codec, "rust-hlx-buffered-flat");
        assert_eq!(receipt.array_count, 1);

        let mut file = File::open(output.join("kv_cache.hlx")).unwrap();
        let (manifest, _) = read_hlx_manifest(&mut file).unwrap();
        assert_eq!(manifest.merkle_root, "pending");
        assert_eq!(manifest.arrays[0].sha256, "");
        assert!(verify_hlx_session(&output).is_err());
    }

    #[test]
    fn hlx_roundtrip_preserves_array_metadata_and_merkle() {
        let temp = tempdir().unwrap();
        let staging = temp.path().join("staging");
        let output = temp.path().join("session");
        fs::create_dir_all(&staging).unwrap();
        let arrays = vec![
            write_array(&staging, "float32_values", "float32", &[2, 2], &[0, 0, 0, 0, 1, 2, 3, 4]),
            write_array(&staging, "int8_values", "int8", &[4], &[1, 2, 3, 255]),
            write_array(&staging, "uint8_values", "uint8", &[3], &[4, 5, 6]),
            write_array(&staging, "int64_values", "int64", &[1], &[7, 0, 0, 0, 0, 0, 0, 0]),
        ];
        fs::write(
            staging.join("arrays.json"),
            serde_json::to_vec(&serde_json::json!({"arrays": arrays})).unwrap(),
        )
        .unwrap();
        let session_json = staging.join("session.json");
        fs::write(&session_json, br#"{"format":"test"}"#).unwrap();

        let receipt = pack_hlx_bundle(&staging, &session_json, &output).unwrap();
        assert_eq!(receipt.array_count, 4);
        assert!(!receipt.merkle_root.is_empty());
        assert!(verify_hlx_session(&output).unwrap().ok);

        let unpacked = temp.path().join("unpacked");
        let manifest = unpack_hlx_bundle(&output.join("kv_cache.hlx"), &unpacked).unwrap();
        assert_eq!(manifest.arrays.len(), 4);
        assert!(unpacked.join("arrays.json").exists());
    }

    #[test]
    fn tampering_one_byte_fails_verification() {
        let temp = tempdir().unwrap();
        let staging = temp.path().join("staging");
        let output = temp.path().join("session");
        fs::create_dir_all(&staging).unwrap();
        let arrays = vec![write_array(&staging, "values", "uint8", &[3], &[1, 2, 3])];
        fs::write(
            staging.join("arrays.json"),
            serde_json::to_vec(&serde_json::json!({"arrays": arrays})).unwrap(),
        )
        .unwrap();
        let session_json = staging.join("session.json");
        fs::write(&session_json, br#"{"format":"test"}"#).unwrap();
        pack_hlx_bundle(&staging, &session_json, &output).unwrap();

        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(output.join("kv_cache.hlx"))
            .unwrap();
        let len = file.metadata().unwrap().len();
        file.seek(SeekFrom::Start(len - 1)).unwrap();
        file.write_all(&[9]).unwrap();

        assert!(verify_hlx_session(&output).is_err());
    }

    #[test]
    fn rejects_object_dtype() {
        let temp = tempdir().unwrap();
        let staging = temp.path().join("staging");
        fs::create_dir_all(&staging).unwrap();
        let arrays = vec![write_array(&staging, "values", "object", &[1], &[0])];
        fs::write(
            staging.join("arrays.json"),
            serde_json::to_vec(&serde_json::json!({"arrays": arrays})).unwrap(),
        )
        .unwrap();
        assert!(prepare_arrays(&staging).is_err());
    }
}
