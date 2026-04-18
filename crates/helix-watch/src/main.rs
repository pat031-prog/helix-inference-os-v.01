use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use flate2::read::GzDecoder;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Parser)]
struct Args {
    /// Path to a stress dashboard JSON, a mission JSON, a receipt bundle, or a directory containing the dashboard.
    path: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
struct Receipt {
    run_id: String,
    layer_index: i64,
    token_index: i64,
    state_kind: String,
    ratio: f64,
    rel_rmse: f64,
    clip_pct: f64,
    finite_after: bool,
    fallback_precision: String,
    #[serde(default)]
    fallback_reason: Option<String>,
}

#[derive(Debug, Default)]
struct ReceiptSummary {
    run_id: String,
    receipts: usize,
    max_token_index: i64,
    avg_ratio: f64,
    max_clip_pct: f64,
    max_rel_rmse: f64,
    fallback_counts: BTreeMap<String, usize>,
    unstable_layers: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Default)]
struct MissionView {
    mission_id: String,
    title: String,
    strongest_claim: String,
    headline_metrics: Vec<(String, String)>,
    preview_lines: Vec<String>,
    footer_lines: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct DashboardView {
    title: String,
    profile: String,
    source_path: PathBuf,
    missions: Vec<MissionView>,
}

#[derive(Debug)]
struct AppState {
    dashboard: DashboardView,
    selected: usize,
}

fn open_reader(path: &PathBuf) -> std::io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path.extension().and_then(|value| value.to_str()) == Some("gz") {
        return Ok(Box::new(BufReader::new(GzDecoder::new(file))));
    }
    Ok(Box::new(BufReader::new(file)))
}

fn load_receipt_summary(path: &PathBuf) -> std::io::Result<ReceiptSummary> {
    let mut summary = ReceiptSummary::default();
    let reader = open_reader(path)?;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let receipt: Receipt = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        summary.run_id = receipt.run_id.clone();
        summary.receipts += 1;
        summary.max_token_index = summary.max_token_index.max(receipt.token_index);
        summary.avg_ratio += receipt.ratio;
        summary.max_clip_pct = summary.max_clip_pct.max(receipt.clip_pct);
        summary.max_rel_rmse = summary.max_rel_rmse.max(receipt.rel_rmse);
        *summary
            .fallback_counts
            .entry(receipt.fallback_precision.clone())
            .or_insert(0) += 1;
        if !receipt.finite_after || receipt.fallback_precision != "int4" {
            let key = format!(
                "layer {} {} ({})",
                receipt.layer_index, receipt.state_kind, receipt.fallback_precision
            );
            *summary.unstable_layers.entry(key).or_insert(0) += 1;
        }
        if let Some(reason) = receipt.fallback_reason.as_deref() {
            let key = format!("reason: {reason}");
            *summary.unstable_layers.entry(key).or_insert(0) += 1;
        }
    }
    if summary.receipts > 0 {
        summary.avg_ratio /= summary.receipts as f64;
    }
    Ok(summary)
}

fn scalar_string(value: &Value) -> String {
    match value {
        Value::Null => "--".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

fn collect_headline_metrics(value: &Value) -> Vec<(String, String)> {
    let mut metrics = Vec::new();
    if let Some(object) = value.as_object() {
        for (key, value) in object.iter().take(8) {
            metrics.push((key.replace('_', " "), scalar_string(value)));
        }
    }
    metrics
}

fn preview_lines_for_mission(value: &Value) -> Vec<String> {
    let mission_id = value
        .get("mission_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match mission_id {
        "long-context-coder" => {
            let mut lines = Vec::new();
            if let Some(tasks) = value.get("tasks").and_then(Value::as_array) {
                for task in tasks.iter().take(4) {
                    let task_id = task
                        .get("task_id")
                        .and_then(Value::as_str)
                        .unwrap_or("task");
                    let ratio = task
                        .get("hybrid_total_runtime_cache_ratio_vs_native")
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string());
                    let hits = task
                        .get("identifier_hits")
                        .and_then(Value::as_array)
                        .map(|items| items.len())
                        .unwrap_or(0);
                    let preview = task
                        .get("combined")
                        .and_then(|item| item.get("answer_preview"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    lines.push(format!("{task_id}: ratio={ratio} identifier_hits={hits}"));
                    if !preview.is_empty() {
                        lines.push(preview.to_string());
                    }
                }
            }
            lines
        }
        "state-juggler" => {
            let mut lines = Vec::new();
            if let Some(phase_a) = value.get("phase_a") {
                let preview = phase_a
                    .get("answer_preview")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let keywords = phase_a
                    .get("context_keywords_hit")
                    .and_then(Value::as_array)
                    .map(|items| items.len())
                    .unwrap_or(0);
                lines.push(format!("phase-a keywords={keywords}"));
                if !preview.is_empty() {
                    lines.push(preview.to_string());
                }
            }
            if let Some(phase_b) = value.get("phase_b") {
                let preview = phase_b
                    .get("answer_preview")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let hash_match = value
                    .get("headline_metrics")
                    .and_then(|item| item.get("hash_match"))
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                lines.push(format!("restore hash_match={hash_match}"));
                if !preview.is_empty() {
                    lines.push(preview.to_string());
                }
            }
            lines
        }
        "context-switcher" => {
            let mut lines = Vec::new();
            if let Some(preview) = value
                .get("combined")
                .and_then(|item| item.get("answer_preview"))
                .and_then(Value::as_str)
            {
                lines.push(preview.to_string());
            }
            if let Some(steps) = value.get("step_windows").and_then(Value::as_array) {
                for step in steps.iter().take(5) {
                    let index = step.get("step_index").map(scalar_string).unwrap_or_else(|| "0".to_string());
                    let ratio = step
                        .get("step_time_ratio_vs_native")
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string());
                    let promoted = step
                        .get("promoted_block_count")
                        .map(scalar_string)
                        .unwrap_or_else(|| "0".to_string());
                    lines.push(format!("step {index}: time_ratio={ratio} promoted_blocks={promoted}"));
                }
            }
            lines
        }
        "restore-equivalence" => {
            let mut lines = Vec::new();
            if let Some(metrics) = value.get("headline_metrics") {
                let hash = metrics
                    .get("hash_match")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let tokens = metrics
                    .get("generated_ids_match")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let top1 = metrics
                    .get("top1_match_all")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let max_delta = metrics
                    .get("max_abs_logit_delta")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                lines.push(format!("hash_match={hash} generated_ids_match={tokens} top1_match_all={top1}"));
                lines.push(format!("max_abs_logit_delta={max_delta}"));
            }
            if let Some(preview) = value
                .get("pre_restore")
                .and_then(|item| item.get("answer_preview"))
                .and_then(Value::as_str)
            {
                lines.push(format!("pre: {preview}"));
            }
            if let Some(preview) = value
                .get("post_restore")
                .and_then(|item| item.get("answer_preview"))
                .and_then(Value::as_str)
            {
                lines.push(format!("post: {preview}"));
            }
            lines
        }
        _ => {
            let mut lines = Vec::new();
            if let Some(text) = value.get("strongest_claim").and_then(Value::as_str) {
                lines.push(text.to_string());
            }
            lines
        }
    }
}

fn mission_from_json(value: &Value) -> MissionView {
    let mission_id = value
        .get("mission_id")
        .and_then(Value::as_str)
        .unwrap_or("artifact")
        .to_string();
    let title = value
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("Artifact")
        .to_string();
    let strongest_claim = value
        .get("strongest_claim")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let headline_metrics = collect_headline_metrics(value.get("headline_metrics").unwrap_or(&Value::Null));
    let preview_lines = preview_lines_for_mission(value);
    let mut footer_lines = Vec::new();
    if let Some(profile) = value.get("profile").and_then(Value::as_str) {
        footer_lines.push(format!("profile: {profile}"));
    }
    if let Some(model_ref) = value.get("model_ref").and_then(Value::as_str) {
        footer_lines.push(format!("model: {model_ref}"));
    }
    if let Some(artifact_path) = value.get("artifact_path").and_then(Value::as_str) {
        footer_lines.push(format!("artifact: {artifact_path}"));
    }
    MissionView {
        mission_id,
        title,
        strongest_claim,
        headline_metrics,
        preview_lines,
        footer_lines,
    }
}

fn dashboard_from_json(value: &Value, source_path: &Path) -> DashboardView {
    if value
        .get("title")
        .and_then(Value::as_str)
        .map(|title| title.contains("Session Core"))
        .unwrap_or(false)
    {
        let empty_models = Vec::new();
        let missions = value
            .get("models")
            .and_then(Value::as_array)
            .unwrap_or(&empty_models)
            .iter()
            .map(|model| {
                let model_key = model
                    .get("model_key")
                    .and_then(Value::as_str)
                    .unwrap_or("model");
                let status = model
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("--");
                let rust_save = model
                    .get("rust_hlx_save_time_ms")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let py_save = model
                    .get("python_npz_save_time_ms")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let hash = model
                    .get("hash_match")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let token_match = model
                    .get("generated_ids_match")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let null_value = Value::Null;
                let repeat = model.get("repeat_benchmark").unwrap_or(&null_value);
                let p50 = repeat
                    .get("save_time_ms_p50")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let p95 = repeat
                    .get("save_time_ms_p95")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let flatten_copy = repeat
                    .get("flatten_copy_time_ms_p50")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let original_arrays = repeat
                    .get("original_array_count_median")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let flat_groups = repeat
                    .get("flat_group_count_median")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let buffer_specs = repeat
                    .get("buffer_spec_count_median")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let pending_p50 = repeat
                    .get("time_to_pending_ms_p50")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let verified_p50 = repeat
                    .get("time_to_verified_ms_p50")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                let audit_policy = model
                    .get("audit_policy")
                    .map(scalar_string)
                    .unwrap_or_else(|| "--".to_string());
                MissionView {
                    mission_id: format!("session-core-{model_key}"),
                    title: format!("Session Core / {model_key}"),
                    strongest_claim: format!("status={status} rust_hlx_save_ms={rust_save} python_npz_save_ms={py_save} p50={p50}"),
                    headline_metrics: vec![
                        ("status".to_string(), status.to_string()),
                        ("rust save ms".to_string(), rust_save),
                        ("python save ms".to_string(), py_save),
                        ("save p50 ms".to_string(), p50),
                        ("save p95 ms".to_string(), p95),
                        ("pending p50 ms".to_string(), pending_p50.clone()),
                        ("verified p50 ms".to_string(), verified_p50.clone()),
                        ("flatten copy p50 ms".to_string(), flatten_copy.clone()),
                        ("arrays to groups".to_string(), format!("{original_arrays} -> {flat_groups}")),
                        ("hash match".to_string(), hash),
                        ("token match".to_string(), token_match),
                    ],
                    preview_lines: vec![
                        format!("model: {}", model.get("model_ref").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                        format!("merkle: {}", model.get("merkle_root").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                        format!("audit policy: {audit_policy}"),
                        format!("pending -> verified p50 ms: {pending_p50} -> {verified_p50}"),
                        format!("buffer specs: {buffer_specs}"),
                        format!("flatten copy p50 ms: {flatten_copy}"),
                        format!("pre: {}", model.get("pre_answer_preview").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                        format!("post: {}", model.get("post_answer_preview").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                    ],
                    footer_lines: vec![format!("artifact: {}", source_path.display())],
                }
            })
            .collect::<Vec<_>>();
        return DashboardView {
            title: value
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("HeliX Local Session Core")
                .to_string(),
            profile: value
                .get("profile")
                .and_then(Value::as_str)
                .unwrap_or("--")
                .to_string(),
            source_path: source_path.to_path_buf(),
            missions,
        };
    }
    if value.get("title").and_then(Value::as_str) == Some("HeliX Local Multimodel Hypervisor v0") {
        let task_events = value
            .get("task_events")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let models_used = value
            .get("models_used")
            .and_then(Value::as_array)
            .map(|items| items.iter().map(scalar_string).collect::<Vec<_>>().join(", "))
            .unwrap_or_else(|| "--".to_string());
        let preview_lines = task_events
            .iter()
            .take(12)
            .map(|event| {
                format!(
                    "{}:{} model={} restored={} save_ms={} load_ms={} preview={}",
                    event.get("task_id").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("agent_id").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("model_id").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event
                        .get("restore_hash_match")
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string()),
                    event.get("save_time_ms").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("load_time_ms").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event
                        .get("handoff_out")
                        .or_else(|| event.get("answer_preview"))
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string()),
                )
            })
            .collect::<Vec<_>>();
        return DashboardView {
            title: "HeliX Local Multimodel Hypervisor v0".to_string(),
            profile: format!(
                "{} / models: {}",
                value.get("scenario").map(scalar_string).unwrap_or_else(|| "demo".to_string()),
                models_used
            ),
            source_path: source_path.to_path_buf(),
            missions: vec![MissionView {
                mission_id: "multimodel-hypervisor".to_string(),
                title: "Multimodel Session Hypervisor".to_string(),
                strongest_claim: format!(
                    "swaps={} restored={} final audits={}",
                    value.get("model_swaps").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    value
                        .get("all_restore_hash_matches")
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string()),
                    value
                        .get("all_final_audits_verified")
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string())
                ),
                headline_metrics: vec![
                    ("status".to_string(), value.get("status").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                    ("models".to_string(), models_used),
                    ("model swaps".to_string(), value.get("model_swaps").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                    (
                        "restored sessions".to_string(),
                        value
                            .get("restored_session_count")
                            .map(scalar_string)
                            .unwrap_or_else(|| "--".to_string()),
                    ),
                    (
                        "restore hash match".to_string(),
                        value
                            .get("all_restore_hash_matches")
                            .map(scalar_string)
                            .unwrap_or_else(|| "--".to_string()),
                    ),
                    (
                        "final audits".to_string(),
                        value
                            .get("all_final_audits_verified")
                            .map(scalar_string)
                            .unwrap_or_else(|| "--".to_string()),
                    ),
                    (
                        "wall time s".to_string(),
                        value.get("total_wall_time_s").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    ),
                ],
                preview_lines,
                footer_lines: vec![
                    format!("artifact: {}", source_path.display()),
                    value
                        .get("state_sharing")
                        .map(scalar_string)
                        .unwrap_or_else(|| "state sharing: none".to_string()),
                ],
            }],
        };
    }
    if value.get("title").and_then(Value::as_str) == Some("HeliX Local Agent Hypervisor v0") {
        let events = value.get("events").and_then(Value::as_array).cloned().unwrap_or_default();
        let preview_lines = events
            .iter()
            .filter(|event| event.get("phase").and_then(Value::as_str) == Some("timeslice"))
            .take(12)
            .map(|event| {
                format!(
                    "{}:{} round {} loaded={} audit={} save_ms={} load_ms={} preview={}",
                    event.get("agent_id").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("role").map(scalar_string).unwrap_or_else(|| "agent".to_string()),
                    event.get("round_index").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("hash_match").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("audit_status").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("save_time_ms").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event.get("load_time_ms").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    event
                        .get("handoff_out")
                        .or_else(|| event.get("answer_preview"))
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string()),
                )
            })
            .collect::<Vec<_>>();
        return DashboardView {
            title: "HeliX Local Agent Hypervisor v0".to_string(),
            profile: format!(
                "{} / {} agents / {} rounds",
                value.get("scenario").map(scalar_string).unwrap_or_else(|| "demo".to_string()),
                value.get("agents").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                value.get("rounds").map(scalar_string).unwrap_or_else(|| "--".to_string())
            ),
            source_path: source_path.to_path_buf(),
            missions: vec![MissionView {
                mission_id: "agent-hypervisor".to_string(),
                title: "Agent Hypervisor".to_string(),
                strongest_claim: format!(
                    "pending loaded={} final audits={}",
                    value.get("all_pending_receipts_loaded")
                        .or_else(|| value.get("all_restore_hash_matches"))
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string()),
                    value.get("all_final_audits_verified")
                        .map(scalar_string)
                        .unwrap_or_else(|| "--".to_string())
                ),
                headline_metrics: vec![
                    ("status".to_string(), value.get("status").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                    ("agents".to_string(), value.get("agents").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                    ("rounds".to_string(), value.get("rounds").map(scalar_string).unwrap_or_else(|| "--".to_string())),
                    (
                        "wall time s".to_string(),
                        value.get("total_wall_time_s").map(scalar_string).unwrap_or_else(|| "--".to_string()),
                    ),
                    (
                        "pending loaded".to_string(),
                        value.get("all_pending_receipts_loaded")
                            .map(scalar_string)
                            .unwrap_or_else(|| "--".to_string()),
                    ),
                    (
                        "final audits".to_string(),
                        value.get("all_final_audits_verified")
                            .map(scalar_string)
                            .unwrap_or_else(|| "--".to_string()),
                    ),
                ],
                preview_lines,
                footer_lines: vec![format!("artifact: {}", source_path.display())],
            }],
        };
    }
    if let Some(missions) = value.get("missions").and_then(Value::as_array) {
        return DashboardView {
            title: value
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("HeliX Stress Missions")
                .to_string(),
            profile: value
                .get("profile")
                .and_then(Value::as_str)
                .unwrap_or("--")
                .to_string(),
            source_path: source_path.to_path_buf(),
            missions: missions.iter().map(mission_from_json).collect(),
        };
    }
    DashboardView {
        title: value
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("HeliX Mission")
            .to_string(),
        profile: value
            .get("profile")
            .and_then(Value::as_str)
            .unwrap_or("--")
            .to_string(),
        source_path: source_path.to_path_buf(),
        missions: vec![mission_from_json(value)],
    }
}

fn receipt_dashboard(path: &Path) -> Result<DashboardView, Box<dyn std::error::Error>> {
    let summary = load_receipt_summary(&path.to_path_buf())?;
    let mut preview_lines = vec![
        format!("receipts: {}", summary.receipts),
        format!("max token index: {}", summary.max_token_index),
        format!("avg ratio: {:.2}x", summary.avg_ratio),
        format!("max clip pct: {:.3}", summary.max_clip_pct),
        format!("max rel rmse: {:.5}", summary.max_rel_rmse),
    ];
    for (key, value) in summary.unstable_layers.iter().take(8) {
        preview_lines.push(format!("{key}: {value}"));
    }
    Ok(DashboardView {
        title: "HeliX Watch".to_string(),
        profile: "receipt-playback".to_string(),
        source_path: path.to_path_buf(),
        missions: vec![MissionView {
            mission_id: "receipt-summary".to_string(),
            title: "Receipt Summary".to_string(),
            strongest_claim: format!("avg ratio {:.2}x across {} receipts", summary.avg_ratio, summary.receipts),
            headline_metrics: vec![
                ("run id".to_string(), summary.run_id),
                ("receipts".to_string(), summary.receipts.to_string()),
                ("max token".to_string(), summary.max_token_index.to_string()),
                ("avg ratio".to_string(), format!("{:.2}x", summary.avg_ratio)),
            ],
            preview_lines,
            footer_lines: vec![format!("artifact: {}", path.display())],
        }],
    })
}

fn load_dashboard(path: &Path) -> Result<DashboardView, Box<dyn std::error::Error>> {
    if path.is_dir() {
        let dashboard_path = path.join("local-zamba2-stress-dashboard.json");
        return load_dashboard(&dashboard_path);
    }
    let is_receipt = path
        .file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.ends_with(".jsonl") || value.ends_with(".jsonl.gz"))
        .unwrap_or(false);
    if is_receipt {
        return receipt_dashboard(path);
    }
    let content = fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&content)?;
    Ok(dashboard_from_json(&value, path))
}

fn draw_dashboard(frame: &mut ratatui::Frame<'_>, app: &AppState) {
    let mission = app
        .dashboard
        .missions
        .get(app.selected)
        .cloned()
        .unwrap_or_default();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(8),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(frame.area());

    let selector = app
        .dashboard
        .missions
        .iter()
        .enumerate()
        .map(|(index, item)| {
            if index == app.selected {
                format!("> {} <", item.title)
            } else {
                item.title.clone()
            }
        })
        .collect::<Vec<_>>()
        .join("   |   ");
    let hero = Paragraph::new(vec![
        Line::from(vec![Span::styled(
            app.dashboard.title.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(format!("profile: {}", app.dashboard.profile)),
        Line::from(selector),
    ])
    .block(Block::default().title("Mission Selector").borders(Borders::ALL));

    let metric_lines = if mission.headline_metrics.is_empty() {
        vec![Line::from("No headline metrics available.")]
    } else {
        mission
            .headline_metrics
            .iter()
            .map(|(key, value)| Line::from(format!("{key}: {value}")))
            .collect::<Vec<_>>()
    };
    let metrics = Paragraph::new(metric_lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().title("Headline Metrics").borders(Borders::ALL));

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(outer[2]);

    let claim = Paragraph::new(vec![
        Line::from(vec![Span::styled(
            mission.strongest_claim.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(format!("mission_id: {}", mission.mission_id)),
    ])
    .wrap(Wrap { trim: false })
    .block(Block::default().title("Claim").borders(Borders::ALL));

    let preview = Paragraph::new(
        mission
            .preview_lines
            .iter()
            .map(|line| Line::from(line.clone()))
            .collect::<Vec<_>>(),
    )
    .wrap(Wrap { trim: false })
    .block(Block::default().title("Playback").borders(Borders::ALL));

    let mut footer_text = mission.footer_lines.clone();
    footer_text.push(format!("source: {}", app.dashboard.source_path.display()));
    let footer = Paragraph::new(
        footer_text
            .iter()
            .map(|line| Line::from(line.clone()))
            .collect::<Vec<_>>(),
    )
    .block(Block::default().title("Footer").borders(Borders::ALL));

    frame.render_widget(hero, outer[0]);
    frame.render_widget(metrics, outer[1]);
    frame.render_widget(claim, middle[0]);
    frame.render_widget(preview, middle[1]);
    frame.render_widget(footer, outer[3]);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let dashboard = load_dashboard(&args.path)?;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = AppState { dashboard, selected: 0 };

    loop {
        terminal.draw(|frame| draw_dashboard(frame, &app))?;
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Left | KeyCode::Char('h') => {
                    if app.selected > 0 {
                        app.selected -= 1;
                    }
                }
                KeyCode::Right | KeyCode::Char('l') => {
                    if app.selected + 1 < app.dashboard.missions.len() {
                        app.selected += 1;
                    }
                }
                KeyCode::Char('1') => app.selected = 0,
                KeyCode::Char('2') => {
                    if app.dashboard.missions.len() > 1 {
                        app.selected = 1;
                    }
                }
                KeyCode::Char('3') => {
                    if app.dashboard.missions.len() > 2 {
                        app.selected = 2;
                    }
                }
                KeyCode::Char('4') => {
                    if app.dashboard.missions.len() > 3 {
                        app.selected = 3;
                    }
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dashboard_from_single_mission_json() {
        let value = json!({
            "mission_id": "context-switcher",
            "title": "Context Switcher",
            "profile": "laptop-12gb",
            "strongest_claim": "kept logits finite",
            "headline_metrics": {"speedup_vs_native": 1.02}
        });
        let dashboard = dashboard_from_json(&value, Path::new("single.json"));
        assert_eq!(dashboard.missions.len(), 1);
        assert_eq!(dashboard.missions[0].mission_id, "context-switcher");
        assert_eq!(dashboard.missions[0].title, "Context Switcher");
    }

    #[test]
    fn dashboard_from_dashboard_json_keeps_multiple_missions() {
        let value = json!({
            "title": "HeliX Stress Missions v1",
            "profile": "laptop-12gb",
            "missions": [
                {
                    "mission_id": "long-context-coder",
                    "title": "Long-Context Coder",
                    "strongest_claim": "claim a",
                    "headline_metrics": {"best_runtime_ratio": 3.4}
                },
                {
                    "mission_id": "state-juggler",
                    "title": "State Juggler",
                    "strongest_claim": "claim b",
                    "headline_metrics": {"hash_match": true}
                },
                {
                    "mission_id": "restore-equivalence",
                    "title": "Restore Equivalence",
                    "strongest_claim": "claim c",
                    "headline_metrics": {"generated_ids_match": true}
                }
            ]
        });
        let dashboard = dashboard_from_json(&value, Path::new("dashboard.json"));
        assert_eq!(dashboard.missions.len(), 3);
        assert_eq!(dashboard.profile, "laptop-12gb");
        assert_eq!(dashboard.missions[1].mission_id, "state-juggler");
        assert_eq!(dashboard.missions[2].mission_id, "restore-equivalence");
    }

    #[test]
    fn restore_equivalence_preview_reports_matches_and_delta() {
        let value = json!({
            "mission_id": "restore-equivalence",
            "headline_metrics": {
                "hash_match": true,
                "generated_ids_match": true,
                "top1_match_all": true,
                "max_abs_logit_delta": 0.0
            },
            "pre_restore": {"answer_preview": "4.1"},
            "post_restore": {"answer_preview": "4.1"}
        });
        let lines = preview_lines_for_mission(&value);

        assert!(lines.iter().any(|line| line.contains("hash_match=true")));
        assert!(lines.iter().any(|line| line.contains("generated_ids_match=true")));
        assert!(lines.iter().any(|line| line.contains("max_abs_logit_delta=0.0")));
        assert!(lines.iter().any(|line| line == "pre: 4.1"));
        assert!(lines.iter().any(|line| line == "post: 4.1"));
    }

    #[test]
    fn session_core_flattened_summary_reports_group_collapse() {
        let value = json!({
            "title": "HeliX Session Core v2 Tensor Flattening",
            "profile": "laptop-12gb",
            "models": [{
                "model_key": "gpt2",
                "status": "completed",
                "model_ref": "gpt2",
                "rust_hlx_save_time_ms": 55.0,
                "python_npz_save_time_ms": 65.0,
                "hash_match": true,
                "generated_ids_match": true,
                "repeat_benchmark": {
                    "save_time_ms_p50": 55.0,
                    "save_time_ms_p95": 70.0,
                    "flatten_copy_time_ms_p50": 2.0,
                    "original_array_count_median": 72,
                    "flat_group_count_median": 3,
                    "buffer_spec_count_median": 3
                }
            }]
        });
        let dashboard = dashboard_from_json(&value, Path::new("flat.json"));

        assert_eq!(dashboard.missions.len(), 1);
        assert!(dashboard.missions[0].headline_metrics.iter().any(|(_, value)| value == "72 -> 3"));
        assert!(dashboard.missions[0].preview_lines.iter().any(|line| line.contains("buffer specs: 3")));
    }

    #[test]
    fn agent_hypervisor_preview_uses_handoff_out_when_available() {
        let value = json!({
            "title": "HeliX Local Agent Hypervisor v0",
            "scenario": "pr-war-room-long",
            "agents": 5,
            "rounds": 1,
            "status": "completed",
            "total_wall_time_s": 10.0,
            "all_restore_hash_matches": true,
            "events": [{
                "phase": "timeslice",
                "agent_id": "agent-0",
                "role": "bug_hunter",
                "round_index": 0,
                "hash_match": true,
                "save_time_ms": 12.0,
                "load_time_ms": 3.0,
                "answer_preview": "ignored preview",
                "handoff_out": "Observation: cache metadata risk"
            }]
        });
        let dashboard = dashboard_from_json(&value, Path::new("war-room.json"));

        assert!(dashboard.missions[0]
            .preview_lines
            .iter()
            .any(|line| line.contains("Observation: cache metadata risk")));
    }
}
