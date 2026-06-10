use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, UNIX_EPOCH};

#[derive(Clone, Debug)]
struct Record {
    suite_id: String,
    case_id: String,
    catalog_scope: String,
    kind: String,
    path: String,
    name: String,
    bytes: u64,
    updated_utc: String,
    mtime_ms: u128,
    run_id: String,
    status: String,
    score: String,
    case_count: String,
    artifact_payload_sha256: String,
    transcript_exports: String,
    snippet: String,
}

fn main() {
    let started = Instant::now();
    let args: Vec<String> = env::args().skip(1).collect();
    let result = dispatch(&args);
    match result {
        Ok(mut out) => {
            append_field(
                &mut out,
                "rust_core_ms",
                &format!("{:.3}", started.elapsed().as_secs_f64() * 1000.0),
                false,
            );
            println!("{out}");
        }
        Err(err) => {
            println!(
                "{{\"status\":\"error\",\"error\":{},\"rust_core_ms\":{:.3}}}",
                q(&err),
                started.elapsed().as_secs_f64() * 1000.0
            );
            std::process::exit(1);
        }
    }
}

fn dispatch(args: &[String]) -> Result<String, String> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "route" => route(
            &flag(args, "--text").unwrap_or_default(),
            &flag(args, "--latency-mode").unwrap_or_else(|| "fast".to_string()),
            &flag(args, "--interaction-mode").unwrap_or_else(|| "balanced".to_string()),
        ),
        "evidence-index" => evidence_index(
            &path_flag(args, "--evidence-root")?,
            &path_flag(args, "--repo-root")?,
        ),
        "suite-list" => suite_list(&path_flag(args, "--evidence-root")?),
        "suite-search" => suite_search(
            &path_flag(args, "--evidence-root")?,
            &flag(args, "--query").unwrap_or_default(),
            flag(args, "--limit")
                .and_then(|item| item.parse::<usize>().ok())
                .unwrap_or(12),
        ),
        "opencode-status" => opencode_status(flag(args, "--opencode-bin")),
        "opencode-run" => opencode_run(
            &path_flag(args, "--repo-root").or_else(|_| env::current_dir().map_err(|err| err.to_string()))?,
            &flag(args, "--goal").unwrap_or_default(),
            flag(args, "--opencode-bin"),
            flag(args, "--run-id"),
            flag(args, "--evidence-root").map(PathBuf::from),
        ),
        "opencode-install" => opencode_install(args),
        "verify-capsule" => verify_capsule(&path_flag(args, "--artifact")?),
        "lab-profiles" => lab_profiles(),
        "lab-run" => lab_run(
            &flag(args, "--profile").unwrap_or_else(|| "patch-safety".to_string()),
            &path_flag(args, "--evidence-root").unwrap_or_else(|_| PathBuf::from("verification")),
            &path_flag(args, "--repo-root").or_else(|_| env::current_dir().map_err(|err| err.to_string()))?,
        ),
        "mcp-stdio" => mcp_stdio(
            flag(args, "--workspace-root")
                .map(PathBuf::from)
                .or_else(|| env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from(".")),
            flag(args, "--evidence-root").map(PathBuf::from),
        ),
        "latency-report" => Ok("{\"status\":\"ok\",\"rust_core\":true,\"commands\":[\"route\",\"evidence-index\",\"suite-list\",\"suite-search\",\"opencode-status\",\"opencode-run\",\"opencode-install\",\"verify-capsule\",\"lab-profiles\",\"lab-run\",\"mcp-stdio\",\"latency-report\"]}".to_string()),
        _ => Err("usage: helix-cli-core route|evidence-index|suite-list|suite-search|opencode-status|opencode-run|opencode-install|verify-capsule|lab-profiles|lab-run|mcp-stdio|latency-report".to_string()),
    }
}

fn route(text: &str, latency_mode: &str, interaction_mode: &str) -> Result<String, String> {
    let started = Instant::now();
    let lowered = text.to_lowercase();
    let has = |terms: &[&str]| terms.iter().any(|term| lowered.contains(term));
    let url = lowered.contains("http://") || lowered.contains("https://");
    let suite = has(&[
        "/verify",
        "suite",
        "suites",
        "artifact",
        "artefacto",
        "manifest",
        "transcript",
        "nuclear",
        "hard anchor",
        "branch pruning",
        "policy rag",
    ]);
    let repo = has(&[
        "lee repo",
        "leer repo",
        "código",
        "codigo",
        "archivo",
        "src/",
        "pytest",
        "cargo",
        "test",
        "patch",
        "diff",
        "implement",
        "refactor",
        "bug",
    ]);
    let evidence = has(&[
        "evidencia",
        "evidence",
        "audit",
        "audita",
        "verifica",
        "demostra",
        "demostrá",
        "certifica",
        "receipt",
        "hash",
        "memoria",
        "memory",
    ]);
    let agentic = has(&[
        "/task",
        "agentic",
        "agent",
        "multi archivo",
        "multi-file",
        "hacelo",
        "arreglalo",
        "terminal",
        "tools",
    ]);
    let helix = lowered.contains("helix") || lowered.contains("hélix");
    let path = if suite {
        "nuclear"
    } else if agentic || repo {
        "agentic"
    } else if latency_mode == "deep"
        || url
        || evidence
        || (latency_mode == "balanced" && helix && interaction_mode == "technical")
    {
        "grounded"
    } else {
        "lightweight"
    };
    let tools = [
        ("suite.index", suite),
        ("web", url),
        ("evidence", evidence),
        ("repo", repo),
    ]
    .iter()
    .filter_map(|(name, enabled)| enabled.then_some(q(name)))
    .collect::<Vec<_>>()
    .join(",");
    let model_hint = match path {
        "lightweight" => "chat",
        "grounded" => "research",
        "agentic" => "coder",
        "nuclear" => "suite-run-analyst",
        _ => "chat",
    };
    Ok(format!(
        "{{\"status\":\"ok\",\"path\":{},\"model_hint\":{},\"tools_needed\":[{}],\"deep_required\":{},\"signals\":{{\"helix\":{},\"suite\":{},\"repo\":{},\"evidence\":{},\"agentic\":{},\"url\":{}}},\"routing_ms\":{:.3}}}",
        q(path),
        q(model_hint),
        tools,
        path != "lightweight",
        helix,
        suite,
        repo,
        evidence,
        agentic,
        url,
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn evidence_index(evidence_root: &Path, repo_root: &Path) -> Result<String, String> {
    let started = Instant::now();
    let mut records = Vec::new();
    let nuclear_root = evidence_root.join("nuclear-methodology");
    let base_root = if nuclear_root.exists() {
        nuclear_root
    } else {
        evidence_root.to_path_buf()
    };
    if base_root.exists() {
        walk_records(
            &base_root,
            evidence_root,
            repo_root,
            &base_root,
            &mut records,
        )?;
    }
    if evidence_root.exists() {
        for entry in fs::read_dir(evidence_root).map_err(|err| err.to_string())? {
            let path = entry.map_err(|err| err.to_string())?.path();
            if path.is_file() && is_text_file(&path) {
                records.push(record_for(&path, evidence_root, repo_root, None, "global"));
            }
        }
    }
    let index_dir = evidence_root.join(".helix-index");
    fs::create_dir_all(&index_dir).map_err(|err| err.to_string())?;
    let jsonl_path = index_dir.join("suites.jsonl");
    let json_path = index_dir.join("suites.json");
    let mut jsonl = String::new();
    for record in &records {
        jsonl.push_str(&record_json(record));
        jsonl.push('\n');
    }
    fs::write(&jsonl_path, jsonl).map_err(|err| err.to_string())?;
    let meta = format!(
        "{{\"version\":1,\"source\":\"rust-index\",\"evidence_root\":{},\"repo_root\":{},\"generated_utc\":{},\"record_count\":{},\"records_jsonl\":{}}}",
        q(&evidence_root.to_string_lossy()),
        q(&repo_root.to_string_lossy()),
        q(&millis_now().to_string()),
        records.len(),
        q(&jsonl_path.to_string_lossy())
    );
    fs::write(&json_path, meta).map_err(|err| err.to_string())?;
    Ok(format!(
        "{{\"status\":\"ok\",\"source\":\"rust-index\",\"index_path\":{},\"records_jsonl\":{},\"evidence_root\":{},\"record_count\":{},\"index_ms\":{:.3}}}",
        q(&json_path.to_string_lossy()),
        q(&jsonl_path.to_string_lossy()),
        q(&evidence_root.to_string_lossy()),
        records.len(),
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn suite_list(evidence_root: &Path) -> Result<String, String> {
    let records = read_jsonl_index(evidence_root)?;
    #[derive(Default)]
    struct SuiteAgg {
        path: String,
        counts: BTreeMap<String, u64>,
        latest: Option<Record>,
        prereg: String,
    }
    let mut suites: BTreeMap<String, SuiteAgg> = BTreeMap::new();
    for record in records {
        if record.catalog_scope != "suite" || record.suite_id.is_empty() {
            continue;
        }
        let entry = suites.entry(record.suite_id.clone()).or_default();
        if entry.path.is_empty() {
            entry.path = record.path.clone();
        }
        *entry.counts.entry(record.kind.clone()).or_default() += 1;
        if record.kind == "preregistered" {
            entry.prereg = record.path.clone();
        }
        if entry
            .latest
            .as_ref()
            .map(|latest| record.mtime_ms > latest.mtime_ms)
            .unwrap_or(true)
        {
            entry.latest = Some(record);
        }
    }
    let rows = suites
        .into_iter()
        .map(|(suite_id, agg)| {
            let counts = agg
                .counts
                .iter()
                .map(|(key, value)| format!("{}:{}", q(key), value))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"suite_id\":{},\"path\":{},\"registered\":false,\"script\":null,\"description\":null,\"preregistered_path\":{},\"counts\":{{{}}},\"latest\":{}}}",
                q(&suite_id),
                q(&agg.path),
                null_or_q(&agg.prereg),
                counts,
                agg.latest.map(|record| record_json(&record)).unwrap_or_else(|| "null".to_string())
            )
        })
        .collect::<Vec<_>>();
    Ok(format!(
        "{{\"status\":\"ok\",\"source\":\"rust-index\",\"evidence_root\":{},\"suite_count\":{},\"suites\":[{}]}}",
        q(&evidence_root.to_string_lossy()),
        rows.len(),
        rows.join(",")
    ))
}

fn suite_search(evidence_root: &Path, query: &str, limit: usize) -> Result<String, String> {
    let query_l = query.trim().to_lowercase();
    if query_l.is_empty() {
        return Err("query is required".to_string());
    }
    let mut scored = Vec::new();
    for record in read_jsonl_index(evidence_root)? {
        let name_l = record.name.to_lowercase();
        let path_l = record.path.to_lowercase();
        let snippet_l = record.snippet.to_lowercase();
        let mut score = 0;
        if name_l == query_l {
            score += 400;
        } else if name_l.contains(&query_l) {
            score += 300;
        }
        if path_l.contains(&query_l) {
            score += 250;
        }
        if snippet_l.contains(&query_l) {
            score += 180;
        }
        if record.catalog_scope == "global" {
            score += 30;
        }
        if score > 0 {
            scored.push((score, record));
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.mtime_ms.cmp(&a.1.mtime_ms)));
    let rows = scored
        .into_iter()
        .take(limit)
        .map(|(_, record)| record_json(&record))
        .collect::<Vec<_>>();
    Ok(format!(
        "{{\"status\":\"ok\",\"source\":\"rust-index\",\"query\":{},\"result_count\":{},\"results\":[{}]}}",
        q(query),
        rows.len(),
        rows.join(",")
    ))
}

fn opencode_status(configured: Option<String>) -> Result<String, String> {
    let started = Instant::now();
    let binary = configured
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(find_opencode_binary);
    Ok(format!(
        "{{\"status\":\"ok\",\"available\":{},\"binary\":{},\"version\":null,\"source\":\"rust-opencode\",\"probe_ms\":{:.3}}}",
        binary.is_some(),
        binary
            .as_ref()
            .map(|path| q(&path.to_string_lossy()))
            .unwrap_or_else(|| "null".to_string()),
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn opencode_run(
    repo_root: &Path,
    goal: &str,
    configured_bin: Option<String>,
    run_id_arg: Option<String>,
    evidence_root_arg: Option<PathBuf>,
) -> Result<String, String> {
    let started = Instant::now();
    if goal.trim().is_empty() {
        return Err("opencode-run requires --goal".to_string());
    }
    let repo_root = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let opencode_bin = configured_bin
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(find_opencode_binary)
        .ok_or_else(|| {
            "opencode binary not found; set HELIX_OPENCODE_BIN or add opencode to PATH".to_string()
        })?;
    let requested_run_id = run_id_arg.unwrap_or_else(|| format!("opencode-{}", millis_now()));
    let run_id = sanitize_run_id(&requested_run_id);
    if run_id.is_empty() {
        return Err("opencode-run received an invalid run id".to_string());
    }
    let run_root = repo_root.join(".helix").join("opencode-runs").join(&run_id);
    let sandbox = run_root.join("worktree");
    let artifact_dir = repo_root
        .join("verification")
        .join("opencode-agent")
        .join(&run_id);
    let evidence_root = evidence_root_arg.unwrap_or_else(|| repo_root.join("verification"));

    if run_root.exists() {
        ensure_child_path(&repo_root.join(".helix").join("opencode-runs"), &run_root)?;
        fs::remove_dir_all(&run_root).map_err(|err| format!("remove existing run dir: {err}"))?;
    }
    fs::create_dir_all(&sandbox)
        .map_err(|err| format!("create sandbox {}: {err}", sandbox.to_string_lossy()))?;
    let ignore_patterns = load_gitignore_patterns(&repo_root);
    let full_workspace_copy = is_project_root(&repo_root);
    if full_workspace_copy {
        copy_workspace(&repo_root, &sandbox, &run_root, &ignore_patterns)?;
    } else {
        copy_workspace_minimal(&repo_root, &sandbox, &run_root, &ignore_patterns)?;
    }
    let original_dirty = full_workspace_copy && git_dirty(&repo_root);
    let task_brief_path = sandbox.join("HELIX_TASK_BRIEF.md");
    fs::write(
        &task_brief_path,
        format!(
            "# HeliX OpenCode Task Brief\n\n{}\n\n## Execution Contract\n\n- Work only inside this sandbox/worktree.\n- HeliX will capture the patch and trust card after the run.\n- Do not assume changes are applied to the real repository.\n",
            goal
        ),
    )
    .map_err(|err| format!("write task brief: {err}"))?;
    init_sandbox_git(&sandbox)?;
    let engine_home = run_root.join("engine-home");
    let engine_config = engine_home.join("config");
    let engine_data = engine_home.join("data");
    let engine_cache = engine_home.join("cache");
    fs::create_dir_all(&engine_config).map_err(|err| {
        format!(
            "create isolated opencode config {}: {err}",
            engine_config.to_string_lossy()
        )
    })?;
    fs::create_dir_all(&engine_data).map_err(|err| {
        format!(
            "create isolated opencode data {}: {err}",
            engine_data.to_string_lossy()
        )
    })?;
    fs::create_dir_all(&engine_cache).map_err(|err| {
        format!(
            "create isolated opencode cache {}: {err}",
            engine_cache.to_string_lossy()
        )
    })?;
    seed_opencode_auth(&engine_home)?;

    let command_prompt = "Read HELIX_TASK_BRIEF.md in the current directory and execute the HeliX task brief. Keep changes inside this sandbox/worktree.";
    let command = vec![
        opencode_bin.to_string_lossy().to_string(),
        "run".to_string(),
        "--format".to_string(),
        "json".to_string(),
        command_prompt.to_string(),
    ];
    let call_started = Instant::now();
    let mut process = opencode_run_command(&opencode_bin, command_prompt);
    let output = process
        .current_dir(&sandbox)
        .env("HELIX_OPENCODE_RUN_ID", &run_id)
        .env("HELIX_REPO_ROOT", &repo_root)
        .env("HELIX_SANDBOX_ROOT", &sandbox)
        .env("HELIX_EVIDENCE_ROOT", &evidence_root)
        .env("HOME", &engine_home)
        .env("USERPROFILE", &engine_home)
        .env("XDG_CONFIG_HOME", &engine_config)
        .env("XDG_DATA_HOME", &engine_data)
        .env("XDG_CACHE_HOME", &engine_cache)
        .output()
        .map_err(|err| {
            format!(
                "failed to execute opencode {} from sandbox {}: {err}",
                opencode_bin.to_string_lossy(),
                sandbox.to_string_lossy()
            )
        })?;
    let opencode_ms = call_started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let _ = run_git(&sandbox, &["add", "-N", "-f", "."]);
    let patch = git_output(&sandbox, &["diff", "--binary", "--"])?;
    let changed_files = git_output(&sandbox, &["diff", "--name-only", "--"])?;

    fs::create_dir_all(&artifact_dir).map_err(|err| err.to_string())?;
    let patch_path = artifact_dir.join("patch.diff");
    let transcript_jsonl_path = artifact_dir.join("transcript.jsonl");
    let transcript_md_path = artifact_dir.join("transcript.md");
    let opencode_events_path = artifact_dir.join("opencode_events.jsonl");
    let sandbox_manifest_path = artifact_dir.join("sandbox_manifest.json");
    let artifact_path = artifact_dir.join("artifact.json");
    let trust_card_path = artifact_dir.join("trust_card.json");
    let task_capsule_path = artifact_dir.join("task_capsule.json");

    fs::write(&patch_path, &patch).map_err(|err| err.to_string())?;
    let transcript_event = format!(
        "{{\"event\":\"opencode_run\",\"run_id\":{},\"goal_sha256\":{},\"exit_code\":{},\"stdout_sha256\":{},\"stderr_sha256\":{},\"patch_sha256\":{},\"opencode_ms\":{:.3},\"stdout_preview\":{},\"stderr_preview\":{}}}\n",
        q(&run_id),
        q(&sha256_hex(goal.as_bytes())),
        output.status.code().unwrap_or(-1),
        q(&sha256_hex(stdout.as_bytes())),
        q(&sha256_hex(stderr.as_bytes())),
        q(&sha256_hex(patch.as_bytes())),
        opencode_ms,
        q(&truncate(&stdout, 2000)),
        q(&truncate(&stderr, 2000))
    );
    fs::write(&transcript_jsonl_path, transcript_event).map_err(|err| err.to_string())?;
    fs::write(&opencode_events_path, normalize_opencode_events(&stdout, &run_id)).map_err(|err| err.to_string())?;
    fs::write(
        &transcript_md_path,
        format!(
            "# OpenCode Run {}\n\n- Exit code: `{}`\n- Opencode ms: `{:.3}`\n- Patch bytes: `{}`\n\n## Goal\n\n```text\n{}\n```\n\n## Stdout\n\n```text\n{}\n```\n\n## Stderr\n\n```text\n{}\n```\n",
            run_id,
            output.status.code().unwrap_or(-1),
            opencode_ms,
            patch.len(),
            goal,
            truncate(&stdout, 12000),
            truncate(&stderr, 12000)
        ),
    )
    .map_err(|err| err.to_string())?;
    let changed_json = changed_files
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(q)
        .collect::<Vec<_>>()
        .join(",");
    let manifest = format!(
        "{{\"run_id\":{},\"repo_root\":{},\"sandbox_root\":{},\"engine_home\":{},\"task_brief_path\":{},\"base\":\"sandbox-initial-commit\",\"dirty_overlay_applied\":{},\"gitignore_patterns_loaded\":{},\"changed_files\":[{}]}}",
        q(&run_id),
        q(&repo_root.to_string_lossy()),
        q(&sandbox.to_string_lossy()),
        q(&engine_home.to_string_lossy()),
        q(&task_brief_path.to_string_lossy()),
        original_dirty,
        ignore_patterns.len(),
        changed_json
    );
    fs::write(&sandbox_manifest_path, &manifest).map_err(|err| err.to_string())?;
    let status = if output.status.success() {
        "passed"
    } else {
        "failed"
    };
    let artifact = format!(
        "{{\"artifact\":\"helix-opencode-agent-run-v1\",\"status\":{},\"engine\":\"opencode\",\"run_id\":{},\"claim_boundary\":{},\"sandbox\":{},\"opencode_call\":{{\"command\":[{}],\"exit_code\":{},\"latency_ms\":{:.3},\"stdout_sha256\":{},\"stderr_sha256\":{},\"stdout_json_detected\":{}}},\"patch\":{{\"path\":{},\"sha256\":{},\"bytes\":{},\"changed_files\":[{}]}},\"helix_mcp_calls\":[],\"transcript_exports\":{{\"jsonl_path\":{},\"md_path\":{},\"opencode_events_path\":{}}},\"artifact_payload_sha256\":{}}}",
        q(status),
        q(&run_id),
        q("OpenCode ran inside a HeliX sandbox. This artifact proves local command/provenance, captured diff, and bounded transcript; it does not prove semantic correctness until a reviewer or suite validates the patch."),
        manifest,
        command.iter().map(|item| q(item)).collect::<Vec<_>>().join(","),
        output.status.code().unwrap_or(-1),
        opencode_ms,
        q(&sha256_hex(stdout.as_bytes())),
        q(&sha256_hex(stderr.as_bytes())),
        stdout.trim_start().starts_with('{') || stdout.trim_start().starts_with('['),
        q(&patch_path.to_string_lossy()),
        q(&sha256_hex(patch.as_bytes())),
        patch.len(),
        changed_json,
        q(&transcript_jsonl_path.to_string_lossy()),
        q(&transcript_md_path.to_string_lossy()),
        q(&opencode_events_path.to_string_lossy()),
        q(&sha256_hex(format!("{}{}{}{}", run_id, stdout, stderr, patch).as_bytes()))
    );
    fs::write(&artifact_path, &artifact).map_err(|err| err.to_string())?;
    let trust_card = opencode_trust_card_json(
        status,
        &run_id,
        "quick",
        &artifact_path,
        &patch_path,
        &sandbox,
        &changed_json,
        patch.len(),
        &patch,
        output.status.code().unwrap_or(-1),
        opencode_ms,
    );
    fs::write(&trust_card_path, &trust_card).map_err(|err| err.to_string())?;
    let task_capsule = format!(
        "{{\"kind\":\"helix-task-capsule-v1\",\"status\":{},\"goal_sha256\":{},\"lane\":\"work\",\"engine\":\"opencode\",\"assurance\":\"quick\",\"models_used\":{{\"planner_model\":null,\"coder_engine\":\"opencode\",\"critic_model\":null,\"verifier_model\":\"helix-rust-core\"}},\"inputs\":{{\"repo_root\":{},\"evidence_root\":{}}},\"anchors\":[],\"sandbox\":{},\"patch\":{{\"path\":{},\"sha256\":{},\"bytes\":{},\"changed_files\":[{}]}},\"checks\":{},\"trust_card_path\":{},\"artifact_path\":{},\"claim_boundary\":{}}}",
        q(status),
        q(&sha256_hex(goal.as_bytes())),
        q(&repo_root.to_string_lossy()),
        q(&evidence_root.to_string_lossy()),
        manifest,
        q(&patch_path.to_string_lossy()),
        q(&sha256_hex(patch.as_bytes())),
        patch.len(),
        changed_json,
        trust_card_checks_json(status, patch.len(), output.status.code().unwrap_or(-1)),
        q(&trust_card_path.to_string_lossy()),
        q(&artifact_path.to_string_lossy()),
        q("This capsule records local sandbox execution, captured diff, hashes, and bounded provenance. It does not prove the patch is semantically correct.")
    );
    fs::write(&task_capsule_path, &task_capsule).map_err(|err| err.to_string())?;

    Ok(format!(
        "{{\"status\":{},\"engine\":\"opencode\",\"run_id\":{},\"repo_root\":{},\"sandbox_root\":{},\"artifact_path\":{},\"patch_path\":{},\"trust_card_path\":{},\"task_capsule_path\":{},\"opencode_events_path\":{},\"trust_card\":{},\"assurance\":\"quick\",\"changed_files\":[{}],\"opencode_trace\":{{\"exit_code\":{},\"latency_ms\":{:.3},\"stdout_preview\":{},\"stderr_preview\":{}}},\"patch\":{},\"patch_sha256\":{},\"total_ms\":{:.3}}}",
        q(status),
        q(&run_id),
        q(&repo_root.to_string_lossy()),
        q(&sandbox.to_string_lossy()),
        q(&artifact_path.to_string_lossy()),
        q(&patch_path.to_string_lossy()),
        q(&trust_card_path.to_string_lossy()),
        q(&task_capsule_path.to_string_lossy()),
        q(&opencode_events_path.to_string_lossy()),
        trust_card,
        changed_json,
        output.status.code().unwrap_or(-1),
        opencode_ms,
        q(&truncate(&stdout, 1000)),
        q(&truncate(&stderr, 1000)),
        q(&patch),
        q(&sha256_hex(patch.as_bytes())),
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn opencode_trust_card_json(
    status: &str,
    run_id: &str,
    assurance: &str,
    artifact_path: &Path,
    patch_path: &Path,
    sandbox: &Path,
    changed_json: &str,
    patch_bytes: usize,
    patch: &str,
    exit_code: i32,
    opencode_ms: f64,
) -> String {
    let warnings = if status == "passed" {
        "[]".to_string()
    } else {
        "[\"OpenCode exited non-zero; patch is not trusted for apply.\"]".to_string()
    };
    format!(
        "{{\"kind\":\"helix-trust-card-v1\",\"subject_type\":\"task\",\"status\":{},\"assurance\":{},\"engine\":\"opencode\",\"run_id\":{},\"models_used\":{{\"planner_model\":null,\"coder_engine\":\"opencode\",\"critic_model\":null,\"verifier_model\":\"helix-rust-core\"}},\"changed_files\":[{}],\"checks_passed\":{},\"warnings\":{},\"sandbox\":{{\"path\":{},\"repo_real_changed\":false}},\"patch\":{{\"path\":{},\"sha256\":{},\"bytes\":{}}},\"artifact_paths\":{{\"artifact\":{},\"patch\":{},\"trust_card\":{},\"task_capsule\":{}}},\"latency\":{{\"opencode_ms\":{:.3}}},\"claim_boundary\":{}}}",
        q(status),
        q(assurance),
        q(run_id),
        changed_json,
        trust_card_checks_json(status, patch_bytes, exit_code),
        warnings,
        q(&sandbox.to_string_lossy()),
        q(&patch_path.to_string_lossy()),
        q(&sha256_hex(patch.as_bytes())),
        patch_bytes,
        q(&artifact_path.to_string_lossy()),
        q(&patch_path.to_string_lossy()),
        q(&artifact_path.with_file_name("trust_card.json").to_string_lossy()),
        q(&artifact_path.with_file_name("task_capsule.json").to_string_lossy()),
        opencode_ms,
        q("This card proves local sandbox provenance, patch capture, and hash integrity for this run; semantic correctness still requires review, tests, or stricter verification.")
    )
}

fn trust_card_checks_json(status: &str, patch_bytes: usize, exit_code: i32) -> String {
    let run_status = if exit_code == 0 && status == "passed" {
        "passed"
    } else {
        "failed"
    };
    let patch_status = if patch_bytes > 0 { "passed" } else { "warning" };
    let patch_summary = if patch_bytes > 0 {
        "Patch diff captured and hashed."
    } else {
        "No patch diff was produced."
    };
    format!(
        "[{{\"id\":\"sandbox_provenance\",\"status\":\"passed\",\"summary\":\"OpenCode ran inside a HeliX worktree sandbox.\"}},{{\"id\":\"opencode_exit\",\"status\":{},\"summary\":\"Backend process completed under HeliX capture.\"}},{{\"id\":\"patch_integrity\",\"status\":{},\"summary\":{} }},{{\"id\":\"provider_disagreement\",\"status\":\"not_run\",\"summary\":\"Multi-model critic is reserved for balanced/strict assurance.\"}},{{\"id\":\"deep_nuclear\",\"status\":\"not_run\",\"summary\":\"Deep nuclear suites require explicit --level deep.\"}}]",
        q(run_status),
        q(patch_status),
        q(patch_summary),
    )
}

fn normalize_opencode_events(stdout: &str, run_id: &str) -> String {
    let mut out = String::new();
    for (index, line) in stdout.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            out.push_str(trimmed);
            out.push('\n');
        } else {
            out.push_str(&format!(
                "{{\"type\":\"stdout_line\",\"run_id\":{},\"index\":{},\"text\":{}}}\n",
                q(run_id),
                index,
                q(&truncate(trimmed, 2000))
            ));
        }
    }
    if out.is_empty() {
        out.push_str(&format!(
            "{{\"type\":\"opencode_events_empty\",\"run_id\":{},\"text\":\"OpenCode produced no stdout events.\"}}\n",
            q(run_id)
        ));
    }
    out
}

fn opencode_install(args: &[String]) -> Result<String, String> {
    let dry_run = args.iter().any(|item| item == "--dry-run");
    let force = args.iter().any(|item| item == "--force");
    let bin = flag(args, "--bin")
        .map(PathBuf::from)
        .or_else(|| env::current_exe().ok())
        .unwrap_or_else(|| PathBuf::from("helix-cli-core"));
    let config_path = flag(args, "--config-path")
        .map(PathBuf::from)
        .unwrap_or_else(default_opencode_config_path);
    let rendered = format!(
        "{{\n  \"mcp\": {{\n    \"helix\": {{\n      \"type\": \"local\",\n      \"command\": [{}, \"mcp-stdio\"],\n      \"enabled\": true,\n      \"timeout\": 2000\n    }}\n  }},\n  \"permission\": {{\n    \"helix_*\": \"allow\"\n  }}\n}}\n",
        q(&bin.to_string_lossy())
    );
    if dry_run {
        return Ok(format!(
            "{{\"status\":\"ok\",\"dry_run\":true,\"config_path\":{},\"config\":{}}}",
            q(&config_path.to_string_lossy()),
            q(&rendered)
        ));
    }
    if config_path.exists() && !force {
        let backup = config_path.with_extension(format!("json.backup-{}", millis_now()));
        fs::copy(&config_path, &backup).map_err(|err| err.to_string())?;
        fs::write(&config_path, rendered).map_err(|err| err.to_string())?;
        return Ok(format!(
            "{{\"status\":\"ok\",\"dry_run\":false,\"config_path\":{},\"backup_path\":{},\"merged\":false,\"note\":\"existing config was backed up before writing HeliX MCP config\"}}",
            q(&config_path.to_string_lossy()),
            q(&backup.to_string_lossy())
        ));
    }
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(&config_path, rendered).map_err(|err| err.to_string())?;
    Ok(format!(
        "{{\"status\":\"ok\",\"dry_run\":false,\"config_path\":{},\"backup_path\":null,\"merged\":false}}",
        q(&config_path.to_string_lossy())
    ))
}

fn verify_capsule(artifact_path: &Path) -> Result<String, String> {
    let started = Instant::now();
    let artifact_path = artifact_path
        .canonicalize()
        .unwrap_or_else(|_| artifact_path.to_path_buf());
    let artifact = fs::read_to_string(&artifact_path)
        .map_err(|err| format!("read artifact {}: {err}", artifact_path.to_string_lossy()))?;
    let run_id = json_string_field(&artifact, "run_id").unwrap_or_else(|| "unknown".to_string());
    let status = json_string_field(&artifact, "status").unwrap_or_else(|| "unknown".to_string());
    let patch_path = json_string_field(&artifact, "path")
        .map(PathBuf::from)
        .unwrap_or_else(|| artifact_path.with_file_name("patch.diff"));
    let patch = fs::read(&patch_path).unwrap_or_default();
    let expected_patch_sha = json_string_field(&artifact, "sha256").unwrap_or_default();
    let actual_patch_sha = sha256_hex(&patch);
    let patch_hash_match = !expected_patch_sha.is_empty() && expected_patch_sha == actual_patch_sha;
    let verified_status = if status == "passed" && patch_hash_match {
        "passed"
    } else if patch_hash_match {
        "partial"
    } else {
        "failed"
    };
    let trust_card = opencode_trust_card_json(
        verified_status,
        &run_id,
        "quick",
        &artifact_path,
        &patch_path,
        &artifact_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("unknown-sandbox"),
        "",
        patch.len(),
        &String::from_utf8_lossy(&patch),
        if status == "passed" { 0 } else { 1 },
        0.0,
    );
    Ok(format!(
        "{{\"status\":{},\"kind\":\"helix-capsule-verification-v1\",\"run_id\":{},\"artifact_path\":{},\"patch_path\":{},\"checks\":[{{\"id\":\"patch_integrity\",\"status\":{},\"expected_sha256\":{},\"actual_sha256\":{}}}],\"trust_card\":{},\"verify_ms\":{:.3}}}",
        q(verified_status),
        q(&run_id),
        q(&artifact_path.to_string_lossy()),
        q(&patch_path.to_string_lossy()),
        q(if patch_hash_match { "passed" } else { "failed" }),
        null_or_q(&expected_patch_sha),
        q(&actual_patch_sha),
        trust_card,
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn lab_profiles() -> Result<String, String> {
    Ok("{\"status\":\"ok\",\"kind\":\"helix-lab-profiles-v1\",\"profiles\":[{\"id\":\"patch-safety\",\"default_level\":\"quick\",\"description\":\"Sandbox, patch hash, trust card, and apply-readiness checks for agentic code tasks.\"},{\"id\":\"doc-grounding\",\"default_level\":\"balanced\",\"description\":\"Hard-anchor style document grounding without loading full corpora into context.\"},{\"id\":\"memory-isolation\",\"default_level\":\"balanced\",\"description\":\"Thread-only memory, quarantine, signed/unsigned source handling, and branch contamination checks.\"},{\"id\":\"provider-audit\",\"default_level\":\"strict\",\"description\":\"Provider/model identity, disagreement, and substitution checks.\"},{\"id\":\"deep-nuclear\",\"default_level\":\"deep\",\"description\":\"Explicit full nuclear verification suites; never runs implicitly in hot chat/task paths.\"}]}".to_string())
}

fn lab_run(profile: &str, evidence_root: &Path, repo_root: &Path) -> Result<String, String> {
    let started = Instant::now();
    let normalized = sanitize_profile(profile);
    if normalized == "deep-nuclear" {
        return Ok(format!(
            "{{\"status\":\"requires_explicit_deep\",\"profile\":\"deep-nuclear\",\"message\":\"deep-nuclear is an explicit expensive profile; run the existing suite command intentionally.\",\"lab_ms\":{:.3}}}",
            started.elapsed().as_secs_f64() * 1000.0
        ));
    }
    let index = suite_list(evidence_root).unwrap_or_else(|_| "{\"suites\":[]}".to_string());
    let rust_core = rust_core_self_available();
    let opencode = opencode_status(None).unwrap_or_else(|_| "{\"available\":false}".to_string());
    let checks = match normalized.as_str() {
        "patch-safety" => format!(
            "[{{\"id\":\"rust_core\",\"status\":{},\"summary\":\"Rust core command path is available.\"}},{{\"id\":\"suite_index\",\"status\":\"passed\",\"summary\":\"Suite index/list path responds without deep artifact replay.\"}},{{\"id\":\"opencode_available\",\"status\":{},\"summary\":\"OpenCode backend is optional but detected when installed.\"}}]",
            q(if rust_core { "passed" } else { "failed" }),
            q(if opencode.contains("\"available\":true") { "passed" } else { "warning" })
        ),
        "doc-grounding" => "[{\"id\":\"hard_anchor_recall\",\"status\":\"not_run\",\"summary\":\"Runtime document anchors are exercised by file/doc tasks; this profile stays non-mutating.\"}]".to_string(),
        "memory-isolation" => "[{\"id\":\"source_poison_guard\",\"status\":\"not_run\",\"summary\":\"Use strict task verification or memory-isolation suites for adversarial replay.\"},{\"id\":\"branch_quarantine\",\"status\":\"not_run\",\"summary\":\"Branch quarantine checks are available as selected nuclear cases.\"}]".to_string(),
        "provider-audit" => "[{\"id\":\"provider_disagreement\",\"status\":\"not_run\",\"summary\":\"No provider call is made by quick lab profiles.\"}]".to_string(),
        _ => return Err(format!("unknown lab profile: {profile}")),
    };
    Ok(format!(
        "{{\"status\":\"ok\",\"kind\":\"helix-lab-run-v1\",\"profile\":{},\"repo_root\":{},\"evidence_root\":{},\"checks\":{},\"suite_index_probe\":{},\"claim_boundary\":{},\"lab_ms\":{:.3}}}",
        q(&normalized),
        q(&repo_root.to_string_lossy()),
        q(&evidence_root.to_string_lossy()),
        checks,
        index,
        q("Lab profiles are commercial runtime readiness checks. They do not prove semantic truth and do not run deep nuclear suites unless explicitly requested."),
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn mcp_stdio(workspace_root: PathBuf, evidence_root: Option<PathBuf>) -> Result<String, String> {
    let evidence_root = evidence_root.unwrap_or_else(|| workspace_root.join("verification"));
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    loop {
        let Some(message) = read_mcp_message(&mut reader)? else {
            break;
        };
        let response = handle_mcp_message(&message, &workspace_root, &evidence_root);
        if !response.is_empty() {
            write_mcp_message(&response)?;
        }
    }
    Ok("{\"status\":\"ok\",\"mcp\":\"stdio\",\"ended\":true}".to_string())
}

fn walk_records(
    root: &Path,
    evidence_root: &Path,
    repo_root: &Path,
    base_root: &Path,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    for entry in fs::read_dir(root).map_err(|err| err.to_string())? {
        let path = entry.map_err(|err| err.to_string())?.path();
        let name = path
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("");
        if name.starts_with('.') || name.starts_with('_') {
            continue;
        }
        if path.is_dir() {
            walk_records(&path, evidence_root, repo_root, base_root, records)?;
        } else if is_text_file(&path) {
            let suite_dir = suite_dir_for_path(&path, base_root);
            let scope = if suite_dir.is_some() {
                "suite"
            } else {
                "global"
            };
            records.push(record_for(
                &path,
                evidence_root,
                repo_root,
                suite_dir.as_deref(),
                scope,
            ));
        }
    }
    Ok(())
}

fn record_for(
    path: &Path,
    _evidence_root: &Path,
    repo_root: &Path,
    suite_dir: Option<&Path>,
    scope: &str,
) -> Record {
    let metadata = fs::metadata(path).ok();
    let bytes = metadata.as_ref().map(|item| item.len()).unwrap_or(0);
    let mtime_ms = metadata
        .and_then(|item| item.modified().ok())
        .and_then(|item| item.duration_since(UNIX_EPOCH).ok())
        .map(|item| item.as_millis())
        .unwrap_or(0);
    let text = if path
        .extension()
        .and_then(|item| item.to_str())
        .unwrap_or("")
        == "json"
        && bytes <= 2_000_000
    {
        fs::read_to_string(path).unwrap_or_default()
    } else {
        String::new()
    };
    let suite_id = suite_dir
        .and_then(|item| item.file_name())
        .and_then(|item| item.to_str())
        .map(str::to_string)
        .or_else(|| json_string_field(&text, "suite_id"))
        .unwrap_or_default();
    let case_id = json_string_field(&text, "case_id").unwrap_or_else(|| {
        suite_dir
            .and_then(|suite| {
                path.parent()
                    .and_then(|parent| parent.strip_prefix(suite).ok())
            })
            .map(|rel| rel.to_string_lossy().replace('\\', "/"))
            .filter(|rel| rel != ".")
            .unwrap_or_default()
    });
    let snippet = if !text.is_empty() {
        [
            ("run_id", json_string_field(&text, "run_id")),
            ("case_id", json_string_field(&text, "case_id")),
            ("status", json_string_field(&text, "status")),
            ("summary", json_string_field(&text, "summary")),
        ]
        .iter()
        .filter_map(|(key, value)| value.as_ref().map(|v| format!("{key}={v}")))
        .collect::<Vec<_>>()
        .join(" ")
    } else if bytes <= 200_000 {
        fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(900)
            .collect()
    } else {
        String::new()
    };
    Record {
        suite_id,
        case_id,
        catalog_scope: scope.to_string(),
        kind: kind_for(path),
        path: rel(path, repo_root),
        name: path
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("")
            .to_string(),
        bytes,
        updated_utc: mtime_ms.to_string(),
        mtime_ms,
        run_id: json_string_field(&text, "run_id")
            .or_else(|| timestamp_from_name(path))
            .unwrap_or_default(),
        status: json_string_field(&text, "status").unwrap_or_default(),
        score: json_numberish_field(&text, "score").unwrap_or_default(),
        case_count: json_numberish_field(&text, "case_count").unwrap_or_default(),
        artifact_payload_sha256: json_string_field(&text, "artifact_payload_sha256")
            .unwrap_or_default(),
        transcript_exports: json_objectish_field(&text, "transcript_exports").unwrap_or_default(),
        snippet,
    }
}

fn read_jsonl_index(evidence_root: &Path) -> Result<Vec<Record>, String> {
    let jsonl = evidence_root.join(".helix-index").join("suites.jsonl");
    let text = fs::read_to_string(&jsonl).map_err(|_| {
        format!(
            "suite index missing; run `/suite index refresh` ({})",
            jsonl.to_string_lossy()
        )
    })?;
    Ok(text.lines().filter_map(record_from_json_line).collect())
}

fn record_from_json_line(line: &str) -> Option<Record> {
    Some(Record {
        suite_id: json_string_field(line, "suite_id").unwrap_or_default(),
        case_id: json_string_field(line, "case_id").unwrap_or_default(),
        catalog_scope: json_string_field(line, "catalog_scope").unwrap_or_default(),
        kind: json_string_field(line, "kind").unwrap_or_default(),
        path: json_string_field(line, "path").unwrap_or_default(),
        name: json_string_field(line, "name").unwrap_or_default(),
        bytes: json_numberish_field(line, "bytes")?.parse().ok()?,
        updated_utc: json_string_field(line, "updated_utc").unwrap_or_default(),
        mtime_ms: json_numberish_field(line, "mtime_ms")?.parse().ok()?,
        run_id: json_string_field(line, "run_id").unwrap_or_default(),
        status: json_string_field(line, "status").unwrap_or_default(),
        score: json_string_field(line, "score").unwrap_or_default(),
        case_count: json_string_field(line, "case_count").unwrap_or_default(),
        artifact_payload_sha256: json_string_field(line, "artifact_payload_sha256")
            .unwrap_or_default(),
        transcript_exports: json_string_field(line, "transcript_exports").unwrap_or_default(),
        snippet: json_string_field(line, "snippet").unwrap_or_default(),
    })
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn path_flag(args: &[String], name: &str) -> Result<PathBuf, String> {
    flag(args, name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} is required"))
}

fn is_text_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|item| item.to_str())
            .unwrap_or("")
            .to_lowercase()
            .as_str(),
        "json" | "jsonl" | "md" | "log" | "txt"
    )
}

fn suite_dir_for_path(path: &Path, base_root: &Path) -> Option<PathBuf> {
    let rel = path.strip_prefix(base_root).ok()?;
    if rel.components().count() < 2 {
        return None;
    }
    let first = rel.components().next()?;
    Some(base_root.join(first.as_os_str()))
}

fn is_lightweight_seed_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(
        name.as_str(),
        "readme.md" | "readme.txt" | "package.json" | "pyproject.toml" | "cargo.toml"
    ) {
        return true;
    }
    matches!(
        path.extension()
            .and_then(|item| item.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "md" | "txt"
            | "json"
            | "jsonl"
            | "csv"
            | "tsv"
            | "html"
            | "htm"
            | "css"
            | "js"
            | "ts"
            | "toml"
            | "yaml"
            | "yml"
    )
}

fn is_project_root(path: &Path) -> bool {
    [
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
    ]
    .iter()
    .any(|name| path.join(name).exists())
}

fn kind_for(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("")
        .to_lowercase();
    let suffix = path
        .extension()
        .and_then(|item| item.to_str())
        .unwrap_or("")
        .to_lowercase();
    if name == "preregistered.md" {
        "preregistered".to_string()
    } else if suffix == "log" {
        "log".to_string()
    } else if suffix == "jsonl" {
        if name.contains("transcript") {
            "transcript_jsonl".to_string()
        } else {
            "jsonl".to_string()
        }
    } else if suffix == "md" {
        if name.contains("transcript") {
            "transcript_md".to_string()
        } else {
            "markdown".to_string()
        }
    } else if suffix == "json" {
        if name.ends_with("-run.json") {
            "manifest".to_string()
        } else if name.contains("integrity-correction") {
            "integrity_correction".to_string()
        } else {
            "artifact".to_string()
        }
    } else {
        "other".to_string()
    }
}

fn rel(path: &Path, repo_root: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn timestamp_from_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    let bytes = name.as_bytes();
    for index in 0..bytes.len().saturating_sub(8) {
        if bytes[index..].starts_with(b"20") {
            let end = name.len().min(index + 15);
            let candidate = &name[index..end];
            if candidate.chars().filter(|ch| ch.is_ascii_digit()).count() >= 8 {
                return Some(candidate.replace('_', "-"));
            }
        }
    }
    None
}

fn json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = text.find(&needle)?;
    let after_key = &text[start + needle.len()..];
    let colon = after_key.find(':')?;
    let mut rest = after_key[colon + 1..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    if !rest.starts_with('"') {
        return json_numberish_field(text, key);
    }
    rest = &rest[1..];
    let mut out = String::new();
    let mut escaped = false;
    for ch in rest.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_numberish_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = text.find(&needle)?;
    let after_key = &text[start + needle.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let value = rest
        .chars()
        .take_while(|ch| !matches!(ch, ',' | '}' | ']'))
        .collect::<String>()
        .trim()
        .trim_matches('"')
        .to_string();
    if value.is_empty() || value == "null" {
        None
    } else {
        Some(value)
    }
}

fn json_objectish_field(text: &str, key: &str) -> Option<String> {
    json_string_field(text, key).or_else(|| json_numberish_field(text, key))
}

fn record_json(record: &Record) -> String {
    format!(
        "{{\"suite_id\":{},\"case_id\":{},\"catalog_scope\":{},\"kind\":{},\"path\":{},\"name\":{},\"bytes\":{},\"updated_utc\":{},\"mtime_ms\":{},\"run_id\":{},\"status\":{},\"score\":{},\"case_count\":{},\"artifact_payload_sha256\":{},\"transcript_exports\":{},\"snippet\":{}}}",
        q(&record.suite_id),
        q(&record.case_id),
        q(&record.catalog_scope),
        q(&record.kind),
        q(&record.path),
        q(&record.name),
        record.bytes,
        q(&record.updated_utc),
        record.mtime_ms,
        q(&record.run_id),
        q(&record.status),
        q(&record.score),
        q(&record.case_count),
        q(&record.artifact_payload_sha256),
        q(&record.transcript_exports),
        q(&record.snippet)
    )
}

fn q(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push(' '),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn null_or_q(text: &str) -> String {
    if text.is_empty() {
        "null".to_string()
    } else {
        q(text)
    }
}

fn sanitize_run_id(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else if matches!(ch, '/' | '\\' | ':' | ' ' | '\t' | '\r' | '\n') {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches(&['.', '-'][..]).to_string();
    if trimmed.is_empty() {
        format!("opencode-{}", millis_now())
    } else {
        trimmed.chars().take(96).collect()
    }
}

fn sanitize_profile(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .collect::<String>()
        .to_lowercase()
}

fn ensure_child_path(parent: &Path, child: &Path) -> Result<(), String> {
    let parent = parent
        .canonicalize()
        .unwrap_or_else(|_| parent.to_path_buf());
    let child_parent = child
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .unwrap_or_else(|_| {
            child
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    if !child_parent.starts_with(&parent) && child_parent != parent {
        return Err(format!(
            "refusing to operate outside {}: {}",
            parent.to_string_lossy(),
            child.to_string_lossy()
        ));
    }
    Ok(())
}

fn append_field(json: &mut String, key: &str, value: &str, quoted: bool) {
    if let Some(pos) = json.rfind('}') {
        let rendered = if quoted { q(value) } else { value.to_string() };
        json.insert_str(pos, &format!(",{}:{}", q(key), rendered));
    }
}

fn millis_now() -> u128 {
    UNIX_EPOCH.elapsed().unwrap_or_default().as_millis()
}

fn find_opencode_binary() -> Option<PathBuf> {
    if let Ok(configured) = env::var("HELIX_OPENCODE_BIN") {
        let path = PathBuf::from(configured);
        if path.exists() {
            return Some(path);
        }
    }
    let path_var = env::var_os("PATH")?;
    let names: &[&str] = if cfg!(windows) {
        &["opencode.exe", "opencode.cmd", "opencode.bat"]
    } else {
        &["opencode"]
    };
    for dir in env::split_paths(&path_var) {
        for name in names {
            let candidate = dir.join(name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn opencode_run_command(opencode_bin: &Path, command_prompt: &str) -> Command {
    if cfg!(windows) && is_windows_batch_file(opencode_bin) {
        let mut command = Command::new("cmd");
        command
            .arg("/D")
            .arg("/C")
            .arg("call")
            .arg(opencode_bin)
            .arg("run")
            .arg("--format")
            .arg("json")
            .arg(command_prompt);
        return command;
    }
    let mut command = Command::new(opencode_bin);
    command
        .arg("run")
        .arg("--format")
        .arg("json")
        .arg(command_prompt);
    command
}

fn is_windows_batch_file(path: &Path) -> bool {
    path.extension()
        .and_then(|item| item.to_str())
        .map(|item| {
            let ext = item.to_ascii_lowercase();
            ext == "cmd" || ext == "bat"
        })
        .unwrap_or(false)
}

fn seed_opencode_auth(engine_home: &Path) -> Result<(), String> {
    let Some(source_auth) = host_opencode_auth_path() else {
        return Ok(());
    };
    if !source_auth.exists() {
        return Ok(());
    }
    let target_dir = engine_home.join(".local").join("share").join("opencode");
    fs::create_dir_all(&target_dir)
        .map_err(|err| format!("create isolated opencode auth dir: {err}"))?;
    fs::copy(&source_auth, target_dir.join("auth.json")).map_err(|err| {
        format!(
            "copy opencode auth from {}: {err}",
            source_auth.to_string_lossy()
        )
    })?;
    Ok(())
}

fn host_opencode_auth_path() -> Option<PathBuf> {
    let mut homes = Vec::new();
    if let Ok(home) = env::var("USERPROFILE") {
        homes.push(PathBuf::from(home));
    }
    if let Ok(home) = env::var("HOME") {
        homes.push(PathBuf::from(home));
    }
    homes
        .into_iter()
        .map(|home| {
            home.join(".local")
                .join("share")
                .join("opencode")
                .join("auth.json")
        })
        .find(|path| path.exists())
}

fn default_opencode_config_path() -> PathBuf {
    if cfg!(windows) {
        if let Ok(appdata) = env::var("APPDATA") {
            return PathBuf::from(appdata)
                .join("opencode")
                .join("opencode.json");
        }
    }
    env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".config")
        .join("opencode")
        .join("opencode.json")
}

fn load_gitignore_patterns(repo_root: &Path) -> Vec<String> {
    let path = repo_root.join(".gitignore");
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#') && !line.starts_with('!'))
        .map(|line| line.trim_start_matches('/').to_string())
        .collect()
}

fn copy_workspace(
    src: &Path,
    dst: &Path,
    run_root: &Path,
    ignore_patterns: &[String],
) -> Result<(), String> {
    let entries = fs::read_dir(src)
        .map_err(|err| format!("read workspace root {}: {err}", src.to_string_lossy()))?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("");
        if should_skip_workspace_entry(name, &path, src, run_root, ignore_patterns) {
            continue;
        }
        let target = dst.join(name);
        if path.is_dir() {
            fs::create_dir_all(&target)
                .map_err(|err| format!("create sandbox dir {}: {err}", target.to_string_lossy()))?;
            copy_dir_recursive(&path, &target, src, run_root, ignore_patterns)?;
        } else if path.is_file() {
            if let Err(err) = fs::copy(&path, &target) {
                if err.kind() == io::ErrorKind::PermissionDenied {
                    continue;
                }
                return Err(format!("copy {}: {err}", path.to_string_lossy()));
            }
        }
    }
    Ok(())
}

fn copy_workspace_minimal(
    src: &Path,
    dst: &Path,
    run_root: &Path,
    ignore_patterns: &[String],
) -> Result<(), String> {
    let entries = fs::read_dir(src)
        .map_err(|err| format!("read workspace root {}: {err}", src.to_string_lossy()))?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("");
        if should_skip_workspace_entry(name, &path, src, run_root, ignore_patterns) {
            continue;
        }
        if path.is_file() && is_lightweight_seed_file(&path) {
            let target = dst.join(name);
            if let Err(err) = fs::copy(&path, &target) {
                if err.kind() == io::ErrorKind::PermissionDenied {
                    continue;
                }
                return Err(format!("copy {}: {err}", path.to_string_lossy()));
            }
        }
    }
    Ok(())
}

fn copy_dir_recursive(
    src: &Path,
    dst: &Path,
    workspace_root: &Path,
    run_root: &Path,
    ignore_patterns: &[String],
) -> Result<(), String> {
    let entries = match fs::read_dir(src) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::PermissionDenied => return Ok(()),
        Err(err) => {
            return Err(format!(
                "read workspace dir {}: {err}",
                src.to_string_lossy()
            ))
        }
    };
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("");
        if should_skip_workspace_entry(name, &path, workspace_root, run_root, ignore_patterns) {
            continue;
        }
        let target = dst.join(name);
        if path.is_dir() {
            fs::create_dir_all(&target)
                .map_err(|err| format!("create sandbox dir {}: {err}", target.to_string_lossy()))?;
            copy_dir_recursive(&path, &target, workspace_root, run_root, ignore_patterns)?;
        } else if path.is_file() {
            if let Err(err) = fs::copy(&path, &target) {
                if err.kind() == io::ErrorKind::PermissionDenied {
                    continue;
                }
                return Err(format!("copy {}: {err}", path.to_string_lossy()));
            }
        }
    }
    Ok(())
}

fn should_skip_workspace_entry(
    name: &str,
    path: &Path,
    workspace_root: &Path,
    run_root: &Path,
    ignore_patterns: &[String],
) -> bool {
    if path.starts_with(run_root) {
        return true;
    }
    if fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        return true;
    }
    let relative = path
        .strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    if matches!(
        relative.as_str(),
        "verification"
            | "evidence"
            | "datasets"
            | "models"
            | "dist"
            | "site-dist"
            | "benchmark-output"
            | "transcripts"
            | "tmp"
            | ".codex"
            | ".agents"
            | ".cache"
            | ".config"
            | ".local"
            | "AppData"
            | "Application Data"
            | "Contacts"
            | "Cookies"
            | "Local Settings"
            | "NetHood"
            | "PrintHood"
            | "Recent"
            | "SendTo"
            | "Start Menu"
            | "Templates"
    ) || relative.starts_with("verification/")
        || relative.starts_with("evidence/")
        || relative.starts_with("datasets/")
        || relative.starts_with("models/")
        || relative.starts_with("dist/")
        || relative.starts_with("site-dist/")
        || relative.starts_with("benchmark-output/")
        || relative.starts_with("transcripts/")
        || relative.starts_with("tmp/")
        || relative.starts_with(".codex/")
        || relative.starts_with(".agents/")
        || relative.starts_with(".cache/")
        || relative.starts_with(".config/")
        || relative.starts_with(".local/")
        || relative.starts_with("AppData/")
        || relative.starts_with("Application Data/")
    {
        return true;
    }
    if ignore_patterns
        .iter()
        .any(|pattern| gitignore_pattern_matches(pattern, name, &relative))
    {
        return true;
    }
    matches!(
        name,
        ".git"
            | ".helix"
            | ".pytest_cache"
            | "__pycache__"
            | "target"
            | "node_modules"
            | ".venv"
            | "venv"
            | ".codex"
            | ".agents"
            | ".cache"
            | ".config"
            | ".local"
            | "AppData"
            | "Application Data"
            | "Contacts"
            | "Cookies"
            | "Local Settings"
            | "NetHood"
            | "PrintHood"
            | "Recent"
            | "SendTo"
            | "Start Menu"
            | "Templates"
    )
}

fn gitignore_pattern_matches(pattern: &str, name: &str, relative: &str) -> bool {
    let pattern = pattern.trim().trim_start_matches('/');
    if pattern.is_empty() {
        return false;
    }
    if let Some(dir) = pattern.strip_suffix('/') {
        return name == dir || relative == dir || relative.starts_with(&format!("{dir}/"));
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        return name.ends_with(&format!(".{suffix}"));
    }
    name == pattern
        || relative == pattern
        || relative.starts_with(&format!("{pattern}/"))
        || relative.ends_with(&format!("/{pattern}"))
}

fn rust_core_self_available() -> bool {
    env::current_exe()
        .map(|path| path.exists())
        .unwrap_or(false)
}

fn git_dirty(repo_root: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("status")
        .arg("--porcelain")
        .output()
        .map(|output| !String::from_utf8_lossy(&output.stdout).trim().is_empty())
        .unwrap_or(false)
}

fn init_sandbox_git(sandbox: &Path) -> Result<(), String> {
    run_git(sandbox, &["init"])?;
    run_git(
        sandbox,
        &["config", "user.email", "helix-opencode@example.local"],
    )?;
    run_git(sandbox, &["config", "user.name", "HeliX OpenCode Sandbox"])?;
    run_git(sandbox, &["add", "-A", "-f"])?;
    let _ = run_git(sandbox, &["commit", "-m", "helix opencode sandbox base"]);
    Ok(())
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|err| format!("git {:?}: {err}", args))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}{}",
            args,
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn git_output(cwd: &Path, args: &[&str]) -> Result<String, String> {
    Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|err| format!("git {:?}: {err}", args))
        .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(max_chars) {
        out.push(ch);
    }
    if text.chars().count() > max_chars {
        out.push_str("\n...<truncated>...");
    }
    out
}

fn read_mcp_message<R: BufRead>(reader: &mut R) -> Result<Option<String>, String> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).map_err(|err| err.to_string())?;
        if read == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            break;
        }
        let lower = trimmed.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse::<usize>().ok();
        }
    }
    let len = content_length.ok_or_else(|| "missing MCP Content-Length".to_string())?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|err| err.to_string())?;
    Ok(Some(String::from_utf8_lossy(&buf).to_string()))
}

fn write_mcp_message(message: &str) -> Result<(), String> {
    let mut stdout = io::stdout();
    write!(
        stdout,
        "Content-Length: {}\r\n\r\n{}",
        message.as_bytes().len(),
        message
    )
    .map_err(|err| err.to_string())?;
    stdout.flush().map_err(|err| err.to_string())
}

fn handle_mcp_message(message: &str, workspace_root: &Path, evidence_root: &Path) -> String {
    let id = json_id_field(message).unwrap_or_else(|| "null".to_string());
    let method = json_string_field(message, "method").unwrap_or_default();
    let result = match method.as_str() {
        "initialize" => "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"helix\",\"version\":\"0.1.0-rust\"}}".to_string(),
        "tools/list" => format!("{{\"tools\":[{}]}}", mcp_tools_json()),
        "tools/call" => {
            let name = json_string_field(message, "name").unwrap_or_default();
            let tool_result = match name.as_str() {
                "helix_trust" => helix_trust_tool(workspace_root, message),
                "evidence_latest" => evidence_latest_tool(evidence_root, message),
                "suite_search" => suite_search_tool(evidence_root, message),
                "memory_search" => memory_search_tool(workspace_root, message),
                _ => format!("{{\"status\":\"error\",\"error\":{}}}", q("unknown HeliX MCP tool")),
            };
            format!("{{\"content\":[{{\"type\":\"text\",\"text\":{}}}],\"isError\":false}}", q(&tool_result))
        }
        "notifications/initialized" => return String::new(),
        _ => "{\"status\":\"ignored\"}".to_string(),
    };
    if result.is_empty() {
        String::new()
    } else {
        format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{}}}",
            id, result
        )
    }
}

fn json_id_field(text: &str) -> Option<String> {
    let needle = "\"id\"";
    let start = text.find(needle)?;
    let after_key = &text[start + needle.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    if rest.starts_with('"') {
        return json_string_field(text, "id").map(|value| q(&value));
    }
    json_numberish_field(text, "id")
}

fn mcp_tools_json() -> String {
    [
        r#"{"name":"helix_trust","description":"Compact local HeliX trust summary for a thread.","inputSchema":{"type":"object","properties":{"thread_id":{"type":"string"},"include_quarantined":{"type":"boolean"}}}}"#,
        r#"{"name":"evidence_latest","description":"Latest compact HeliX evidence records from the Rust suite index.","inputSchema":{"type":"object","properties":{"limit":{"type":"integer"}}}}"#,
        r#"{"name":"suite_search","description":"Search indexed nuclear verification artifacts without deep scanning.","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}}"#,
        r#"{"name":"memory_search","description":"Search HeliX memory journal with session scope by default.","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"thread_id":{"type":"string"},"retrieval_scope":{"type":"string"},"limit":{"type":"integer"},"include_quarantined":{"type":"boolean"}},"required":["query"]}}"#,
    ]
    .join(",")
}

fn helix_trust_tool(workspace_root: &Path, message: &str) -> String {
    let thread_id =
        json_string_field(message, "thread_id").unwrap_or_else(|| "current".to_string());
    let trust_root = workspace_root
        .join("session-os")
        .join("trust")
        .join("trust_root.json");
    let active_key = fs::read_to_string(&trust_root)
        .ok()
        .and_then(|text| json_string_field(&text, "active_key_id"));
    format!(
        "{{\"kind\":\"helix-local-trust-summary\",\"thread_id\":{},\"status\":\"available\",\"trust_root_active_key_id\":{},\"full_report\":\"Use HeliX forensic commands for full proof; MCP keeps trust compact.\"}}",
        q(&thread_id),
        active_key.as_ref().map(|item| q(item)).unwrap_or_else(|| "null".to_string())
    )
}

fn evidence_latest_tool(evidence_root: &Path, message: &str) -> String {
    let limit = json_numberish_field(message, "limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .min(20);
    let records = read_jsonl_index(evidence_root).unwrap_or_default();
    let mut rows = records;
    rows.sort_by(|a, b| b.mtime_ms.cmp(&a.mtime_ms));
    let rendered = rows
        .into_iter()
        .take(limit)
        .map(|record| record_json(&record))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"status\":\"ok\",\"records\":[{}]}}", rendered)
}

fn suite_search_tool(evidence_root: &Path, message: &str) -> String {
    let query = json_string_field(message, "query").unwrap_or_default();
    let limit = json_numberish_field(message, "limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
        .min(20);
    suite_search(evidence_root, &query, limit)
        .unwrap_or_else(|err| format!("{{\"status\":\"error\",\"error\":{}}}", q(&err)))
}

fn memory_search_tool(workspace_root: &Path, message: &str) -> String {
    let query = json_string_field(message, "query")
        .unwrap_or_default()
        .to_lowercase();
    let thread_id = json_string_field(message, "thread_id");
    let scope =
        json_string_field(message, "retrieval_scope").unwrap_or_else(|| "session".to_string());
    let include_quarantined = message.contains("\"include_quarantined\":true");
    let limit = json_numberish_field(message, "limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .min(20);
    let journal = workspace_root
        .join("session-os")
        .join("memory.journal.jsonl");
    let text = fs::read_to_string(&journal).unwrap_or_default();
    let mut rows = Vec::new();
    for line in text.lines() {
        if rows.len() >= limit {
            break;
        }
        let lowered = line.to_lowercase();
        if !query.split_whitespace().all(|term| lowered.contains(term)) {
            continue;
        }
        if scope == "session" {
            if let Some(thread) = &thread_id {
                if !line.contains(&format!("\"session_id\":\"{}\"", thread))
                    && !line.contains(&format!("\"session_id\": \"{}\"", thread))
                {
                    continue;
                }
            }
        }
        if !include_quarantined && lowered.contains("\"quarantined\":true") {
            continue;
        }
        rows.push(format!(
            "{{\"source\":\"memory.journal.jsonl\",\"preview\":{},\"sha256\":{}}}",
            q(&truncate(line, 1200)),
            q(&sha256_hex(line.as_bytes()))
        ));
    }
    format!(
        "{{\"status\":\"ok\",\"retrieval_scope\":{},\"query\":{},\"result_count\":{},\"results\":[{}]}}",
        q(&scope),
        q(&query),
        rows.len(),
        rows.join(",")
    )
}

fn sha256_hex(data: &[u8]) -> String {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = H0;
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("helix_cli_core_{}_{}", name, millis_now()));
        fs::create_dir_all(&dir).expect("create temp test dir");
        dir
    }

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn write_fake_opencode(bin_dir: &Path) -> PathBuf {
        fs::create_dir_all(bin_dir).expect("create fake bin dir");
        #[cfg(windows)]
        {
            let path = bin_dir.join("opencode.cmd");
            fs::write(
                &path,
                "@echo off\r\nif \"%1\"==\"run\" (\r\n  echo patched by fake opencode> hello.txt\r\n  echo new file from fake opencode> created.txt\r\n  echo ignored html from fake opencode> generated.html\r\n  echo {\"final_answer\":\"patched\"}\r\n  exit /B 0\r\n)\r\necho unexpected args 1>&2\r\nexit /B 2\r\n",
            )
            .expect("write fake opencode cmd");
            path
        }
        #[cfg(not(windows))]
        {
            let path = bin_dir.join("opencode");
            fs::write(
                &path,
                "#!/usr/bin/env sh\nif [ \"$1\" = \"run\" ]; then\n  printf 'patched by fake opencode\\n' > hello.txt\n  printf 'new file from fake opencode\\n' > created.txt\n  printf 'ignored html from fake opencode\\n' > generated.html\n  printf '{\"final_answer\":\"patched\"}\\n'\n  exit 0\nfi\necho unexpected args >&2\nexit 2\n",
            )
            .expect("write fake opencode shell");
            let mut perms = fs::metadata(&path).expect("fake metadata").permissions();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                perms.set_mode(0o755);
                fs::set_permissions(&path, perms).expect("chmod fake opencode");
            }
            path
        }
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn mcp_tools_are_compact_and_read_only() {
        let tools = mcp_tools_json();
        assert!(tools.contains("\"helix_trust\""));
        assert!(tools.contains("\"evidence_latest\""));
        assert!(tools.contains("\"suite_search\""));
        assert!(tools.contains("\"memory_search\""));
        assert!(!tools.contains("patch"));
        assert!(!tools.contains("exec"));
    }

    #[test]
    fn opencode_backend_runs_in_sandbox_and_captures_patch() {
        if !git_available() {
            return;
        }
        let root = temp_test_dir("opencode_sandbox");
        let repo = root.join("repo");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&repo).expect("create repo");
        fs::write(repo.join("hello.txt"), "original\n").expect("write source file");
        let fake_bin = write_fake_opencode(&bin_dir);

        let result = opencode_run(
            &repo,
            "change hello",
            Some(fake_bin.to_string_lossy().to_string()),
            Some("test-run".to_string()),
            None,
        )
        .expect("opencode run succeeds");

        assert!(result.contains("\"status\":\"passed\""));
        assert_eq!(
            fs::read_to_string(repo.join("hello.txt")).expect("read real repo"),
            "original\n"
        );
        let patch_path = repo
            .join("verification")
            .join("opencode-agent")
            .join("test-run")
            .join("patch.diff");
        let patch = fs::read_to_string(&patch_path).expect("read patch");
        assert!(patch.contains("patched by fake opencode"));
        assert!(patch.contains("created.txt"));
        assert!(patch.contains("new file from fake opencode"));
        assert!(repo
            .join("verification")
            .join("opencode-agent")
            .join("test-run")
            .join("artifact.json")
            .exists());
        assert!(repo
            .join("verification")
            .join("opencode-agent")
            .join("test-run")
            .join("trust_card.json")
            .exists());
        assert!(repo
            .join("verification")
            .join("opencode-agent")
            .join("test-run")
            .join("task_capsule.json")
            .exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn opencode_run_sanitizes_run_id_and_respects_gitignore() {
        if !git_available() {
            return;
        }
        let root = temp_test_dir("opencode_hardened");
        let repo = root.join("repo");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&repo).expect("create repo");
        fs::write(repo.join("hello.txt"), "original\n").expect("write source file");
        fs::write(repo.join(".gitignore"), "secret.txt\n*.tmp\n").expect("write gitignore");
        fs::write(repo.join("secret.txt"), "do not copy\n").expect("write ignored file");
        fs::write(repo.join("scratch.tmp"), "do not copy\n").expect("write ignored glob");
        fs::create_dir_all(repo.join("web")).expect("create web");
        fs::write(
            repo.join("web").join("existing.html"),
            "baseline ignored html\n",
        )
        .expect("write baseline ignored html");
        let fake_bin = write_fake_opencode(&bin_dir);

        let result = opencode_run(
            &repo,
            "change hello",
            Some(fake_bin.to_string_lossy().to_string()),
            Some("../bad run".to_string()),
            None,
        )
        .expect("opencode run succeeds");

        assert!(result.contains("\"run_id\":\"bad-run\""));
        let sandbox = repo
            .join(".helix")
            .join("opencode-runs")
            .join("bad-run")
            .join("worktree");
        assert!(!sandbox.join("secret.txt").exists());
        assert!(!sandbox.join("scratch.tmp").exists());
        assert!(repo
            .join("verification")
            .join("opencode-agent")
            .join("bad-run")
            .join("trust_card.json")
            .exists());
        let patch = fs::read_to_string(
            repo.join("verification")
                .join("opencode-agent")
                .join("bad-run")
                .join("patch.diff"),
        )
        .expect("read patch");
        assert!(!patch.contains("existing.html"));
        assert!(patch.contains("generated.html"));
        assert!(patch.contains("ignored html from fake opencode"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn opencode_run_skips_user_profile_private_dirs() {
        if !git_available() {
            return;
        }
        let root = temp_test_dir("opencode_private_dirs");
        let repo = root.join("repo");
        let bin_dir = root.join("bin");
        fs::create_dir_all(repo.join(".codex").join("memories")).expect("create codex dir");
        fs::create_dir_all(repo.join("AppData").join("Local")).expect("create appdata dir");
        fs::create_dir_all(repo.join("public-folder")).expect("create public folder");
        fs::write(repo.join("hello.txt"), "original\n").expect("write source file");
        fs::write(repo.join("README.md"), "seed readme\n").expect("write readme");
        fs::write(
            repo.join(".codex")
                .join("memories")
                .join("pytest-helix-cli"),
            "private memory\n",
        )
        .expect("write private memory");
        fs::write(
            repo.join("AppData").join("Local").join("cache.db"),
            "private cache\n",
        )
        .expect("write private cache");
        fs::write(
            repo.join("public-folder").join("nested.txt"),
            "should not copy whole home-like tree\n",
        )
        .expect("write nested public file");
        let fake_bin = write_fake_opencode(&bin_dir);

        let result = opencode_run(
            &repo,
            "change hello",
            Some(fake_bin.to_string_lossy().to_string()),
            Some("private-dir-run".to_string()),
            None,
        )
        .expect("opencode run succeeds");

        assert!(result.contains("\"status\":\"passed\""));
        let sandbox = repo
            .join(".helix")
            .join("opencode-runs")
            .join("private-dir-run")
            .join("worktree");
        assert!(!sandbox.join(".codex").exists());
        assert!(!sandbox.join("AppData").exists());
        assert!(!sandbox.join("public-folder").exists());
        assert!(sandbox.join("README.md").exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn opencode_run_handles_multiline_goal_via_task_brief() {
        if !git_available() {
            return;
        }
        let root = temp_test_dir("opencode_multiline_goal");
        let repo = root.join("repo");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&repo).expect("create repo");
        fs::write(repo.join("hello.txt"), "original\n").expect("write source file");
        let fake_bin = write_fake_opencode(&bin_dir);

        let result = opencode_run(
            &repo,
            "line one\nline two with quotes \"and\" unicode cafe",
            Some(fake_bin.to_string_lossy().to_string()),
            Some("multi-run".to_string()),
            None,
        )
        .expect("opencode run succeeds");

        assert!(result.contains("\"status\":\"passed\""));
        let brief = fs::read_to_string(
            repo.join(".helix")
                .join("opencode-runs")
                .join("multi-run")
                .join("worktree")
                .join("HELIX_TASK_BRIEF.md"),
        )
        .expect("read task brief");
        assert!(brief.contains("line one"));
        assert!(brief.contains("line two"));
        assert!(brief.contains("Execution Contract"));
        let artifact = fs::read_to_string(
            repo.join("verification")
                .join("opencode-agent")
                .join("multi-run")
                .join("artifact.json"),
        )
        .expect("read artifact");
        assert!(artifact.contains("HELIX_TASK_BRIEF.md"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lab_profiles_and_deep_profile_are_explicit() {
        let profiles = lab_profiles().expect("profiles");
        assert!(profiles.contains("patch-safety"));
        assert!(profiles.contains("deep-nuclear"));
        let root = temp_test_dir("lab_profile");
        let report = lab_run("deep-nuclear", &root.join("verification"), &root).expect("lab run");
        assert!(report.contains("requires_explicit_deep"));
        let _ = fs::remove_dir_all(root);
    }
}
