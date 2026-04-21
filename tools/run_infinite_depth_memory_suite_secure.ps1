param(
    [string]$Case = "all",
    [int]$Depth = 5000,
    [int]$SmallDepth = 128,
    [int]$MidDepth = 1024,
    [int]$Repeats = 7,
    [int]$BudgetTokens = 800,
    [int]$Limit = 5,
    [double]$MaxEmptyQueryMs = 75.0,
    [double]$MaxBoundedContextMs = 150.0,
    [double]$MaxAuditChainMs = 250.0,
    [double]$BaselineMinSpeedup = 1.05,
    [int]$BaselineRuns = 1,
    [string]$OutputDir = "verification/nuclear-methodology/infinite-depth-memory",
    [string]$LegacyTelemetry = "verification/infinite_loop_benchmarks.json",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$allowedCases = @(
    "all",
    "legacy-telemetry-boundary",
    "empty-retrieval-fast-path",
    "bounded-context-under-depth",
    "scale-gradient-vs-naive-copy",
    "deep-parent-chain-audit",
    "claim-boundary-detector"
)
if ($allowedCases -notcontains $Case) {
    throw "Unsupported Case '$Case'. Allowed: $($allowedCases -join ', ')"
}
if ($Depth -lt 16) {
    throw "Depth must be at least 16."
}
if ($Repeats -lt 1) {
    throw "Repeats must be at least 1."
}
if ($BaselineRuns -lt 1) {
    throw "BaselineRuns must be at least 1."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "infinite-depth-memory-$stamp"
}
$logPath = Join-Path $verificationDir "local-infinite-depth-memory-suite-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-infinite-depth-memory-suite-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-infinite-depth-memory-suite.log"
$stableManifestPath = Join-Path $verificationDir "local-infinite-depth-memory-suite-run.json"

$previous = @{
    HELIX_RUN_STARTED_AT_UTC = $env:HELIX_RUN_STARTED_AT_UTC
    HELIX_RUN_DATE_UTC = $env:HELIX_RUN_DATE_UTC
    HELIX_RUN_TIMEZONE = $env:HELIX_RUN_TIMEZONE
    HELIX_RUN_ID = $env:HELIX_RUN_ID
}

$exitCode = 1
$artifactPath = $null
$artifactHash = $null
$artifactBytes = 0

try {
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Case: $Case"
    Write-Host "[helix] Depth: $Depth"
    Write-Host "[helix] Repeats: $Repeats"
    Write-Host "[helix] Baseline runs: $BaselineRuns"
    Write-Host "[helix] Budget tokens: $BudgetTokens"
    Write-Host "[helix] Limit: $Limit"
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_infinite_depth_memory_suite_v1.py",
        "--output-dir", $verificationDir,
        "--legacy-telemetry", $LegacyTelemetry,
        "--run-id", $RunId,
        "--case", $Case,
        "--depth", [string]$Depth,
        "--small-depth", [string]$SmallDepth,
        "--mid-depth", [string]$MidDepth,
        "--repeats", [string]$Repeats,
        "--budget-tokens", [string]$BudgetTokens,
        "--limit", [string]$Limit,
        "--max-empty-query-ms", [string]$MaxEmptyQueryMs,
        "--max-bounded-context-ms", [string]$MaxBoundedContextMs,
        "--max-audit-chain-ms", [string]$MaxAuditChainMs,
        "--baseline-min-speedup", [string]$BaselineMinSpeedup,
        "--baseline-runs", [string]$BaselineRuns
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @pyArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
} finally {
    $endedAtUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    $logHash = $null
    $logBytes = 0
    if (Test-Path -LiteralPath $logPath) {
        Copy-Item -LiteralPath $logPath -Destination $stableLogPath -Force
        $logHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $logBytes = (Get-Item -LiteralPath $logPath).Length
    }

    $candidate = Join-Path $verificationDir "local-infinite-depth-memory-suite-$RunId.json"
    if (($BaselineRuns -gt 1) -and -not (Test-Path -LiteralPath $candidate)) {
        $candidate = Join-Path $verificationDir "local-infinite-depth-memory-baseline-$RunId.json"
    }
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-infinite-depth-memory-suite-run"
        generated_by = "tools/run_infinite_depth_memory_suite_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        case = $Case
        depth = $Depth
        small_depth = $SmallDepth
        mid_depth = $MidDepth
        repeats = $Repeats
        baseline_runs = $BaselineRuns
        budget_tokens = $BudgetTokens
        limit = $Limit
        thresholds = [ordered]@{
            max_empty_query_ms = $MaxEmptyQueryMs
            max_bounded_context_ms = $MaxBoundedContextMs
            max_audit_chain_ms = $MaxAuditChainMs
            baseline_min_speedup = $BaselineMinSpeedup
        }
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        log_path = $logPath
        stable_log_path = $stableLogPath
        log_sha256 = $logHash
        log_bytes = $logBytes
        artifact_path = $artifactPath
        artifact_sha256 = $artifactHash
        artifact_bytes = $artifactBytes
        token_handling = [ordered]@{
            token_required = $false
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        transcript_sidecars = "Each case artifact records transcript_exports.jsonl_path and transcript_exports.md_path."
        claim_boundary = "Bounded context over deep stores, not literal infinite memory or physical zero latency."
    }
    $manifestJson = $manifest | ConvertTo-Json -Depth 8
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($manifestPath, $manifestJson, $utf8NoBom)
    Copy-Item -LiteralPath $manifestPath -Destination $stableManifestPath -Force
    Write-Host "[helix] Evidence manifest: $manifestPath"
    if ($null -ne $artifactPath) {
        Write-Host "[helix] Artifact: $artifactPath"
    }

    foreach ($key in $previous.Keys) {
        if ($null -ne $previous[$key]) {
            Set-Item -Path "Env:\$key" -Value $previous[$key]
        } else {
            Remove-Item -Path "Env:\$key" -ErrorAction SilentlyContinue
        }
    }
}

exit $exitCode
