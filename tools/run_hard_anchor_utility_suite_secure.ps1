param(
    [string]$Case = "all",
    [int]$Depth = 5000,
    [int]$BytesPerNode = 8192,
    [int]$Repeats = 7,
    [double]$MaxAnchorMs = 25.0,
    [double]$MinSpeedup = 9.0,
    [double]$MaxCompressionRatio = 0.05,
    [switch]$UseDeepInfra,
    [string]$SolverModel = "Qwen/Qwen3.6-35B-A3B",
    [string]$AuditorModel = "anthropic/claude-4-sonnet",
    [int]$Tokens = 2200,
    [string]$OutputDir = "verification/nuclear-methodology/hard-anchor-utility",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$allowedCases = @(
    "all",
    "rust-identity-lane-benchmark",
    "exact-anchor-recovery-under-lossy-summary",
    "auditor-visible-evidence-bridge",
    "tombstone-metabolism-routing",
    "multi-hop-policy-resolution",
    "claim-boundary-detector"
)
if ($allowedCases -notcontains $Case) {
    throw "Unsupported Case '$Case'. Allowed: $($allowedCases -join ', ')"
}
if ($Depth -lt 64) {
    throw "Depth must be at least 64."
}
if ($BytesPerNode -lt 256) {
    throw "BytesPerNode must be at least 256."
}
if ($Repeats -lt 1) {
    throw "Repeats must be at least 1."
}
function Convert-SecureStringToPlainText {
    param([Parameter(Mandatory = $true)][securestring]$Secure)
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
}

function Assert-DeepInfraModelRef {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -notmatch "^[^/\s]+/[^/\s]+$") {
        throw "$Name must be a full DeepInfra model ref like 'owner/model'. Got: '$Value'"
    }
}
if ($UseDeepInfra) {
    Assert-DeepInfraModelRef -Name "SolverModel" -Value $SolverModel
    Assert-DeepInfraModelRef -Name "AuditorModel" -Value $AuditorModel
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "hard-anchor-utility-$stamp"
}
$logPath = Join-Path $verificationDir "local-hard-anchor-utility-suite-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-hard-anchor-utility-suite-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-hard-anchor-utility-suite.log"
$stableManifestPath = Join-Path $verificationDir "local-hard-anchor-utility-suite-run.json"

$previous = @{
    DEEPINFRA_API_TOKEN = $env:DEEPINFRA_API_TOKEN
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
    if ($UseDeepInfra) {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."
    } else {
        Write-Host "[helix] Local deterministic mode selected; no token required."
    }

    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Case: $Case"
    Write-Host "[helix] Depth: $Depth"
    Write-Host "[helix] Bytes per node: $BytesPerNode"
    Write-Host "[helix] Repeats: $Repeats"
    Write-Host "[helix] DeepInfra enabled: $UseDeepInfra"
    if ($UseDeepInfra) {
        Write-Host "[helix] Solver model: $SolverModel"
        Write-Host "[helix] Auditor model: $AuditorModel"
        Write-Host "[helix] Tokens: $Tokens"
    }
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_hard_anchor_utility_suite_v1.py",
        "--output-dir", $verificationDir,
        "--run-id", $RunId,
        "--case", $Case,
        "--depth", [string]$Depth,
        "--bytes-per-node", [string]$BytesPerNode,
        "--repeats", [string]$Repeats,
        "--max-anchor-ms", [string]$MaxAnchorMs,
        "--min-speedup", [string]$MinSpeedup,
        "--max-compression-ratio", [string]$MaxCompressionRatio
    )
    if ($UseDeepInfra) {
        $pyArgs += @(
            "--use-deepinfra",
            "--solver-model", $SolverModel,
            "--auditor-model", $AuditorModel,
            "--tokens", [string]$Tokens
        )
    }

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

    $candidate = Join-Path $verificationDir "local-hard-anchor-utility-suite-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-hard-anchor-utility-suite-run"
        generated_by = "tools/run_hard_anchor_utility_suite_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        case = $Case
        depth = $Depth
        bytes_per_node = $BytesPerNode
        repeats = $Repeats
        thresholds = [ordered]@{
            max_anchor_ms = $MaxAnchorMs
            min_speedup = $MinSpeedup
            max_compression_ratio = $MaxCompressionRatio
        }
        deepinfra_enabled = [bool]$UseDeepInfra
        models = [ordered]@{
            solver_requested = $(if ($UseDeepInfra) { $SolverModel } else { $null })
            auditor_requested = $(if ($UseDeepInfra) { $AuditorModel } else { $null })
        }
        tokens = $(if ($UseDeepInfra) { $Tokens } else { 0 })
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
            token_required = [bool]$UseDeepInfra
            token_prompt_hidden = [bool]$UseDeepInfra
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        transcript_sidecars = "Each case artifact records transcript_exports.jsonl_path and transcript_exports.md_path."
        claim_boundary = "Rust hard-anchor identity-lane utility under deep stores, not literal infinite memory or zero-cost context."
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
