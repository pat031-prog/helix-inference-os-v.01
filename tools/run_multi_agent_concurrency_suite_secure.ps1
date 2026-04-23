param(
[string]$Case = "all",
[switch]$UseDeepInfra,
[string]$AlphaModel = "Qwen/Qwen3.5-122B-A10B",
[string]$BetaModel = "mistralai/Devstral-Small-2507",
[string]$GammaModel = "anthropic/claude-4-sonnet",
    [int]$MaxTokens = 512,
    [string]$OutputDir = "verification/nuclear-methodology/multi-agent-concurrency",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$allowedCases = @(
    "all",
    "concurrent-branch-quarantine",
    "gamma-evidence-merge",
    "naive-baseline-collapse"
)
if ($allowedCases -notcontains $Case) {
    throw "Unsupported Case '$Case'. Allowed: $($allowedCases -join ', ')"
}
if ($MaxTokens -lt 64) {
    throw "MaxTokens must be at least 64."
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
    Assert-DeepInfraModelRef -Name "AlphaModel" -Value $AlphaModel
    Assert-DeepInfraModelRef -Name "BetaModel" -Value $BetaModel
    Assert-DeepInfraModelRef -Name "GammaModel" -Value $GammaModel
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "multi-agent-concurrency-$stamp"
}
$logPath = Join-Path $verificationDir "local-multi-agent-concurrency-suite-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-multi-agent-concurrency-suite-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-multi-agent-concurrency-suite.log"
$stableManifestPath = Join-Path $verificationDir "local-multi-agent-concurrency-suite-run.json"

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
    Write-Host "[helix] DeepInfra enabled: $UseDeepInfra"
    if ($UseDeepInfra) {
        Write-Host "[helix] Alpha model: $AlphaModel"
        Write-Host "[helix] Beta model: $BetaModel"
        Write-Host "[helix] Gamma model: $GammaModel"
        Write-Host "[helix] Max tokens: $MaxTokens"
    }
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_multi_agent_concurrency_suite_v1.py",
        "--output-dir", $verificationDir,
        "--run-id", $RunId,
        "--case", $Case,
        "--max-tokens", [string]$MaxTokens
    )
    if ($UseDeepInfra) {
        $pyArgs += @(
            "--use-deepinfra",
            "--alpha-model", $AlphaModel,
            "--beta-model", $BetaModel,
            "--gamma-model", $GammaModel
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

    $candidate = Join-Path $verificationDir "local-multi-agent-concurrency-suite-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-multi-agent-concurrency-suite-run"
        generated_by = "tools/run_multi_agent_concurrency_suite_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        case = $Case
        deepinfra_enabled = [bool]$UseDeepInfra
        models = [ordered]@{
            alpha_requested = $(if ($UseDeepInfra) { $AlphaModel } else { $null })
            beta_requested = $(if ($UseDeepInfra) { $BetaModel } else { $null })
            gamma_requested = $(if ($UseDeepInfra) { $GammaModel } else { $null })
        }
        max_tokens = $MaxTokens
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
        claim_boundary = "Local multi-agent stale-parent race preservation with canonical head and quarantine, not distributed consensus or semantic truth."
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
