param(
    [string]$Case = "all",
    [string]$ForensicModel = "Qwen/Qwen3.6-35B-A3B",
    [string]$AuditorModel = "anthropic/claude-4-sonnet",
    [int]$Tokens = 3600,
    [int]$ChainLength = 48,
    [string]$OutputDir = "verification/nuclear-methodology/long-horizon-checkpoints",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$allowedCases = @(
    "all",
    "long-chain-summary-fidelity",
    "summary-only-continuation",
    "selective-expansion-boundary",
    "recursive-summary-drift",
    "adversarial-checkpoint-injection",
    "cross-model-checkpoint-graft",
    "cost-utility-comparison",
    "correction-resummary-lineage",
    "needle-decoy-stress",
    "temporal-rollback-ambiguity",
    "checkpoint-of-checkpoints-consensus"
)
if ($allowedCases -notcontains $Case) {
    throw "Unsupported Case '$Case'. Allowed: $($allowedCases -join ', ')"
}
if ($ChainLength -lt 12) {
    throw "ChainLength must be at least 12."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "long-horizon-checkpoints-$stamp"
}
$logPath = Join-Path $verificationDir "local-long-horizon-checkpoint-suite-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-long-horizon-checkpoint-suite-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-long-horizon-checkpoint-suite.log"
$stableManifestPath = Join-Path $verificationDir "local-long-horizon-checkpoint-suite-run.json"

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

Assert-DeepInfraModelRef -Name "ForensicModel" -Value $ForensicModel
Assert-DeepInfraModelRef -Name "AuditorModel" -Value $AuditorModel

$previous = @{
    DEEPINFRA_API_TOKEN = $env:DEEPINFRA_API_TOKEN
    HELIX_RECEIPT_SIGNING_MODE = $env:HELIX_RECEIPT_SIGNING_MODE
    HELIX_RECEIPT_SIGNING_SEED = $env:HELIX_RECEIPT_SIGNING_SEED
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
    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if ([string]::IsNullOrWhiteSpace($plainToken)) {
        throw "DeepInfra token cannot be empty."
    }
    $env:DEEPINFRA_API_TOKEN = $plainToken
    $plainToken = $null
    Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."

    $env:HELIX_RECEIPT_SIGNING_MODE = "ephemeral_preregistered"
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Case: $Case"
    Write-Host "[helix] Forensic model: $ForensicModel"
    Write-Host "[helix] Auditor model: $AuditorModel"
    Write-Host "[helix] Tokens: $Tokens"
    Write-Host "[helix] Chain length: $ChainLength"
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_long_horizon_checkpoint_suite_v1.py",
        "--output-dir", $verificationDir,
        "--forensic-model", $ForensicModel,
        "--auditor-model", $AuditorModel,
        "--tokens", [string]$Tokens,
        "--chain-length", [string]$ChainLength,
        "--run-id", $RunId,
        "--case", $Case
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

    $candidate = Join-Path $verificationDir "local-long-horizon-checkpoint-suite-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-long-horizon-checkpoint-suite-run"
        generated_by = "tools/run_long_horizon_checkpoint_suite_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        case = $Case
        models = [ordered]@{
            forensic_requested = $ForensicModel
            auditor_requested = $AuditorModel
        }
        tokens = $Tokens
        chain_length = $ChainLength
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
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        transcript_sidecars = "Each case artifact records transcript_exports.jsonl_path and transcript_exports.md_path."
        claim_boundary = "Long-horizon checkpoint suite. Signed checkpoints support auditable bounded-memory claims, not perfect recall."
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
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
