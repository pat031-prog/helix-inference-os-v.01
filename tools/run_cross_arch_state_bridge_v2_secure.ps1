param(
    [string]$AnalystModel = "meta-llama/Llama-3.2-3B-Instruct",
    [string]$ContinuistModel = "Qwen/Qwen2.5-7B-Instruct",
    [int]$TokensPerRound = 320,
    [int]$LocalTokens = 16,
    [switch]$CloudOnly,
    [ValidateSet("standard", "nuclear")]
    [string]$Scenario = "standard",
    [string]$OutputDir = "verification",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "cross-arch-state-bridge-v2-$stamp"
}
$logPath = Join-Path $verificationDir "local-cross-arch-state-bridge-v2-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-cross-arch-state-bridge-v2-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-cross-arch-state-bridge-v2.log"
$stableManifestPath = Join-Path $verificationDir "local-cross-arch-state-bridge-v2-run.json"

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

Assert-DeepInfraModelRef -Name "AnalystModel" -Value $AnalystModel
Assert-DeepInfraModelRef -Name "ContinuistModel" -Value $ContinuistModel

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
    Write-Host "[helix] Analyst model: $AnalystModel"
    Write-Host "[helix] Continuist model: $ContinuistModel"
    Write-Host "[helix] Tokens per round (cloud): $TokensPerRound"
    Write-Host "[helix] Local tokens (state proof): $LocalTokens"
    Write-Host "[helix] Cloud only: $([bool]$CloudOnly)"
    Write-Host "[helix] Scenario: $Scenario"
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_cross_arch_state_bridge_v2.py",
        "--output-dir", $verificationDir,
        "--tokens-per-round", [string]$TokensPerRound,
        "--local-tokens", [string]$LocalTokens,
        "--analyst-model", $AnalystModel,
        "--continuist-model", $ContinuistModel,
        "--scenario", $Scenario,
        "--run-id", $RunId
    )
    if ($CloudOnly) {
        $pyArgs += "--cloud-only"
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

    $candidate = Join-Path $verificationDir "local-cross-arch-state-bridge-v2-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $suspiciousShortLog = ($exitCode -eq 0 -and $logBytes -lt 2000 -and $artifactBytes -lt 2000)
    if ($suspiciousShortLog) {
        $exitCode = 2
        Write-Host "[helix] suspicious_short_log: passed run produced less than 2000 log bytes and less than 2000 artifact bytes."
    }

    $manifest = [ordered]@{
        artifact = "local-cross-arch-state-bridge-v2-run"
        generated_by = "tools/run_cross_arch_state_bridge_v2_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        models = [ordered]@{
            analyst_requested = $AnalystModel
            continuist_requested = $ContinuistModel
            local_state_proofs = $(if ($CloudOnly) { @() } else { @("gpt2 (transformer)", "Zyphra/Zamba2-1.2B-Instruct-v2 (hybrid-ssm)") })
        }
        tokens_per_round = $TokensPerRound
        local_tokens = $LocalTokens
        cloud_only = [bool]$CloudOnly
        scenario = $Scenario
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        suspicious_short_log = $suspiciousShortLog
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
        claim_boundary = $(if ($CloudOnly) { "Cloud-only run: local .hlx bit-identity evidence is skipped. Claim-B (cross_model_task_continuity) uses two heterogeneous DeepInfra cloud models with strict signed-memory retrieval and deterministic coverage/novelty/repetition gates. No bijective KV<->SSM transfer is claimed." } else { "Claim-A (per_arch_bit_identity) uses local GPT-2 + Zamba2. Claim-B (cross_model_task_continuity) uses two heterogeneous DeepInfra cloud models with strict signed-memory retrieval and deterministic coverage/novelty/repetition gates. No bijective KV<->SSM transfer is claimed." })
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
