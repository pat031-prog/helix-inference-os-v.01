param(
    [switch]$Synthetic,
    [switch]$EnableThinking,
    [switch]$RequireAllModels,
    [switch]$SkipBuild,
    [string]$GemmaModel = "google/gemma-4-26B-A4B-it",
    [string]$ClaudeModel = "anthropic/claude-4-sonnet",
    [string]$QwenModel = "Qwen/Qwen3.5-9B",
    [int]$OuroborosCycles = 20,
    [string]$PytestArgs = "tests/test_identity_trust_gauntlet_v2.py -v -s"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
$runId = "identity-trust-v2-$stamp"
$logPath = Join-Path $verificationDir "local-identity-trust-gauntlet-v2-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-identity-trust-gauntlet-v2-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-identity-trust-gauntlet-v2.log"
$stableManifestPath = Join-Path $verificationDir "local-identity-trust-gauntlet-v2-run.json"
$pytestArgList = @($PytestArgs -split '\s+' | Where-Object { $_ })

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

$rustManifest = Join-Path $repo "crates\helix-state-server\Cargo.toml"
$rustSource = Join-Path $repo "crates\helix-state-server\src\main.rs"
$rustBin = Join-Path $repo "crates\helix-state-server\target\x86_64-pc-windows-gnullvm\release\helix-state-server.exe"

$previousToken = $env:DEEPINFRA_API_TOKEN
$previousGemma = $env:HELIX_V2_GEMMA_MODEL
$previousClaude = $env:HELIX_V2_CLAUDE_MODEL
$previousQwen = $env:HELIX_V2_QWEN_MODEL
$previousThinking = $env:HELIX_IDENTITY_DISABLE_THINKING
$previousCycles = $env:HELIX_OUROBOROS_CYCLES
$previousRunStarted = $env:HELIX_RUN_STARTED_AT_UTC
$previousRunDate = $env:HELIX_RUN_DATE_UTC
$previousRunTimezone = $env:HELIX_RUN_TIMEZONE
$previousRunId = $env:HELIX_RUN_ID
$previousRequireAll = $env:HELIX_REQUIRE_ALL_MODELS
$exitCode = 1
$mode = "real"
$buildRan = $false

try {
    if (-not $SkipBuild) {
        $needsBuild = -not (Test-Path -LiteralPath $rustBin)
        if (-not $needsBuild) {
            $needsBuild = (Get-Item -LiteralPath $rustSource).LastWriteTimeUtc -gt (Get-Item -LiteralPath $rustBin).LastWriteTimeUtc
        }
        if ($needsBuild) {
            Write-Host "[helix] Building helix-state-server because source is newer than binary."
            cargo +stable-x86_64-pc-windows-gnullvm build --release --target x86_64-pc-windows-gnullvm --manifest-path $rustManifest
            $buildRan = $true
        }
    }

    $env:HELIX_V2_GEMMA_MODEL = $GemmaModel
    $env:HELIX_V2_CLAUDE_MODEL = $ClaudeModel
    $env:HELIX_V2_QWEN_MODEL = $QwenModel
    $env:HELIX_OUROBOROS_CYCLES = [string]$OuroborosCycles
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $runId
    $env:HELIX_REQUIRE_ALL_MODELS = $(if ($RequireAllModels) { "1" } else { "0" })
    if ($EnableThinking) {
        $env:HELIX_IDENTITY_DISABLE_THINKING = "0"
    } else {
        $env:HELIX_IDENTITY_DISABLE_THINKING = "1"
    }

    if ($Synthetic) {
        $mode = "synthetic"
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
        Write-Host "[helix] Running Identity Trust Gauntlet v2 in synthetic mode."
    } else {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        Write-Host "[helix] Token loaded for this pytest process only. It will not be written to disk."
    }

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Gemma model: $GemmaModel"
    Write-Host "[helix] Claude model: $ClaudeModel"
    Write-Host "[helix] Qwen model: $QwenModel"
    Write-Host "[helix] Ouroboros cycles: $OuroborosCycles"
    Write-Host "[helix] Require all models: $RequireAllModels"
    Write-Host "[helix] Disable thinking: $(-not $EnableThinking)"

    python -m pytest @pytestArgList 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
} finally {
    $endedAtUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    $hash = $null
    $bytes = 0
    if (Test-Path -LiteralPath $logPath) {
        Copy-Item -LiteralPath $logPath -Destination $stableLogPath -Force
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $bytes = (Get-Item -LiteralPath $logPath).Length
    }
    $evidenceArtifacts = @(
        (Join-Path $verificationDir "local-identity-trust-gauntlet-v2.json"),
        (Join-Path $verificationDir "local-identity-trust-conversation-ledger.json"),
        (Join-Path $verificationDir "local-tri-model-governance-ledger-v2.json"),
        (Join-Path $verificationDir "local-cross-model-ouroboros-relay.json"),
        (Join-Path $verificationDir "local-provider-trust-network-v2.json"),
        (Join-Path $verificationDir "local-conversation-fork-forensics.json")
    )
    $artifactBytes = 0
    foreach ($artifactPath in $evidenceArtifacts) {
        if (Test-Path -LiteralPath $artifactPath) {
            $artifactBytes += (Get-Item -LiteralPath $artifactPath).Length
        }
    }
    $suspiciousShortLog = ($exitCode -eq 0 -and $bytes -lt 5000 -and $artifactBytes -lt 5000)
    if ($suspiciousShortLog) {
        $exitCode = 2
        Write-Host "[helix] suspicious_short_log: passed run produced less than 5000 log bytes and less than 5000 artifact bytes."
    }
    $manifest = [ordered]@{
        artifact = "local-identity-trust-gauntlet-v2-run"
        generated_by = "tools/run_identity_trust_gauntlet_v2_secure.ps1"
        mode = $mode
        run_id = $runId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        models = [ordered]@{
            gemma = $GemmaModel
            claude = $ClaudeModel
            qwen = $QwenModel
        }
        model_audit_status = "delegated_to_stable_artifacts"
        model_audit_artifacts = $evidenceArtifacts
        ouroboros_cycles = $OuroborosCycles
        require_all_models = [bool]$RequireAllModels
        build_ran = $buildRan
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        suspicious_short_log = $suspiciousShortLog
        log_path = $logPath
        stable_log_path = $stableLogPath
        log_sha256 = $hash
        log_bytes = $bytes
        evidence_artifact_bytes = $artifactBytes
        token_handling = [ordered]@{
            token_prompt_hidden = (-not [bool]$Synthetic)
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        date_policy = "model_must_use_artifact_date_only"
        claim_boundary = "Auditable cross-model continuity; not consciousness, legal compliance, or cloud private .hlx state."
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Copy-Item -LiteralPath $manifestPath -Destination $stableManifestPath -Force
    Write-Host "[helix] Evidence manifest: $manifestPath"

    if ($null -ne $previousToken) { $env:DEEPINFRA_API_TOKEN = $previousToken } else { Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue }
    if ($null -ne $previousGemma) { $env:HELIX_V2_GEMMA_MODEL = $previousGemma } else { Remove-Item Env:\HELIX_V2_GEMMA_MODEL -ErrorAction SilentlyContinue }
    if ($null -ne $previousClaude) { $env:HELIX_V2_CLAUDE_MODEL = $previousClaude } else { Remove-Item Env:\HELIX_V2_CLAUDE_MODEL -ErrorAction SilentlyContinue }
    if ($null -ne $previousQwen) { $env:HELIX_V2_QWEN_MODEL = $previousQwen } else { Remove-Item Env:\HELIX_V2_QWEN_MODEL -ErrorAction SilentlyContinue }
    if ($null -ne $previousThinking) { $env:HELIX_IDENTITY_DISABLE_THINKING = $previousThinking } else { Remove-Item Env:\HELIX_IDENTITY_DISABLE_THINKING -ErrorAction SilentlyContinue }
    if ($null -ne $previousCycles) { $env:HELIX_OUROBOROS_CYCLES = $previousCycles } else { Remove-Item Env:\HELIX_OUROBOROS_CYCLES -ErrorAction SilentlyContinue }
    if ($null -ne $previousRunStarted) { $env:HELIX_RUN_STARTED_AT_UTC = $previousRunStarted } else { Remove-Item Env:\HELIX_RUN_STARTED_AT_UTC -ErrorAction SilentlyContinue }
    if ($null -ne $previousRunDate) { $env:HELIX_RUN_DATE_UTC = $previousRunDate } else { Remove-Item Env:\HELIX_RUN_DATE_UTC -ErrorAction SilentlyContinue }
    if ($null -ne $previousRunTimezone) { $env:HELIX_RUN_TIMEZONE = $previousRunTimezone } else { Remove-Item Env:\HELIX_RUN_TIMEZONE -ErrorAction SilentlyContinue }
    if ($null -ne $previousRunId) { $env:HELIX_RUN_ID = $previousRunId } else { Remove-Item Env:\HELIX_RUN_ID -ErrorAction SilentlyContinue }
    if ($null -ne $previousRequireAll) { $env:HELIX_REQUIRE_ALL_MODELS = $previousRequireAll } else { Remove-Item Env:\HELIX_REQUIRE_ALL_MODELS -ErrorAction SilentlyContinue }
}

exit $exitCode
