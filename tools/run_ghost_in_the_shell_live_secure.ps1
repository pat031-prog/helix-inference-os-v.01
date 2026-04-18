param(
    [switch]$EnableThinking,
    [switch]$RequireExactModels,
    [switch]$SkipBuild,
    [string]$LlamaModel = "meta-llama/Llama-3.2-3B-Instruct",
    [string]$MistralModel = "mistralai/Mistral-7B-Instruct-v0.3",
    [string]$QwenSmallModel = "Qwen/Qwen2.5-7B-Instruct",
    [string]$GemmaModel = "google/gemma-4-26B-A4B-it",
    [string]$ClaudeModel = "anthropic/claude-4-sonnet",
    [string]$QwenModel = "Qwen/Qwen3.5-9B",
    [string]$PytestArgs = "tests/test_ghost_in_the_shell_live.py -v -s"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
$runId = "ghost-in-the-shell-live-$stamp"
$logPath = Join-Path $verificationDir "local-ghost-in-the-shell-live-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-ghost-in-the-shell-live-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-ghost-in-the-shell-live.log"
$stableManifestPath = Join-Path $verificationDir "local-ghost-in-the-shell-live-run.json"
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

function Copy-ArtifactWithStamp {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Stamp
    )
    $source = Join-Path $verificationDir $Name
    if (-not (Test-Path -LiteralPath $source)) {
        return $null
    }
    $base = [System.IO.Path]::GetFileNameWithoutExtension($Name)
    $ext = [System.IO.Path]::GetExtension($Name)
    $dest = Join-Path $verificationDir "$base-$Stamp$ext"
    Copy-Item -LiteralPath $source -Destination $dest -Force
    return $dest
}

$rustManifest = Join-Path $repo "crates\helix-state-server\Cargo.toml"
$rustSource = Join-Path $repo "crates\helix-state-server\src\main.rs"
$rustBin = Join-Path $repo "crates\helix-state-server\target\x86_64-pc-windows-gnullvm\release\helix-state-server.exe"

$previous = @{
    DEEPINFRA_API_TOKEN = $env:DEEPINFRA_API_TOKEN
    HELIX_OBSERVATORY_LLAMA_MODEL = $env:HELIX_OBSERVATORY_LLAMA_MODEL
    HELIX_OBSERVATORY_MISTRAL_MODEL = $env:HELIX_OBSERVATORY_MISTRAL_MODEL
    HELIX_OBSERVATORY_QWEN_SMALL_MODEL = $env:HELIX_OBSERVATORY_QWEN_SMALL_MODEL
    HELIX_OBSERVATORY_GEMMA_MODEL = $env:HELIX_OBSERVATORY_GEMMA_MODEL
    HELIX_OBSERVATORY_CLAUDE_MODEL = $env:HELIX_OBSERVATORY_CLAUDE_MODEL
    HELIX_OBSERVATORY_QWEN_MODEL = $env:HELIX_OBSERVATORY_QWEN_MODEL
    HELIX_OBSERVATORY_DISABLE_THINKING = $env:HELIX_OBSERVATORY_DISABLE_THINKING
    HELIX_OBSERVATORY_SIMULATE_SUBSTITUTIONS = $env:HELIX_OBSERVATORY_SIMULATE_SUBSTITUTIONS
    HELIX_REQUIRE_EXACT_MODELS = $env:HELIX_REQUIRE_EXACT_MODELS
    HELIX_RUN_STARTED_AT_UTC = $env:HELIX_RUN_STARTED_AT_UTC
    HELIX_RUN_DATE_UTC = $env:HELIX_RUN_DATE_UTC
    HELIX_RUN_TIMEZONE = $env:HELIX_RUN_TIMEZONE
    HELIX_RUN_ID = $env:HELIX_RUN_ID
}

$exitCode = 1
$buildRan = $false
$timestampedArtifacts = @()
$artifactHashes = [ordered]@{}

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

    $env:HELIX_OBSERVATORY_LLAMA_MODEL = $LlamaModel
    $env:HELIX_OBSERVATORY_MISTRAL_MODEL = $MistralModel
    $env:HELIX_OBSERVATORY_QWEN_SMALL_MODEL = $QwenSmallModel
    $env:HELIX_OBSERVATORY_GEMMA_MODEL = $GemmaModel
    $env:HELIX_OBSERVATORY_CLAUDE_MODEL = $ClaudeModel
    $env:HELIX_OBSERVATORY_QWEN_MODEL = $QwenModel
    $env:HELIX_OBSERVATORY_SIMULATE_SUBSTITUTIONS = "0"
    $env:HELIX_REQUIRE_EXACT_MODELS = $(if ($RequireExactModels) { "1" } else { "0" })
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $runId
    if ($EnableThinking) {
        $env:HELIX_OBSERVATORY_DISABLE_THINKING = "0"
    } else {
        $env:HELIX_OBSERVATORY_DISABLE_THINKING = "1"
    }

    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if ([string]::IsNullOrWhiteSpace($plainToken)) {
        throw "DeepInfra token cannot be empty."
    }
    $env:DEEPINFRA_API_TOKEN = $plainToken
    $plainToken = $null
    Write-Host "[helix] Token loaded for this pytest process only. It will not be written to disk."

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Ghost-in-the-Shell Live Gauntlet"
    Write-Host "[helix] Provider probes: $LlamaModel | $MistralModel | $QwenSmallModel"
    Write-Host "[helix] Shell models: $GemmaModel | $ClaudeModel | $QwenModel"
    Write-Host "[helix] Require exact models: $RequireExactModels"
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
    $stableArtifacts = @(
        "local-ghost-in-the-shell-live.json",
        "local-ghost-shell-conversation-ledger.json",
        "local-ghost-shell-task-scores.json"
    )
    foreach ($name in $stableArtifacts) {
        $copied = Copy-ArtifactWithStamp -Name $name -Stamp $stamp
        if ($null -ne $copied) {
            $timestampedArtifacts += $copied
            $artifactHashes[[System.IO.Path]::GetFileName($copied)] = (Get-FileHash -Algorithm SHA256 -LiteralPath $copied).Hash.ToLowerInvariant()
        }
    }
    $artifactBytes = 0
    foreach ($artifactPath in $timestampedArtifacts) {
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
        artifact = "local-ghost-in-the-shell-live-run"
        generated_by = "tools/run_ghost_in_the_shell_live_secure.ps1"
        mode = "real"
        run_id = $runId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        provider_probe_models = @($LlamaModel, $MistralModel, $QwenSmallModel)
        shell_models = @($GemmaModel, $ClaudeModel, $QwenModel)
        model_audit_status = "delegated_to_timestamped_artifacts"
        model_audit_artifacts = $timestampedArtifacts
        require_exact_models = [bool]$RequireExactModels
        build_ran = $buildRan
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        suspicious_short_log = $suspiciousShortLog
        log_path = $logPath
        stable_log_path = $stableLogPath
        log_sha256 = $hash
        log_bytes = $bytes
        timestamped_artifact_bytes = $artifactBytes
        timestamped_artifacts = $timestampedArtifacts
        timestamped_artifact_sha256 = $artifactHashes
        token_handling = [ordered]@{
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        claim_boundary = "Ghost means operational continuity through HeliX memory and Merkle lineage, not sentience."
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Copy-Item -LiteralPath $manifestPath -Destination $stableManifestPath -Force
    Write-Host "[helix] Evidence manifest: $manifestPath"

    foreach ($key in $previous.Keys) {
        if ($null -ne $previous[$key]) {
            Set-Item -Path "Env:\$key" -Value $previous[$key]
        } else {
            Remove-Item -Path "Env:\$key" -ErrorAction SilentlyContinue
        }
    }
}

exit $exitCode
