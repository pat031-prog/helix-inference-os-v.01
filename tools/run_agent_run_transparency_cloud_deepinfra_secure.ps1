param(
    [string]$Models = "Qwen/Qwen3.6-35B-A3B,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct",
    [int]$Tokens = 450,
    [double]$Temperature = 0.2,
    [int]$Timeout = 240,
    [string]$RunId = "",
    [string]$OutputDir = "verification\nuclear-methodology\agent-run-transparency-gauntlet"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "cloud-deepinfra-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

$modelList = @($Models -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($modelList.Count -eq 0) {
    throw "At least one DeepInfra model ref is required."
}
foreach ($model in $modelList) {
    if ($model -notmatch "^[^/\s,]+/[^,\s]+$") {
        throw "Models entry must be a full DeepInfra model ref like 'owner/model'. Got: '$model'"
    }
}

if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $verificationDir = $OutputDir
} else {
    $verificationDir = Join-Path $repo $OutputDir
}
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $verificationDir "local-agent-run-transparency-cloud-deepinfra-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-agent-run-transparency-cloud-deepinfra-$stamp-run.json"
$artifactSlug = "local-agent-run-transparency-cloud-deepinfra-v1"
$artifactPath = Join-Path $verificationDir "$artifactSlug-$RunId.json"
$extractPath = Join-Path $verificationDir "$artifactSlug-$RunId-extract.md"
$bundlePath = Join-Path $verificationDir "$artifactSlug-$RunId-bundle.json"

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

$previousToken = $env:DEEPINFRA_API_TOKEN
$previousRunId = $env:HELIX_RUN_ID
$previousStartedAt = $env:HELIX_RUN_STARTED_AT_UTC
$previousDate = $env:HELIX_RUN_DATE_UTC
$previousTimezone = $env:HELIX_RUN_TIMEZONE
$startedAt = [DateTimeOffset]::Now
$exitCode = 1

try {
    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if ([string]::IsNullOrWhiteSpace($plainToken)) {
        throw "DeepInfra token cannot be empty."
    }

    $env:DEEPINFRA_API_TOKEN = $plainToken
    $plainToken = $null
    $env:HELIX_RUN_ID = $RunId
    $env:HELIX_RUN_STARTED_AT_UTC = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    $env:HELIX_RUN_DATE_UTC = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"

    Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."
    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Models: $($modelList -join ',')"
    Write-Host "[helix] Tokens per model: $Tokens"
    Write-Host "[helix] Temperature: $Temperature"
    Write-Host "[helix] Timeout seconds: $Timeout"
    Write-Host "[helix] Output dir: $verificationDir"

    $pythonArgs = @(
        "tools\run_agent_run_transparency_gauntlet_v1.py",
        "--mode", "cloud-deepinfra",
        "--output-dir", $verificationDir,
        "--run-id", $RunId,
        "--models", ($modelList -join ","),
        "--tokens", "$Tokens",
        "--temperature", "$Temperature",
        "--timeout", "$Timeout"
    )
    & python @pythonArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
} finally {
    $endedAt = [DateTimeOffset]::Now
    $logHash = $null
    $logBytes = 0
    if (Test-Path -LiteralPath $logPath) {
        $logHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $logBytes = (Get-Item -LiteralPath $logPath).Length
    }
    $artifactHash = $null
    $bundleHash = $null
    if (Test-Path -LiteralPath $artifactPath) {
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifactPath).Hash.ToLowerInvariant()
    }
    if (Test-Path -LiteralPath $bundlePath) {
        $bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundlePath).Hash.ToLowerInvariant()
    }

    [ordered]@{
        artifact = "local-agent-run-transparency-cloud-deepinfra-run"
        generated_by = "tools/run_agent_run_transparency_cloud_deepinfra_secure.ps1"
        started_at = $startedAt.ToString("o")
        ended_at = $endedAt.ToString("o")
        duration_s = [Math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        run_id = $RunId
        models = $modelList
        tokens_per_model = $Tokens
        temperature_milli = [int][Math]::Round($Temperature * 1000)
        timeout_s = $Timeout
        log_path = $logPath
        log_sha256 = $logHash
        log_bytes = $logBytes
        artifact_path = $(if (Test-Path -LiteralPath $artifactPath) { $artifactPath } else { $null })
        artifact_sha256 = $artifactHash
        extract_markdown_path = $(if (Test-Path -LiteralPath $extractPath) { $extractPath } else { $null })
        standalone_bundle_path = $(if (Test-Path -LiteralPath $bundlePath) { $bundlePath } else { $null })
        standalone_bundle_sha256 = $bundleHash
        token_handling = [ordered]@{
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        claim_boundary = "Cloud DeepInfra evidence proves local attestability of observed calls, not model semantic truth or global non-equivocation."
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"
    if (Test-Path -LiteralPath $artifactPath) {
        Write-Host "[helix] Artifact: $artifactPath"
    }
    if (Test-Path -LiteralPath $bundlePath) {
        Write-Host "[helix] Bundle: $bundlePath"
    }

    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }
    if ($null -ne $previousRunId) { $env:HELIX_RUN_ID = $previousRunId } else { Remove-Item Env:\HELIX_RUN_ID -ErrorAction SilentlyContinue }
    if ($null -ne $previousStartedAt) { $env:HELIX_RUN_STARTED_AT_UTC = $previousStartedAt } else { Remove-Item Env:\HELIX_RUN_STARTED_AT_UTC -ErrorAction SilentlyContinue }
    if ($null -ne $previousDate) { $env:HELIX_RUN_DATE_UTC = $previousDate } else { Remove-Item Env:\HELIX_RUN_DATE_UTC -ErrorAction SilentlyContinue }
    if ($null -ne $previousTimezone) { $env:HELIX_RUN_TIMEZONE = $previousTimezone } else { Remove-Item Env:\HELIX_RUN_TIMEZONE -ErrorAction SilentlyContinue }
}

exit $exitCode
