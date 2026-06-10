param(
    [switch]$DryRun,
    [string]$Models = "anthropic/claude-4-sonnet,Qwen/Qwen3.6-35B-A3B,google/gemma-4-31B-it,stepfun-ai/Step-3.5-Flash,meta-llama/Llama-3.2-3B-Instruct,mistralai/Mistral-7B-Instruct-v0.3",
    [string]$LocalAliases = "",
    [int]$MaxTokens = 650,
    [double]$Temperature = 0.0,
    [string]$OutputDir = "verification/nuclear-methodology/philosophical-model-probe",
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
    $RunId = "philosophical-probe-$stamp"
}
$logPath = Join-Path $verificationDir "local-philosophical-model-probe-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-philosophical-model-probe-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-philosophical-model-probe.log"
$stableManifestPath = Join-Path $verificationDir "local-philosophical-model-probe-run.json"

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

if (-not [string]::IsNullOrWhiteSpace($Models)) {
    foreach ($model in $Models.Split(",")) {
        Assert-DeepInfraModelRef -Name "Models entry" -Value $model.Trim()
    }
}

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
$mode = $(if ($DryRun) { "dry-run" } else { "real" })
$tokenPrompted = $false

try {
    if (-not $DryRun -and -not [string]::IsNullOrWhiteSpace($Models)) {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty when cloud Models are configured."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        $tokenPrompted = $true
        Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."
    } elseif ($DryRun) {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }

    $env:HELIX_RECEIPT_SIGNING_MODE = "ephemeral_preregistered"
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Mode: $mode"
    Write-Host "[helix] Models: $Models"
    Write-Host "[helix] Local aliases: $LocalAliases"
    Write-Host "[helix] Max tokens: $MaxTokens"
    Write-Host "[helix] Temperature: $Temperature"
    Write-Host "[helix] Output dir: $verificationDir"

    $modelsArg = $(if ([string]::IsNullOrWhiteSpace($Models)) { " " } else { $Models })
    $pyArgs = @(
        "tools\run_philosophical_model_probe_v1.py",
        "--output-dir", $verificationDir,
        "--models", $modelsArg,
        "--max-tokens", [string]$MaxTokens,
        "--temperature", [string]$Temperature,
        "--run-id", $RunId
    )
    if (-not [string]::IsNullOrWhiteSpace($LocalAliases)) {
        $pyArgs += @("--local-aliases", $LocalAliases)
    }
    if ($DryRun) {
        $pyArgs += "--dry-run"
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

    $candidate = Join-Path $verificationDir "local-philosophical-model-probe-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-philosophical-model-probe-run"
        generated_by = "tools/run_philosophical_model_probe_secure.ps1"
        mode = $mode
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        models = $Models
        local_aliases = $LocalAliases
        max_tokens = $MaxTokens
        temperature = $Temperature
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
            token_prompt_hidden = $tokenPrompted
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        claim_boundary = "Closed qualitative model-output probe. No sentience, hidden identity, provider intent, philosophical truth, or persistent internal memory claim."
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
