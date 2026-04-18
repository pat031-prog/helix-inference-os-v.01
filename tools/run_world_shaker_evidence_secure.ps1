param(
    [switch]$Synthetic,
    [switch]$EnableThinking,
    [string]$FrontierModel = "Qwen/Qwen3.5-122B-A10B",
    [string]$WorldLlama3B = "meta-llama/Llama-3.2-3B-Instruct",
    [string]$WorldLlama8B = "meta-llama/Meta-Llama-3-8B-Instruct",
    [string]$WorldMistral = "mistralai/Mistral-7B-Instruct-v0.3",
    [string]$WorldQwen = "Qwen/Qwen2.5-7B-Instruct",
    [string]$PytestArgs = "tests/test_world_shaker.py tests/test_frontier_workloads.py -v -s",
    [string]$EvidenceName = "world-shaker-frontier"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$pytestArgList = @($PytestArgs -split '\s+' | Where-Object { $_ })
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$safeEvidenceName = ($EvidenceName -replace '[^A-Za-z0-9_.-]', '-')
$logPath = Join-Path $verificationDir ("local-{0}-{1}.log" -f $safeEvidenceName, $stamp)
$artifactPath = Join-Path $verificationDir ("local-{0}-{1}.json" -f $safeEvidenceName, $stamp)

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
$previousFrontierFast = $env:HELIX_FRONTIER_FAST_MODEL
$previousFrontierThink = $env:HELIX_FRONTIER_THINK_MODEL
$previousFrontierDisableThinking = $env:HELIX_FRONTIER_DISABLE_THINKING
$previousWorldLlama3B = $env:HELIX_WORLD_LLAMA_3B
$previousWorldLlama8B = $env:HELIX_WORLD_LLAMA_8B
$previousWorldMistral = $env:HELIX_WORLD_MISTRAL
$previousWorldQwen = $env:HELIX_WORLD_QWEN
$previousWorldDisableThinking = $env:HELIX_WORLD_DISABLE_THINKING

$startedAt = [DateTimeOffset]::Now
$pytestExitCode = 1
$mode = "real"

try {
    $env:HELIX_FRONTIER_FAST_MODEL = $FrontierModel
    $env:HELIX_FRONTIER_THINK_MODEL = $FrontierModel
    $env:HELIX_WORLD_LLAMA_3B = $WorldLlama3B
    $env:HELIX_WORLD_LLAMA_8B = $WorldLlama8B
    $env:HELIX_WORLD_MISTRAL = $WorldMistral
    $env:HELIX_WORLD_QWEN = $WorldQwen
    if ($EnableThinking) {
        $env:HELIX_FRONTIER_DISABLE_THINKING = "0"
        $env:HELIX_WORLD_DISABLE_THINKING = "0"
    } else {
        $env:HELIX_FRONTIER_DISABLE_THINKING = "1"
        $env:HELIX_WORLD_DISABLE_THINKING = "1"
    }

    if ($Synthetic) {
        $mode = "synthetic"
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
        Write-Host "[helix] Running World-Shaker + Frontier evidence in synthetic mode."
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
    Write-Host "[helix] Frontier model: $FrontierModel"
    Write-Host "[helix] World models: $WorldLlama3B | $WorldMistral | $WorldQwen | synthesis=$WorldLlama8B"
    Write-Host "[helix] Disable thinking: $(-not $EnableThinking)"

    & python -m pytest @pytestArgList 2>&1 | Tee-Object -FilePath $logPath
    $pytestExitCode = $LASTEXITCODE
} finally {
    $endedAt = [DateTimeOffset]::Now
    $logHash = $null
    $logLength = 0
    if (Test-Path -LiteralPath $logPath) {
        $logHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $logLength = (Get-Item -LiteralPath $logPath).Length
    }
    $suspiciousShortLog = ((-not [bool]$Synthetic) -and $pytestExitCode -eq 0 -and $logLength -lt 5000)
    if ($suspiciousShortLog) {
        $pytestExitCode = 2
        Write-Host "[helix] suspicious_short_log: passed run produced less than 5000 log bytes."
    }

    $artifact = [ordered]@{
        artifact = "local-$safeEvidenceName"
        generated_by = "tools/run_world_shaker_evidence_secure.ps1"
        mode = $mode
        llm_synthetic_mode = [bool]$Synthetic
        started_at = $startedAt.ToString("o")
        ended_at = $endedAt.ToString("o")
        duration_s = [Math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
        pytest_exit_code = $pytestExitCode
        pytest_passed = ($pytestExitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        suspicious_short_log = $suspiciousShortLog
        pytest_args = $PytestArgs
        test_files = @(
            "tests/test_world_shaker.py",
            "tests/test_frontier_workloads.py"
        )
        expected_test_count = 8
        log_path = $logPath
        log_sha256 = $logHash
        log_bytes = $logLength
        token_handling = [ordered]@{
            token_prompt_hidden = (-not [bool]$Synthetic)
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        models = [ordered]@{
            frontier_fast_model = $FrontierModel
            frontier_think_model = $FrontierModel
            world_llama_3b = $WorldLlama3B
            world_llama_8b = $WorldLlama8B
            world_mistral = $WorldMistral
            world_qwen = $WorldQwen
            disable_thinking = (-not [bool]$EnableThinking)
        }
        public_claim_level = $(if ($Synthetic) { "synthetic_mechanics_verified" } elseif ($pytestExitCode -eq 0) { "real_world_shaker_frontier_observed" } else { "real_run_failed_artifact_preserved" })
        claims_allowed = @(
            "The run captures stdout plus a SHA-256 receipt for the evidence log.",
            "The DeepInfra token is read through a hidden prompt and is not written to disk.",
            "Passing real mode supports a real-cloud evidence claim for the two suites.",
            "The log may include model outputs, but never API keys or request headers."
        )
        claims_not_allowed = @(
            "This wrapper does not prove below-prompt KV persistence for cloud models.",
            "This wrapper does not replace per-test structured artifacts; it preserves the executable evidence trail."
        )
    }

    $artifact | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $artifactPath -Encoding UTF8
    Write-Host "[helix] Evidence artifact: $artifactPath"

    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }
    if ($null -ne $previousFrontierFast) { $env:HELIX_FRONTIER_FAST_MODEL = $previousFrontierFast } else { Remove-Item Env:\HELIX_FRONTIER_FAST_MODEL -ErrorAction SilentlyContinue }
    if ($null -ne $previousFrontierThink) { $env:HELIX_FRONTIER_THINK_MODEL = $previousFrontierThink } else { Remove-Item Env:\HELIX_FRONTIER_THINK_MODEL -ErrorAction SilentlyContinue }
    if ($null -ne $previousFrontierDisableThinking) { $env:HELIX_FRONTIER_DISABLE_THINKING = $previousFrontierDisableThinking } else { Remove-Item Env:\HELIX_FRONTIER_DISABLE_THINKING -ErrorAction SilentlyContinue }
    if ($null -ne $previousWorldLlama3B) { $env:HELIX_WORLD_LLAMA_3B = $previousWorldLlama3B } else { Remove-Item Env:\HELIX_WORLD_LLAMA_3B -ErrorAction SilentlyContinue }
    if ($null -ne $previousWorldLlama8B) { $env:HELIX_WORLD_LLAMA_8B = $previousWorldLlama8B } else { Remove-Item Env:\HELIX_WORLD_LLAMA_8B -ErrorAction SilentlyContinue }
    if ($null -ne $previousWorldMistral) { $env:HELIX_WORLD_MISTRAL = $previousWorldMistral } else { Remove-Item Env:\HELIX_WORLD_MISTRAL -ErrorAction SilentlyContinue }
    if ($null -ne $previousWorldQwen) { $env:HELIX_WORLD_QWEN = $previousWorldQwen } else { Remove-Item Env:\HELIX_WORLD_QWEN -ErrorAction SilentlyContinue }
    if ($null -ne $previousWorldDisableThinking) { $env:HELIX_WORLD_DISABLE_THINKING = $previousWorldDisableThinking } else { Remove-Item Env:\HELIX_WORLD_DISABLE_THINKING -ErrorAction SilentlyContinue }
}

exit $pytestExitCode
