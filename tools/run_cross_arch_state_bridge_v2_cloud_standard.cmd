@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cross_arch_state_bridge_v2_secure.ps1" -CloudOnly -Scenario "standard" -AnalystModel "Qwen/Qwen3.6-35B-A3B" -ContinuistModel "zai-org/GLM-5.1" -TokensPerRound 640
exit /b %ERRORLEVEL%
