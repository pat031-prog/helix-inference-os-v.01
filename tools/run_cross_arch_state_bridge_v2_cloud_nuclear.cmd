@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cross_arch_state_bridge_v2_secure.ps1" -CloudOnly -Scenario "nuclear" -AnalystModel "zai-org/GLM-5.1" -ContinuistModel "Qwen/Qwen3.6-35B-A3B" -TokensPerRound 768
exit /b %ERRORLEVEL%
