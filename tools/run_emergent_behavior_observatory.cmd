@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_emergent_behavior_observatory_secure.ps1" -Rounds 12 -TokensPerTurn 700 -AnalysisTokens 2200
exit /b %ERRORLEVEL%
