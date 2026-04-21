@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cognitive_drift_rollback_secure.ps1" -Rounds 14 -EventRound 5 -TokensPerTurn 2400 -AuditTokens 3800
exit /b %ERRORLEVEL%
