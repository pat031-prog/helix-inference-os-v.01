@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_helix_freeform_drift_secure.ps1" -Scenario hosted-in-helix -Rounds 12 -TokensPerTurn 2200 -AuditTokens 3600
exit /b %ERRORLEVEL%
