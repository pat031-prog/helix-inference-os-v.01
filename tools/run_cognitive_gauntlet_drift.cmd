@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cognitive_gauntlet_secure.ps1" -Scenario drift -Rounds 8 -AttackRound 4 -TokensPerTurn 2200 -AuditTokens 3200
exit /b %ERRORLEVEL%
