@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cognitive_gauntlet_secure.ps1" -Scenario gauntlet -Rounds 8 -AttackRound 4 -TokensPerTurn 1400 -AuditTokens 2600
exit /b %ERRORLEVEL%
