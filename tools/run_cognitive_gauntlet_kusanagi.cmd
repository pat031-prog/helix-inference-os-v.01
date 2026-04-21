@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cognitive_gauntlet_secure.ps1" -Scenario kusanagi -Rounds 8 -AttackRound 3 -TokensPerTurn 1600 -AuditTokens 2800
exit /b %ERRORLEVEL%
