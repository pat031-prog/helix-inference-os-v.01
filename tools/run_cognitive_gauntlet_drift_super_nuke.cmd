@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cognitive_gauntlet_secure.ps1" -Scenario drift-nuke -Rounds 12 -AttackRound 4 -TokensPerTurn 3000 -AuditTokens 4800
exit /b %ERRORLEVEL%
