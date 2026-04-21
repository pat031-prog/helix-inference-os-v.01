@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_cognitive_gauntlet_secure.ps1" -Scenario kusanagi-nuke -Rounds 12 -AttackRound 3 -TokensPerTurn 2200 -AuditTokens 4200
exit /b %ERRORLEVEL%
