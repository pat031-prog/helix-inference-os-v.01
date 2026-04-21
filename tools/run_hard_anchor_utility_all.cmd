@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_hard_anchor_utility_suite_secure.ps1" -Case "all" -Depth 5000 -BytesPerNode 8192 -Repeats 7 -MaxAnchorMs 25.0 -MinSpeedup 9.0 -MaxCompressionRatio 0.05
exit /b %ERRORLEVEL%
