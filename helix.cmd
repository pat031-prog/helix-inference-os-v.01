@echo off
setlocal
set "HELIX_REPO=%~dp0"
set "PYTHONPATH=%HELIX_REPO%src;%HELIX_REPO%;%PYTHONPATH%"
python -m helix_proto.helix_cli %*
exit /b %ERRORLEVEL%
