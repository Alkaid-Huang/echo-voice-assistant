@echo off
rem One-click launcher: prefer the project venv, fall back to system python
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto run
set "PY=%~dp0..\Live2D-anget\.venv\Scripts\python.exe"
if exist "%PY%" goto run
set "PY=python"
:run
"%PY%" "%~dp0main.py" %*
