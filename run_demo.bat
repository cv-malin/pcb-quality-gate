@echo off
chcp 65001 >nul
title PCB Quality Gate - 缺陷智能质检台
cd /d "%~dp0"

set PY=D:\xtrain-venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo.
echo   PCB Quality Gate  v1.0.0
echo   Starting local demo server...
echo.
start "" http://127.0.0.1:8765
"%PY%" server.py
pause
