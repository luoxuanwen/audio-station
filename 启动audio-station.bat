@echo off
title audio-station 统一网关 (18000) - 关闭本窗口即停止全部服务
cd /d "%~dp0"
set "PY=D:/audio-station/runtime/python.exe"
set "PYTHONIOENCODING=utf-8"
echo ============================================================
echo   audio-station 统一网关
echo   TTS  : POST /v1/audio/speech   model=breeze-tts-2
echo   ASR  : POST /v1/audio/transcriptions
echo   翻唱 : POST /v1/audio/speech   model=seed-svc + Gradio /convert
echo   管理台: http://127.0.0.1:18000  (Gradio UI)
echo   Breeze 子进程随本窗口启动/退出
echo ============================================================
echo.
"%PY%" "%~dp0gateway.py"
echo.
echo [网关已退出] code=%errorlevel%
pause
