@echo off
rem 启动 Web 界面（环境已经装好时用这一个）
rem 双击即可，窗口留着看日志，关了窗口服务就停
rem 优先用项目里的便携环境 env\python.exe；没有才回退到系统 conda 的命名环境
chcp 65001 >nul
title 视频固定位置水印去除
setlocal

set "ENV_NAME=watermark-remover"
cd /d "%~dp0"
set "PY=%~dp0env\python.exe"

rem ---- 1) 便携环境优先 ----
rem 认 env\.wmr_ready 这个标记（setup_portable.bat 装成功后写的），
rem 光看 python.exe 存在不够：装到一半失败时它也在。
if exist "%~dp0env\.wmr_ready" (
    echo 使用便携环境：%PY%
    echo.
    goto :run
)

rem ---- 2) 回退到命名环境，需要先找 conda ----
set "CONDA="
where conda >nul 2>&1
if %errorlevel%==0 (
    set "CONDA=conda"
) else (
    for %%P in ("D:\conda\Scripts\conda.exe" ^
                "C:\ProgramData\Miniconda3\Scripts\conda.exe" ^
                "C:\ProgramData\Anaconda3\Scripts\conda.exe" ^
                "%USERPROFILE%\miniconda3\Scripts\conda.exe" ^
                "%USERPROFILE%\anaconda3\Scripts\conda.exe") do (
        if not defined CONDA if exist %%~P set "CONDA=%%~fP"
    )
)

if not defined CONDA (
    echo [错误] 既没有便携环境 env\python.exe，也找不到 conda。
    echo        第一次使用请先双击「安装并启动.bat」。
    pause
    exit /b 1
)

call "%CONDA%" run -n %ENV_NAME% python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] conda 环境 %ENV_NAME% 还不存在。
    echo        第一次使用请先双击「安装并启动.bat」，或者双击「setup_portable.bat」建便携环境。
    pause
    exit /b 1
)

echo 使用 conda 命名环境：%ENV_NAME%
echo.
call "%CONDA%" run --no-capture-output -n %ENV_NAME% python gradio_app.py
goto :done

:run
echo 正在启动 Web 界面，浏览器会自动打开 http://127.0.0.1:7860
echo 想停止服务：直接关掉这个窗口，或按 Ctrl+C。
echo.
"%PY%" gradio_app.py

:done
echo.
echo 服务已退出。
pause
