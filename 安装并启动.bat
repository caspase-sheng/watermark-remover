@echo off
rem 首次使用：装环境 + 装依赖 + 下模型 + 启动 Web 界面
rem 环境优先级：项目里的便携环境 env\ > 已有的 conda 命名环境 > 现建一个便携环境
rem 支持断点续跑：环境已存在就跳过创建，权重已存在就跳过下载
chcp 65001 >nul
title 安装并启动 - 视频固定位置水印去除
setlocal

set "ENV_NAME=watermark-remover"
cd /d "%~dp0"
set "PY=%~dp0env\python.exe"

rem ---- 先看便携环境在不在（认 setup_portable.bat 写的就绪标记）----
if exist "%~dp0env\.wmr_ready" (
    echo [1/3] 检测到便携环境 env\，跳过安装
    goto :launch_portable
)

rem ---- 再看 conda 命名环境在不在（之前装过就不用重复建）----
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

if defined CONDA (
    call "%CONDA%" run -n %ENV_NAME% python --version >nul 2>&1
    if not errorlevel 1 (
        echo [1/3] 检测到 conda 命名环境 %ENV_NAME%，跳过安装
        goto :launch_named
    )
)

rem ---- 两个都没有：走便携环境（可整包拷贝，推荐）----
echo [1/3] 没有可用环境，开始创建便携环境 env\（会一并装依赖、下权重）...
echo.
call "%~dp0setup_portable.bat"

if not exist "%~dp0env\.wmr_ready" (
    echo.
    echo [错误] 便携环境没建成功，请根据上面的报错排查后重新运行本脚本。
    pause
    exit /b 1
)
goto :launch_portable

:launch_portable
echo [2/3] 环境就绪（便携环境）
echo [3/3] 启动 Web 界面，浏览器会自动打开 http://127.0.0.1:7860
echo.
"%PY%" gradio_app.py
goto :done

:launch_named
echo [2/3] 环境就绪（conda 命名环境 %ENV_NAME%）
echo [3/3] 启动 Web 界面，浏览器会自动打开 http://127.0.0.1:7860
echo.
call "%CONDA%" run --no-capture-output -n %ENV_NAME% python gradio_app.py

:done
echo.
echo 服务已退出。
pause
