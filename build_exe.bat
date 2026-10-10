@echo off
rem 用 PyInstaller 把 gradio_app.py 打包成 exe（实验性）
rem
rem 说明几个取舍：
rem   1) 用 --onedir（一个文件夹）而不是 --onefile。torch 用 onefile 每次启动都要
rem      把几个 GB 解压到临时目录，慢得没法用。
rem   2) 模型权重（models\）和 ffmpeg 不打进 exe：
rem      - 权重用 download_models.py 下，或者把已有的 models\ 拷到 exe 旁边；
rem      - ffmpeg 走 imageio-ffmpeg 包自带的，会被收进包里。
rem   3) 打包体积大概 3GB 上下（torch + gradio），能不能在别的机器上跑起来
rem      受 CUDA / 驱动版本影响，不保证。真要挪机器，优先用「便携模式」
rem      （setup_portable.bat），那个更稳。
chcp 65001 >nul
title 打包 exe - 视频固定位置水印去除
setlocal

cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "ENV_NAME=watermark-remover"
set "PYRUN="
set "USE_CONDA="

rem ---- 找 conda：先 PATH，再几个常见安装位置 ----
rem 这段要放在顶层，别塞进 if/else 块里：
rem 块里的 %errorlevel% 是在解析整块时展开的，会拿到上一次命令的旧值。
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

rem ---- 挑一个能用的 Python：便携环境优先 ----
if exist "%ROOT%\env\.wmr_ready" (
    set "PYRUN=%ROOT%\env\python.exe"
    echo 使用便携环境：%ROOT%\env\python.exe
    goto :have_py
)

if not defined CONDA (
    echo [错误] 找不到便携环境 env\，也找不到 conda。先跑「安装并启动.bat」。
    pause
    exit /b 1
)

call "%CONDA%" run -n %ENV_NAME% python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] conda 环境 %ENV_NAME% 不存在。先跑「安装并启动.bat」。
    pause
    exit /b 1
)
set "PYRUN=%CONDA%"
set "USE_CONDA=1"
echo 使用 conda 命名环境：%ENV_NAME%

:have_py
echo.
echo [1/3] 检查 PyInstaller...
call :py -c "import PyInstaller"
if errorlevel 1 (
    echo       没装，正在安装 pyinstaller...
    call :py -m pip install pyinstaller
    if errorlevel 1 (
        echo [错误] PyInstaller 安装失败，检查网络后重试。
        pause
        exit /b 1
    )
)

echo [2/3] 开始打包（几分钟到十几分钟，看机器）...
rem --collect-all 把 gradio 的前端资源、lama 包的数据、imageio-ffmpeg 里的
rem ffmpeg 可执行文件都收进来；--onedir 输出到 dist\watermark-remover\。
call :py -m PyInstaller ^
    --noconfirm --clean --onedir ^
    --name watermark-remover ^
    --collect-all gradio ^
    --collect-all gradio_client ^
    --collect-all simple_lama_inpainting ^
    --collect-all imageio_ffmpeg ^
    --collect-submodules scipy ^
    --add-data "config.yaml;." ^
    gradio_app.py
if errorlevel 1 (
    echo.
    echo [错误] 打包失败。PyInstaller 配 gradio + torch 本来就容易缺东西，
    echo        如果折腾不动，直接用「便携模式」：整个项目目录拷走 + setup_portable.bat。
    pause
    exit /b 1
)

rem ---- 3/3 把权重挪到 exe 旁边（不打进包里）----
echo [3/3] 整理模型权重...
if exist "%ROOT%\models" (
    if not exist "%ROOT%\dist\watermark-remover\models" mkdir "%ROOT%\dist\watermark-remover\models"
    xcopy /e /i /y "%ROOT%\models" "%ROOT%\dist\watermark-remover\models" >nul
    echo       已把 models\ 拷到 dist\watermark-remover\models\
) else (
    echo       没找到 models\，跳过。跑 exe 前记得把权重放到 exe 同级的 models\ 里。
)

echo.
echo 打包完成，产物在：dist\watermark-remover\
echo 双击 dist\watermark-remover\watermark-remover.exe 即可（控制台窗口会一起出来，方便看日志）。
echo 注意：models\ 必须和 exe 放在一起，否则要先联网下权重。
echo.
pause
exit /b 0

rem ---- 用选定的环境跑一条命令，参数原样透传 ----
:py
if defined USE_CONDA (
    call "%PYRUN%" run --no-capture-output -n %ENV_NAME% python %*
) else (
    "%PYRUN%" %*
)
exit /b %errorlevel%
