@echo off
rem 便携环境安装：在项目目录下建一个 env\ 路径式 conda 环境
rem 好处是整个项目（含 env\ 和 models\）拷到另一台 Windows 机器就能跑，
rem 不用去动那台机器的系统级 conda 环境。
rem 支持断点续跑：env\python.exe 在就跳过创建，torch 能 import 就跳过安装，
rem 权重已存在 download_models.py 会自己跳过。
chcp 65001 >nul
title 便携环境安装 - 视频固定位置水印去除
setlocal

cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PENV=%ROOT%\env"
set "PY=%PENV%\python.exe"

rem ---- 找 conda：先 PATH，再几个常见安装位置 ----
rem 注意：便携模式仍然需要机器上装了 conda / Miniconda，
rem 我们只是借它建一个放在项目里的环境，环境本身不依赖系统 conda。
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
    echo [错误] 找不到 conda。便携模式的 env\ 也需要用 conda 来创建。
    echo        请先装一个 Miniconda：https://docs.conda.io/en/latest/miniconda.html
    echo        装完重开一个窗口再运行本脚本。
    pause
    exit /b 1
)

echo 项目目录：%ROOT%
echo 便携环境：%PENV%
echo 使用的 conda：%CONDA%
echo.

rem ---- 1/4 建路径式环境 ----
if exist "%PY%" (
    echo [1/4] 便携环境已存在，跳过创建
) else (
    echo [1/4] 创建路径式环境（conda create -p env python=3.10）...
    echo       路径式环境直接建在项目里，不占用 conda 的命名环境列表。
    call "%CONDA%" create -p "%PENV%" python=3.10 -y
    if errorlevel 1 (
        echo       默认源没通，换清华镜像重试...
        call "%CONDA%" create -p "%PENV%" python=3.10 -y ^
            -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ --override-channels
        if errorlevel 1 (
            echo [错误] 环境创建失败。检查网络后重新运行本脚本（已完成的步骤会跳过）。
            pause
            exit /b 1
        )
    )
)

if not exist "%PY%" (
    echo [错误] 没找到 %PY%，环境创建似乎没成功。
    pause
    exit /b 1
)

rem ---- 2/4 依赖 ----
rem 注意这里不 conda activate，直接用 env\python.exe 跑，省得依赖 shell 状态
echo [2/4] 安装依赖...
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo       安装 PyTorch 2.1.2 + CUDA 12.1（约 2.5GB，耐心等）...
    "%PY%" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [错误] PyTorch 安装失败，检查网络后重新运行本脚本（已完成的步骤会跳过）。
        pause
        exit /b 1
    )
) else (
    echo       PyTorch 已安装，跳过
)

"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] requirements.txt 安装失败，检查网络后重新运行本脚本。
    pause
    exit /b 1
)

rem ---- 3/4 模型权重（已存在的会自动跳过）----
echo [3/4] 下载模型权重（已下载的会自动跳过）...
"%PY%" download_models.py
if errorlevel 1 (
    echo [错误] 权重下载失败。网络不好的话可以稍后再跑一次本脚本，
    echo        或按 README「模型权重」一节手动下载放进 models\。
    pause
    exit /b 1
)

rem ---- 4/4 自检 ----
echo [4/4] 自检...
"%PY%" -c "import torch, gradio, cv2; print('torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available()); print('gradio', gradio.__version__); print('opencv', cv2.__version__)"
if errorlevel 1 (
    echo [错误] 自检没通过，上面的报错信息可以拿去排查。
    pause
    exit /b 1
)

rem 写一个就绪标记。启动脚本认这个文件，不认 python.exe：
rem 万一装到一半失败了，env\python.exe 可能已经存在但不完整，
rem 有标记在就不会误用半成品环境。
> "%PENV%\.wmr_ready" echo ok

echo.
echo 便携环境装好了。以后直接双击「启动.bat」就会优先用这个 env\。
echo 想拷到另一台机器：整个项目目录复制过去，目标机器装好 conda / Miniconda 后
echo 重新跑一次本脚本即可（或直接 ./env/python.exe gradio_app.py 试跑）。
echo.
pause
