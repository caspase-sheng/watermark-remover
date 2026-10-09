@echo off
REM ============================================================
REM  watermark-remover - one click setup for Windows
REM
REM  This script creates the conda environment and checks the
REM  dependencies. Messages are kept in English on purpose:
REM  cmd.exe reads .bat files with the OEM code page, and UTF-8
REM  Chinese text in a batch file often shows up as garbage.
REM ============================================================

setlocal
set "ENV_NAME=watermark-remover"
set "SCRIPT_DIR=%~dp0"

echo.
echo ============================================================
echo  watermark-remover  environment setup
echo ============================================================
echo.

REM --- locate conda ------------------------------------------
REM Prefer whatever is on PATH; otherwise fall back to common install
REM locations, including D:\conda which is where this machine keeps it.
set "CONDA_EXE="
for %%C in (conda.exe) do if not defined CONDA_EXE (
    for /f "delims=" %%P in ('where %%C 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%P"
)
if not defined CONDA_EXE (
    for %%D in ("D:\conda" "C:\ProgramData\Miniconda3" "C:\ProgramData\Anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3") do (
        if not defined CONDA_EXE if exist "%%~D\Scripts\conda.exe" set "CONDA_EXE=%%~D\Scripts\conda.exe"
    )
)
if not defined CONDA_EXE (
    echo [ERROR] conda was not found.
    echo         Install Miniconda / Anaconda first, or edit CONDA_EXE in this script.
    echo.
    pause
    exit /b 1
)
echo Using conda: %CONDA_EXE%
echo.

REM --- network: conda's default read timeout is about 9s, which is often
REM     too short for the win-64 repodata on a slow connection -----------
set "CONDA_REMOTE_READ_TIMEOUT_SECS=120"
set "CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30"
set "CONDA_ALWAYS_YES=true"

REM --- create folders ----------------------------------------
if not exist "%SCRIPT_DIR%input"  mkdir "%SCRIPT_DIR%input"
if not exist "%SCRIPT_DIR%output" mkdir "%SCRIPT_DIR%output"

REM --- step 1: create the environment ------------------------
REM Route A: the standard way, everything from environment.yml
REM Route B: conda only provides Python 3.10, everything else comes from pip
REM          (needed when repo.anaconda.com is unreachable / keeps timing out)
echo [1/4] Creating conda environment "%ENV_NAME%" ...
echo       python 3.10 + pytorch 2.1.2 + cuda 12.1
echo       this downloads a few GB and may take a while.
echo.

set "ENV_PY=%CONDA_PREFIX%\envs\%ENV_NAME%\python.exe"
if not defined CONDA_PREFIX set "ENV_PY="
if not defined ENV_PY (
    for %%D in ("D:\conda" "C:\ProgramData\Miniconda3" "C:\ProgramData\Anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3") do (
        if not defined ENV_PY if exist "%%~D\envs\%ENV_NAME%\python.exe" set "ENV_PY=%%~D\envs\%ENV_NAME%\python.exe"
    )
)

call "%CONDA_EXE%" env create -f "%SCRIPT_DIR%environment.yml"
if errorlevel 1 (
    echo.
    echo [WARN] "conda env create" did not work. Falling back to the pip route.
    echo        conda will only provide Python 3.10; PyTorch comes from the
    echo        official CUDA 12.1 wheel index and the rest from PyPI.
    echo.

    REM clean up whatever half-created env is lying around
    call "%CONDA_EXE%" env remove -n %ENV_NAME% -y >nul 2>nul

    echo   ^- creating the environment ...
    call "%CONDA_EXE%" create -n %ENV_NAME% python=3.10 -y
    if errorlevel 1 (
        echo   ^- default channel failed, retrying with the Tsinghua mirror ...
        call "%CONDA_EXE%" create -n %ENV_NAME% python=3.10 -y ^
            -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ --override-channels
        if errorlevel 1 (
            echo [ERROR] Could not create the environment. See the messages above.
            pause
            exit /b 1
        )
    )

    REM work out where the env's python.exe ended up
    for /f "delims=" %%P in ('call "%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys;print(sys.executable)" 2^>nul') do set "ENV_PY=%%P"

    echo   ^- installing PyTorch 2.1.2 + CUDA 12.1 from download.pytorch.org ...
    call "%ENV_PY%" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed.
        pause
        exit /b 1
    )

    echo   ^- installing the rest from PyPI ...
    call "%ENV_PY%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Installing the remaining dependencies failed.
        pause
        exit /b 1
    )
)

REM --- step 2: check pytorch / cuda --------------------------
echo.
echo [2/4] Checking PyTorch and CUDA ...
call "%CONDA_EXE%" run -n %ENV_NAME% python -c "import torch; print('    torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print('    device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"

REM --- step 3: check base deps -------------------------------
echo.
echo [3/4] Checking base dependencies ...
call "%CONDA_EXE%" run -n %ENV_NAME% python -c "import cv2, numpy, yaml, tqdm, PIL, imageio_ffmpeg; print('    opencv', cv2.__version__, '| numpy', numpy.__version__, '| imageio-ffmpeg OK')"

REM --- step 4: optional LaMa backend -------------------------
echo.
echo [4/4] Checking the LaMa inpainting backend ...
call "%CONDA_EXE%" run -n %ENV_NAME% python -c "from simple_lama_inpainting import SimpleLama; print('    simple-lama-inpainting OK')"
if errorlevel 1 (
    echo.
    echo [WARN] simple-lama-inpainting is NOT available.
    echo        The project still works: open config.yaml and set
    echo            inpaint:
    echo              backend: opencv
    echo        to use the OpenCV inpainting fallback instead.
)

echo.
echo ============================================================
echo  Setup finished.
echo ============================================================
echo.
echo  Next steps:
echo    1) conda activate %ENV_NAME%
echo    2) copy your .mp4 files into the "input" folder
echo    3) python select_region.py     (drag to select the watermark)
echo    4) python main.py              (batch process everything)
echo.
pause
