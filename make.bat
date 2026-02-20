@echo off
REM Windows batch file equivalent to Makefile
REM Usage: make.bat <target>
REM Available targets: install, train, format, lint

if "%1"=="" (
    echo Available targets:
    echo   install - Install dependencies and set up project
    echo   train   - Train the model
    echo   format  - Format code with ruff
    echo   lint    - Lint code with ruff
    echo.
    echo Usage: make.bat ^<target^>
    goto :eof
)

if "%1"=="install" goto install
if "%1"=="train" goto train
if "%1"=="format" goto format
if "%1"=="lint" goto lint

echo Error: Unknown target "%1"
echo Run "make.bat" without arguments to see available targets
goto :eof

:install
uv sync
echo.
echo ------------------ PROJECT READY ----------------------
echo Activate env: .venv\Scripts\activate
echo Train model: make.bat train
echo.
goto :eof

:train
python -m src.train
goto :eof

:format
ruff src
goto :eof

:lint
ruff check src
goto :eof