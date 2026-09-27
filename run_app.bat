@echo off
REM GeoAI Studio - one-click launcher for Windows.
REM First run creates a virtual environment in .venv and installs the core packages.
cd /d "%~dp0"
if not exist .venv (
    echo Creating Python environment...
    python -m venv .venv || (echo Python 3.10+ not found. Install from python.org & pause & exit /b 1)
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)
streamlit run app.py
pause
