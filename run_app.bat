@echo off
REM GeoAI Studio - one-click launcher for Windows.
REM First run creates a virtual environment in .venv and installs the core packages.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creating Python environment...
    python -m venv .venv || (echo Python 3.10+ not found. Install from python.org & pause & exit /b 1)
    .venv\Scripts\python.exe -m pip install --upgrade pip
    REM --only-binary avoids slow/failing source builds on new Python versions
    .venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt || (echo Install failed - see messages above & pause & exit /b 1)
)
echo Starting GeoAI Studio at http://localhost:8501 ...
REM headless skips the first-run email prompt, so open the browser ourselves
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8501"
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --browser.gatherUsageStats false --server.port 8501
pause
