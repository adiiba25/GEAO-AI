@echo off
REM GeoAI Studio - install AI extras (PyTorch, YOLO26, LangSAM) into .venv.
REM Uses the NVIDIA GPU build when an NVIDIA GPU is found, otherwise the CPU build.
REM Download is ~3 GB for GPU. Run run_app.bat once first to create .venv.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run run_app.bat first to create the environment. & pause & exit /b 1)
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
where nvidia-smi >nul 2>nul && set TORCH_INDEX=https://download.pytorch.org/whl/cu130
echo Installing PyTorch from %TORCH_INDEX% ...
.venv\Scripts\python.exe -m pip install --only-binary=:all: torch torchvision --index-url %TORCH_INDEX% || goto :fail
echo Installing YOLO and LangSAM ...
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-ai.txt || goto :fail
.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU mode)')"
echo Done. Restart run_app.bat to use the AI tools.
pause
exit /b 0
:fail
echo Install failed - see messages above.
pause
exit /b 1
