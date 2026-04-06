@echo off
echo SynthyperVAE Setup
echo.

if exist venv (
    echo venv already exists, skipping creation
) else (
    echo Creating virtual environment...
    python -m venv venv
)

echo.
echo Installing PyTorch with CUDA...
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo Installing requirements...
venv\Scripts\pip install -r requirements.txt

echo.
echo Setup complete. Run gui.bat to launch.
pause
