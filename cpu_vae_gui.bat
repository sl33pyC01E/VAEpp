@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m experiments.cpu_vae_gui
