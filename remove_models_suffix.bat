@echo off
cd /d "%~dp0"
python remove_models_suffix.py %*
pause
