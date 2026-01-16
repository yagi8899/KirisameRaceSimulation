@echo off
echo ======================================
echo remove_2016_2025: remove_suffix.ps1 を実行します
echo 実行中...
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0remove_suffix.ps1" -Dir "D:\src\Python\KirisameRaceSimulation2\models" -Suffix "_2016-2025" -Ext ".sav"
set RC=%ERRORLEVEL%
echo.
echo 終了コード: %RC%
if %RC% neq 0 (
	echo エラーが発生しました。PowerShell とスクリプトの出力を確認してください。
)
echo.
pause