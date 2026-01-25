@echo off
chcp 65001 > nul
echo ========================================
echo 疑似速報データ生成 - 2024年
echo ========================================
echo.

set PGHOST=localhost
set PGUSER=postgres
set PGDATABASE=keiba_2024
set PGPASSWORD=ahtaht88

echo 接続先: %PGHOST% / %PGDATABASE%
echo 対象年: 2024年
echo.

set /p CONFIRM="実行しますか？ (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo キャンセルしました
    pause
    exit /b 0
)

echo.
echo 疑似速報データを生成中...
echo.

psql -h %PGHOST% -U %PGUSER% -d %PGDATABASE% ^
  -v target_year_start=2024 ^
  -v target_year_end=2024 ^
  -f "%~dp0create_pseudo_sokuho_data.sql"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ エラーが発生しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 2024年の疑似速報データ生成完了！
echo ========================================
echo.
echo 次のステップ: validate_pseudo_sokuho_2024.bat で検証してください
echo.
pause
