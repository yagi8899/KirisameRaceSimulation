@echo off
chcp 65001 > nul
echo ========================================
echo 疑似速報データ検証 - 2024年
echo ========================================
echo.

set PGHOST=localhost
set PGUSER=postgres
set PGDATABASE=keiba_2024
set PGPASSWORD=ahtaht88

echo 接続先: %PGHOST% / %PGDATABASE%
echo 対象年: 2024年
echo.

echo 疑似速報データを検証中...
echo.

psql -h %PGHOST% -U %PGUSER% -d %PGDATABASE% ^
  -v target_year_start=2024 ^
  -v target_year_end=2024 ^
  -f "%~dp0..\validate_pseudo_sokuho.sql"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ エラーが発生しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 2024年の検証完了！
echo ========================================
echo.
pause
