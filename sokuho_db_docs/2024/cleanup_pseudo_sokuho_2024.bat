@echo off
chcp 65001 > nul
echo ========================================
echo 疑似速報データ削除 - 2024年
echo ========================================
echo.

set PGHOST=localhost
set PGUSER=postgres
set PGDATABASE=keiba_2024
set PGPASSWORD=ahtaht88

echo 接続先: %PGHOST% / %PGDATABASE%
echo 対象年: 2024年
echo.
echo ⚠️  この操作は元に戻せません！
echo.

set /p CONFIRM="本当に削除しますか？ (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo キャンセルしました
    pause
    exit /b 0
)

echo.
echo 疑似速報データを削除中...
echo.

psql -h %PGHOST% -U %PGUSER% -d %PGDATABASE% ^
  -v target_year_start=2024 ^
  -v target_year_end=9999 ^
  -f "%~dp0..\cleanup_pseudo_sokuho.sql"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ エラーが発生しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 2024年の疑似速報データ削除完了！
echo ========================================
echo.
pause
