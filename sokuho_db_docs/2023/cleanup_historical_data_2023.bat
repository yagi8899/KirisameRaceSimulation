@echo off
chcp 65001 > nul
echo ========================================
echo 過去成績データ削除 - 2023年
echo 対象: jvd_ra, jvd_se
echo ========================================
echo.

set PGHOST=localhost
set PGUSER=postgres
set PGDATABASE=keiba_2023
set PGPASSWORD=ahtaht88

echo 接続先: %PGHOST% / %PGDATABASE%
echo 対象年: 2023年
echo.
echo ⚠️  警告: この操作は元に戻せません！
echo ⚠️  jvd_ra, jvd_se の2023年データが完全に削除されます
echo ⚠️  本番環境では絶対に実行しないでください！
echo.

set /p CONFIRM="本当に削除しますか？ (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo キャンセルしました
    pause
    exit /b 0
)

echo.
set /p CONFIRM2="再確認: 過去成績データを削除します。よろしいですか？ (yes/no): "
if /i not "%CONFIRM2%"=="yes" (
    echo キャンセルしました
    pause
    exit /b 0
)

echo.
echo 過去成績データを削除中...
echo.

psql -h %PGHOST% -U %PGUSER% -d %PGDATABASE% ^
  -v target_year_start=2023 ^
  -v target_year_end=9999 ^
  -f "%~dp0..\cleanup_historical_data.sql"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ エラーが発生しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 2023年の過去成績データ削除完了！
echo ========================================
echo.
echo 疑似速報データ（sokuho_ra, sokuho_se）は残っています
echo.
pause
