@echo off
echo ======================================
echo モデルファイルのサフィックス削除
echo ======================================
echo.

cd /d "%~dp0.."
python -c "import os; from pathlib import Path; models_dir = Path('models'); suffix = '_2015-2024'; count = 0; files = list(models_dir.glob(f'*{suffix}.sav')); [print(f'{f.name} -> {f.name.replace(suffix, \"\")}') or f.rename(models_dir / f.name.replace(suffix, '')) or (count := count + 1) for f in files]; print(f'\n{count}個のファイルをリネームしました')"

echo.
echo 完了しました！
pause
