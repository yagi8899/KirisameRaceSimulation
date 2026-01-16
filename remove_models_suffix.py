#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファイル名サフィックス一括削除スクリプト

Usage:
    # モデルファイルの_2015-2024を削除
    python remove_models_suffix.py
    
    # カスタムサフィックス指定
    python remove_models_suffix.py --suffix "_2020-2024"
    
    # 別のディレクトリ指定
    python remove_models_suffix.py --dir results --suffix "_trainunknown_test2023"
    
    # 拡張子指定（全ファイル対象は --ext "*"）
    python remove_models_suffix.py --ext ".tsv" --suffix "_all"
    
    # 全ファイル対象
    python remove_models_suffix.py --dir results --ext "*" --suffix "_skipped"
"""

import argparse
from pathlib import Path

def remove_suffix(target_dir='models', suffix='_2015-2024', ext='.sav'):
    """
    ファイル名のサフィックスを削除
    
    Args:
        target_dir: 対象ディレクトリパス
        suffix: 削除するサフィックス
        ext: 対象ファイルの拡張子（*で全ファイル）
    """
    dir_path = Path(target_dir)
    
    if not dir_path.exists():
        print(f"[ERROR] ディレクトリが見つかりません: {dir_path}")
        return
    
    # ファイル検索パターン作成
    if ext == '*':
        pattern = f'*{suffix}*'
    else:
        # 拡張子の前にサフィックスがある場合
        ext_clean = ext if ext.startswith('.') else f'.{ext}'
        pattern = f'*{suffix}{ext_clean}'
    
    files = list(dir_path.glob(pattern))
    
    if not files:
        print(f"[INFO] 対象ファイルが見つかりません")
        print(f"  ディレクトリ: {dir_path.absolute()}")
        print(f"  パターン: {pattern}")
        return
    
    print('=' * 60)
    print('ファイル名サフィックス一括削除')
    print('=' * 60)
    print(f'対象ディレクトリ: {dir_path.absolute()}')
    print(f'削除サフィックス: {suffix}')
    print(f'対象拡張子: {ext}')
    print(f'対象ファイル数: {len(files)}個')
    print()
    
    count = 0
    for file in files:
        new_name = file.name.replace(suffix, '')
        print(f'  {file.name} -> {new_name}')
        file.rename(dir_path / new_name)
        count += 1
    
    print()
    print(f'[DONE] {count}個のファイルをリネームしました')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ファイル名サフィックス一括削除',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python remove_models_suffix.py
  python remove_models_suffix.py --suffix "_2020-2024"
  python remove_models_suffix.py --dir results --suffix "_all" --ext ".tsv"
  python remove_models_suffix.py --dir results --ext "*" --suffix "_skipped"
        """
    )
    parser.add_argument('--suffix', type=str, default='_2015-2024',
                        help='削除するサフィックス（デフォルト: _2015-2024）')
    parser.add_argument('--dir', type=str, default='models',
                        help='対象ディレクトリ（デフォルト: models）')
    parser.add_argument('--ext', type=str, default='.sav',
                        help='対象ファイルの拡張子（デフォルト: .sav、全ファイルは *）')
    
    args = parser.parse_args()
    remove_suffix(target_dir=args.dir, suffix=args.suffix, ext=args.ext)
