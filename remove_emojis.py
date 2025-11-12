#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows cp932エンコード対応: 全pyファイルからemoji除去
"""

import os
import re
from pathlib import Path

# 置換マッピング
EMOJI_REPLACEMENTS = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[!]',
    '⚠': '[!]',
    '📋': '[LIST]',
    '📁': '[FILE]',
    '🔧': '[TOOL]',
    '🎯': '[TARGET]',
    '📏': '[DIST]',
    '📅': '[DATE]',
    '🔬': '[*]',
    '📊': '[+]',
    '🏁': '[DONE]',
    '🔍': '[TEST]',
    '📚': '[RUN]',
    '🧪': '[RUN]',
    '📈': '[STATS]',
    '📉': '[-]',
    '💡': '[TIP]',
    '🏇': '[RACE]',
    '🚀': '[START]',
    '💪': '[POWER]',
    '📝': '[NOTE]',
    '🏟️': '[TRACK]',
    '🏟': '[TRACK]',
    '🌱': '[TURF]',
    '🆕': '[NEW]',
    '🔥': '',
    '📌': '[PIN]',
    '≥': '>=',
    '≤': '<=',
}

def remove_emojis_from_file(filepath):
    """ファイルからemojiを除去"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 全てのemojiを置換
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)
        
        # 変更があった場合のみ書き込み
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return False

def main():
    """全pyファイルを処理"""
    current_dir = Path(__file__).parent
    py_files = list(current_dir.glob('*.py'))
    
    modified_count = 0
    
    print(f"[START] {len(py_files)}個のPythonファイルをチェック中...")
    
    for py_file in py_files:
        if py_file.name == 'remove_emojis.py':
            continue  # このスクリプト自体はスキップ
        
        if remove_emojis_from_file(py_file):
            print(f"[FIXED] {py_file.name}")
            modified_count += 1
    
    print(f"\n[DONE] {modified_count}個のファイルを修正しました!")

if __name__ == '__main__':
    main()
