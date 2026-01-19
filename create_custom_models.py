#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1用: custom_models一括作成スクリプト

穴馬予測Phase 1用の4つのモデルを一括作成します。
- 阪神芝長距離（ベースライン）
- 阪神芝短距離（短距離特性検証）
- 函館芝長距離（最も穴が出やすい）
- 東京芝長距離（最も堅い）
"""

from batch_model_creator import create_custom_models

if __name__ == '__main__':
    print("="*60)
    print("Phase 1用 custom_models 一括作成")
    print("="*60)
    print("作成モデル:")
    print("  1. hanshin_turf_3ageup_long.sav  (阪神芝長距離)")
    print("  2. hanshin_turf_3ageup_short.sav (阪神芝短距離)")
    print("  3. hakodate_turf_3ageup_long.sav (函館芝長距離)")
    print("  4. tokyo_turf_3ageup_long.sav    (東京芝長距離)")
    print("="*60)
    
    # custom_modelsを一括作成
    # 学習期間: 2020-2022年（速度優先で3年間）
    create_custom_models(
        output_dir='models',
        year_start=2020,
        year_end=2022
    )
    
    print("\n" + "="*60)
    print("モデル作成完了!")
    print("次のコマンドで穴馬検出テストを実行できます:")
    print("  python upset_detector.py models/hanshin_turf_3ageup_long.sav --test-year 2023")
    print("  python upset_detector.py models/hakodate_turf_3ageup_long.sav --test-year 2023 --track-code 02")
    print("  python upset_detector.py models/tokyo_turf_3ageup_long.sav --test-year 2023 --track-code 05")
    print("="*60)
