#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1用: 阪神芝長距離モデル作成スクリプト

穴馬予測Phase 1のベースラインモデルを作成します。
"""

from model_creator import create_universal_model

if __name__ == '__main__':
    print("="*60)
    print("Phase 1ベースラインモデル作成")
    print("="*60)
    
    # 阪神芝中長距離3歳以上モデル
    # 学習期間: 2020-2022年（速度優先で3年間）
    create_universal_model(
        track_code='09',  # 阪神
        kyoso_shubetsu_code='13',  # 3歳以上
        surface_type='turf',  # 芝
        min_distance=1700,  # 中長距離
        max_distance=9999,  # 上限なし
        model_filename='hanshin_turf_3ageup_long.sav',
        output_dir='models',
        year_start=2020,
        year_end=2022
    )
    
    print("\n" + "="*60)
    print("モデル作成完了!")
    print("次のコマンドで穴馬検出テストを実行できます:")
    print("python upset_detector.py models/hanshin_turf_3ageup_long.sav --test-year 2023")
    print("="*60)
