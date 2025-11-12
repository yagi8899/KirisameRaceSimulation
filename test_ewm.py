#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数加重平均(EWM)機能のテストスクリプト
"""
from model_creator import create_universal_model

if __name__ == '__main__':
    print("="*80)
    print("指数加重平均(EWM)機能のテスト - フルデータ版")
    print("="*80)
    print("\nテストモデル: 東京芝1700m以上3歳以上")
    print("学習期間: 2013-2021年（フルデータ）")
    print("テスト期間: 2022-2023年")
    print("="*80)
    
    # フルデータでモデルを作成
    create_universal_model(
        track_code='05',           # 東京
        kyoso_shubetsu_code='13',  # 3歳以上
        surface_type='turf',       # 芝
        min_distance=1700,
        max_distance=9999,
        model_filename='test_ewm_model.sav',
        output_dir='models',
        year_start=2013,           # フルデータ: 2013-2021
        year_end=2021
    )
    
    print("\n" + "="*80)
    print("[OK] テスト完了!")
    print("="*80)
    print("\n確認事項:")
    print("  1. EWM変換のログが表示されたか?")
    print("  2. SQL平均 vs EWM平均の比較が表示されたか?")
    print("  3. モデルが正常に作成されたか?")
    print("\n期待される改善:")
    print("  - 調子の上昇/下降を検知")
    print("  - 最新成績を重視した予測")
    print("  - 的中率 +0.5~1.5% 向上")
