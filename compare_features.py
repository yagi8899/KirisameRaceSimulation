#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト時の特徴量分布を学習時と比較
"""

import pandas as pd
import numpy as np

def compare_feature_distributions():
    """学習時とテスト時の特徴量分布を比較"""
    
    # モデルの特徴量リスト
    model_features = [
        'predicted_rank', 'predicted_score', 'popularity_rank', 'tansho_odds', 'value_gap',
        'past_score', 'past_avg_sotai_chakujun', 'kohan_3f_index', 'time_index', 'relative_ability',
        'current_class_score', 'class_score_change', 'past_score_mean',
        'estimated_running_style', 'avg_4corner_position', 'distance_change',
        'wakuban_inner', 'wakuban_outer',
        'kyori', 'tenko_code', 'keibajo_code_numeric'
    ]
    
    # 学習データ
    train_df = pd.read_csv("results/upset_training_data_universal.tsv", sep='\t')
    
    # テスト結果（穴馬確率が出力されているファイル）
    test_df = pd.read_csv("results/predicted_results_hanshin_turf_3ageup_short_trainunknown_test2023_all.tsv", sep='\t')
    
    print("=" * 80)
    print("【学習データ vs テストデータ 特徴量比較】")
    print("=" * 80)
    
    print(f"\n[DATA] 学習データ: {len(train_df)}件")
    print(f"[DATA] テストデータ: {len(test_df)}件")
    
    # テストデータの列名確認
    print(f"\n[TEST] テストデータの列名:")
    for col in test_df.columns:
        print(f"  - {col}")
    
    # 特徴量がテストデータにあるか確認
    print(f"\n[FEATURE] 特徴量の存在確認:")
    missing_in_test = []
    for feat in model_features:
        if feat in train_df.columns and feat in test_df.columns:
            train_mean = train_df[feat].mean()
            test_mean = test_df[feat].mean() if feat in test_df.columns else np.nan
            print(f"  {feat:30}: 学習={train_mean:10.4f}, テスト={test_mean:10.4f}")
        elif feat in train_df.columns:
            print(f"  {feat:30}: 学習={train_df[feat].mean():10.4f}, テスト=★存在しない★")
            missing_in_test.append(feat)
        else:
            print(f"  {feat:30}: ★両方に存在しない★")
    
    if missing_in_test:
        print(f"\n⚠️ テストデータに存在しない特徴量: {missing_in_test}")
        print("→ これらは0で埋められて予測に使われている可能性あり！")

if __name__ == "__main__":
    compare_feature_distributions()