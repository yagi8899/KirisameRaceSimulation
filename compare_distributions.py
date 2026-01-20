#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習時とテスト時の特徴量分布を詳細比較
"""

import pandas as pd
import numpy as np
import pickle

def compare_distributions():
    """学習時とテスト時の特徴量分布を比較"""
    
    # モデルの特徴量リスト
    with open("models/upset_classifier_universal.sav", 'rb') as f:
        model_data = pickle.load(f)
    feature_cols = model_data['feature_cols']
    
    # 学習データ
    train_df = pd.read_csv("results/upset_training_data_universal.tsv", sep='\t')
    
    print("=" * 80)
    print("[比較] 学習データ vs テストデータ（阪神中長距離）特徴量分布")
    print("=" * 80)
    print(f"\n学習データ: {len(train_df)}件, 穴馬: {train_df['is_upset'].sum()}頭")
    
    # テストで報告された値との比較
    # テスト時の値（universal_test.pyのデバッグ出力から）
    test_values = {
        'predicted_rank': 6.5417,
        'predicted_score': 0.4460,
        'popularity_rank': 6.5891,
        'tansho_odds': 52.2619,
        'value_gap': -0.0474,
        'past_score': 81.5172,
        'past_avg_sotai_chakujun': 0.4246,
        'kohan_3f_index': 0.1562,
        'time_index': 16.4743,
        'relative_ability': 0.0000,
        'current_class_score': 0.8277,
        'class_score_change': -0.0207,
        'past_score_mean': 27.2799,
        'estimated_running_style': 0.9971,
        'avg_4corner_position': 6.0345,
        'distance_change': 41.6810,
        'wakuban_inner': 0.3032,
        'wakuban_outer': 0.4511,
        'kyori': 2093.6782,
        'tenko_code': 1.3793,
        'keibajo_code_numeric': 9.0000
    }
    
    print(f"\n{'特徴量':<30} {'学習平均':>12} {'テスト平均':>12} {'差異':>10}")
    print("-" * 70)
    
    for col in feature_cols:
        if col in train_df.columns:
            train_mean = train_df[col].mean()
            test_mean = test_values.get(col, np.nan)
            if not np.isnan(test_mean):
                diff = test_mean - train_mean
                diff_pct = diff / train_mean * 100 if train_mean != 0 else 0
                flag = " <<<" if abs(diff_pct) > 50 else ""
                print(f"{col:<30} {train_mean:>12.4f} {test_mean:>12.4f} {diff_pct:>9.1f}%{flag}")
            else:
                print(f"{col:<30} {train_mean:>12.4f} {'N/A':>12}")
        else:
            print(f"{col:<30} {'N/A':>12}")
    
    # 穴馬と非穴馬の特徴量差異を確認
    print("\n" + "=" * 80)
    print("[分析] 学習データ内での穴馬 vs 非穴馬の特徴量差異")
    print("=" * 80)
    
    upset = train_df[train_df['is_upset'] == 1]
    non_upset = train_df[train_df['is_upset'] == 0]
    
    print(f"\n{'特徴量':<30} {'穴馬平均':>12} {'非穴馬平均':>12} {'差異':>10}")
    print("-" * 70)
    
    for col in feature_cols:
        if col in train_df.columns:
            upset_mean = upset[col].mean()
            non_upset_mean = non_upset[col].mean()
            diff = upset_mean - non_upset_mean
            print(f"{col:<30} {upset_mean:>12.4f} {non_upset_mean:>12.4f} {diff:>10.4f}")

if __name__ == "__main__":
    compare_distributions()