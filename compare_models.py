#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旧モデル vs 新モデル(EWM版)のSHAP比較スクリプト
"""
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

def compare_models():
    """モデル比較分析"""
    print("="*80)
    print("📊 旧モデル vs 新モデル(EWM版) 比較分析")
    print("="*80)
    
    # モデル読み込み
    print("\n📦 モデル読み込み中...")
    with open('models/tokyo_turf_3ageup_long.sav', 'rb') as f:
        old_model = pickle.load(f)
    print("  ✅ 旧モデル: tokyo_turf_3ageup_long.sav")
    
    with open('models/test_ewm_model.sav', 'rb') as f:
        new_model = pickle.load(f)
    print("  ✅ 新モデル: test_ewm_model.sav (EWM版)")
    
    # 特徴量取得
    old_features = old_model.feature_name()
    new_features = new_model.feature_name()
    
    print(f"\n📋 特徴量数:")
    print(f"  旧モデル: {len(old_features)}個")
    print(f"  新モデル: {len(new_features)}個")
    
    # 特徴量重要度取得
    old_importance = old_model.feature_importance(importance_type='gain')
    new_importance = new_model.feature_importance(importance_type='gain')
    
    # DataFrame化
    old_df = pd.DataFrame({
        'feature': old_features,
        'importance_old': old_importance
    }).sort_values('importance_old', ascending=False)
    
    new_df = pd.DataFrame({
        'feature': new_features,
        'importance_new': new_importance
    }).sort_values('importance_new', ascending=False)
    
    # マージ
    comparison = pd.merge(old_df, new_df, on='feature', how='outer').fillna(0)
    comparison['diff'] = comparison['importance_new'] - comparison['importance_old']
    comparison['diff_ratio'] = ((comparison['importance_new'] / comparison['importance_old']) - 1) * 100
    comparison['diff_ratio'] = comparison['diff_ratio'].replace([np.inf, -np.inf], 0)
    
    # Top15表示
    print("\n" + "="*80)
    print("【特徴量重要度 Top15 比較】")
    print("="*80)
    comparison_top = comparison.sort_values('importance_new', ascending=False).head(15)
    
    for idx, row in comparison_top.iterrows():
        old_val = row['importance_old']
        new_val = row['importance_new']
        diff = row['diff']
        diff_ratio = row['diff_ratio']
        
        if diff > 0:
            arrow = "↗️"
        elif diff < 0:
            arrow = "↘️"
        else:
            arrow = "→"
        
        print(f"{row['feature']:30s} {arrow}")
        print(f"  旧: {old_val:8.2f} → 新: {new_val:8.2f} (差分: {diff:+7.2f}, {diff_ratio:+6.1f}%)")
    
    # past_avg_sotai_chakujunの変化を特に注目
    print("\n" + "="*80)
    print("【🔥 past_avg_sotai_chakujun の変化】")
    print("="*80)
    past_row = comparison[comparison['feature'] == 'past_avg_sotai_chakujun'].iloc[0]
    print(f"旧モデル重要度: {past_row['importance_old']:.2f}")
    print(f"新モデル重要度: {past_row['importance_new']:.2f}")
    print(f"変化率: {past_row['diff_ratio']:+.1f}%")
    
    if past_row['diff'] < 0:
        print("⚠️ 重要度が低下しています!")
        print("原因候補:")
        print("  1. EWMで過度に平滑化され、情報量が減った")
        print("  2. 過去データが少ない馬でNaNが増えた")
        print("  3. span=3が適切ではない(span=5など試す必要)")
    elif past_row['diff'] > 0:
        print("✅ 重要度が向上しています!")
    
    # 最も変化した特徴量
    print("\n" + "="*80)
    print("【最も増加/減少した特徴量】")
    print("="*80)
    
    print("\n増加Top5:")
    increased = comparison.sort_values('diff', ascending=False).head(5)
    for idx, row in increased.iterrows():
        print(f"  {row['feature']:30s} {row['diff']:+7.2f} ({row['diff_ratio']:+6.1f}%)")
    
    print("\n減少Top5:")
    decreased = comparison.sort_values('diff', ascending=True).head(5)
    for idx, row in decreased.iterrows():
        print(f"  {row['feature']:30s} {row['diff']:+7.2f} ({row['diff_ratio']:+6.1f}%)")
    
    # 結果ファイルを読み込んで的中率比較
    print("\n" + "="*80)
    print("【性能比較】")
    print("="*80)
    
    # 結果ファイル読み込み(あれば)
    results_file = Path('results/model_comparison.tsv')
    if results_file.exists():
        results_df = pd.read_csv(results_file, sep='\t')
        print(results_df.to_string(index=False))
    
    # 保存
    comparison.to_csv('model_feature_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 比較結果を保存: model_feature_comparison.csv")
    
    # 考察
    print("\n" + "="*80)
    print("【考察】")
    print("="*80)
    print("的中率が悪化した原因候補:")
    print("  1. 🔥 EWMで情報が平滑化されすぎた可能性")
    print("     → span=3が小さすぎる? span=5, 7で試す")
    print("  2. 🔥 過去データが少ない馬でEWMがうまく機能していない")
    print("     → min_periods=1が原因? min_periods=2に変更")
    print("  3. 🔥 学習データが少ない(2020-2021のみ)")
    print("     → 2013-2021で再学習して比較")
    print("  4. 他の特徴量とのバランスが崩れた")
    print("     → 特徴量重要度の変化を確認")


if __name__ == '__main__':
    compare_models()
