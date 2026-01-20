#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
穴馬予測モデルの学習時とテスト時の不整合を調査
"""

import pandas as pd
import numpy as np
import pickle
import os

def investigate_model_discrepancy():
    """学習時とテスト時の違いを調査"""
    
    print("=" * 60)
    print("【調査1】モデルの特徴量を確認")
    print("=" * 60)
    
    # モデルをロード
    model_path = "models/upset_classifier_universal.sav"
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    upset_models = model_data['models']
    model_features = model_data['feature_cols']
    
    print(f"\n[MODEL] モデル数: {len(upset_models)}個")
    print(f"[MODEL] 特徴量数: {len(model_features)}個")
    print(f"[MODEL] 特徴量リスト:")
    for i, feat in enumerate(model_features):
        print(f"  [{i:2d}] {feat}")
    
    print("\n" + "=" * 60)
    print("【調査2】学習データの特徴量分布を確認")
    print("=" * 60)
    
    # 学習データをロード
    train_data_path = "results/upset_training_data_universal.tsv"
    train_df = pd.read_csv(train_data_path, sep='\t')
    
    print(f"\n[TRAIN] 学習データ件数: {len(train_df)}件")
    print(f"[TRAIN] 穴馬数: {train_df['is_upset'].sum()}頭")
    
    # 展開要因特徴量の分布
    upset_features = ['estimated_running_style', 'avg_4corner_position', 'distance_change', 
                      'wakuban_inner', 'wakuban_outer']
    
    print(f"\n[TRAIN] 展開要因特徴量の分布:")
    for feat in upset_features:
        if feat in train_df.columns:
            print(f"  {feat}: min={train_df[feat].min():.4f}, max={train_df[feat].max():.4f}, mean={train_df[feat].mean():.4f}")
        else:
            print(f"  {feat}: ★列が存在しない★")
    
    print("\n" + "=" * 60)
    print("【調査3】モデルの出力形式を確認")
    print("=" * 60)
    
    # 学習データから一部を取得して予測してみる
    X_sample = train_df[model_features].head(100).copy()
    X_sample = X_sample.fillna(0).replace([np.inf, -np.inf], 0)
    
    # 各モデルの出力を確認
    for i, model in enumerate(upset_models):
        pred = model.predict(X_sample, num_iteration=model.best_iteration)
        print(f"\n[MODEL{i}] predict()の出力:")
        print(f"  形状: {pred.shape}")
        print(f"  範囲: {pred.min():.4f} 〜 {pred.max():.4f}")
        print(f"  平均: {pred.mean():.4f}")
        print(f"  サンプル: {pred[:5]}")
        
        # predict_proba があるか確認
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_sample)
            print(f"  predict_proba()の出力形状: {proba.shape}")
            if len(proba.shape) > 1:
                print(f"  クラス1確率範囲: {proba[:, 1].min():.4f} 〜 {proba[:, 1].max():.4f}")
        
        # 1つ目のモデルだけ詳細確認
        if i == 0:
            break
    
    print("\n" + "=" * 60)
    print("【調査4】学習データでの穴馬の確率分布")
    print("=" * 60)
    
    # 学習データ全体で予測
    X_train = train_df[model_features].copy()
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    
    # アンサンブル予測
    proba_list = []
    for model in upset_models:
        proba = model.predict(X_train, num_iteration=model.best_iteration)
        proba_list.append(proba)
    
    avg_proba = np.mean(proba_list, axis=0)
    train_df['predicted_proba'] = avg_proba
    
    # 穴馬（is_upset=1）の確率分布
    upset_horses = train_df[train_df['is_upset'] == 1]
    non_upset_horses = train_df[train_df['is_upset'] == 0]
    
    print(f"\n[穴馬] 予測確率分布（is_upset=1）:")
    print(f"  件数: {len(upset_horses)}頭")
    print(f"  範囲: {upset_horses['predicted_proba'].min():.4f} 〜 {upset_horses['predicted_proba'].max():.4f}")
    print(f"  平均: {upset_horses['predicted_proba'].mean():.4f}")
    print(f"  中央値: {upset_horses['predicted_proba'].median():.4f}")
    
    print(f"\n[非穴馬] 予測確率分布（is_upset=0）:")
    print(f"  件数: {len(non_upset_horses)}頭")
    print(f"  範囲: {non_upset_horses['predicted_proba'].min():.4f} 〜 {non_upset_horses['predicted_proba'].max():.4f}")
    print(f"  平均: {non_upset_horses['predicted_proba'].mean():.4f}")
    print(f"  中央値: {non_upset_horses['predicted_proba'].median():.4f}")
    
    # 閾値別の精度
    print(f"\n[閾値分析] 学習データでの閾値別適合率:")
    for threshold in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]:
        candidates = train_df[train_df['predicted_proba'] > threshold]
        if len(candidates) > 0:
            hits = candidates['is_upset'].sum()
            precision = hits / len(candidates) * 100
            print(f"  閾値{threshold:.2f}: 候補{len(candidates):5d}頭, 的中{hits:4d}頭, 適合率{precision:5.2f}%")
        else:
            print(f"  閾値{threshold:.2f}: 候補0頭")

if __name__ == "__main__":
    investigate_model_discrepancy()