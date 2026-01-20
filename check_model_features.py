#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデルファイルから特徴量リストを確認
"""

import pickle
import sys

def check_model_features(model_path):
    """モデルファイルから特徴量リストを確認"""
    
    print("=" * 80)
    print(f"モデル: {model_path}")
    print("=" * 80)
    
    # モデル読み込み
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nモデルの型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"キー: {list(data.keys())}")
        
        if 'feature_cols' in data:
            features = data['feature_cols']
            print(f"\n特徴量数: {len(features)}")
            
            # Phase 1の新特徴量チェック
            print("\n" + "=" * 80)
            print("Phase 1 新特徴量チェック")
            print("=" * 80)
            new_features = [
                'is_turf_bad_condition',
                'is_turf_heavy',
                'is_local_track',
                'is_open_class',
                'is_3win_class',
                'is_age_prime',
                'zenso_top6',
                'rest_days_fresh'
            ]
            
            for f in new_features:
                status = "OK - 含まれてる" if f in features else "NG - 含まれてない"
                print(f"  {f:30s} : {status}")
            
            # 全特徴量リスト
            print("\n" + "=" * 80)
            print(f"全特徴量リスト ({len(features)}個)")
            print("=" * 80)
            for i, f in enumerate(features, 1):
                print(f"  {i:3d}. {f}")
        else:
            print("\nERROR: 'feature_cols' キーが見つかりません")
    else:
        # LightGBM Boosterの場合
        if hasattr(data, 'feature_name'):
            features = data.feature_name()
            print(f"\n特徴量数: {len(features)}")
            
            # Phase 1の新特徴量チェック
            print("\n" + "=" * 80)
            print("Phase 1 新特徴量チェック")
            print("=" * 80)
            new_features = [
                'is_turf_bad_condition',
                'is_turf_heavy',
                'is_local_track',
                'is_open_class',
                'is_3win_class',
                'is_age_prime',
                'zenso_top6',
                'rest_days_fresh'
            ]
            
            for f in new_features:
                status = "OK - 含まれてる" if f in features else "NG - 含まれてない"
                print(f"  {f:30s} : {status}")
            
            # 全特徴量リスト
            print("\n" + "=" * 80)
            print(f"全特徴量リスト ({len(features)}個)")
            print("=" * 80)
            for i, f in enumerate(features, 1):
                print(f"  {i:3d}. {f}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # デフォルトのモデルパス
        model_path = 'walk_forward_results_custom2/period_10/models/2025/upset_classifier_2015-2024.sav'
    
    check_model_features(model_path)
