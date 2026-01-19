#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
universal_test.pyのテスト時の特徴量値を直接確認
"""

import pandas as pd
import numpy as np
import pickle
import psycopg2
from pathlib import Path
import sys
sys.path.append('.')

from db_query_builder import build_race_data_query
from feature_engineering import add_upset_features

def check_actual_test_features():
    """テスト時の実際の特徴量値を確認"""
    
    print("=" * 60)
    print("【調査】テスト時の実際の特徴量値を確認")
    print("=" * 60)
    
    # 阪神短距離のパラメータ
    track_code = '09'
    surface_type = 'turf'
    min_distance = 1000
    max_distance = 1600
    test_year = 2023
    
    # SQL取得
    sql = build_race_data_query(
        track_code=track_code,
        year_start=test_year - 3,
        year_end=test_year,
        surface_type=surface_type,
        distance_min=min_distance,
        distance_max=max_distance,
        kyoso_shubetsu_code='13',
        include_payout=True
    )
    
    # テスト年のみフィルタ
    sql = f"""
    select * from (
        {sql}
    ) filtered_data
    where cast(filtered_data.kaisai_nen as integer) = {test_year}
    """
    
    # DB接続
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="keiba",
        user="postgres",
        password="postgres"
    )
    
    df = pd.read_sql(sql, conn)
    conn.close()
    
    print(f"\n[DATA] 取得データ: {len(df)}件")
    
    # 展開要因特徴量の元データを確認
    print(f"\n[BEFORE] add_upset_features() 呼び出し前:")
    required_cols = ['corner_4_numeric', 'zenso_kyori', 'zenso_chakujun', 'kyori', 'wakuban', 'kakutei_chakujun_numeric']
    for col in required_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            sample = df[col].dropna().head(5).tolist()
            print(f"  {col}: {non_null}/{len(df)}件 (非NULL)")
            print(f"    サンプル: {sample}")
        else:
            print(f"  {col}: 列なし!")
    
    # add_upset_featuresを呼び出し
    df = add_upset_features(df)
    
    print(f"\n[AFTER] add_upset_features() 呼び出し後:")
    upset_cols = ['estimated_running_style', 'avg_4corner_position', 'distance_change',
                  'wakuban_inner', 'wakuban_outer', 'prev_rank_change']
    for col in upset_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            unique = df[col].nunique()
            sample = df[col].head(10).tolist()
            print(f"  {col}:")
            print(f"    非NULL: {non_null}/{len(df)}件")
            print(f"    ユニーク値数: {unique}")
            print(f"    サンプル: {sample}")
        else:
            print(f"  {col}: 列なし!")
    
    # モデルの特徴量でテスト用DataFrameを作成して確認
    print("\n" + "=" * 60)
    print("【調査】モデル特徴量とのマッピング確認")
    print("=" * 60)
    
    # モデルをロード
    model_path = "models/upset_classifier_universal.sav"
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    upset_models = model_data['models']
    model_features = model_data['feature_cols']
    
    print(f"\n[MODEL] 必要な特徴量: {len(model_features)}個")
    
    # 各特徴量の存在確認
    print(f"\n[MAPPING] 特徴量マッピング状況:")
    missing_features = []
    for feat in model_features:
        if feat in df.columns:
            print(f"  OK {feat}")
        else:
            print(f"  NG {feat} - 存在しない!")
            missing_features.append(feat)
    
    if missing_features:
        print(f"\n[WARNING] 欠損特徴量: {missing_features}")
        print("これらは0で埋められて予測に使われる!")

if __name__ == "__main__":
    check_actual_test_features()