#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト時のデータに展開要因特徴量の元データがあるか確認
"""

import pandas as pd
import numpy as np
import sys
import os

# SQLクエリを取得するためにdb_query_builderをインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_query_builder import build_race_data_query

def check_test_data_columns():
    """テストデータに必要な列があるか確認"""
    
    print("=" * 60)
    print("【調査】テスト用SQLクエリの列を確認")
    print("=" * 60)
    
    # 阪神短距離のテストパラメータ（直接指定）
    keibajo_code = '09'  # 阪神
    kyori_low = 1000
    kyori_high = 1600
    test_years = [2023]
    track_type = 'turf'
    
    print(f"\n[CONFIG] 阪神芝短距離")
    print(f"[CONFIG] keibajo: {keibajo_code}")
    print(f"[CONFIG] 距離: {kyori_low}〜{kyori_high}")
    
    # SQLクエリを取得
    sql = build_race_data_query(
        track_code=keibajo_code,
        year_start=2020,
        year_end=2023,
        surface_type=track_type,
        distance_min=kyori_low,
        distance_max=kyori_high,
        kyoso_shubetsu_code='13',
        include_payout=True
    )
    
    # 必要な列がSQLに含まれているか確認
    required_cols = ['corner_4_numeric', 'zenso_kyori', 'zenso_chakujun']
    
    print(f"\n[SQL] 必要な列の存在確認:")
    for col in required_cols:
        if col in sql:
            print(f"  OK {col}: SQLに含まれている")
        else:
            print(f"  NG {col}: SQLに含まれていない！")
    
    # SQLクエリの一部を表示
    print(f"\n[SQL] クエリの一部 (最初の3000文字):")
    print(sql[:3000])
    
    # 実際にDBから取得してカラムを確認
    print(f"\n" + "=" * 60)
    print("【調査】実際にDBからデータを取得して確認")
    print("=" * 60)
    
    import psycopg2
    import json
    
    # DB接続情報をdb_config.jsonから取得
    with open('db_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    db_config = config['database']
    
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        df = pd.read_sql(sql, conn)
        conn.close()
        
        print(f"\n[DB] 取得データ: {len(df)}件")
        
        print(f"\n[DB] 全列名:")
        for i, col in enumerate(df.columns):
            print(f"  [{i:2d}] {col}")
        
        # 展開要因特徴量の元データを確認
        print(f"\n[DB] 展開要因特徴量の元データ確認:")
        for col in required_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"  OK {col}: {non_null}/{len(df)}件 (非NULL), 例: {df[col].dropna().head(3).tolist()}")
            else:
                print(f"  NG {col}: 列が存在しない！")
        
    except Exception as e:
        print(f"[ERROR] DB接続エラー: {e}")

if __name__ == "__main__":
    check_test_data_columns()