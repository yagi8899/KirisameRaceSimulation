#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習データとテストデータのpast_score乖離の原因を調査
"""

import pandas as pd
import numpy as np
import json
import psycopg2

def investigate_past_score():
    """past_scoreの乖離原因を調査"""
    
    print("=" * 70)
    print("【調査】past_scoreの乖離原因を特定")
    print("=" * 70)
    
    # 学習データ
    train_df = pd.read_csv('results/upset_training_data_universal.tsv', sep='\t')
    
    print(f"\n[学習データ] {len(train_df)}件")
    print(f"  past_score: mean={train_df['past_score'].mean():.2f}, median={train_df['past_score'].median():.2f}")
    
    # 年別のpast_score分布
    print(f"\n[学習データ] 年別past_score平均:")
    # kaisai_nenがあるか確認
    if 'kaisai_nen' in train_df.columns:
        for year in sorted(train_df['kaisai_nen'].unique()):
            year_df = train_df[train_df['kaisai_nen'] == year]
            print(f"  {year}年: mean={year_df['past_score'].mean():.2f}, n={len(year_df)}")
    else:
        print("  kaisai_nen列なし")
    
    # 競馬場別のpast_score分布
    print(f"\n[学習データ] 競馬場別past_score平均:")
    if 'keibajo_code_numeric' in train_df.columns:
        for code in sorted(train_df['keibajo_code_numeric'].unique()):
            code_df = train_df[train_df['keibajo_code_numeric'] == code]
            print(f"  競馬場{int(code)}: mean={code_df['past_score'].mean():.2f}, n={len(code_df)}")
    
    # 7-12番人気のみの分布（穴馬候補範囲）
    print(f"\n[学習データ] 人気別past_score平均:")
    if 'popularity_rank' in train_df.columns:
        for ninki in range(1, 13):
            ninki_df = train_df[train_df['popularity_rank'] == ninki]
            if len(ninki_df) > 0:
                print(f"  {ninki:2d}番人気: mean={ninki_df['past_score'].mean():.2f}, n={len(ninki_df)}")
    
    # 距離別の分布
    print(f"\n[学習データ] 距離帯別past_score平均:")
    if 'kyori' in train_df.columns:
        train_df['distance_band'] = pd.cut(train_df['kyori'], bins=[0, 1400, 1800, 2200, 9999], labels=['短距離', 'マイル', '中距離', '長距離'])
        for band in ['短距離', 'マイル', '中距離', '長距離']:
            band_df = train_df[train_df['distance_band'] == band]
            if len(band_df) > 0:
                print(f"  {band}: mean={band_df['past_score'].mean():.2f}, n={len(band_df)}")

if __name__ == "__main__":
    investigate_past_score()