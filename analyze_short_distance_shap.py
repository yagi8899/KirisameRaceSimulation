#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短距離モデル専用SHAP分析スクリプト

東京芝短距離モデル(tokyo_turf_3ageup_short.sav)のSHAP分析を実行し、
中長距離モデルとの特徴量重要度の違いを比較する。
"""

import psycopg2
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

# プロット保存用ディレクトリ
PLOT_DIR = Path('shap_analysis')
PLOT_DIR.mkdir(exist_ok=True)


def load_model_and_data(model_filename, track_code, kyoso_shubetsu_code, surface_type, 
                        min_distance, max_distance, test_year=2023, sample_size=500):
    """
    モデルとテストデータを読み込む
    """
    # モデル読み込み
    model_path = Path('models') / model_filename
    print(f"📦 モデル読み込み: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # PostgreSQL接続
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )
    
    # トラック条件を動的に設定
    if surface_type.lower() == 'turf':
        track_condition = "cast(rase.track_code as integer) between 10 and 22"
        baba_condition = "ra.babajotai_code_shiba"
    else:
        track_condition = "cast(rase.track_code as integer) between 23 and 29"
        baba_condition = "ra.babajotai_code_dirt"

    # 距離条件を設定
    if max_distance == 9999:
        distance_condition = f"cast(rase.kyori as integer) >= {min_distance}"
    else:
        distance_condition = f"cast(rase.kyori as integer) between {min_distance} and {max_distance}"
    
    # 競争種別条件を動的に設定
    if kyoso_shubetsu_code == "all":
        kyoso_shubetsu_condition = "1=1"
    else:
        kyoso_shubetsu_condition = f"rase.kyoso_shubetsu_code = '{kyoso_shubetsu_code}'"

    # SQLクエリ（model_creator.pyと同じ）
    sql = f"""
    select * from (
        select
        ra.kaisai_nen,
        ra.kaisai_tsukihi,
        ra.keibajo_code,
        ra.race_bango,
        ra.kyori,
        ra.tenko_code,
        {baba_condition} as babajotai_code,
        ra.grade_code,
        ra.kyoso_joken_code,
        ra.kyoso_shubetsu_code,
        ra.track_code,
        ra.shusso_tosu,
        seum.ketto_toroku_bango,
        trim(seum.bamei),
        seum.wakuban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.futan_juryo,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        seum.seibetsu_code,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        18 - cast(seum.kakutei_chakujun as integer) + 1 as kakutei_chakujun_numeric, 
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,
        AVG(
            (1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float)))
            * CASE
                WHEN seum.time_sa LIKE '-%' THEN 1.00
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                ELSE 0.20
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS past_avg_sotai_chakujun,
        AVG(
            cast(ra.kyori as integer) /
            NULLIF(
                FLOOR(cast(seum.soha_time as integer) / 1000) * 60 +
                FLOOR((cast(seum.soha_time as integer) % 1000) / 10) +
                (cast(seum.soha_time as integer) % 10) * 0.1,
                0
            )
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS time_index,
        SUM(
            CASE 
                WHEN seum.kakutei_chakujun = '01' THEN 100
                WHEN seum.kakutei_chakujun = '02' THEN 80
                WHEN seum.kakutei_chakujun = '03' THEN 60
                WHEN seum.kakutei_chakujun = '04' THEN 40
                WHEN seum.kakutei_chakujun = '05' THEN 30
                WHEN seum.kakutei_chakujun = '06' THEN 20
                WHEN seum.kakutei_chakujun = '07' THEN 10
                ELSE 5 
            END
            * CASE 
                WHEN ra.grade_code = 'A' THEN 3.00
                WHEN ra.grade_code = 'B' THEN 2.00
                WHEN ra.grade_code = 'C' THEN 1.50
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '999' THEN 1.00
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '016' THEN 0.80
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '010' THEN 0.60
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '005' THEN 0.40
                ELSE 0.20
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING  
        ) AS past_score,
        AVG(
            CASE 
                WHEN seum.kohan_3f = '000' OR seum.kohan_3f = '999' THEN NULL
                ELSE 600.0 / nullif(cast(seum.kohan_3f as integer), 0)
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS kohan_3f_index
    from
        jvd_ra ra 
        inner join ( 
            select
                se.kaisai_nen
                , se.kaisai_tsukihi
                , se.keibajo_code
                , se.race_bango
                , se.kakutei_chakujun
                , se.ketto_toroku_bango
                , se.bamei
                , se.wakuban
                , se.umaban
                , se.barei
                , se.seibetsu_code
                , se.kishu_code
                , se.chokyoshi_code
                , se.futan_juryo
                , se.tansho_odds
                , se.tansho_ninkijun
                , se.kohan_3f
                , se.soha_time
                , se.time_sa
            from
                jvd_se se
            where 
                se.kohan_3f <> '000' 
                and se.kohan_3f <> '999'
        ) seum 
            on ra.kaisai_nen = seum.kaisai_nen 
            and ra.kaisai_tsukihi = seum.kaisai_tsukihi 
            and ra.keibajo_code = seum.keibajo_code 
            and ra.race_bango = seum.race_bango 
    where
        cast(ra.kaisai_nen as integer) = {test_year}
    ) rase 
    where 
    rase.keibajo_code = '{track_code}'
    and {kyoso_shubetsu_condition}
    and {track_condition}
    and {distance_condition}
    """
    
    df = pd.read_sql_query(sql=sql, con=conn)
    conn.close()
    
    if len(df) == 0:
        print("❌ データが見つかりません")
        return None, None, None
    
    print(f"📊 データ件数: {len(df)}件")
    
    # サンプリング
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"📊 サンプリング後: {len(df)}件")
    
    # データ前処理
    df = df[df['chakujun_score'] > 0]
    
    numeric_columns = [
        'wakuban', 'umaban_numeric', 'barei', 'futan_juryo', 'tansho_odds',
        'kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'kyori', 'shusso_tosu',
        'tenko_code', 'babajotai_code', 'grade_code', 'kyoso_joken_code',
        'kyoso_shubetsu_code', 'track_code', 'seibetsu_code',
        'kakutei_chakujun_numeric', 'chakujun_score', 'past_avg_sotai_chakujun',
        'time_index', 'past_score', 'kohan_3f_index'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # 特徴量作成（model_creator.pyと同じ）
    X = df.loc[:, [
        "past_score",
        "kohan_3f_index",
        "past_avg_sotai_chakujun",
        "time_index",
    ]].astype(float)
    
    # 派生特徴量
    df['wakuban_ratio'] = df['wakuban'].astype(int) / df['shusso_tosu']
    X['wakuban_ratio'] = df['wakuban_ratio']
    
    df['futan_per_barei'] = df['futan_juryo'] / df['barei'].replace(0, 1)
    X['futan_per_barei'] = df['futan_per_barei']
    
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    df['futan_per_barei_log'] = np.log(df['futan_per_barei'].clip(lower=0.1))
    X['futan_per_barei_log'] = df['futan_per_barei_log']
    
    expected_weight_by_age = {2: 48, 3: 52, 4: 55, 5: 57, 6: 57, 7: 56, 8: 55}
    df['futan_deviation'] = df.apply(
        lambda row: row['futan_juryo'] - expected_weight_by_age.get(row['barei'], 55), 
        axis=1
    )
    X['futan_deviation'] = df['futan_deviation']
    
    # 以下、model_creator.pyと同じ特徴量を追加
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango'])['umaban_numeric'].rank(pct=True)
    X['umaban_percentile'] = df['umaban_percentile']
    
    # futan_zscore を計算（groupby の結果を reset_index して merge）
    futan_stats = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango'])['futan_juryo'].agg(['mean', 'std']).reset_index()
    df = df.merge(futan_stats, on=['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango'], how='left')
    df['futan_zscore'] = (df['futan_juryo'] - df['mean']) / df['std'].replace(0, 1)
    df['futan_zscore'] = df['futan_zscore'].fillna(0)
    X['futan_zscore'] = df['futan_zscore']
    
    df['futan_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango'])['futan_juryo'].rank(pct=True)
    X['futan_percentile'] = df['futan_percentile']
    
    distance_categories = {1000: 1, 1200: 2, 1400: 3, 1600: 4, 1800: 5, 2000: 6, 2200: 7, 2400: 8, 2500: 9, 3000: 10}
    df['distance_category_score'] = df['kyori'].apply(lambda x: distance_categories.get(x, 0))
    X['distance_category_score'] = df['distance_category_score']
    
    df['similar_distance_score'] = df.apply(
        lambda row: 1.0 if abs(row['kyori'] - 1600) <= 200 else 0.5,
        axis=1
    )
    X['similar_distance_score'] = df['similar_distance_score']
    
    track_aptitude = {'10': 1.0, '11': 1.0, '12': 0.9, '13': 0.8, '14': 0.9, '15': 0.8, '16': 0.7, '17': 1.0, '18': 0.9, '19': 0.8, '20': 0.9, '21': 0.8, '22': 0.7}
    df['surface_aptitude_score'] = df['track_code'].astype(str).map(track_aptitude).fillna(0.5)
    X['surface_aptitude_score'] = df['surface_aptitude_score']
    
    df['baba_change_adaptability'] = df.apply(
        lambda row: 0.8 if row['babajotai_code'] in [3, 4] else 1.0,
        axis=1
    )
    X['baba_change_adaptability'] = df['baba_change_adaptability']
    
    # 騎手・調教師特徴量（簡易版）
    kishu_stats = df.groupby('kishu_code').agg({
        'chakujun_score': 'mean'
    }).to_dict()['chakujun_score']
    df['kishu_skill_score'] = df['kishu_code'].map(kishu_stats).fillna(0.5)
    X['kishu_skill_score'] = df['kishu_skill_score']
    
    df['kishu_popularity_score'] = df['tansho_ninkijun_numeric'] / df['shusso_tosu']
    X['kishu_popularity_score'] = df['kishu_popularity_score']
    
    df['kishu_surface_score'] = df['kishu_code'].map(kishu_stats).fillna(0.5)
    X['kishu_surface_score'] = df['kishu_surface_score']
    
    chokyoshi_stats = df.groupby('chokyoshi_code').agg({
        'chakujun_score': 'mean'
    }).to_dict()['chakujun_score']
    df['chokyoshi_recent_score'] = df['chokyoshi_code'].map(chokyoshi_stats).fillna(0.5)
    X['chokyoshi_recent_score'] = df['chokyoshi_recent_score']
    
    y = df['kakutei_chakujun_numeric']
    
    return model, X, y


def analyze_shap(model, X, model_name):
    """
    SHAP分析を実行
    """
    print(f"\n🔍 SHAP分析開始: {model_name}")
    
    # SHAP explainer作成
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # SHAP値の平均絶対値を計算
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_mean_abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_mean_abs', ascending=False)
    
    print("\n📊 特徴量重要度（SHAP平均絶対値）:")
    print(shap_importance.to_string(index=False))
    
    # CSVで保存
    output_file = PLOT_DIR / f'{model_name}_importance.csv'
    shap_importance.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 保存完了: {output_file}")
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plot_file = PLOT_DIR / f'{model_name}_summary.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Summary plot保存: {plot_file}")
    
    return shap_importance


def main():
    print("=" * 80)
    print("🎯 短距離モデルSHAP分析")
    print("=" * 80)
    
    # 短距離モデル
    print("\n📌 東京芝短距離3歳以上モデル")
    model_short, X_short, y_short = load_model_and_data(
        model_filename='tokyo_turf_3ageup_short.sav',
        track_code='05',
        kyoso_shubetsu_code='13',
        surface_type='turf',
        min_distance=1000,
        max_distance=1600,
        test_year=2023,
        sample_size=500
    )
    
    if model_short is not None:
        shap_short = analyze_shap(model_short, X_short, 'tokyo_turf_3ageup_short')
    
    # 中長距離モデル（比較用）
    print("\n\n📌 東京芝中長距離3歳以上モデル（比較用）")
    model_long, X_long, y_long = load_model_and_data(
        model_filename='tokyo_turf_3ageup_long.sav',
        track_code='05',
        kyoso_shubetsu_code='13',
        surface_type='turf',
        min_distance=1700,
        max_distance=9999,
        test_year=2023,
        sample_size=500
    )
    
    if model_long is not None:
        shap_long = analyze_shap(model_long, X_long, 'tokyo_turf_3ageup_long')
    
    # 比較
    if model_short is not None and model_long is not None:
        print("\n" + "=" * 80)
        print("📊 短距離 vs 中長距離 特徴量重要度比較")
        print("=" * 80)
        
        comparison = pd.merge(
            shap_short[['feature', 'shap_mean_abs']].rename(columns={'shap_mean_abs': 'short'}),
            shap_long[['feature', 'shap_mean_abs']].rename(columns={'shap_mean_abs': 'long'}),
            on='feature',
            how='outer'
        ).fillna(0)
        
        comparison['diff'] = comparison['short'] - comparison['long']
        comparison = comparison.sort_values('diff', ascending=False)
        
        print("\n短距離で重要度が高い特徴量:")
        print(comparison.head(10).to_string(index=False))
        
        print("\n中長距離で重要度が高い特徴量:")
        print(comparison.tail(10).to_string(index=False))
        
        # 比較結果を保存
        comparison_file = PLOT_DIR / 'short_vs_long_comparison.csv'
        comparison.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 比較結果保存: {comparison_file}")
    
    print("\n" + "=" * 80)
    print("✅ SHAP分析完了!")
    print(f"📁 結果保存先: {PLOT_DIR.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
