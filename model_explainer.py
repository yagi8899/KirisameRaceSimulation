#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP分析による競馬予測モデル説明スクリプト

学習済みモデルの予測理由をSHAPで可視化・分析します。
- 個別レースの予測理由を詳細表示
- 特徴量の全体的な影響度を可視化
- 特徴量間の相互作用を分析
"""

import psycopg2
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
import os
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from model_config_loader import get_all_models
from keiba_constants import format_model_description

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

# プロット保存用ディレクトリ
PLOT_DIR = Path('shap_analysis')
PLOT_DIR.mkdir(exist_ok=True)


def load_model_and_data(model_filename, track_code, kyoso_shubetsu_code, surface_type, 
                        min_distance, max_distance, test_year=2022, sample_size=None):
    """
    モデルとテストデータを読み込む
    
    Args:
        model_filename (str): モデルファイル名
        track_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競争種別コード
        surface_type (str): 'turf' or 'dirt'
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        test_year (int): テスト対象年 (デフォルト: 2022)
        sample_size (int): サンプル数制限 (None=全件)
        
    Returns:
        tuple: (model, X_test, y_test, test_df_full)
    """
    
    # モデル読み込み
    model_path = Path('models') / model_filename
    if not model_path.exists():
        model_path = Path(model_filename)
    
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

    # 競争種別を設定
    if kyoso_shubetsu_code == '12':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) = 12"
    elif kyoso_shubetsu_code == '13':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) >= 13"

    # SQLクエリ（model_creator.pyと完全に同じ構造）
    sql = f"""
    select * from (
        select
        ra.kaisai_nen,
        ra.kaisai_tsukihi,
        ra.race_bango,
        seum.umaban,
        seum.bamei,
        ra.keibajo_code,
        CASE 
            WHEN ra.keibajo_code = '01' THEN '札幌' 
            WHEN ra.keibajo_code = '02' THEN '函館' 
            WHEN ra.keibajo_code = '03' THEN '福島' 
            WHEN ra.keibajo_code = '04' THEN '新潟' 
            WHEN ra.keibajo_code = '05' THEN '東京' 
            WHEN ra.keibajo_code = '06' THEN '中山' 
            WHEN ra.keibajo_code = '07' THEN '中京' 
            WHEN ra.keibajo_code = '08' THEN '京都' 
            WHEN ra.keibajo_code = '09' THEN '阪神' 
            WHEN ra.keibajo_code = '10' THEN '小倉' 
            ELSE '' 
        END keibajo_name,
        ra.kyori,
        ra.shusso_tosu,
        ra.tenko_code,
        {baba_condition} as babajotai_code,
        ra.grade_code,
        ra.kyoso_joken_code,
        ra.kyoso_shubetsu_code,
        ra.track_code,
        seum.ketto_toroku_bango,
        seum.wakuban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.kishu_name,
        seum.chokyoshi_name,
        seum.futan_juryo,
        seum.seibetsu_code,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        18 - cast(seum.kakutei_chakujun as integer) + 1 as kakutei_chakujun_numeric,
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,
        AVG(
            1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float))
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
                WHEN ra.grade_code = 'A' THEN 1.00
                WHEN ra.grade_code = 'B' THEN 0.80
                WHEN ra.grade_code = 'C' THEN 0.60
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '999' THEN 0.50
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '016' THEN 0.40
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '010' THEN 0.30
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '005' THEN 0.20
                ELSE 0.10
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING  
        ) AS past_score,
        CASE 
            WHEN AVG(
                CASE 
                    WHEN cast(seum.kohan_3f as integer) > 0 AND cast(seum.kohan_3f as integer) < 999 THEN
                    CAST(seum.kohan_3f AS FLOAT) / 10
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) IS NOT NULL THEN
            AVG(
                CASE 
                    WHEN cast(seum.kohan_3f as integer) > 0 AND cast(seum.kohan_3f as integer) < 999 THEN
                    CAST(seum.kohan_3f AS FLOAT) / 10
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) - 
            CASE
                WHEN cast(ra.kyori as integer) <= 1600 THEN 33.5
                WHEN cast(ra.kyori as integer) <= 2000 THEN 35.0
                WHEN cast(ra.kyori as integer) <= 2400 THEN 36.0
                ELSE 37.0
            END
            ELSE 0
        END AS kohan_3f_index,
        seum.kakutei_chakujun,
        seum.kohan_3f
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
                , trim(se.kishumei_ryakusho) as kishu_name
                , trim(se.chokyoshimei_ryakusho) as chokyoshi_name
                , se.futan_juryo
                , se.tansho_odds
                , se.tansho_ninkijun
                , se.kohan_3f
                , se.soha_time
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
    
    print(f"[+] データ取得: {test_year}年")
    df_raw = pd.read_sql(sql, conn)
    conn.close()
    
    print(f"取得レコード数: {len(df_raw)}")
    
    if len(df_raw) == 0:
        print("[ERROR] データが取得できませんでした")
        return None, None, None, None
    
    # データ前処理
    df = df_raw.copy()
    
    # 文字列として保持すべきカラム
    string_columns = ['kishu_code', 'chokyoshi_code', 'bamei']
    
    # 数値カラムを明示的に定義（string_columnsを除く）
    numeric_columns = [col for col in df.columns if col not in string_columns + 
                      ['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango', 
                       'keibajo_name', 'ketto_toroku_bango', 'seibetsu_code', 
                       'kyoso_joken_code', 'kyoso_shubetsu_code', 
                       'grade_code', 'track_code']]
    
    # 数値カラムのみを数値型に変換
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # グループキー作成
    df['group_key'] = (df['kaisai_nen'].astype(str) + '_' + 
                       df['kaisai_tsukihi'].astype(str) + '_' + 
                       df['keibajo_code'].astype(str) + '_' + 
                       df['race_bango'].astype(str))
    
    # 特徴量計算
    X = calculate_features(df, model)
    
    # モデルの実際の特徴量名を取得して順序を合わせる
    if hasattr(model, 'feature_name'):
        actual_features = model.feature_name()
        print(f"[LIST] モデルの実際の特徴量: {len(actual_features)}個")
        
        # 不足している特徴量をチェック
        missing = [f for f in actual_features if f not in X.columns]
        if missing:
            raise ValueError(f"[ERROR] 必須特徴量が不足しています: {missing}")
        
        # 特徴量の順序をモデルと合わせる
        X = X[actual_features]
    else:
        print("[ERROR] モデルから特徴量名を取得できませんでした")
        return None, None, None, None
    
    y = df['kakutei_chakujun'].values
    
    # サンプリング
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[indices]
        y = y[indices]
        df = df.iloc[indices]
    
    print(f"[OK] データ準備完了: {len(X)}件")
    
    return model, X, y, df


def calculate_features(df, model):
    """
    model_creator.pyと同じ特徴量を計算
    """
    print("🔄 model_creator.pyと同じ特徴量計算を実行中...")
    
    # past_avg_sotai_chakujunはSQLで計算済みの単純移動平均を使用
    
    # 基本特徴量（SQLで計算済み）
    base_features = ["futan_juryo", "past_score", "kohan_3f_index", "past_avg_sotai_chakujun", "time_index"]
    
    # 不足チェック
    missing = [feat for feat in base_features if feat not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] 必須特徴量が不足しています: {missing}")
    
    X = df.loc[:, base_features].astype(float).copy()
    
    # 派生特徴量の計算
    # 枠番と頭数の比率
    max_wakuban = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['wakuban'].transform('max')
    X['wakuban_ratio'] = df['wakuban'] / max_wakuban
    
    # 斤量と馬齢の比率
    df['futan_per_barei'] = df['futan_juryo'] / df['barei'].replace(0, 1)
    X['futan_per_barei'] = df['futan_per_barei']
    
    # 馬番×距離の相互作用
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    # futan_per_bareiの非線形変換
    df['futan_per_barei_log'] = np.log(df['futan_per_barei'].clip(lower=0.1))
    X['futan_per_barei_log'] = df['futan_per_barei_log']
    
    # 期待斤量からの差分
    expected_weight_by_age = {2: 48, 3: 52, 4: 55, 5: 57, 6: 57, 7: 56, 8: 55}
    df['futan_deviation'] = df.apply(
        lambda row: row['futan_juryo'] - expected_weight_by_age.get(row['barei'], 55), 
        axis=1
    )
    X['futan_deviation'] = df['futan_deviation']
    
    # ピーク年齢パターン
    X['barei_peak_distance'] = abs(df['barei'] - 4)
    X['barei_peak_short'] = abs(df['barei'] - 3)
    
    # 枠番バイアススコア
    wakuban_stats = df.groupby('wakuban').agg({
        'kakutei_chakujun_numeric': ['mean', 'std', 'count']
    }).round(4)
    wakuban_stats.columns = ['waku_avg_rank', 'waku_std_rank', 'waku_count']
    wakuban_stats = wakuban_stats.reset_index()
    
    overall_avg_rank = df['kakutei_chakujun_numeric'].mean()
    wakuban_stats['wakuban_bias_score'] = (overall_avg_rank - wakuban_stats['waku_avg_rank']) / wakuban_stats['waku_std_rank']
    wakuban_stats['wakuban_bias_score'] = wakuban_stats['wakuban_bias_score'].fillna(0)
    
    df = df.merge(wakuban_stats[['wakuban', 'wakuban_bias_score']], on='wakuban', how='left')
    X['wakuban_bias_score'] = df['wakuban_bias_score']
    
    # 馬番相対位置
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # 斤量偏差値
    race_group = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['futan_juryo']
    df['futan_mean'] = race_group.transform('mean')
    df['futan_std'] = race_group.transform('std')
    
    df['futan_zscore'] = np.where(
        df['futan_std'] > 0,
        (df['futan_juryo'] - df['futan_mean']) / df['futan_std'],
        0
    )
    X['futan_zscore'] = df['futan_zscore']
    X['futan_percentile'] = race_group.transform(lambda x: x.rank(pct=True))
    
    # 距離・馬場カテゴリ分類
    def categorize_distance(kyori):
        if kyori <= 1400: return 'short'
        elif kyori <= 1800: return 'mile'
        elif kyori <= 2400: return 'middle'
        else: return 'long'
    
    def categorize_surface(track_code):
        track_code_int = int(track_code)
        if 10 <= track_code_int <= 22: return 'turf'
        elif 23 <= track_code_int <= 24: return 'dirt'
        else: return 'unknown'
    
    def categorize_baba_condition(baba_code):
        if baba_code == 1: return 'good'
        elif baba_code == 2: return 'slightly'
        elif baba_code == 3: return 'heavy'
        elif baba_code == 4: return 'bad'
        else: return 'unknown'
    
    df['distance_category'] = df['kyori'].apply(categorize_distance)
    df['surface_type'] = df['track_code'].apply(categorize_surface)
    df['baba_condition'] = df['babajotai_code'].apply(categorize_baba_condition)
    
    # 時系列スコア計算（距離適性）
    df_sorted = df.sort_values(['ketto_toroku_bango', 'kaisai_nen', 'kaisai_tsukihi']).copy()
    
    def calc_distance_category_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_category = group.iloc[idx]['distance_category']
            past_same = group.iloc[:idx][group.iloc[:idx]['distance_category'] == current_category].tail(5)
            if len(past_same) > 0:
                scores.append((1 - (past_same['kakutei_chakujun_numeric'] / 18.0)).mean())
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    def calc_similar_distance_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_kyori = group.iloc[idx]['kyori']
            past_similar = group.iloc[:idx][abs(group.iloc[:idx]['kyori'] - current_kyori) <= 200].tail(10)
            if len(past_similar) > 0:
                scores.append((1 - (past_similar['kakutei_chakujun_numeric'] / 18.0)).mean())
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    def calc_surface_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_surface = group.iloc[idx]['surface_type']
            past_same = group.iloc[:idx][group.iloc[:idx]['surface_type'] == current_surface].tail(10)
            if len(past_same) > 0:
                scores.append((1 - (past_same['kakutei_chakujun_numeric'] / 18.0)).mean())
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    print("  - 距離適性スコア計算中...")
    df_sorted['distance_category_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_category_score
    ).values
    
    df_sorted['similar_distance_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_similar_distance_score
    ).values
    
    print("  - 馬場適性スコア計算中...")
    df_sorted['surface_aptitude_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_surface_score
    ).values
    
    # 元の順序に戻す
    df['distance_category_score'] = df_sorted.sort_index()['distance_category_score']
    df['similar_distance_score'] = df_sorted.sort_index()['similar_distance_score']
    df['surface_aptitude_score'] = df_sorted.sort_index()['surface_aptitude_score']
    
    # distance_change_adaptability追加
    def calc_distance_change_adaptability(group):
        scores = []
        for idx in range(len(group)):
            if idx < 2:
                scores.append(0.5)
                continue
            past_races = group.iloc[max(0, idx-6):idx].copy()
            if len(past_races) >= 3:
                past_races['kyori_diff'] = past_races['kyori'].diff().abs()
                past_races_eval = past_races.tail(5)
                changed_races = past_races_eval[past_races_eval['kyori_diff'] >= 100]
                if len(changed_races) > 0:
                    scores.append((1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean())
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_change_adaptability
    ).values
    df['distance_change_adaptability'] = df_sorted.sort_index()['distance_change_adaptability']
    
    # baba_condition_score追加
    def calc_baba_condition_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_condition = group.iloc[idx]['baba_condition']
            past_same = group.iloc[:idx][group.iloc[:idx]['baba_condition'] == current_condition].tail(10)
            if len(past_same) > 0:
                scores.append((1 - (past_same['kakutei_chakujun_numeric'] / 18.0)).mean())
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_condition_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_condition_score
    ).values
    df['baba_condition_score'] = df_sorted.sort_index()['baba_condition_score']
    
    # baba_change_adaptability追加
    def calc_baba_change_adaptability(group):
        scores = []
        for idx in range(len(group)):
            if idx < 2:
                scores.append(0.5)
                continue
            past_races = group.iloc[max(0, idx-6):idx].copy()
            if len(past_races) >= 3:
                past_races['baba_changed'] = past_races['baba_condition'].shift(1) != past_races['baba_condition']
                past_races_eval = past_races.tail(5)
                changed_races = past_races_eval[past_races_eval['baba_changed'] == True]
                if len(changed_races) > 0:
                    scores.append((1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean())
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_change_adaptability
    ).values
    df['baba_change_adaptability'] = df_sorted.sort_index()['baba_change_adaptability']
    
    X['distance_category_score'] = df['distance_category_score']
    X['similar_distance_score'] = df['similar_distance_score']
    X['surface_aptitude_score'] = df['surface_aptitude_score']
    X['distance_change_adaptability'] = df['distance_change_adaptability']
    X['baba_condition_score'] = df['baba_condition_score']
    X['baba_change_adaptability'] = df['baba_change_adaptability']
    
    # 騎手・調教師スコア（簡易版 - 全体統計ベース）
    print("  - 騎手・調教師スコア計算中...")
    df_sorted_kishu = df.sort_values(['kishu_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    def calc_kishu_skill_score(group):
        scores = []
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
            past_races = group.iloc[:idx]
            if len(past_races) >= 3:
                avg_score = (1.0 - ((18 - past_races['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                scores.append(max(0.0, min(1.0, avg_score)))
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    def calc_kishu_surface_score(group):
        scores = []
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
            current_surface = group.iloc[idx]['surface_type']
            past_races = group.iloc[:idx]
            past_same_surface = past_races[past_races['surface_type'] == current_surface]
            if len(past_same_surface) >= 5:
                avg_score = (1 - ((18 - past_same_surface['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_skill_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_skill_score
    ).values
    
    df_sorted_kishu['kishu_surface_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_surface_score
    ).values
    
    # kishu_popularity_score追加
    def calc_kishu_popularity_score(group):
        scores = []
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
            past_races = group.iloc[:idx]
            if len(past_races) >= 3:
                valid_races = past_races[past_races['tansho_odds'] > 0]
                if len(valid_races) >= 3:
                    max_odds = valid_races['tansho_odds'].max()
                    valid_races = valid_races.copy()
                    valid_races['odds_expectation'] = 1.0 - (valid_races['tansho_odds'] / (max_odds + 1.0))
                    valid_races['actual_score'] = 1.0 - ((18 - valid_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                    valid_races['performance_diff'] = valid_races['actual_score'] - valid_races['odds_expectation']
                    avg_diff = valid_races['performance_diff'].mean()
                    normalized_score = 0.5 + (avg_diff * 0.5)
                    scores.append(max(0.0, min(1.0, normalized_score)))
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_popularity_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_popularity_score
    ).values
    
    df['kishu_skill_score'] = df_sorted_kishu.sort_index()['kishu_skill_score']
    df['kishu_surface_score'] = df_sorted_kishu.sort_index()['kishu_surface_score']
    df['kishu_popularity_score'] = df_sorted_kishu.sort_index()['kishu_popularity_score']
    
    # chokyoshi_recent_score追加
    df_sorted_chokyoshi = df.sort_values(['chokyoshi_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    def calc_chokyoshi_recent_score(group):
        scores = []
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['chokyoshi_code']) or group.iloc[idx]['chokyoshi_code'] == '':
                scores.append(0.5)
                continue
            past_races = group.iloc[:idx]
            if len(past_races) >= 5:
                avg_score = (1 - ((18 - past_races['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted_chokyoshi['chokyoshi_recent_score'] = df_sorted_chokyoshi.groupby('chokyoshi_code', group_keys=False).apply(
        calc_chokyoshi_recent_score
    ).values
    
    df['chokyoshi_recent_score'] = df_sorted_chokyoshi.sort_index()['chokyoshi_recent_score']
    
    X['kishu_skill_score'] = df['kishu_skill_score']
    X['kishu_surface_score'] = df['kishu_surface_score']
    X['kishu_popularity_score'] = df['kishu_popularity_score']
    X['chokyoshi_recent_score'] = df['chokyoshi_recent_score']
    
    print(f"[OK] 特徴量計算完了: {len(X.columns)}個")
    
    return X


def analyze_shap_global(model, X, feature_names, output_prefix):
    """
    SHAP全体分析（特徴量重要度、依存性プロット）
    
    Args:
        model: LightGBMモデル
        X: 特徴量データ
        feature_names: 特徴量名リスト
        output_prefix: 出力ファイル名プレフィックス
    """
    print("\n[+] SHAP全体分析を実行中...")
    
    # SHAP値計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. Summary Plot（特徴量重要度と分布）
    print("  - Summary Plot作成中...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - 特徴量の影響度と分布', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_prefix}_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 保存: {PLOT_DIR / f'{output_prefix}_summary.png'}")
    
    # 2. Bar Plot（平均絶対SHAP値）
    print("  - Bar Plot作成中...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Bar Plot - 特徴量の平均影響度', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_prefix}_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 保存: {PLOT_DIR / f'{output_prefix}_bar.png'}")
    
    # 3. 上位5特徴量の依存性プロット
    print("  - Dependence Plot作成中...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_features_idx):
        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, 
                            ax=axes[i], show=False)
        axes[i].set_title(f'{feature_names[idx]} の依存性', fontsize=12)
    
    # 最後のサブプロットを非表示
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_prefix}_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 保存: {PLOT_DIR / f'{output_prefix}_dependence.png'}")
    
    # 4. 特徴量重要度をCSV出力
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'lgb_gain': model.feature_importance(importance_type='gain')
    }).sort_values('mean_abs_shap', ascending=False)
    
    csv_path = PLOT_DIR / f'{output_prefix}_importance.csv'
    feature_importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    [OK] 特徴量重要度保存: {csv_path}")
    
    print("\n[LIST] 特徴量重要度トップ10:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    return shap_values, explainer


def analyze_shap_individual(shap_values, explainer, X, df_full, feature_names, 
                            output_prefix, num_samples=5):
    """
    個別レースのSHAP分析
    
    Args:
        shap_values: SHAP値配列
        explainer: SHAPExplainer
        X: 特徴量データ
        df_full: 元データフレーム（馬名などの情報含む）
        feature_names: 特徴量名リスト
        output_prefix: 出力ファイル名プレフィックス
        num_samples: 分析するサンプル数
    """
    print(f"\n[TEST] 個別レース分析（サンプル{num_samples}件）...")
    
    # ランダムにサンプル選択
    sample_indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n--- サンプル {i+1}/{num_samples} ---")
        
        # レース情報
        race_info = df_full.iloc[idx]
        print(f"日付: {race_info['kaisai_nen']}/{race_info['kaisai_tsukihi']}")
        print(f"競馬場: {race_info['keibajo_name']} R{race_info['race_bango']}")
        print(f"馬名: {race_info['bamei']}")
        print(f"実際の着順: {race_info['kakutei_chakujun']:.0f}着")
        print(f"人気: {race_info['tansho_ninkijun_numeric']:.0f}番人気")
        
        # Force Plot
        shap.force_plot(
            explainer.expected_value, 
            shap_values[idx], 
            X.iloc[idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"{race_info['bamei']} - SHAP Force Plot", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f'{output_prefix}_force_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 貢献度トップ10を表示
        shap_contributions = pd.DataFrame({
            'feature': feature_names,
            'value': X.iloc[idx].values,
            'shap_value': shap_values[idx]
        })
        shap_contributions['abs_shap'] = np.abs(shap_contributions['shap_value'])
        shap_contributions = shap_contributions.sort_values('abs_shap', ascending=False)
        
        print("\n貢献度トップ10:")
        for _, row in shap_contributions.head(10).iterrows():
            direction = "↑" if row['shap_value'] > 0 else "↓"
            print(f"  {row['feature']:30s}: {row['value']:8.2f} → SHAP={row['shap_value']:+8.4f} {direction}")
        
        print(f"  [OK] Force Plot保存: {PLOT_DIR / f'{output_prefix}_force_{i+1}.png'}")


def main():
    """
    メイン処理
    """
    import sys
    
    print("=" * 80)
    print("[TARGET] SHAP分析による競馬予測モデル説明")
    print("=" * 80)
    
    # 分析対象モデルを選択
    models = get_all_models()
    
    if not models:
        print("[ERROR] model_configs.jsonにモデルが定義されていません")
        return
    
    print("\n利用可能なモデル:")
    for i, model_info in enumerate(models, 1):
        desc = format_model_description(
            model_info['track_code'],
            model_info['kyoso_shubetsu_code'],
            model_info['surface_type'],
            model_info['min_distance'],
            model_info['max_distance']
        )
        print(f"  {i}. {desc}")
    
    # コマンドライン引数からモデルファイル名と対象年を取得
    target_model_filename = None
    test_year = 2023  # デフォルトは2023年
    
    if len(sys.argv) >= 2:
        target_model_filename = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            test_year = int(sys.argv[2])
        except ValueError:
            print(f"[WARNING] 年の指定が不正です: {sys.argv[2]}. デフォルト2023年を使用します")
    
    # 指定されたモデルを検索（指定なしの場合は最初のモデル）
    model_info = None
    if target_model_filename:
        for m in models:
            if m['model_filename'] == target_model_filename:
                model_info = m
                break
        if not model_info:
            print(f"[WARNING] モデル {target_model_filename} が見つかりません。最初のモデルを使用します")
            model_info = models[0]
    else:
        model_info = models[0]
    
    print(f"\n[PIN] 分析対象: {format_model_description(model_info['track_code'], model_info['kyoso_shubetsu_code'], model_info['surface_type'], model_info['min_distance'], model_info['max_distance'])}")
    print(f"[PIN] 対象年: {test_year}年")
    
    # モデルとデータ読み込み
    model, X, y, df_full = load_model_and_data(
        model_filename=model_info['model_filename'],
        track_code=model_info['track_code'],
        kyoso_shubetsu_code=model_info['kyoso_shubetsu_code'],
        surface_type=model_info['surface_type'],
        min_distance=model_info['min_distance'],
        max_distance=model_info['max_distance'],
        test_year=test_year,
        sample_size=500  # 計算時間短縮のため500件に制限
    )
    
    if model is None:
        print("[ERROR] データ取得に失敗しました")
        return
    
    # 出力ファイル名のプレフィックス
    output_prefix = Path(model_info['model_filename']).stem
    
    # SHAP全体分析
    shap_values, explainer = analyze_shap_global(
        model=model,
        X=X,
        feature_names=X.columns.tolist(),
        output_prefix=output_prefix
    )
    
    # 個別レース分析
    analyze_shap_individual(
        shap_values=shap_values,
        explainer=explainer,
        X=X,
        df_full=df_full,
        feature_names=X.columns.tolist(),
        output_prefix=output_prefix,
        num_samples=5
    )
    
    print("\n" + "=" * 80)
    print("[OK] SHAP分析完了!")
    print(f"[FILE] 結果保存先: {PLOT_DIR.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
