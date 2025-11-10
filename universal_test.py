#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汎用競馬予測テストスクリプト

このスクリプトは、複数のモデルファイルに対応した競馬予測テストを実行します。
model_creator.pyで作成したモデルを使用して予測を行い、結果を保存します。
"""

import psycopg2
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from keiba_constants import get_track_name, format_model_description
from model_config_loader import get_all_models, get_legacy_model


def save_results_with_append(df, filename, append_mode=True, output_dir='results'):
    """
    結果をTSVファイルに保存（追記モード対応）
    
    Args:
        df (DataFrame): 保存するデータフレーム
        filename (str): 保存先ファイル名
        append_mode (bool): True=追記モード、False=上書きモード
        output_dir (str): 出力先ディレクトリ（デフォルト: 'results'）
    """
    # 出力ディレクトリを作成（存在しない場合）
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # ファイルパスを作成
    filepath = output_path / filename
    
    if append_mode and filepath.exists():
        # ファイルが既に存在する場合は追記（ヘッダーなし）
        print(f"📝 既存ファイルに追記: {filepath}")
        df.to_csv(filepath, mode='a', header=False, index=False, sep='\t', encoding='utf-8-sig')
    else:
        # ファイルが存在しない場合は新規作成（ヘッダーあり）
        print(f"📋 新規ファイル作成: {filepath}")
        df.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')


def predict_with_model(model_filename, track_code, kyoso_shubetsu_code, surface_type, 
                      min_distance, max_distance, test_year_start=2023, test_year_end=2023):
    """
    指定したモデルで予測を実行する汎用関数
    
    Args:
        model_filename (str): 使用するモデルファイル名
        track_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競争種別コード
        surface_type (str): 'turf' or 'dirt'
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        test_year_start (int): テスト対象開始年 (デフォルト: 2023)
        test_year_end (int): テスト対象終了年 (デフォルト: 2023)
        
    Returns:
        tuple: (予測結果DataFrame, サマリーDataFrame, レース数)
    """
    
    # PostgreSQL コネクションの作成
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )

    # トラック条件を動的に設定
    if surface_type.lower() == 'turf':
        # 芝の場合
        track_condition = "cast(rase.track_code as integer) between 10 and 22"
        baba_condition = "ra.babajotai_code_shiba"
    else:
        # ダートの場合
        track_condition = "cast(rase.track_code as integer) between 23 and 29"
        baba_condition = "ra.babajotai_code_dirt"

    # 距離条件を設定
    if max_distance == 9999:
        distance_condition = f"cast(rase.kyori as integer) >= {min_distance}"
    else:
        distance_condition = f"cast(rase.kyori as integer) between {min_distance} and {max_distance}"

    # 競争種別を設定
    if kyoso_shubetsu_code == '12':
        # 3歳戦
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) = 12"
    elif kyoso_shubetsu_code == '13':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) >= 13"

    # SQLクエリを動的に生成
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
        nullif(cast(seum.kakutei_chakujun as integer), 0) as kakutei_chakujun_numeric,
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,
        AVG(
            (1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float)))
            * CASE
                WHEN seum.time_sa LIKE '-%' THEN 1.00  -- 1着(マイナス値) → 係数1.00(満点)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85   -- 0.5秒差以内 → 0.85倍(15%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70  -- 1.0秒差以内 → 0.70倍(30%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50  -- 2.0秒差以内 → 0.50倍(50%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30  -- 3.0秒差以内 → 0.30倍(70%減)
                ELSE 0.20  -- 3.0秒超 → 0.20倍(大敗はほぼ無視)
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
        END AS kohan_3f_index
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1a), '') as integer), 0) as 複勝1着馬番
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1b), '') as float), 0) / 100 as 複勝1着オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1c), '') as integer), 0) as 複勝1着人気
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_2a), '') as integer), 0) as 複勝2着馬番
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_2b), '') as float), 0) / 100 as 複勝2着オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_2c), '') as integer), 0) as 複勝2着人気
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_3a), '') as integer), 0) as 複勝3着馬番
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_3b), '') as float), 0) / 100 as 複勝3着オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_3c), '') as integer), 0) as 複勝3着人気
        ,cast(substring(trim(hr.haraimodoshi_umaren_1a), 1, 2) as integer) as 馬連馬番1
        ,cast(substring(trim(hr.haraimodoshi_umaren_1a), 3, 2) as integer) as 馬連馬番2
        ,nullif(cast(nullif(trim(hr.haraimodoshi_umaren_1b), '') as float), 0) / 100 as 馬連オッズ
        ,cast(substring(trim(hr.haraimodoshi_wide_1a), 1, 2) as integer) as ワイド1_2馬番1
        ,cast(substring(trim(hr.haraimodoshi_wide_1a), 3, 2) as integer) as ワイド1_2馬番2
        ,cast(substring(trim(hr.haraimodoshi_wide_2a), 1, 2) as integer) as ワイド2_3着馬番1
        ,cast(substring(trim(hr.haraimodoshi_wide_2a), 3, 2) as integer) as ワイド2_3着馬番2
        ,cast(substring(trim(hr.haraimodoshi_wide_3a), 1, 2) as integer) as ワイド1_3着馬番1
        ,cast(substring(trim(hr.haraimodoshi_wide_3a), 3, 2) as integer) as ワイド1_3着馬番2
        ,nullif(cast(nullif(trim(hr.haraimodoshi_wide_1b), '') as float), 0) / 100 as ワイド1_2オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_wide_2b), '') as float), 0) / 100 as ワイド2_3オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_wide_3b), '') as float), 0) / 100 as ワイド1_3オッズ
        ,cast(substring(trim(hr.haraimodoshi_umatan_1a), 1, 2) as integer) as 馬単馬番1
        ,cast(substring(trim(hr.haraimodoshi_umatan_1a), 3, 2) as integer) as 馬単馬番2
        ,nullif(cast(nullif(trim(hr.haraimodoshi_umatan_1b), '') as float), 0) / 100 as 馬単オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_sanrenpuku_1b), '') as float), 0) / 100 as ３連複オッズ
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
                , se.futan_juryo
                , se.kishu_code
                , se.chokyoshi_code
                , trim(se.kishumei_ryakusho) as kishu_name
                , trim(se.chokyoshimei_ryakusho) as chokyoshi_name
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
        inner join jvd_hr hr
            on ra.kaisai_nen = hr.kaisai_nen 
            and ra.kaisai_tsukihi = hr.kaisai_tsukihi 
            and ra.keibajo_code = hr.keibajo_code 
            and ra.race_bango = hr.race_bango
    where
        cast(ra.kaisai_nen as integer) between {test_year_start - 3} and {test_year_end}  --テスト用データの対象年範囲
    ) rase 
    where 
    rase.keibajo_code = '{track_code}'
    and cast(rase.kaisai_nen as integer) between {test_year_start} and {test_year_end}  --テスト年範囲
    and {kyoso_shubetsu_condition}
    and {track_condition}
    and {distance_condition}
    """
    
    # テスト用のSQLをログファイルに出力（常に上書き）
    log_filepath = Path('sql_log_test.txt')
    with open(log_filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== テスト用SQL ===\n")
        f.write(f"モデル: {model_filename}\n")
        f.write(f"テスト期間: {test_year_start}年〜{test_year_end}年\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{sql}\n")
    print(f"📝 テスト用SQLをログファイルに出力: {log_filepath}")

    # データを取得
    df = pd.read_sql_query(sql=sql, con=conn)
    conn.close()
    
    if len(df) == 0:
        print(f"❌ {model_filename} に対応するテストデータが見つかりませんでした。")
        return None, None, 0

    print(f"📊 テストデータ件数: {len(df)}件")

    # 🔥修正: データ前処理を適切に実施（model_creator.pyと同じロジック）
    # 騎手コード・調教師コード・馬名などの文字列列を保持したまま、数値列のみを処理
    print("🔍 データ型確認...")
    print(f"  kishu_code型（修正前）: {df['kishu_code'].dtype}")
    print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
    print(f"  kishu_codeユニーク数: {df['kishu_code'].nunique()}")
    
    # 数値化する列を明示的に指定（文字列列は除外）
    numeric_columns = [
        'wakuban', 'umaban_numeric', 'barei', 'futan_juryo', 'tansho_odds',
        'kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'kyori', 'shusso_tosu',
        'tenko_code', 'babajotai_code', 'grade_code', 'kyoso_joken_code',
        'kyoso_shubetsu_code', 'track_code', 'seibetsu_code',
        'kakutei_chakujun_numeric', 'chakujun_score', 'past_avg_sotai_chakujun',
        'time_index', 'past_score', 'kohan_3f_index'
    ]
    
    # 数値化する列のみ処理（文字列列は保持）
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値を0で埋める（数値列のみ）
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # 文字列型の列はそのまま保持（kishu_code, chokyoshi_code, bamei など）
    print(f"  kishu_code型（修正後）: {df['kishu_code'].dtype}")
    print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
    print("✅ データ前処理完了（文字列列を保持）")

    # past_avg_sotai_chakujunはSQLで計算済みの単純移動平均を使用
    # (EWM実験の結果、単純平均の方が複勝・三連複で安定した性能を示した)

    # 特徴量を選択（model_creator.pyと同じ特徴量）
    X = df.loc[:, [
        # "futan_juryo",
        "past_score",
        "kohan_3f_index",
        "past_avg_sotai_chakujun",
        "time_index",
    ]].astype(float)
    
    # 高性能な派生特徴量を追加！（model_creator.pyと同じ）
    # 枠番と頭数の比率（内枠有利度）
    max_wakuban = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['wakuban'].transform('max')
    df['wakuban_ratio'] = df['wakuban'] / max_wakuban
    X['wakuban_ratio'] = df['wakuban_ratio']
    
    # 斤量と馬齢の比率（若馬の負担能力）
    df['futan_per_barei'] = df['futan_juryo'] / df['barei'].replace(0, 1)
    X['futan_per_barei'] = df['futan_per_barei']
    
    # 🔥改善された特徴量🔥
    # 2. futan_per_bareiの非線形変換
    df['futan_per_barei_log'] = np.log(df['futan_per_barei'].clip(lower=0.1))
    X['futan_per_barei_log'] = df['futan_per_barei_log']
    
    # 期待斤量からの差分（年齢別期待斤量との差）
    expected_weight_by_age = {2: 48, 3: 52, 4: 55, 5: 57, 6: 57, 7: 56, 8: 55}
    df['futan_deviation'] = df.apply(
        lambda row: row['futan_juryo'] - expected_weight_by_age.get(row['barei'], 55), 
        axis=1
    )
    X['futan_deviation'] = df['futan_deviation']

    # 馬番×距離の相互作用（内外枠の距離適性）
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000  # スケール調整
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    # 4. 複数のピーク年齢パターン
    # df['barei_peak_distance'] = abs(df['barei'] - 4)  # 4歳をピークと仮定（既存）
    # X['barei_peak_distance'] = df['barei_peak_distance']
    
    # 3歳短距離ピーク（早熟型）
    # df['barei_peak_short'] = abs(df['barei'] - 3)
    # X['barei_peak_short'] = df['barei_peak_short']
    
    # # 5歳長距離ピーク（晩成型）
    # df['barei_peak_long'] = abs(df['barei'] - 5)
    # X['barei_peak_long'] = df['barei_peak_long']

    # 5. 枠番バイアススコア（枠番の歴史的優位性を数値化）
    # 枠番別の歴史的着順分布を計算
    wakuban_stats = df.groupby('wakuban').agg({
        'kakutei_chakujun_numeric': ['mean', 'std', 'count']
    }).round(4)
    wakuban_stats.columns = ['waku_avg_rank', 'waku_std_rank', 'waku_count']
    wakuban_stats = wakuban_stats.reset_index()
    
    # 全体平均からの偏差でバイアススコアを計算
    overall_avg_rank = df['kakutei_chakujun_numeric'].mean()
    wakuban_stats['wakuban_bias_score'] = (overall_avg_rank - wakuban_stats['waku_avg_rank']) / wakuban_stats['waku_std_rank']
    wakuban_stats['wakuban_bias_score'] = wakuban_stats['wakuban_bias_score'].fillna(0)  # NaNを0で埋める
    
    # DataFrameにマージ
    df = df.merge(wakuban_stats[['wakuban', 'wakuban_bias_score']], on='wakuban', how='left')
    # X['wakuban_bias_score'] = df['wakuban_bias_score']

    # レース内での馬番相対位置（頭数による正規化）
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # 研究用特徴量 追加
    # 斤量偏差値（レース内で標準化）
    # レース内の平均と標準偏差を計算して、各馬の斤量がどれくらい重い/軽いかを表現
    race_group = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['futan_juryo']
    df['futan_mean'] = race_group.transform('mean')
    df['futan_std'] = race_group.transform('std')
    
    # 標準偏差が0の場合（全頭同じ斤量）は0にする
    df['futan_zscore'] = np.where(
        df['futan_std'] > 0,
        (df['futan_juryo'] - df['futan_mean']) / df['futan_std'],
        0
    )
    X['futan_zscore'] = df['futan_zscore']
    
    # レース内での斤量順位（パーセンタイル）
    # 0.0=最軽量、1.0=最重量
    df['futan_percentile'] = race_group.transform(lambda x: x.rank(pct=True))
    X['futan_percentile'] = df['futan_percentile']

    # 🔥新機能: 距離適性スコアを追加（3種類）🔥
    # model_creator.pyと同じ処理を実行
    
    # 距離カテゴリ分類関数
    def categorize_distance(kyori):
        """距離を4カテゴリに分類"""
        if kyori <= 1400:
            return 'short'  # 短距離
        elif kyori <= 1800:
            return 'mile'   # マイル
        elif kyori <= 2400:
            return 'middle' # 中距離
        else:
            return 'long'   # 長距離
    
    # 今回のレースの距離カテゴリを追加
    df['distance_category'] = df['kyori'].apply(categorize_distance)
    
    # 🔥重要: 馬場情報も先に追加（df_sortedで使うため）🔥
    # 芝/ダート分類関数
    def categorize_surface(track_code):
        """トラックコードから芝/ダートを判定"""
        track_code_int = int(track_code)
        if 10 <= track_code_int <= 22:
            return 'turf'
        elif 23 <= track_code_int <= 24:
            return 'dirt'
        else:
            return 'unknown'
    
    # 馬場状態分類関数
    def categorize_baba_condition(baba_code):
        """馬場状態コードを分類"""
        if baba_code == 1:
            return 'good'
        elif baba_code == 2:
            return 'slightly'
        elif baba_code == 3:
            return 'heavy'
        elif baba_code == 4:
            return 'bad'
        else:
            return 'unknown'
    
    # 今回のレースの馬場情報を追加
    df['surface_type'] = df['track_code'].apply(categorize_surface)
    df['baba_condition'] = df['babajotai_code'].apply(categorize_baba_condition)
    
    # 時系列順にソート（馬ごとに過去データを参照するため）
    df_sorted = df.sort_values(['ketto_toroku_bango', 'kaisai_nen', 'kaisai_tsukihi']).copy()
    
    # 1️⃣ 距離カテゴリ別適性スコア
    def calc_distance_category_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_category = group.iloc[idx]['distance_category']
            past_same_category = group.iloc[:idx][
                group.iloc[:idx]['distance_category'] == current_category
            ].tail(5)
            
            if len(past_same_category) > 0:
                avg_score = (1 - (past_same_category['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_category_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_category_score
    ).values
    
    # 2️⃣ 近似距離での成績
    def calc_similar_distance_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_kyori = group.iloc[idx]['kyori']
            past_similar = group.iloc[:idx][
                abs(group.iloc[:idx]['kyori'] - current_kyori) <= 200
            ].tail(10)
            
            if len(past_similar) > 0:
                avg_score = (1 - (past_similar['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['similar_distance_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_similar_distance_score
    ).values
    
    # 3️⃣ 距離変化対応力
    def calc_distance_change_adaptability(group):
        scores = []
        for idx in range(len(group)):
            if idx < 2:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            # ✅ 修正: 過去6走分を取得（前走との差分を見るため）
            past_races = group.iloc[max(0, idx-6):idx].copy()
            
            if len(past_races) >= 3:  # ✅ 修正: 最低3走必要（差分2個）
                past_races['kyori_diff'] = past_races['kyori'].diff().abs()
                
                # ✅ 修正: 最新5走のみを評価（最初の1行はNaNなので除外）
                past_races_eval = past_races.tail(5)
                changed_races = past_races_eval[past_races_eval['kyori_diff'] >= 100]
                
                if len(changed_races) > 0:
                    avg_score = (1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # ✅ 修正: 変化なしは中立
            else:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_change_adaptability
    ).values
    
    # 元のインデックス順に戻す
    df = df.copy()
    df['distance_category_score'] = df_sorted.sort_index()['distance_category_score']
    df['similar_distance_score'] = df_sorted.sort_index()['similar_distance_score']
    df['distance_change_adaptability'] = df_sorted.sort_index()['distance_change_adaptability']
    
    # 特徴量に追加
    X['distance_category_score'] = df['distance_category_score']
    X['similar_distance_score'] = df['similar_distance_score']
    # X['distance_change_adaptability'] = df['distance_change_adaptability']

    # 🔥新機能: 馬場適性スコアを追加（3種類）🔥
    # 馬場情報は既にdf_sortedに含まれているので、そのまま使用
    
    # 1️⃣ 芝/ダート別適性スコア
    def calc_surface_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_surface = group.iloc[idx]['surface_type']
            past_same_surface = group.iloc[:idx][
                group.iloc[:idx]['surface_type'] == current_surface
            ].tail(10)
            
            if len(past_same_surface) > 0:
                avg_score = (1 - (past_same_surface['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['surface_aptitude_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_surface_score
    ).values
    
    # 2️⃣ 馬場状態別適性スコア
    def calc_baba_condition_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_condition = group.iloc[idx]['baba_condition']
            past_same_condition = group.iloc[:idx][
                group.iloc[:idx]['baba_condition'] == current_condition
            ].tail(10)
            
            if len(past_same_condition) > 0:
                avg_score = (1 - (past_same_condition['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_condition_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_condition_score
    ).values
    
    # 3️⃣ 馬場変化対応力
    def calc_baba_change_adaptability(group):
        scores = []
        for idx in range(len(group)):
            if idx < 2:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            # ✅ 修正: 過去6走分を取得（前走との変化を見るため）
            past_races = group.iloc[max(0, idx-6):idx].copy()
            
            if len(past_races) >= 3:  # ✅ 修正: 最低3走必要
                past_races['baba_changed'] = past_races['baba_condition'].shift(1) != past_races['baba_condition']
                
                # ✅ 修正: 最新5走のみを評価
                past_races_eval = past_races.tail(5)
                changed_races = past_races_eval[past_races_eval['baba_changed'] == True]
                
                if len(changed_races) > 0:
                    avg_score = (1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # ✅ 修正: 変化なしは中立
            else:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_change_adaptability
    ).values
    
    # 元のインデックス順に戻す
    df['surface_aptitude_score'] = df_sorted.sort_index()['surface_aptitude_score']
    df['baba_condition_score'] = df_sorted.sort_index()['baba_condition_score']
    df['baba_change_adaptability'] = df_sorted.sort_index()['baba_change_adaptability']
    
    # 特徴量に追加
    X['surface_aptitude_score'] = df['surface_aptitude_score']
    # X['baba_condition_score'] = df['baba_condition_score']
    X['baba_change_adaptability'] = df['baba_change_adaptability']

    # 🔥新機能: 騎手・調教師の動的能力スコアを追加（4種類）🔥
    # model_creator.pyと完全に同じロジック
    
    # ✅ 修正: race_bangoを追加して時系列リークを防止
    df_sorted_kishu = df.sort_values(['kishu_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    # 1️⃣ 騎手の実力補正スコア（期待着順との差分、直近3ヶ月）
    def calc_kishu_skill_adjusted_score(group):
        """騎手の純粋な技術を評価（馬の実力を補正）"""
        scores = []
        
        for idx in range(len(group)):
            # 騎手コードがない場合はスキップ
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            
            # 3ヶ月前の日付
            three_months_ago = current_date - pd.DateOffset(months=3)
            
            # 過去3ヶ月のレースを抽出（未来のデータは見ない！）
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                recent_races = past_races[past_races['kaisai_date'] >= three_months_ago]
                
                if len(recent_races) >= 3:  # 最低3レース必要
                    # ✅ 修正: 騎手の純粋な成績を評価（馬の実力補正ではなく、騎手の平均成績）
                    # 着順をスコア化（1着=1.0, 18着=0.0）
                    recent_races['rank_score'] = 1.0 - ((18 - recent_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                    
                    # 騎手の平均スコアを計算
                    avg_score = recent_races['rank_score'].mean()
                    
                    # 0-1の範囲にクリップ（既に範囲内だが念のため）
                    normalized_score = max(0.0, min(1.0, avg_score))
                    
                    scores.append(normalized_score)
                else:
                    scores.append(0.5)  # データ不足は中立
            else:
                scores.append(0.5)  # 初回は中立
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_skill_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_skill_adjusted_score
    ).values
    
    # 2️⃣ 騎手の人気差スコア（オッズ補正、直近3ヶ月）
    def calc_kishu_popularity_adjusted_score(group):
        """騎手の人気補正スコア（人気より上位に来れるか）"""
        scores = []
        
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            
            three_months_ago = current_date - pd.DateOffset(months=3)
            
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                recent_races = past_races[past_races['kaisai_date'] >= three_months_ago]
                
                if len(recent_races) >= 3:
                    # オッズが0や異常値の場合を除外
                    valid_races = recent_races[recent_races['tansho_odds'] > 0]
                    
                    if len(valid_races) >= 3:
                        # ✅ 修正: オッズベースの期待成績と実際の成績を比較
                        # オッズが低い = 期待値が高い（1に近い）
                        # オッズが高い = 期待値が低い（0に近い）
                        max_odds = valid_races['tansho_odds'].max()
                        valid_races['odds_expectation'] = 1.0 - (valid_races['tansho_odds'] / (max_odds + 1.0))
                        
                        # 実際の成績スコア
                        valid_races['actual_score'] = 1.0 - ((18 - valid_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                        
                        # 期待を上回った度合い（プラスなら期待以上）
                        valid_races['performance_diff'] = valid_races['actual_score'] - valid_races['odds_expectation']
                        
                        # 平均差分をスコア化（0.5が中立）
                        avg_diff = valid_races['performance_diff'].mean()
                        normalized_score = 0.5 + (avg_diff * 0.5)  # ±0.5の範囲に収める
                        normalized_score = max(0.0, min(1.0, normalized_score))
                        
                        scores.append(normalized_score)
                    else:
                        scores.append(0.5)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_popularity_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_popularity_adjusted_score
    ).values
    
    # 3️⃣ 騎手の芝/ダート別スコア（馬場適性考慮、直近6ヶ月）
    def calc_kishu_surface_score(group):
        """騎手の馬場タイプ別直近6ヶ月成績"""
        scores = []
        
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            current_surface = group.iloc[idx]['surface_type']
            
            six_months_ago = current_date - pd.DateOffset(months=6)
            
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                # 同じ馬場タイプでの直近6ヶ月
                recent_same_surface = past_races[
                    (past_races['kaisai_date'] >= six_months_ago) &
                    (past_races['surface_type'] == current_surface)
                ]
                
                if len(recent_same_surface) >= 5:  # 最低5レース必要
                    avg_score = (1 - ((18 - recent_same_surface['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_surface_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_surface_score
    ).values
    
    # ✅ 修正: race_bangoを追加して時系列リークを防止
    df_sorted_chokyoshi = df.sort_values(['chokyoshi_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    # 4️⃣ 調教師の直近3ヶ月成績スコア
    def calc_chokyoshi_recent_score(group):
        """調教師の直近3ヶ月成績"""
        scores = []
        
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['chokyoshi_code']) or group.iloc[idx]['chokyoshi_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            
            three_months_ago = current_date - pd.DateOffset(months=3)
            
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                recent_races = past_races[past_races['kaisai_date'] >= three_months_ago]
                
                if len(recent_races) >= 5:  # ✅ 修正: 5レースに変更（10レースでは大部分が中立値になる）
                    avg_score = (1 - ((18 - recent_races['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_chokyoshi['chokyoshi_recent_score'] = df_sorted_chokyoshi.groupby('chokyoshi_code', group_keys=False).apply(
        calc_chokyoshi_recent_score
    ).values
    
    # 元のインデックス順に戻す
    df['kishu_skill_score'] = df_sorted_kishu.sort_index()['kishu_skill_score']
    df['kishu_popularity_score'] = df_sorted_kishu.sort_index()['kishu_popularity_score']
    df['kishu_surface_score'] = df_sorted_kishu.sort_index()['kishu_surface_score']
    df['chokyoshi_recent_score'] = df_sorted_chokyoshi.sort_index()['chokyoshi_recent_score']
    
    # 特徴量に追加
    X['kishu_skill_score'] = df['kishu_skill_score']
    X['kishu_popularity_score'] = df['kishu_popularity_score']
    X['kishu_surface_score'] = df['kishu_surface_score']
    X['chokyoshi_recent_score'] = df['chokyoshi_recent_score']

    # 過去レースで「人気薄なのに好走した回数」
    # df['upset_count'] = df.groupby('ketto_toroku_bango').apply(
    #     lambda g: ((g['tansho_ninkijun_numeric'] >= 5) & (g['kakutei_chakujun_numeric'] <= 3)).sum()
    # )
    # X['upset_count'] = df['upset_count']

    # # 研究用特徴量 追加

    # カテゴリ変数を作成
    # X['kyori'] = X['kyori'].astype('category')
    # X['tenko_code'] = X['tenko_code'].astype('category')
    # X['babajotai_code'] = X['babajotai_code'].astype('category')
    # X['seibetsu_code'] = X['seibetsu_code'].astype('category')

    # モデルをロード
    try:
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        print(f"❌ モデルファイル {model_filename} が見つかりません。")
        return None, None, 0

    # シグモイド関数を定義
    def sigmoid(x):
        """値を0-1の範囲に収めるよ～"""
        return 1 / (1 + np.exp(-x))

    # 予測を実行して、シグモイド関数で変換
    raw_scores = model.predict(X)
    df['predicted_chakujun_score'] = sigmoid(raw_scores)

    # データをソート
    df = df.sort_values(by=['kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'umaban'], ascending=True)

    # グループ内でのスコア順位を計算
    df['score_rank'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['predicted_chakujun_score'].rank(method='min', ascending=False)

    # kakutei_chakujun_numeric と score_rank を整数に変換
    df['kakutei_chakujun_numeric'] = df['kakutei_chakujun_numeric'].fillna(0).astype(int)
    df['tansho_ninkijun_numeric'] = df['tansho_ninkijun_numeric'].fillna(0).astype(int)
    df['score_rank'] = df['score_rank'].fillna(0).astype(int)
    
    # surface_type列を追加（芝・ダート区分）
    from keiba_constants import get_surface_name
    df['surface_type_name'] = get_surface_name(surface_type)

    # 必要な列を選択
    output_columns = ['keibajo_name',
                      'kaisai_nen', 
                      'kaisai_tsukihi', 
                      'race_bango',
                      'surface_type_name',
                      'kyori',
                      'umaban', 
                      'bamei', 
                      'tansho_odds', 
                      'tansho_ninkijun_numeric', 
                      'kakutei_chakujun_numeric', 
                      'score_rank', 
                      'predicted_chakujun_score',
                      '複勝1着馬番',
                      '複勝1着オッズ',
                      '複勝1着人気',
                      '複勝2着馬番',
                      '複勝2着オッズ',
                      '複勝2着人気',
                      '複勝3着馬番',
                      '複勝3着オッズ',
                      '複勝3着人気',
                      '馬連馬番1',
                      '馬連馬番2',
                      '馬連オッズ',
                      'ワイド1_2馬番1',
                      'ワイド1_2馬番2',
                      'ワイド2_3着馬番1',
                      'ワイド2_3着馬番2',
                      'ワイド1_3着馬番1',
                      'ワイド1_3着馬番2',
                      'ワイド1_2オッズ',
                      'ワイド2_3オッズ',
                      'ワイド1_3オッズ',
                      '馬単馬番1',
                      '馬単馬番2',
                      '馬単オッズ',
                      '３連複オッズ',]
    output_df = df[output_columns]

    # 列名を変更
    output_df = output_df.rename(columns={
        'keibajo_name': '競馬場',
        'kaisai_nen': '開催年',
        'kaisai_tsukihi': '開催日',
        'race_bango': 'レース番号',
        'surface_type_name': '芝ダ区分',
        'kyori': '距離',
        'umaban': '馬番',
        'bamei': '馬名',
        'tansho_odds': '単勝オッズ',
        'tansho_ninkijun_numeric': '人気順',
        'kakutei_chakujun_numeric': '確定着順',
        'score_rank': '予測順位',
        'predicted_chakujun_score': '予測スコア'
    })

    # 正しいレース数の計算方法はこれ～！
    race_count = len(output_df.groupby(['開催年', '開催日', 'レース番号']))

    # 的中率・回収率計算（元のtest.pyから移植）
    # 単勝の的中率と回収率
    tansho_hit = (output_df['確定着順'] == 1) & (output_df['予測順位'] == 1)
    tansho_hitrate = 100 * tansho_hit.sum() / race_count
    tansho_recoveryrate = 100 * (tansho_hit * output_df['単勝オッズ']).sum() / race_count

    # 複勝の的中率と回収率
    fukusho_hit = (output_df['確定着順'].isin([1, 2, 3])) & (output_df['予測順位'].isin([1, 2, 3]))
    fukusho_hitrate = fukusho_hit.sum() / (race_count * 3) * 100

    # 的中馬だけ取り出す
    hit_rows = output_df[fukusho_hit].copy()

    def extract_odds(row):
        if row['確定着順'] == 1:
            return row['複勝1着オッズ']
        elif row['確定着順'] == 2:
            return row['複勝2着オッズ']
        elif row['確定着順'] == 3:
            return row['複勝3着オッズ']
        else:
            return 0

    # 的中馬に対応する払戻を計算（100円賭けたとして）
    hit_rows['的中オッズ'] = hit_rows.apply(extract_odds, axis=1)
    total_payout = (hit_rows['的中オッズ'] * 100).sum()

    # 総購入額（毎レースで3頭に100円ずつ）
    total_bet = race_count * 3 * 100
    fukusho_recoveryrate = total_payout / total_bet * 100

    # 馬連の的中率と回収率
    umaren_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: set([1, 2]).issubset(set(x.sort_values('予測スコア', ascending=False).head(2)['確定着順'].values))
    )
    umaren_hitrate = 100 * umaren_hit.sum() / race_count
    umaren_recoveryrate = 100 * (umaren_hit * output_df.groupby(['開催年', '開催日', 'レース番号'])['馬連オッズ'].first()).sum() / race_count

    # ワイド的中率・回収率も計算（省略して簡略化）
    wide_hitrate = 0  # 計算が複雑なので省略
    wide_recoveryrate = 0

    # 馬単の的中率と回収率
    umatan_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: list(x.sort_values('予測スコア', ascending=False).head(2)['確定着順'].values) == [1, 2]
    )
    umatan_hitrate = 100 * umatan_hit.sum() / race_count
    
    umatan_odds_sum = 0
    for name, race_group in output_df.groupby(['開催年', '開催日', 'レース番号']):
        top_horses = race_group.sort_values('予測スコア', ascending=False).head(2)
        if list(top_horses['確定着順'].values) == [1, 2]:
            umatan_odds_sum += race_group['馬単オッズ'].iloc[0]

    umatan_recoveryrate = 100 * umatan_odds_sum / race_count

    # 三連複の的中率と回収率
    sanrenpuku_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: set([1, 2, 3]).issubset(set(x.sort_values('予測スコア', ascending=False).head(3)['確定着順'].values))
    )
    sanrenpuku_hitrate = 100 * sanrenpuku_hit.sum() / len(sanrenpuku_hit)
    sanrenpuku_recoveryrate = 100 * (sanrenpuku_hit * output_df.groupby(['開催年', '開催日', 'レース番号'])['３連複オッズ'].first()).sum() / len(sanrenpuku_hit)

    # 結果をデータフレームにまとめる
    summary_df = pd.DataFrame({
        '的中数': [tansho_hit.sum(), fukusho_hit.sum(), umaren_hit.sum(), 0, umatan_hit.sum(), sanrenpuku_hit.sum()],
        '的中率(%)': [tansho_hitrate, fukusho_hitrate, umaren_hitrate, wide_hitrate, umatan_hitrate, sanrenpuku_hitrate],
        '回収率(%)': [tansho_recoveryrate, fukusho_recoveryrate, umaren_recoveryrate, wide_recoveryrate, umatan_recoveryrate, sanrenpuku_recoveryrate]
    }, index=['単勝', '複勝', '馬連', 'ワイド', '馬単', '３連複'])

    return output_df, summary_df, race_count


def test_multiple_models(test_year_start=2023, test_year_end=2023):
    """
    複数のモデルをテストして結果を比較する関数(設定はJSONファイルから読み込み)
    
    Args:
        test_year_start (int): テスト対象開始年 (デフォルト: 2023)
        test_year_end (int): テスト対象終了年 (デフォルト: 2023)
    """
    
    # JSONファイルから全モデル設定を読み込み
    try:
        model_configs = get_all_models()
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    if not model_configs:
        print("⚠️  テスト対象のモデル設定が見つかりませんでした。")
        return
    
    print("🏇 複数モデルテストを開始します！")
    print("=" * 60)
    
    all_results = {}
    # 統合ファイルの初回書き込みフラグ
    first_unified_write = True
    
    for i, config in enumerate(model_configs, 1):
        model_filename = config['model_filename']
        description = config.get('description', f"モデル{i}")
        
        print(f"\n【{i}/{len(model_configs)}】 {description} モデルをテスト中...")
        print(f"📁 モデルファイル: {model_filename}")
        
        # モデルファイルの存在確認（modelsフォルダも確認）
        model_path = model_filename
        if not os.path.exists(model_path):
            models_path = f"models/{model_filename}"
            if os.path.exists(models_path):
                model_path = models_path
                print(f"📂 modelsフォルダ内のファイルを使用: {models_path}")
            else:
                print(f"⚠️  モデルファイル {model_filename} が見つかりません。スキップします。")
                print(f"    確認場所: ./{model_filename}, ./models/{model_filename}")
                continue
        
        try:
            output_df, summary_df, race_count = predict_with_model(
                model_filename=model_path,  # 存在確認済みのパスを使用
                track_code=config['track_code'],
                kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
                surface_type=config['surface_type'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                test_year_start=test_year_start,
                test_year_end=test_year_end
            )
            
            if output_df is not None:
                # 結果を保存（追記モード）
                base_filename = model_filename.replace('.sav', '').replace('models/', '')
                individual_output_file = f"predicted_results_{base_filename}.tsv"
                summary_file = f"betting_summary_{base_filename}.tsv"
                
                # 個別モデル結果を追記保存
                save_results_with_append(output_df, individual_output_file, append_mode=True)
                
                # 全モデル統合ファイルに保存（初回は上書き、以降は追記）
                unified_output_file = "predicted_results.tsv"
                save_results_with_append(output_df, unified_output_file, append_mode=not first_unified_write)
                first_unified_write = False  # 初回書き込み完了
                
                # サマリーは個別ファイルに保存
                results_dir = Path('results')
                results_dir.mkdir(exist_ok=True)
                summary_filepath = results_dir / summary_file
                summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig')
                
                print(f"✅ 完了！レース数: {race_count}")
                print(f"  - 個別結果: {individual_output_file}")
                print(f"  - 統合結果: {unified_output_file}")
                print(f"  - サマリー: {summary_file}")
                
                # 結果を保存（後で比較用）
                all_results[description] = {
                    'summary': summary_df,
                    'race_count': race_count,
                    'model_filename': model_filename
                }
                
                # 主要な結果を表示
                print(f"  - 単勝的中率: {summary_df.loc['単勝', '的中率(%)']:.2f}%")
                print(f"  - 単勝回収率: {summary_df.loc['単勝', '回収率(%)']:.2f}%")
                print(f"  - 複勝的中率: {summary_df.loc['複勝', '的中率(%)']:.2f}%")
                print(f"  - 複勝回収率: {summary_df.loc['複勝', '回収率(%)']:.2f}%")
                
            else:
                print(f"❌ テストデータが見つかりませんでした。")
                
        except Exception as e:
            print(f"❌ エラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # 複数モデルの比較結果を作成
    if len(all_results) > 1:
        print("\n📊 モデル比較結果")
        print("=" * 60)
        
        comparison_data = []
        for description, result in all_results.items():
            summary = result['summary']
            comparison_data.append({
                'モデル': description,
                'レース数': result['race_count'],
                '単勝的中率': f"{summary.loc['単勝', '的中率(%)']:.2f}%",
                '単勝回収率': f"{summary.loc['単勝', '回収率(%)']:.2f}%",
                '複勝的中率': f"{summary.loc['複勝', '的中率(%)']:.2f}%",
                '複勝回収率': f"{summary.loc['複勝', '回収率(%)']:.2f}%",
                '三連複的中率': f"{summary.loc['３連複', '的中率(%)']:.2f}%",
                '三連複回収率': f"{summary.loc['３連複', '回収率(%)']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 比較結果を保存
        comparison_file = 'model_comparison.tsv'
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        comparison_filepath = results_dir / comparison_file
        
        comparison_df.to_csv(comparison_filepath, index=False, sep='\t', encoding='utf-8-sig')
        
        print(comparison_df.to_string(index=False))
        print(f"\n📋 比較結果を {comparison_filepath} に保存しました！")
    
    print("\n🏁 すべてのテストが完了しました！")


def predict_and_save_results():
    """
    旧バージョン互換性のための関数
    阪神競馬場の３歳以上芝中長距離モデルでテスト
    """
    output_df, summary_df, race_count = predict_with_model(
        model_filename='hanshin_shiba_3ageup_model.sav',
        track_code='09',  # 阪神
        kyoso_shubetsu_code='13',  # 3歳以上
        surface_type='turf',  # 芝
        min_distance=1700,  # 中長距離
        max_distance=9999  # 上限なし
    )
    
    if output_df is not None:
        # resultsディレクトリを作成
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 結果をTSVに保存（追記モード）
        output_file = 'predicted_results.tsv'
        save_results_with_append(output_df, output_file, append_mode=True)
        print(f"予測結果を results/{output_file} に保存しました！")

        # 的中率と回収率を別ファイルに保存
        summary_file = 'betting_summary.tsv'
        summary_filepath = results_dir / summary_file
        summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig')
        print(f"的中率・回収率・的中数を results/{summary_file} に保存しました！")


if __name__ == '__main__':
    # 実行方法を選択できるように
    import sys
    
    # デフォルトのテスト年範囲
    test_year_start = 2023
    test_year_end = 2023
    
    # コマンドライン引数を解析
    mode = 'single'  # デフォルトは単一モデルテスト
    
    for arg in sys.argv[1:]:
        if arg == 'multi':
            mode = 'multi'
        elif '-' in arg and arg[0].isdigit():
            # "2020-2023" 形式の年範囲指定
            try:
                years = arg.split('-')
                if len(years) == 2:
                    test_year_start = int(years[0])
                    test_year_end = int(years[1])
                    print(f"📅 テスト年範囲指定: {test_year_start}年~{test_year_end}年")
            except ValueError:
                print(f"⚠️  無効な年範囲フォーマット: {arg} (例: 2020-2023)")
        elif arg.isdigit() and len(arg) == 4:
            # "2023" 形式の単一年指定
            test_year_start = test_year_end = int(arg)
            print(f"📅 テスト年指定: {test_year_start}年")
    
    if mode == 'multi':
        # python universal_test.py multi [年範囲]
        test_multiple_models(test_year_start=test_year_start, test_year_end=test_year_end)
    else:
        # python universal_test.py [年範囲] (デフォルト)
        # 単一モデルテストで年範囲を使用
        output_df, summary_df, race_count = predict_with_model(
            model_filename='hanshin_shiba_3ageup_model.sav',
            track_code='09',  # 阪神
            kyoso_shubetsu_code='13',  # 3歳以上
            surface_type='turf',  # 芝
            min_distance=1700,  # 中長距離
            max_distance=9999,  # 上限なし
            test_year_start=test_year_start,
            test_year_end=test_year_end
        )
        
        if output_df is not None:
            # resultsディレクトリを作成
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # 結果をTSVに保存（追記モード）
            output_file = 'predicted_results.tsv'
            save_results_with_append(output_df, output_file, append_mode=True)
            print(f"予測結果を results/{output_file} に保存しました!")

            # 的中率と回収率を別ファイルに保存
            summary_file = 'betting_summary.tsv'
            summary_filepath = results_dir / summary_file
            summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig')
            print(f"的中率・回収率・的中数を results/{summary_file} に保存しました!")