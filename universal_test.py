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

# Phase 1: 期待値・ケリー基準・信頼度スコアの統合
from expected_value_calculator import ExpectedValueCalculator
from kelly_criterion import KellyCriterion
from race_confidence_scorer import RaceConfidenceScorer


def add_purchase_logic(
    output_df: pd.DataFrame,
    prediction_rank_max: int = 3,
    popularity_rank_max: int = 3,
    min_odds: float = 1.5,
    max_odds: float = 20.0,
    min_score_diff: float = 0.05,
    initial_bankroll: float = 1000000,
    bet_unit: int = 1000
) -> pd.DataFrame:
    """
    予測結果に購入判断・購入額を追加 (新戦略: 本命×予測上位フィルター)
    
    Phase 1新戦略:
    - 予測順位1-3位 AND 人気順1-3位 のみ対象
    - オッズ範囲でフィルタリング (1.5倍～20倍)
    - 予測スコア差が一定以上のレースのみ (本命が明確)
    - 一律ベット (シンプル&確実)
    
    Args:
        output_df (DataFrame): 予測結果データフレーム
        prediction_rank_max (int): 予測順位の上限 (デフォルト: 3)
        popularity_rank_max (int): 人気順の上限 (デフォルト: 3)
        min_odds (float): 最低オッズ (デフォルト: 1.5倍)
        max_odds (float): 最高オッズ (デフォルト: 20倍)
        min_score_diff (float): 予測1位と2位のスコア差の最小値 (デフォルト: 0.05)
        initial_bankroll (float): 初期資金 (デフォルト: 100万円)
        bet_unit (int): 1頭あたりの購入額 (デフォルト: 1000円)
        
    Returns:
        DataFrame: 購入ロジックが追加されたデータフレーム
    """
    df = output_df.copy()
    
    # カラム名をマッピング (日本語 → 英語)
    df_work = df.rename(columns={
        '開催年': 'kaisai_year',
        '開催日': 'kaisai_date',
        '競馬場': 'keibajo_code',
        'レース番号': 'race_number',
        '馬番': 'umaban_numeric',
        '予測順位': 'predicted_rank',
        '予測スコア': 'predicted_score',
        '人気順': 'popularity_rank',
        '単勝オッズ': 'tansho_odds',
        '確定着順': 'chakujun_numeric'
    })
    
    # レースごとにグループ化して処理
    race_groups = df_work.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
    
    all_races = []
    current_bankroll = initial_bankroll
    total_purchased = 0
    total_wins = 0
    
    for race_id, race_df in race_groups:
        race_df = race_df.copy()
        
        # 予測スコアでソート(降順)
        race_df_sorted = race_df.sort_values('predicted_score', ascending=False).reset_index(drop=True)
        
        # 予測1位と2位のスコア差を計算
        if len(race_df_sorted) >= 2:
            score_diff = race_df_sorted.iloc[0]['predicted_score'] - race_df_sorted.iloc[1]['predicted_score']
        else:
            score_diff = 0
        
        # 全馬にレース情報を追加
        race_df['score_diff'] = score_diff
        race_df['skip_reason'] = None
        
        # フィルター1: 予測スコア差が小さいレースはスキップ
        if score_diff < min_score_diff:
            race_df['購入推奨'] = False
            race_df['購入額'] = 0
            race_df['skip_reason'] = 'low_score_diff'
            all_races.append(race_df)
            continue
        
        # フィルター2: 予測順位 AND 人気順 AND オッズ範囲
        race_df['購入推奨'] = (
            (race_df['predicted_rank'] <= prediction_rank_max) &
            (race_df['popularity_rank'] <= popularity_rank_max) &
            (race_df['tansho_odds'] >= min_odds) &
            (race_df['tansho_odds'] <= max_odds)
        )
        
        # スキップ理由を記録
        race_df.loc[~race_df['購入推奨'] & (race_df['predicted_rank'] > prediction_rank_max), 'skip_reason'] = 'low_predicted_rank'
        race_df.loc[~race_df['購入推奨'] & (race_df['popularity_rank'] > popularity_rank_max), 'skip_reason'] = 'low_popularity'
        race_df.loc[~race_df['購入推奨'] & (race_df['tansho_odds'] < min_odds), 'skip_reason'] = 'odds_too_low'
        race_df.loc[~race_df['購入推奨'] & (race_df['tansho_odds'] > max_odds), 'skip_reason'] = 'odds_too_high'
        
        # 購入推奨馬を抽出
        buy_horses = race_df[race_df['購入推奨']].copy()
        
        # 購入額列を初期化
        race_df['購入額'] = 0
        
        if len(buy_horses) > 0:
            # 一律ベット
            total_purchased += len(buy_horses)
            
            # 資金を更新
            total_bet = bet_unit * len(buy_horses)
            total_return = 0
            
            for idx in buy_horses.index:
                race_df.loc[idx, '購入額'] = bet_unit
                if race_df.loc[idx, 'chakujun_numeric'] == 1:
                    total_return += bet_unit * race_df.loc[idx, 'tansho_odds']
                    total_wins += 1
            
            current_bankroll = current_bankroll - total_bet + total_return
        
        # 現在の資金残高を記録
        race_df['現在資金'] = current_bankroll
        
        all_races.append(race_df)
    
    # 全レースを統合
    df_integrated = pd.concat(all_races, ignore_index=True)
    
    # カラム名を日本語に戻す(英語から日本語へ)
    df_integrated = df_integrated.rename(columns={
        'kaisai_year': '開催年',
        'kaisai_date': '開催日',
        'keibajo_code': '競馬場',
        'race_number': 'レース番号',
        'umaban_numeric': '馬番',
        'predicted_rank': '予測順位',
        'predicted_score': '予測スコア',
        'popularity_rank': '人気順',
        'tansho_odds': '単勝オッズ',
        'chakujun_numeric': '確定着順',
        'score_diff': 'スコア差',
        'skip_reason': 'スキップ理由'
    })
    
    return df_integrated


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
        print(f"[NOTE] 既存ファイルに追記: {filepath}")
        df.to_csv(filepath, mode='a', header=False, index=False, sep='\t', encoding='utf-8-sig')
    else:
        # ファイルが存在しない場合は新規作成（ヘッダーあり）
        print(f"[LIST] 新規ファイル作成: {filepath}")
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
        seum.corner_1,
        seum.corner_2,
        seum.corner_3,
        seum.corner_4,
        seum.kyakushitsu_hantei,
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
        -- 馬体重関連の特徴量
        ,nullif(cast(seum.bataiju as integer), 0) as bataiju_current
        ,CASE 
            WHEN seum.zogen_fugo = '-' THEN -1 * nullif(cast(seum.zogen_sa as integer), 0)
            ELSE nullif(cast(seum.zogen_sa as integer), 0)
        END as bataiju_change
        -- 前走の馬体重
        ,LAG(nullif(cast(seum.bataiju as integer), 0)) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
        ) as bataiju_prev
        -- 過去3走の平均馬体重
        ,AVG(nullif(cast(seum.bataiju as integer), 0)) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as bataiju_avg_3races
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
                , se.bataiju
                , se.zogen_fugo
                , se.zogen_sa
                , se.kohan_3f
                , se.soha_time
                , se.time_sa
                , se.corner_1
                , se.corner_2
                , se.corner_3
                , se.corner_4
                , se.kyakushitsu_hantei
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
    print(f"[NOTE] テスト用SQLをログファイルに出力: {log_filepath}")

    # データを取得
    df = pd.read_sql_query(sql=sql, con=conn)
    conn.close()
    
    if len(df) == 0:
        print(f"[ERROR] {model_filename} に対応するテストデータが見つかりませんでした。")
        return None, None, 0

    print(f"[+] テストデータ件数: {len(df)}件")

    # 修正: データ前処理を適切に実施（model_creator.pyと同じロジック）
    # 騎手コード・調教師コード・馬名などの文字列列を保持したまま、数値列のみを処理
    print("[TEST] データ型確認...")
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
        'time_index', 'past_score', 'kohan_3f_index', 'corner_1', 'corner_2',
        'corner_3', 'corner_4', 'kyakushitsu_hantei'
    ]
    
    # 数値化する列のみ処理（文字列列は保持）
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値を0で埋める（数値列のみ、存在する列のみ処理）
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
    df[existing_numeric_columns] = df[existing_numeric_columns].fillna(0)
    
    # 文字列型の列はそのまま保持（kishu_code, chokyoshi_code, bamei など）
    print(f"  kishu_code型（修正後）: {df['kishu_code'].dtype}")
    print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
    print("[OK] データ前処理完了（文字列列を保持）")

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
    
    # 改善された特徴量
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
    
    # 🐴 馬体重関連の特徴量 (Phase 2: 特徴量強化)
    # 前走からの馬体重変化率 (%)
    df['bataiju_change_rate'] = np.where(
        (df['bataiju_prev'].notna()) & (df['bataiju_prev'] > 0),
        (df['bataiju_current'] - df['bataiju_prev']) / df['bataiju_prev'] * 100,
        0
    )
    X['bataiju_change_rate'] = df['bataiju_change_rate']
    
    # 平均馬体重との差の比率 (%)
    df['bataiju_deviation_rate'] = np.where(
        (df['bataiju_avg_3races'].notna()) & (df['bataiju_avg_3races'] > 0),
        (df['bataiju_current'] - df['bataiju_avg_3races']) / df['bataiju_avg_3races'] * 100,
        0
    )
    X['bataiju_deviation_rate'] = df['bataiju_deviation_rate']
    
    # 馬体重増減の絶対値 (kg) - DB由来
    X['bataiju_change'] = df['bataiju_change'].fillna(0)
    
    # 馬体重トレンド (増加傾向/減少傾向)
    df['bataiju_trend'] = np.where(
        (df['bataiju_change_rate'].notna()),
        np.sign(df['bataiju_change_rate']) * np.log1p(abs(df['bataiju_change_rate'])),
        0
    )
    X['bataiju_trend'] = df['bataiju_trend']

    # 馬番×距離の相互作用（内外枠の距離適性）
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000  # スケール調整
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    # 短距離特化特徴量
    # 枠番×距離の相互作用（短距離ほど内枠有利を数値化）
    # 距離が短いほど枠番の影響が大きい: (2000 - 距離) / 1000 で重み付け
    df['wakuban_kyori_interaction'] = df['wakuban'] * (2000 - df['kyori']) / 1000
    X['wakuban_kyori_interaction'] = df['wakuban_kyori_interaction']
    
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

    # 新機能: 距離適性スコアを追加（3種類）
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
    
    # 重要: 馬場情報も先に追加（df_sortedで使うため）
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
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
                continue
            
            current_category = group.iloc[idx]['distance_category']
            past_same_category = group.iloc[:idx][
                group.iloc[:idx]['distance_category'] == current_category
            ].tail(5)
            
            if len(past_same_category) > 0:
                avg_score = (1 - (past_same_category['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # [OK] 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_category_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_category_score
    ).values
    
    # 2️⃣ 近似距離での成績
    def calc_similar_distance_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
                continue
            
            current_kyori = group.iloc[idx]['kyori']
            past_similar = group.iloc[:idx][
                abs(group.iloc[:idx]['kyori'] - current_kyori) <= 200
            ].tail(10)
            
            if len(past_similar) > 0:
                avg_score = (1 - (past_similar['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # [OK] 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['similar_distance_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_similar_distance_score
    ).values
    
    # 3️⃣ 距離変化対応力
    def calc_distance_change_adaptability(group):
        scores = []
        for idx in range(len(group)):
            if idx < 2:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
                continue
            
            # [OK] 修正: 過去6走分を取得（前走との差分を見るため）
            past_races = group.iloc[max(0, idx-6):idx].copy()
            
            if len(past_races) >= 3:  # [OK] 修正: 最低3走必要（差分2個）
                past_races['kyori_diff'] = past_races['kyori'].diff().abs()
                
                # [OK] 修正: 最新5走のみを評価（最初の1行はNaNなので除外）
                past_races_eval = past_races.tail(5)
                changed_races = past_races_eval[past_races_eval['kyori_diff'] >= 100]
                
                if len(changed_races) > 0:
                    avg_score = (1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # [OK] 修正: 変化なしは中立
            else:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_change_adaptability
    ).values
    
    # 短距離特化: 前走距離差を計算
    def calc_zenso_kyori_sa(group):
        """前走からの距離差を計算（短距離の距離変化影響を評価）"""
        diffs = []
        for idx in range(len(group)):
            if idx == 0:
                diffs.append(0)  # 初回は前走なし
            else:
                current_kyori = group.iloc[idx]['kyori']
                previous_kyori = group.iloc[idx-1]['kyori']
                diffs.append(abs(current_kyori - previous_kyori))
        return pd.Series(diffs, index=group.index)
    
    df_sorted['zenso_kyori_sa'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_zenso_kyori_sa
    ).values
    
    # [NEW] 長距離経験回数（2400m以上のレース経験数）
    def calc_long_distance_experience_count(group):
        """長距離(2400m以上)のレース経験回数をカウント"""
        counts = []
        for idx in range(len(group)):
            if idx == 0:
                counts.append(0)  # 初回は経験なし
            else:
                # 過去のレースで2400m以上を走った回数
                past_long_count = (group.iloc[:idx]['kyori'] >= 2400).sum()
                counts.append(past_long_count)
        return pd.Series(counts, index=group.index)
    
    df_sorted['long_distance_experience_count'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_long_distance_experience_count
    ).values
    
    # 元のインデックス順に戻す
    df = df.copy()
    df['distance_category_score'] = df_sorted.sort_index()['distance_category_score']
    df['similar_distance_score'] = df_sorted.sort_index()['similar_distance_score']
    df['distance_change_adaptability'] = df_sorted.sort_index()['distance_change_adaptability']
    df['zenso_kyori_sa'] = df_sorted.sort_index()['zenso_kyori_sa']
    df['long_distance_experience_count'] = df_sorted.sort_index()['long_distance_experience_count']
    
    # 特徴量に追加
    X['distance_category_score'] = df['distance_category_score']
    X['similar_distance_score'] = df['similar_distance_score']
    # X['distance_change_adaptability'] = df['distance_change_adaptability']
    X['zenso_kyori_sa'] = df['zenso_kyori_sa']
    X['long_distance_experience_count'] = df['long_distance_experience_count']

    # 新機能: スタート指数を追加（第1コーナー通過順位から算出）
    if 'corner_1' in df.columns:
        print("[DONE] スタート指数を計算中...")
        
        def calc_start_index(group):
            """
            過去10走の第1コーナー通過順位からスタート能力を評価
            - 早期位置取り能力（通過順位が良い = スタート良好）
            - 一貫性（標準偏差が小さい = スタート安定）
            """
            scores = []
            for idx in range(len(group)):
                if idx == 0:
                    scores.append(0.5)  # 初回は中立値
                    continue
                
                # 過去10走の第1コーナー通過順位を取得（corner_1は既に数値化済み）
                past_corners = group.iloc[max(0, idx-10):idx]['corner_1'].dropna()
                
                if len(past_corners) >= 3:  # 最低3走必要
                    avg_position = past_corners.mean()
                    std_position = past_corners.std()
                    
                    # スコア計算: 
                    # 1. 通過順位が良い（小さい）ほど高スコア → 1.0 - (avg_position / 18)
                    # 2. 安定性ボーナス: std が小さいほど高評価 → 最大0.2のボーナス
                    position_score = max(0, 1.0 - (avg_position / 18.0))
                    stability_bonus = max(0, 0.2 - (std_position / 10.0))
                    
                    total_score = position_score + stability_bonus
                    scores.append(min(1.0, total_score))  # 最大1.0にクリップ
                else:
                    scores.append(0.5)  # データ不足は中立値
            
            return pd.Series(scores, index=group.index)
        
        df_sorted['start_index'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
            calc_start_index
        ).values
        
        df['start_index'] = df_sorted.sort_index()['start_index']
        X['start_index'] = df['start_index']
        
        print(f"[OK] スタート指数を追加しました！")
        print(f"  - start_index: 過去10走の第1コーナー通過順位から算出（早期位置取り能力+安定性）")
    else:
        print("[!] corner_1データが存在しないため、スタート指数はスキップします")
        # ダミーデータで0.5（中立値）を設定
        df['start_index'] = 0.5
        X['start_index'] = 0.5
    
    # 短距離特化: コーナー通過位置スコア（全コーナーの平均）
    if all(col in df.columns for col in ['corner_1', 'corner_2', 'corner_3', 'corner_4']):
        print("[DONE] コーナー通過位置スコアを計算中...")
        
        def calc_corner_position_score(group):
            """
            過去3走の全コーナー(1-4)通過位置の平均と安定性を計算
            - 位置取りが良い(数値が小さい)ほど高スコア
            - 安定性も評価 → 馬連・ワイドの精度向上
            """
            scores = []
            for idx in range(len(group)):
                if idx < 1:  # 最低1走分のデータが必要
                    scores.append(0.5)
                    continue
                
                # 過去3走を取得
                past_3_races = group.iloc[max(0, idx-2):idx+1]
                
                if len(past_3_races) >= 1:
                    # 各レースの全コーナー平均位置を計算
                    corner_averages = []
                    for _, race in past_3_races.iterrows():
                        corners = []
                        for corner_col in ['corner_1', 'corner_2', 'corner_3', 'corner_4']:
                            corner_val = race[corner_col]
                            if pd.notna(corner_val) and corner_val > 0:
                                corners.append(corner_val)
                        if len(corners) > 0:
                            corner_averages.append(np.mean(corners))
                    
                    if len(corner_averages) > 0:
                        avg_position = np.mean(corner_averages)
                        std_position = np.std(corner_averages) if len(corner_averages) > 1 else 0
                        
                        # スコア計算:
                        # 1. 位置取りスコア: 前方ほど高評価
                        position_score = max(0, 1.0 - (avg_position / 18.0))
                        
                        # 2. 安定性ボーナス: stdが小さいほど高評価 (最大+0.3)
                        stability_bonus = max(0, 0.3 - (std_position / 10.0))
                        
                        # 合計スコア (最大1.0にクリップ)
                        total_score = position_score + stability_bonus
                        scores.append(min(1.0, total_score))
                    else:
                        scores.append(0.5)
                else:
                    scores.append(0.5)
            
            return pd.Series(scores, index=group.index)
        
        df_sorted['corner_position_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
            calc_corner_position_score
        ).values
        
        df['corner_position_score'] = df_sorted.sort_index()['corner_position_score']
        X['corner_position_score'] = df['corner_position_score']
        
        print(f"[OK] コーナー通過位置スコアを追加しました！")
        print(f"  - corner_position_score: 過去3走の全コーナー(1-4)通過位置平均+安定性（ポジショニング能力+安定性）")
    else:
        print("[!] corner_2~4データが存在しないため、コーナー通過位置スコアはスキップします")
        df['corner_position_score'] = 0.5
        X['corner_position_score'] = 0.5

    # 新機能: 馬場適性スコアを追加（3種類）
    # 馬場情報は既にdf_sortedに含まれているので、そのまま使用
    
    # 1️⃣ 芝/ダート別適性スコア
    def calc_surface_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
                continue
            
            current_surface = group.iloc[idx]['surface_type']
            past_same_surface = group.iloc[:idx][
                group.iloc[:idx]['surface_type'] == current_surface
            ].tail(10)
            
            if len(past_same_surface) > 0:
                avg_score = (1 - (past_same_surface['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # [OK] 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['surface_aptitude_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_surface_score
    ).values
    
    # 2️⃣ 馬場状態別適性スコア
    def calc_baba_condition_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
                continue
            
            current_condition = group.iloc[idx]['baba_condition']
            past_same_condition = group.iloc[:idx][
                group.iloc[:idx]['baba_condition'] == current_condition
            ].tail(10)
            
            if len(past_same_condition) > 0:
                avg_score = (1 - (past_same_condition['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # [OK] 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_condition_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_condition_score
    ).values
    
    # 3️⃣ 馬場変化対応力
    def calc_baba_change_adaptability(group):
        scores = []
        for idx in range(len(group)):
            if idx < 2:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
                continue
            
            # [OK] 修正: 過去6走分を取得（前走との変化を見るため）
            past_races = group.iloc[max(0, idx-6):idx].copy()
            
            if len(past_races) >= 3:  # [OK] 修正: 最低3走必要
                past_races['baba_changed'] = past_races['baba_condition'].shift(1) != past_races['baba_condition']
                
                # [OK] 修正: 最新5走のみを評価
                past_races_eval = past_races.tail(5)
                changed_races = past_races_eval[past_races_eval['baba_changed'] == True]
                
                if len(changed_races) > 0:
                    avg_score = (1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # [OK] 修正: 変化なしは中立
            else:
                scores.append(0.5)  # [OK] 修正: データ不足は中立値
        
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

    # 新機能: 騎手・調教師の動的能力スコアを追加（4種類）
    # model_creator.pyと完全に同じロジック
    
    # [OK] 修正: race_bangoを追加して時系列リークを防止
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
                    # [OK] 修正: 騎手の純粋な成績を評価（馬の実力補正ではなく、騎手の平均成績）
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
                        # [OK] 修正: オッズベースの期待成績と実際の成績を比較
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
    
    # [OK] 修正: race_bangoを追加して時系列リークを防止
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
                
                if len(recent_races) >= 5:  # [OK] 修正: 5レースに変更（10レースでは大部分が中立値になる）
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

    # [TARGET] 路面×距離別特徴量選択（SHAP分析結果に基づく最適化）
    print(f"\n[RACE] 路面×距離別特徴量選択を実施...")
    print(f"  路面: {surface_type}, 距離: {min_distance}m 〜 {max_distance}m")
    
    # 路面と距離の組み合わせで特徴量を調整
    is_turf = surface_type.lower() == 'turf'
    is_short = max_distance <= 1600
    is_long = min_distance >= 1700
    
    # 短距離専用特徴量の追加
    if is_short:
        print(f"  [TARGET] 短距離モデル: 短距離特化特徴量を追加")
        # wakuban_kyori_interaction, zenso_kyori_sa, start_index, corner_position_scoreは既にdfとXに追加済み
        # 短距離モデルでのみ使用するため、長距離では削除する
        features_added_short = ['wakuban_kyori_interaction', 'zenso_kyori_sa', 'start_index', 'corner_position_score']
        print(f"    [OK] 短距離特化特徴量: {features_added_short}")
        # 長距離特化特徴量は短距離では不要
        if 'long_distance_experience_count' in X.columns:
            X = X.drop(columns=['long_distance_experience_count'])
            print(f"    [OK] 削除（短距離用）: long_distance_experience_count")
    else:
        # 長距離・中距離モデルでは短距離特化特徴量を削除
        print(f"  [PIN] 中長距離モデル: 短距離特化特徴量を削除")
        features_to_remove_for_long = ['wakuban_kyori_interaction', 'zenso_kyori_sa', 'start_index', 'corner_position_score']
        for feature in features_to_remove_for_long:
            if feature in X.columns:
                X = X.drop(columns=[feature])
                print(f"    [OK] 削除（長距離用）: {feature}")
        # 長距離(2200m以上)ではlong_distance_experience_countを使用
        if min_distance >= 2200:
            print(f"  [TARGET] 長距離モデル: 長距離特化特徴量を使用")
            print(f"    [OK] 長距離特化特徴量: ['long_distance_experience_count']")
        else:
            # 中距離では長距離特化特徴量は不要
            if 'long_distance_experience_count' in X.columns:
                X = X.drop(columns=['long_distance_experience_count'])
                print(f"    [OK] 削除（中距離用）: long_distance_experience_count")
    
    features_to_remove = []
    
    if is_turf and is_long:
        # 🌿 芝中長距離（ベースモデル）: 全特徴量を使用
        print("  [PIN] 芝中長距離（ベースモデル）: 全特徴量を使用")
        print(f"  [OK] これが最も成功しているモデルです!")
    
    elif is_turf and is_short:
        # 🌿 芝短距離: SHAP分析で効果が低い特徴量を削除
        print("  [PIN] 芝短距離: 不要な特徴量を削除")
        features_to_remove = [
            'kohan_3f_index',           # SHAP 0.030 → 後半の脚は短距離では重要度低い
            'surface_aptitude_score',   # SHAP 0.000 → 完全に無意味
            'wakuban_ratio',            # SHAP 0.008 → ほぼ無効
        ]
    
    elif not is_turf and is_long:
        # 🏜️ ダート中長距離: 芝特有の特徴量を調整
        print("  [PIN] ダート中長距離: 芝特有の特徴量を調整")
        # ダートでは芝と異なる特性があるため、必要に応じて特徴量を調整
        # 現時点では全特徴量を使用（今後の分析で調整可能）
        pass
    
    elif not is_turf and is_short:
        # 🏜️ ダート短距離: 芝短距離の調整 + ダート特有の調整
        print("  [PIN] ダート短距離: 芝短距離+ダート特有の調整")
        features_to_remove = [
            'kohan_3f_index',           # 短距離では後半の脚は重要度低い
            'surface_aptitude_score',   # 芝/ダート適性スコアは効果薄
            'wakuban_ratio',            # ダート短距離でも効果薄い可能性
        ]
    
    else:
        # マイル距離など中間
        print("  [PIN] 中間距離モデル: 全特徴量を使用")
    
    # 特徴量の削除実行
    if features_to_remove:
        print(f"  削除する特徴量: {features_to_remove}")
        for feature in features_to_remove:
            if feature in X.columns:
                X = X.drop(columns=[feature])
                print(f"    [OK] 削除: {feature}")
    
    print(f"  最終特徴量数: {len(X.columns)}個")
    print(f"  特徴量リスト: {list(X.columns)}")

    # モデルをロード
    try:
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        print(f"[ERROR] モデルファイル {model_filename} が見つかりません。")
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

    # Phase 1統合: 期待値・ケリー基準・信頼度スコアを追加
    print("[PHASE1] 新購入ロジック(本命×予測上位フィルター)を実行中...")
    try:
        output_df_with_logic = add_purchase_logic(
            output_df,
            prediction_rank_max=3,  # 予測順位1-3位
            popularity_rank_max=3,  # 人気順1-3位
            min_odds=1.5,  # 最低オッズ1.5倍
            max_odds=20.0,  # 最高オッズ20倍
            min_score_diff=0.05,  # 予測スコア差0.05以上
            initial_bankroll=1000000,
            bet_unit=1000  # 一律1000円ベット
        )
        print("[PHASE1] 購入ロジック統合完了!")
        
        # 購入推奨馬の統計
        buy_count = output_df_with_logic['購入推奨'].sum()
        total_bet = output_df_with_logic['購入額'].sum()
        final_bankroll = output_df_with_logic['現在資金'].iloc[-1]
        
        # 的中数を計算
        purchased = output_df_with_logic[output_df_with_logic['購入額'] > 0]
        wins = len(purchased[purchased['確定着順'] == 1])
        hit_rate = (wins / len(purchased) * 100) if len(purchased) > 0 else 0
        
        print(f"[STATS] 購入推奨馬数: {buy_count}")
        print(f"[STATS] 実購入馬数: {len(purchased)}")
        print(f"[STATS] 的中数: {wins}")
        print(f"[STATS] 的中率: {hit_rate:.2f}%")
        print(f"[STATS] 総投資額: {total_bet:,.0f}円")
        print(f"[STATS] 最終資金: {final_bankroll:,.0f}円 (初期: 1,000,000円)")
        print(f"[STATS] 損益: {final_bankroll - 1000000:+,.0f}円")
        
        # 回収率を計算
        if total_bet > 0:
            recovery_rate = (final_bankroll - 1000000 + total_bet) / total_bet * 100
            print(f"[STATS] 回収率: {recovery_rate:.2f}%")
        
        output_df = output_df_with_logic
    except Exception as e:
        print(f"[WARNING] Phase 1統合でエラー発生: {e}")
        print("[WARNING] 従来の予測結果のみ返します")
        import traceback
        traceback.print_exc()

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
        print(f"[ERROR] 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    if not model_configs:
        print("[!] テスト対象のモデル設定が見つかりませんでした。")
        return
    
    print("[RACE] 複数モデルテストを開始します！")
    print("=" * 60)
    
    all_results = {}
    # 統合ファイルの初回書き込みフラグ
    first_unified_write = True
    
    for i, config in enumerate(model_configs, 1):
        base_model_filename = config['model_filename']
        description = config.get('description', f"モデル{i}")
        
        print(f"\n【{i}/{len(model_configs)}】 {description} モデルをテスト中...")
        
        # 年範囲が指定されているモデルファイルを探す
        # 例: tokyo_turf_3ageup_long_2020-2022.sav
        import glob
        base_name = base_model_filename.replace('.sav', '')
        model_pattern = f"models/{base_name}_*-*.sav"
        matching_models = glob.glob(model_pattern)
        
        # マッチするモデルがなければ元のファイル名を使用
        if not matching_models:
            model_filename = base_model_filename
            train_year_range = "unknown"
        else:
            # 最新のモデルを使用（ファイル名でソート）
            model_filename = sorted(matching_models)[-1]
            # ファイル名から学習期間を抽出
            import re
            match = re.search(r'_(\d{4})-(\d{4})\.sav$', model_filename)
            if match:
                train_year_range = f"{match.group(1)}-{match.group(2)}"
            else:
                train_year_range = "unknown"
        
        print(f"[FILE] モデルファイル: {model_filename}")
        if train_year_range != "unknown":
            print(f"[RUN] 学習期間: {train_year_range}")
        
        # モデルファイルの存在確認
        model_path = model_filename
        if not os.path.exists(model_path):
            # modelsフォルダも確認
            models_path = f"models/{base_model_filename}"
            if os.path.exists(models_path):
                model_path = models_path
                train_year_range = "unknown"
                print(f"[DIR] modelsフォルダ内のファイルを使用: {models_path}")
            else:
                print(f"[!] モデルファイルが見つかりません。スキップします。")
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
                # 結果ファイル名を生成（学習期間とテスト年を含める）
                base_filename = base_model_filename.replace('.sav', '')
                test_year_str = f"{test_year_start}-{test_year_end}" if test_year_start != test_year_end else str(test_year_start)
                individual_output_file = f"predicted_results_{base_filename}_train{train_year_range}_test{test_year_str}.tsv"
                summary_file = f"betting_summary_{base_filename}_train{train_year_range}_test{test_year_str}.tsv"
                
                # 個別モデル結果を上書き保存（追記ではなく上書き）
                save_results_with_append(output_df, individual_output_file, append_mode=False)
                
                # 全モデル統合ファイルに保存（初回は上書き、以降は追記）
                unified_output_file = "predicted_results.tsv"
                save_results_with_append(output_df, unified_output_file, append_mode=not first_unified_write)
                first_unified_write = False  # 初回書き込み完了
                
                # サマリーは個別ファイルに保存
                results_dir = Path('results')
                results_dir.mkdir(exist_ok=True)
                summary_filepath = results_dir / summary_file
                summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig')
                
                print(f"[OK] 完了！レース数: {race_count}")
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
                print(f"[ERROR] テストデータが見つかりませんでした。")
                
        except Exception as e:
            print(f"[ERROR] エラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # 複数モデルの比較結果を作成
    if len(all_results) > 1:
        print("\n[+] モデル比較結果")
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
        print(f"\n[LIST] 比較結果を {comparison_filepath} に保存しました！")
    
    print("\n[DONE] すべてのテストが完了しました！")


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
                    print(f"[DATE] テスト年範囲指定: {test_year_start}年~{test_year_end}年")
            except ValueError:
                print(f"[!] 無効な年範囲フォーマット: {arg} (例: 2020-2023)")
        elif arg.isdigit() and len(arg) == 4:
            # "2023" 形式の単一年指定
            test_year_start = test_year_end = int(arg)
            print(f"[DATE] テスト年指定: {test_year_start}年")
    
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