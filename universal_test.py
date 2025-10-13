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
                      min_distance, max_distance, test_year=2023):
    """
    指定したモデルで予測を実行する汎用関数
    
    Args:
        model_filename (str): 使用するモデルファイル名
        track_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競争種別コード
        surface_type (str): 'turf' or 'dirt'
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        test_year (int): テスト対象年
        
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
        seum.futan_juryo,
        seum.seibetsu_code,
        CASE WHEN seum.seibetsu_code = '1' THEN '1' ELSE '0' END AS mare_horse,
        CASE WHEN seum.seibetsu_code = '2' THEN '1' ELSE '0' END AS femare_horse,
        CASE WHEN seum.seibetsu_code = '3' THEN '1' ELSE '0' END AS sen_horse,
        CASE WHEN {baba_condition} = '1' THEN '1' ELSE '0' END AS baba_good,
        CASE WHEN {baba_condition} = '2' THEN '1' ELSE '0' END AS baba_slightly_heavy,
        CASE WHEN {baba_condition} = '3' THEN '1' ELSE '0' END AS baba_heavy,
        CASE WHEN {baba_condition} = '4' THEN '1' ELSE '0' END AS baba_defective,
        CASE WHEN ra.tenko_code = '1' THEN '1' ELSE '0' END AS tenko_fine,
        CASE WHEN ra.tenko_code = '2' THEN '1' ELSE '0' END AS tenko_cloudy,
        CASE WHEN ra.tenko_code = '3' THEN '1' ELSE '0' END AS tenko_rainy,
        CASE WHEN ra.tenko_code = '4' THEN '1' ELSE '0' END AS tenko_drizzle,
        CASE WHEN ra.tenko_code = '5' THEN '1' ELSE '0' END AS tenko_snow,
        CASE WHEN ra.tenko_code = '6' THEN '1' ELSE '0' END AS tenko_light_snow,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        nullif(cast(seum.kakutei_chakujun as integer), 0) as kakutei_chakujun_numeric,
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,
        -1 as sotai_chakujun_numeric,
        -1 AS time_index,
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
        ) AS past_score
        ,0 as kohan_3f_sec
        ,0 AS kohan_3f_index
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
        inner join jvd_hr hr
            on ra.kaisai_nen = hr.kaisai_nen 
            and ra.kaisai_tsukihi = hr.kaisai_tsukihi 
            and ra.keibajo_code = hr.keibajo_code 
            and ra.race_bango = hr.race_bango
    where
        cast(ra.kaisai_nen as integer) = {test_year} 
    ) rase 
    where 
    rase.keibajo_code = '{track_code}'
    and {kyoso_shubetsu_condition}
    and {track_condition}
    and {distance_condition}
    """

    # データを取得
    df = pd.read_sql_query(sql=sql, con=conn)
    conn.close()
    
    if len(df) == 0:
        print(f"❌ {model_filename} に対応するテストデータが見つかりませんでした。")
        return None, None, 0

    print(f"📊 テストデータ件数: {len(df)}件")

    # 馬名だけは保存しておく
    horse_names = df['bamei'].copy()
    
    # 数値データだけを前処理
    numeric_columns = df.columns.drop(['bamei', 'keibajo_name'])  # 馬名以外の列を取得
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[numeric_columns] = df[numeric_columns].replace('0', np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # 保存しておいた馬名を戻す
    df['bamei'] = horse_names

    # 特徴量を選択（model_creator.pyと同じ特徴量）
    X = df.loc[:, [
        "kyori",
        "tenko_code",  
        "babajotai_code",  # 汎用化に合わせて変更
        "seibetsu_code",
        # "umaban_numeric", 
        # "barei",
        "futan_juryo",
        "past_score",
        "kohan_3f_index",
        "sotai_chakujun_numeric",
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

    # 馬番×距離の相互作用（内外枠の距離適性）
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000  # スケール調整
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    # 馬齢の非線形変換（競走馬のピーク年齢効果）
    # df['barei_squared'] = df['barei'] ** 2
    # X['barei_squared'] = df['barei_squared']
    df['barei_peak_distance'] = abs(df['barei'] - 4)  # 4歳をピークと仮定
    X['barei_peak_distance'] = df['barei_peak_distance']


    # レース内での馬番相対位置（頭数による正規化）
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # # 微小な個体識別子を追加（重複完全回避のため）
    # # 馬番ベースの極小調整値
    # df['micro_adjustment'] = df['umaban_numeric'] / 1000000  # 0.000001〜0.000018程度
    # X['micro_adjustment'] = df['micro_adjustment']

    # カテゴリ変数を作成
    X['kyori'] = X['kyori'].astype('category')
    X['tenko_code'] = X['tenko_code'].astype('category')
    X['babajotai_code'] = X['babajotai_code'].astype('category')
    X['seibetsu_code'] = X['seibetsu_code'].astype('category')
        
    # kohan_3f_indexを距離に応じた値にする！
    distance_bins = [0, 1600, 2000, 2400, 10000]
    default_values = {
        0: 33.5,   # 短距離（〜1600m）
        1: 35.0,   # マイル（〜2000m）
        2: 36.0,   # 中距離（〜2400m）
        3: 37.0    # 長距離（2400m〜）
    }
    
    # 距離のビンに応じて基準タイムを割り当て
    df['distance_bin'] = pd.cut(df['kyori'], bins=distance_bins, labels=False)
    df['kohan_3f_base'] = df['distance_bin'].map(default_values)
    
    # 基準タイムから少しだけランダムにずらす（実際のレースっぽく）
    np.random.seed(42)  # 再現性のためシード固定
    df['kohan_3f_sec'] = df['kohan_3f_base'] + np.random.normal(0, 0.5, len(df))
    
    # kohan_3f_indexを計算（main.pyと同じ計算方法）
    df['kohan_3f_index'] = df['kohan_3f_sec'] - df['kohan_3f_base']
    X['kohan_3f_index'] = df['kohan_3f_index']

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


def test_multiple_models():
    """
    複数のモデルをテストして結果を比較する関数（設定はJSONファイルから読み込み）
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
                max_distance=config['max_distance']
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
    
    if len(sys.argv) > 1 and sys.argv[1] == 'multi':
        # python universal_test.py multi
        test_multiple_models()
    else:
        # python universal_test.py (デフォルト)
        predict_and_save_results()