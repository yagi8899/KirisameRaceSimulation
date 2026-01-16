#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
速報データ予測スクリプト

このスクリプトは、apd_sokuho_jvd_ra/apd_sokuho_jvd_seから速報データを取得し、
モデルで予測を行い、購入推奨結果を出力します。

Usage:
    python sokuho_prediction.py --model standard
    python sokuho_prediction.py --model custom tokyo_turf_3ageup_long
"""

import psycopg2
import pandas as pd
import pickle
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from keiba_constants import get_track_name, format_model_description, get_surface_name
from model_config_loader import get_all_models, get_custom_models
from db_query_builder import build_sokuho_race_data_query
from data_preprocessing import preprocess_race_data
from feature_engineering import create_features

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('sokuho_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def add_sokuho_purchase_logic(
    output_df: pd.DataFrame,
    prediction_rank_max: int = 3,
    popularity_rank_max: int = 3,
    min_odds: float = 1.5,
    max_odds: float = 20.0,
    min_score_diff: float = 0.05
) -> pd.DataFrame:
    """
    速報予測結果に購入判断を追加（的中率・回収率計算なし）
    
    Args:
        output_df (DataFrame): 予測結果データフレーム
        prediction_rank_max (int): 予測順位の上限 (デフォルト: 3)
        popularity_rank_max (int): 人気順の上限 (デフォルト: 3)
        min_odds (float): 最低オッズ (デフォルト: 1.5倍)
        max_odds (float): 最高オッズ (デフォルト: 20倍)
        min_score_diff (float): 予測1位と2位のスコア差の最小値 (デフォルト: 0.05)
        
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
        '単勝オッズ': 'tansho_odds'
    })
    
    # レースごとにグループ化して処理
    race_groups = df_work.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
    
    all_races = []
    total_recommended = 0
    
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
        
        # スキップ理由を記録（優先順位順に判定）
        race_df.loc[~race_df['購入推奨'] & (race_df['predicted_rank'] > prediction_rank_max), 'skip_reason'] = 'low_predicted_rank'
        race_df.loc[~race_df['購入推奨'] & (race_df['popularity_rank'] > popularity_rank_max), 'skip_reason'] = 'low_popularity'
        race_df.loc[~race_df['購入推奨'] & (race_df['tansho_odds'] < min_odds), 'skip_reason'] = 'odds_too_low'
        race_df.loc[~race_df['購入推奨'] & (race_df['tansho_odds'] > max_odds), 'skip_reason'] = 'odds_too_high'
        
        # 購入推奨がFalseでskip_reasonがまだNoneの場合は「複合条件」として記録
        race_df.loc[~race_df['購入推奨'] & race_df['skip_reason'].isna(), 'skip_reason'] = 'multiple_conditions'
        
        # 購入推奨馬を抽出
        buy_horses = race_df[race_df['購入推奨']].copy()
        total_recommended += len(buy_horses)
        
        all_races.append(race_df)
    
    # 全レースを統合
    df_integrated = pd.concat(all_races, ignore_index=True)
    
    # カラム名を日本語に戻す
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
        'score_diff': 'スコア差',
        'skip_reason': 'スキップ理由'
    })
    
    logger.info(f"購入推奨馬数: {total_recommended}頭")
    
    return df_integrated


def predict_sokuho_model(
    track_code: str,
    surface_type: str,
    distance_min: int,
    distance_max: int,
    kyoso_shubetsu_code: str,
    model_filename: str,
    model_description: str
) -> pd.DataFrame:
    """
    単一モデルで速報データの予測を実行
    
    Args:
        track_code: 競馬場コード
        surface_type: 芝/ダート ('turf' or 'dirt')
        distance_min: 最小距離
        distance_max: 最大距離
        kyoso_shubetsu_code: 競走種別コード
        model_filename: モデルファイル名
        model_description: モデルの説明
        
    Returns:
        DataFrame: 予測結果
    """
    logger.info(f"[START] {model_description} の予測を開始...")
    
    # モデルファイルの存在確認
    model_path = Path('models') / model_filename
    if not model_path.exists():
        logger.warning(f"[SKIP] モデルファイルが見つかりません: {model_path}")
        return None
    
    # SQLクエリ生成
    sql = build_sokuho_race_data_query(
        track_code=track_code,
        surface_type=surface_type,
        distance_min=distance_min,
        distance_max=distance_max,
        kyoso_shubetsu_code=kyoso_shubetsu_code
    )
    
    # データ取得
    try:
        conn = psycopg2.connect(
            dbname='keiba',
            user='postgres',
            password='ahtaht88',
            host='localhost',
            port=5432
        )
        df = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception as e:
        logger.error(f"[ERROR] データ取得エラー: {type(e).__name__}: {str(e)}")
        logger.error(f"[DEBUG] SQL実行エラーの詳細:")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # データが0件の場合
    if len(df) == 0:
        logger.warning(f"[SKIP] 対象データが0件です（競馬場:{track_code}, 馬場:{surface_type}, 距離:{distance_min}-{distance_max}m）")
        return None
    
    logger.info(f"データ取得完了: {len(df)}頭")
    # 前処理
    df = preprocess_race_data(df)
    
    # 基本特徴量生成
    X = create_features(df)
    logger.info(f"[DEBUG] 基本特徴量数: {len(X.columns)}個")
    logger.info(f"[DEBUG] 基本特徴量: {list(X.columns)}")
    
    # 高度な特徴量生成（universal_test.pyから移植）
    X = add_advanced_features(df, X, surface_type, distance_min, distance_max)
    logger.info(f"[DEBUG] 最終特徴量数: {len(X.columns)}個")
    logger.info(f"[DEBUG] 最終特徴量: {list(X.columns)}")
    
    # モデルロード
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        # モデルが期待する特徴量数を確認
        if hasattr(model, 'n_features_'):
            logger.info(f"[DEBUG] モデルが期待する特徴量数: {model.n_features_}個")
    except Exception as e:
        logger.error(f"[ERROR] モデル読み込みエラー: {e}")
        return None
    
    # 予測実行
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    raw_scores = model.predict(X)
    df['predicted_chakujun_score'] = sigmoid(raw_scores)
    
    # データをソート
    df = df.sort_values(by=['kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'umaban'], ascending=True)
    
    # レース内での予測順位を計算
    df['score_rank'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['predicted_chakujun_score'].rank(
        method='min', ascending=False
    )
    
    # surface_type列を追加
    df['surface_type_name'] = get_surface_name(surface_type)
    
    # 必要な列を選択
    output_columns = [
        'keibajo_name',
        'kaisai_nen', 
        'kaisai_tsukihi', 
        'race_bango',
        'surface_type_name',
        'kyori',
        'umaban', 
        'bamei', 
        'tansho_odds', 
        'tansho_ninkijun_numeric',
        'score_rank', 
        'predicted_chakujun_score'
    ]
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
        'score_rank': '予測順位',
        'predicted_chakujun_score': '予測スコア'
    })
    
    # 整数に変換
    output_df['人気順'] = output_df['人気順'].fillna(0).astype(int)
    output_df['予測順位'] = output_df['予測順位'].fillna(0).astype(int)
    
    # 購入推奨ロジックを追加
    output_df = add_sokuho_purchase_logic(output_df)
    
    logger.info(f"[DONE] {model_description} の予測完了")
    
    return output_df


def add_advanced_features(df: pd.DataFrame, X: pd.DataFrame, surface_type: str, min_distance: int, max_distance: int) -> pd.DataFrame:
    """
    高度な特徴量を追加（universal_test.pyから完全移植）
    """
    logger.info("[START] 高度な特徴量生成を開始...")
    
    # ========================================
    # 0️⃣ 基本特徴量（universal_test.pyで追加されている）
    # ========================================
    # レース内での馬番相対位置（頭数による正規化）
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # 斤量偏差値（レース内で標準化）
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
    df['futan_percentile'] = race_group.transform(lambda x: x.rank(pct=True))
    X['futan_percentile'] = df['futan_percentile']
    
    # 距離カテゴリ分類関数
    def categorize_distance(kyori):
        """距離を4カテゴリに分類"""
        if kyori <= 1400:
            return 'short'
        elif kyori <= 1800:
            return 'mile'
        elif kyori <= 2400:
            return 'middle'
        else:
            return 'long'
    
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
    
    # 今回のレースのカテゴリ情報を追加
    df['distance_category'] = df['kyori'].apply(categorize_distance)
    df['surface_type'] = df['track_code'].apply(categorize_surface)
    df['baba_condition'] = df['babajotai_code'].apply(categorize_baba_condition)
    
    # 時系列順にソート
    df_sorted = df.sort_values(['ketto_toroku_bango', 'kaisai_nen', 'kaisai_tsukihi']).copy()
    
    # ========================================
    # 1️⃣ 距離適性スコア（3種類）
    # ========================================
    logger.info("  [1/7] 距離適性スコアを計算中...")
    
    # 1-1: 距離カテゴリ別適性スコア
    def calc_distance_category_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_category = group.iloc[idx]['distance_category']
            past_same_category = group.iloc[:idx][
                group.iloc[:idx]['distance_category'] == current_category
            ].tail(5)
            if len(past_same_category) > 0:
                avg_score = (past_same_category['kakutei_chakujun_numeric'] / 18.0).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_category_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_category_score
    ).values
    
    # 1-2: 近似距離での成績
    def calc_similar_distance_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_kyori = group.iloc[idx]['kyori']
            past_similar = group.iloc[:idx][
                abs(group.iloc[:idx]['kyori'] - current_kyori) <= 200
            ].tail(10)
            if len(past_similar) > 0:
                avg_score = (1 - (past_similar['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['similar_distance_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_similar_distance_score
    ).values
    
    # 1-3: 距離変化対応力
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
                    avg_score = (changed_races['kakutei_chakujun_numeric'] / 18.0).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_change_adaptability
    ).values
    
    # 1-4: 前走距離差（短距離特化）
    def calc_zenso_kyori_sa(group):
        diffs = []
        for idx in range(len(group)):
            if idx == 0:
                diffs.append(0)
            else:
                current_kyori = group.iloc[idx]['kyori']
                previous_kyori = group.iloc[idx-1]['kyori']
                diffs.append(abs(current_kyori - previous_kyori))
        return pd.Series(diffs, index=group.index)
    
    df_sorted['zenso_kyori_sa'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_zenso_kyori_sa
    ).values
    
    # 1-5: 長距離経験回数（2400m以上）
    def calc_long_distance_experience_count(group):
        counts = []
        for idx in range(len(group)):
            if idx == 0:
                counts.append(0)
            else:
                past_long_count = (group.iloc[:idx]['kyori'] >= 2400).sum()
                counts.append(past_long_count)
        return pd.Series(counts, index=group.index)
    
    df_sorted['long_distance_experience_count'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_long_distance_experience_count
    ).values
    
    # dfに戻す
    df['similar_distance_score'] = df_sorted.sort_index()['similar_distance_score']
    df['distance_change_adaptability'] = df_sorted.sort_index()['distance_change_adaptability']
    df['zenso_kyori_sa'] = df_sorted.sort_index()['zenso_kyori_sa']
    df['long_distance_experience_count'] = df_sorted.sort_index()['long_distance_experience_count']
    
    X['similar_distance_score'] = df['similar_distance_score']
    X['zenso_kyori_sa'] = df['zenso_kyori_sa']
    X['long_distance_experience_count'] = df['long_distance_experience_count']
    
    # ========================================
    # 2️⃣ スタート指数（第1コーナー通過順位）
    # ========================================
    logger.info("  [2/7] スタート指数を計算中...")
    
    if 'corner_1' in df.columns:
        def calc_start_index(group):
            scores = []
            for idx in range(len(group)):
                if idx == 0:
                    scores.append(0.5)
                    continue
                past_corners = group.iloc[max(0, idx-10):idx]['corner_1'].dropna()
                if len(past_corners) >= 3:
                    avg_position = past_corners.mean()
                    std_position = past_corners.std()
                    position_score = max(0, 1.0 - (avg_position / 18.0))
                    stability_bonus = max(0, 0.2 - (std_position / 10.0))
                    total_score = position_score + stability_bonus
                    scores.append(min(1.0, total_score))
                else:
                    scores.append(0.5)
            return pd.Series(scores, index=group.index)
        
        df_sorted['start_index'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
            calc_start_index
        ).values
        df['start_index'] = df_sorted.sort_index()['start_index']
        X['start_index'] = df['start_index']
    else:
        df['start_index'] = 0.5
        X['start_index'] = 0.5
    
    # ========================================
    # 3️⃣ コーナー通過位置スコア（短距離特化）
    # ========================================
    logger.info("  [3/7] コーナー通過位置スコアを計算中...")
    
    if all(col in df.columns for col in ['corner_1', 'corner_2', 'corner_3', 'corner_4']):
        def calc_corner_position_score(group):
            scores = []
            for idx in range(len(group)):
                if idx < 1:
                    scores.append(0.5)
                    continue
                past_3_races = group.iloc[max(0, idx-2):idx+1]
                if len(past_3_races) >= 1:
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
                        position_score = max(0, 1.0 - (avg_position / 18.0))
                        stability_bonus = max(0, 0.3 - (std_position / 10.0))
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
    else:
        df['corner_position_score'] = 0.5
        X['corner_position_score'] = 0.5
    
    # ========================================
    # 4️⃣ 馬場適性スコア（3種類）
    # ========================================
    logger.info("  [4/7] 馬場適性スコアを計算中...")
    
    # 4-1: 芝/ダート別適性スコア
    def calc_surface_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_surface = group.iloc[idx]['surface_type']
            past_same_surface = group.iloc[:idx][
                group.iloc[:idx]['surface_type'] == current_surface
            ].tail(10)
            if len(past_same_surface) > 0:
                avg_score = (1 - (past_same_surface['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['surface_aptitude_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_surface_score
    ).values
    
    # 4-2: 馬場状態別適性スコア
    def calc_baba_condition_score(group):
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)
                continue
            current_condition = group.iloc[idx]['baba_condition']
            past_same_condition = group.iloc[:idx][
                group.iloc[:idx]['baba_condition'] == current_condition
            ].tail(10)
            if len(past_same_condition) > 0:
                avg_score = (past_same_condition['kakutei_chakujun_numeric'] / 18.0).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_condition_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_condition_score
    ).values
    
    # 4-3: 馬場変化対応力
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
                    avg_score = (changed_races['kakutei_chakujun_numeric'] / 18.0).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_change_adaptability
    ).values
    
    df['surface_aptitude_score'] = df_sorted.sort_index()['surface_aptitude_score']
    df['baba_condition_score'] = df_sorted.sort_index()['baba_condition_score']
    df['baba_change_adaptability'] = df_sorted.sort_index()['baba_change_adaptability']
    
    X['surface_aptitude_score'] = df['surface_aptitude_score']
    X['baba_change_adaptability'] = df['baba_change_adaptability']
    
    # ========================================
    # 5️⃣ 騎手スコア（3種類）
    # ========================================
    logger.info("  [5/7] 騎手スコアを計算中...")
    
    df_sorted_kishu = df.sort_values(['kishu_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    # 5-1: 騎手の実力補正スコア
    def calc_kishu_skill_adjusted_score(group):
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
                    recent_races['rank_score'] = recent_races['kakutei_chakujun_numeric'] / 18.0
                    avg_score = recent_races['rank_score'].mean()
                    normalized_score = max(0.0, min(1.0, avg_score))
                    scores.append(normalized_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_skill_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_skill_adjusted_score
    ).values
    
    # 5-2: 騎手の人気差スコア
    def calc_kishu_popularity_adjusted_score(group):
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
                    valid_races = recent_races[recent_races['tansho_odds'] > 0]
                    if len(valid_races) >= 3:
                        max_odds = valid_races['tansho_odds'].max()
                        valid_races['odds_expectation'] = 1.0 - (valid_races['tansho_odds'] / (max_odds + 1.0))
                        valid_races['actual_score'] = valid_races['kakutei_chakujun_numeric'] / 18.0
                        valid_races['performance_diff'] = valid_races['actual_score'] - valid_races['odds_expectation']
                        avg_diff = valid_races['performance_diff'].mean()
                        normalized_score = 0.5 + (avg_diff * 0.5)
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
    
    # 5-3: 騎手の芝/ダート別スコア
    def calc_kishu_surface_score(group):
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
                recent_same_surface = past_races[
                    (past_races['kaisai_date'] >= six_months_ago) &
                    (past_races['surface_type'] == current_surface)
                ]
                if len(recent_same_surface) >= 5:
                    avg_score = (recent_same_surface['kakutei_chakujun_numeric'] / 18.0).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_surface_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_surface_score
    ).values
    
    df['kishu_skill_score'] = df_sorted_kishu.sort_index()['kishu_skill_score']
    df['kishu_popularity_score'] = df_sorted_kishu.sort_index()['kishu_popularity_score']
    df['kishu_surface_score'] = df_sorted_kishu.sort_index()['kishu_surface_score']
    
    X['kishu_skill_score'] = df['kishu_skill_score']
    X['kishu_popularity_score'] = df['kishu_popularity_score']
    X['kishu_surface_score'] = df['kishu_surface_score']
    
    # ========================================
    # 6️⃣ 調教師スコア（1種類）
    # ========================================
    logger.info("  [6/7] 調教師スコアを計算中...")
    
    df_sorted_chokyoshi = df.sort_values(['chokyoshi_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    def calc_chokyoshi_recent_score(group):
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
                if len(recent_races) >= 5:
                    avg_score = (recent_races['kakutei_chakujun_numeric'] / 18.0).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return pd.Series(scores, index=group.index)
    
    df_sorted_chokyoshi['chokyoshi_recent_score'] = df_sorted_chokyoshi.groupby('chokyoshi_code', group_keys=False).apply(
        calc_chokyoshi_recent_score
    ).values
    
    df['chokyoshi_recent_score'] = df_sorted_chokyoshi.sort_index()['chokyoshi_recent_score']
    # X['chokyoshi_recent_score'] = df['chokyoshi_recent_score']  # 実際にはモデルによって使用/不使用
    
    # ========================================
    # 7️⃣ 路面×距離別特徴量選択
    # ========================================
    logger.info("  [7/7] 路面×距離別特徴量選択を実施中...")
    logger.info(f"    路面: {surface_type}, 距離: {min_distance}m 〜 {max_distance}m")
    
    is_turf = surface_type.lower() == 'turf'
    is_short = max_distance <= 1600
    is_long = min_distance >= 1700
    
    # 短距離専用特徴量の調整
    if is_short:
        logger.info(f"    [短距離モデル] 短距離特化特徴量を使用")
        # 短距離では長距離特化特徴量を削除
        if 'long_distance_experience_count' in X.columns:
            X = X.drop(columns=['long_distance_experience_count'])
            logger.info(f"      削除: long_distance_experience_count")
    else:
        logger.info(f"    [中長距離モデル] 短距離特化特徴量を削除")
        # 中長距離では短距離特化特徴量を削除
        features_to_remove_for_long = ['start_index', 'corner_position_score', 'zenso_kyori_sa']
        for feature in features_to_remove_for_long:
            if feature in X.columns:
                X = X.drop(columns=[feature])
                logger.info(f"      削除: {feature}")
        
        # 長距離(2200m以上)では長距離特化特徴量を残す
        if min_distance >= 2200:
            logger.info(f"    [長距離モデル] 長距離特化特徴量を使用")
        else:
            # 中距離では長距離特化特徴量も削除
            if 'long_distance_experience_count' in X.columns:
                X = X.drop(columns=['long_distance_experience_count'])
                logger.info(f"      削除: long_distance_experience_count")
    
    # 路面×距離別の特徴量削除
    features_to_remove = []
    
    # wakuban_kyori_interactionは短距離モデル専用なので、中長距離では削除
    if not is_short and 'wakuban_kyori_interaction' in X.columns:
        X = X.drop(columns=['wakuban_kyori_interaction'])
        logger.info(f"      削除: wakuban_kyori_interaction（中長距離では不要）")
    
    if is_turf and is_long:
        logger.info("    [芝中長距離] 全特徴量を使用（ベースモデル）")
    elif is_turf and is_short:
        logger.info("    [芝短距離] 不要な特徴量を削除")
        features_to_remove = ['kohan_3f_index', 'surface_aptitude_score', 'wakuban_ratio']
    elif not is_turf and is_long:
        logger.info("    [ダート中長距離] 全特徴量を使用")
    elif not is_turf and is_short:
        logger.info("    [ダート短距離] 不要な特徴量を削除")
        features_to_remove = ['kohan_3f_index', 'surface_aptitude_score', 'wakuban_ratio']
    else:
        logger.info("    [中間距離] 全特徴量を使用")
    
    if features_to_remove:
        for feature in features_to_remove:
            if feature in X.columns:
                X = X.drop(columns=[feature])
                logger.info(f"      削除: {feature}")
    
    logger.info(f"  [DONE] 最終特徴量数: {len(X.columns)}個")
    
    return X


def save_sokuho_results(df: pd.DataFrame, model_name: str, output_dir: str = 'sokuho_results'):
    """
    速報予測結果をTSVファイルに保存
    
    Args:
        df: 予測結果データフレーム
        model_name: モデル名
        output_dir: 出力先ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'sokuho_prediction_{model_name}_{timestamp}.tsv'
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
    logger.info(f"[SAVE] 結果を保存しました: {filepath}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='速報データ予測スクリプト')
    parser.add_argument('--model', type=str, required=True,
                        help='モデルタイプ: "standard" または "custom"')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("速報データ予測を開始します")
    logger.info("=" * 60)
    
    # モデル設定の取得
    if args.model.lower() == 'standard':
        model_configs = get_all_models()
        logger.info(f"標準モデル {len(model_configs)}個 をテストします")
    elif args.model.lower() == 'custom':
        model_configs = get_custom_models()
        if len(model_configs) == 0:
            logger.warning("[WARNING] カスタムモデルが定義されていません")
            return
        logger.info(f"カスタムモデル {len(model_configs)}個 を一括実行します")
    else:
        logger.error(f"[ERROR] 不明なモデルタイプ: {args.model}")
        logger.info("使用方法: --model standard または --model custom")
        return
    
    # 各モデルで予測を実行
    all_results = []
    for i, config in enumerate(model_configs, 1):
        logger.info(f"\n【{i}/{len(model_configs)}】 {config['description']} の処理中...")
        
        # モデルファイルの存在確認（universal_test.pyと同じ処理）
        model_filename = config['model_filename']
        model_path = Path('models') / model_filename
        
        if not model_path.exists():
            logger.warning(f"[SKIP] モデルファイルが見つかりません: {model_path}")
            continue
        
        result = predict_sokuho_model(
            track_code=config['track_code'],
            surface_type=config['surface_type'],
            distance_min=config['min_distance'],
            distance_max=config['max_distance'],
            kyoso_shubetsu_code=config.get('kyoso_shubetsu_code'),
            model_filename=config['model_filename'],
            model_description=config['description']
        )
        
        if result is not None:
            all_results.append({
                'model_name': config['model_filename'].replace('.sav', ''),
                'dataframe': result
            })
    
    # 結果の保存
    if len(all_results) == 0:
        logger.warning("[WARNING] 予測可能なデータが見つかりませんでした")
        logger.info("モデルファイルが存在しないか、速報データが登録されていない可能性があります")
    else:
        logger.info(f"\n[SUMMARY] {len(all_results)}個のモデルで予測を実行しました")
        
        # 個別ファイルとして保存
        for result in all_results:
            save_sokuho_results(result['dataframe'], result['model_name'])
        
        # 全モデルの結果を統合して保存（universal_test.pyと同じパターン）
        all_dfs = [result['dataframe'] for result in all_results]
        if len(all_dfs) > 0:
            df_all = pd.concat(all_dfs, ignore_index=True)
            
            # 集約ファイルでは不要な列を削除
            columns_to_drop = ['スコア差', 'スキップ理由', '購入推奨']
            df_all_clean = df_all.drop(columns=[col for col in columns_to_drop if col in df_all.columns])
            
            # 統合ファイルを保存
            output_path = Path('sokuho_results')
            output_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            all_filename = f'sokuho_prediction_all_{timestamp}.tsv'
            filepath_all = output_path / all_filename
            
            df_all_clean.to_csv(filepath_all, index=False, sep='\t', encoding='utf-8-sig')
            logger.info(f"[SAVE] 全レース統合ファイルを保存しました: {filepath_all}")
            logger.info(f"       統合レース数: {len(df_all_clean)}件")
        
        logger.info("\n速報予測が完了しました！")


if __name__ == '__main__':
    main()
