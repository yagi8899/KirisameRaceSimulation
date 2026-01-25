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
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from keiba_constants import get_track_name, format_model_description, get_surface_name
from model_config_loader import get_all_models, get_custom_models
from db_query_builder import build_sokuho_race_data_query
from data_preprocessing import preprocess_race_data
from feature_engineering import create_features, add_advanced_features

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
    model_description: str,
    target_date: str = None
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
        target_date: テスト用日付 ('YYYYMMDD'形式)。指定時は過去データ範囲と速報日付を制御
        
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
        kyoso_shubetsu_code=kyoso_shubetsu_code,
        target_date=target_date
    )
    
    # データ取得
    try:
        # DB接続情報をdb_config.jsonから読み込み
        with open('db_config.json', 'r', encoding='utf-8') as f:
            db_config = json.load(f)['database']
        
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
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
    
    # 高度な特徴量生成（feature_engineering.pyの共通関数を使用）
    X = add_advanced_features(df, X, surface_type, distance_min, distance_max, logger=logger, inverse_rank=False)
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
    
    # Phase 2.5: 穴馬予測を追加
    # dfに特徴量Xをマージ（穴馬モデルが必要とする特徴量を含める）
    df_with_features = df.copy()
    for col in X.columns:
        if col not in df_with_features.columns:
            df_with_features[col] = X[col].values
    
    # keibajo_code_numericを追加（穴馬モデルで必要）
    df_with_features['keibajo_code_numeric'] = df_with_features['keibajo_code'].astype(int)
    
    try:
        output_df = add_upset_prediction(df_with_features, output_df, surface_type, track_code, distance_max)
    except Exception as e:
        logger.warning(f"[WARNING] 穴馬予測の追加に失敗: {str(e)}")
        import traceback
        logger.warning(traceback.format_exc())
        # 穴馬予測なしで継続
    
    logger.info(f"[DONE] {model_description} の予測完了")
    
    return output_df


def add_upset_prediction(df_features: pd.DataFrame, output_df: pd.DataFrame, surface_type: str, track_code: str = None, distance_max: int = None) -> pd.DataFrame:
    """
    Phase 2.5: 穴馬予測を追加
    
    Args:
        df_features: 特徴量を含むDataFrame
        output_df: 出力用DataFrame
        surface_type: 芝/ダート ('turf' or 'dirt')
        track_code: 競馬場コード（閾値設定用）
        distance_max: 最大距離（閾値設定用）
        
    Returns:
        穴馬予測カラムが追加されたDataFrame
    """
    import pickle
    import numpy as np
    from pathlib import Path
    
    # surface_typeに応じた穴馬分類器モデルのロード
    if surface_type == 'turf':
        upset_model_path = Path('models/upset_classifier_turf.sav')
    else:
        upset_model_path = Path('models/upset_classifier_dirt.sav')
    
    if not upset_model_path.exists():
        logger.warning(f"[UPSET] 穴馬分類器モデルが見つかりません: {upset_model_path}")
        output_df['穴馬確率'] = 0.0
        output_df['穴馬候補'] = 0
        return output_df
    
    with open(upset_model_path, 'rb') as f:
        upset_model_data = pickle.load(f)
    
    models = upset_model_data['models']
    feature_cols = upset_model_data['feature_cols']
    calibrators = upset_model_data.get('calibrators', [None] * len(models))
    has_calibration = upset_model_data.get('has_calibration', False)
    calibration_method = upset_model_data.get('calibration_method', 'platt')
    
    logger.info(f"[UPSET] 穴馬分類器をロード: {len(models)}個のモデルアンサンブル (校正: {has_calibration})")
    
    # 閾値を設定ファイルから読み込み（競馬場・芝ダ・距離を考慮）
    distance_category = 'short' if distance_max and distance_max <= 1800 else 'long'
    upset_threshold = _load_upset_threshold_for_sokuho(track_code, surface_type, distance_category)
    logger.info(f"[UPSET] 閾値: {upset_threshold} (track={track_code}, surface={surface_type}, dist={distance_category})")
    
    # 7-12番人気のみ対象（穴馬予測のターゲット）
    target_mask = (output_df['人気順'] >= 7) & (output_df['人気順'] <= 12)
    target_indices = output_df[target_mask].index
    
    logger.info(f"[UPSET] 対象馬: {len(target_indices)}頭 (7-12番人気)")
    
    # デフォルト値を設定
    output_df['穴馬確率'] = 0.0
    output_df['穴馬候補'] = 0
    
    if len(target_indices) == 0:
        logger.info("[UPSET] 7-12番人気の馬が存在しません")
        return output_df
    
    # 対象馬のみの特徴量を準備
    df_work = df_features.loc[target_indices].copy()
    
    # 展開要因特徴量を追加
    df_work['estimated_running_style'] = 0  # デフォルト値
    df_work['avg_4corner_position'] = df_work.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban'].transform(lambda x: len(x) / 2)
    df_work['distance_change'] = 0
    df_work['wakuban_inner'] = (df_work.get('wakuban', 0) <= 3).astype(int)
    df_work['wakuban_outer'] = (df_work.get('wakuban', 0) >= 6).astype(int)
    
    # 予測結果から必要な特徴量を追加
    df_work['predicted_rank'] = output_df.loc[target_indices, '予測順位'].values
    df_work['predicted_score'] = output_df.loc[target_indices, '予測スコア'].values
    df_work['popularity_rank'] = output_df.loc[target_indices, '人気順'].values
    df_work['tansho_odds'] = output_df.loc[target_indices, '単勝オッズ'].values
    df_work['value_gap'] = df_work['predicted_rank'] - df_work['popularity_rank']
    
    # 不足特徴量のデフォルト値を設定（中立値または適切な値）
    # ※騎手・調教師統計はSQL側で正しく計算されるように修正済み（NULL除外）
    default_values = {
        # 騎手・調教師統計（中立値）- 上で計算済みだがNULLの場合のフォールバック
        'jockey_win_rate': 0.10,       # 平均的な騎手勝率
        'jockey_place_rate': 0.30,     # 平均的な騎手複勝率
        'jockey_recent_form': 0.5,     # 中立
        'trainer_win_rate': 0.10,      # 平均的な調教師勝率
        'trainer_place_rate': 0.30,    # 平均的な調教師複勝率
        'trainer_recent_form': 0.5,    # 中立
        # 馬のキャリア統計
        'horse_career_win_rate': 0.10,
        'horse_career_place_rate': 0.30,
        # 前走関連（中立値）
        'zenso_oikomi_power': 0.0,
        'zenso_kakoi_komon': 0.0,
        'zenso_ninki_gap': 0.0,
        'zenso_nigeba': 0.0,
        'zenso_taihai': 0.0,
        'zenso_agari_rank': 0.5,
        'zenso_top6': 0,
        'saikin_kaikakuritsu': 0.5,
        # その他
        'past_score_std': 0.0,
        'past_chakujun_variance': 0.0,
        'rest_weeks': 4.0,             # 平均的な休養週数
        'rest_days_fresh': 0,
        'is_turf_bad_condition': 0,
        'is_turf_heavy': 0,
        'is_local_track': 0,
        'is_open_class': 0,
        'is_3win_class': 0,
        'is_age_prime': 1,             # 4-5歳はプライム
    }
    
    # 特徴量の抽出
    missing_features = [col for col in feature_cols if col not in df_work.columns]
    if missing_features:
        logger.warning(f"[UPSET] 不足特徴量: {len(missing_features)}個 - {missing_features[:10]}...")
        for col in missing_features:
            # デフォルト値があれば使用、なければ0
            df_work[col] = default_values.get(col, 0)
    
    X_upset = df_work[feature_cols].copy()
    X_upset = X_upset.fillna(0)
    X_upset = X_upset.replace([np.inf, -np.inf], 0)
    
    # アンサンブル予測
    upset_proba_list = []
    for i, model in enumerate(models):
        try:
            # scikit-learn互換API (XGBClassifier, LGBMClassifier等)
            if hasattr(model, 'predict_proba'):
                # LightGBMの場合はbest_iterationを使用
                if hasattr(model, 'best_iteration'):
                    proba = model.predict_proba(X_upset, num_iteration=model.best_iteration)[:, 1]
                else:
                    proba = model.predict_proba(X_upset)[:, 1]
            # LightGBM Boosterオブジェクトの場合
            elif hasattr(model, 'predict') and 'lightgbm' in str(type(model)).lower():
                if hasattr(model, 'best_iteration'):
                    proba = model.predict(X_upset, num_iteration=model.best_iteration)
                else:
                    proba = model.predict(X_upset)
            # XGBoost Boosterオブジェクトの場合
            elif hasattr(model, 'save_model') and 'xgboost' in str(type(model)).lower():
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X_upset)
                proba = model.predict(dmatrix)
            else:
                # その他のモデル（predict メソッドを持つ場合）
                proba = model.predict(X_upset)
            
            # 確率校正を適用
            if has_calibration and calibrators[i] is not None:
                if calibration_method == 'platt':
                    proba = calibrators[i].predict_proba(proba.reshape(-1, 1))[:, 1]
                else:
                    proba = calibrators[i].predict(proba)
            
            upset_proba_list.append(proba)
        except Exception as e:
            logger.warning(f"[UPSET] モデル予測エラー ({type(model).__name__}): {str(e)}")
            continue
    
    if len(upset_proba_list) == 0:
        logger.warning("[UPSET] 全モデルの予測に失敗しました")
        return output_df
    
    upset_probability = np.mean(upset_proba_list, axis=0)
    
    # 対象馬のみに確率を設定
    output_df.loc[target_indices, '穴馬確率'] = upset_probability
    
    # 閾値で穴馬候補を判定（7-12番人気のみ）
    is_upset_candidate = (upset_probability > upset_threshold).astype(int)
    output_df.loc[target_indices, '穴馬候補'] = is_upset_candidate
    
    upset_count = is_upset_candidate.sum()
    logger.info(f"[UPSET] 穴馬候補: {upset_count}頭 / {len(target_indices)}頭 (7-12番人気)")
    
    return output_df


def _load_upset_threshold_for_sokuho(track_code: str = None, surface_type: str = None, distance_category: str = None) -> float:
    """
    速報用に閾値設定ファイルから読み込み（universal_test.pyと同じロジック）
    
    優先順位:
    1. by_track_surface_distance（最も具体的）
    2. by_track_surface
    3. by_track
    4. by_surface
    5. by_distance
    6. default_threshold（フォールバック）
    """
    config_path = Path('upset_threshold_config.json')
    default_threshold = 0.15
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"[UPSET-THRESHOLD] 設定ファイル読み込みエラー: {e}")
        return default_threshold
    
    default_threshold = config.get('default_threshold', default_threshold)
    thresholds = config.get('thresholds_by_condition', {})
    
    # 1. 最も具体的: 競馬場_芝ダ_距離区分
    if track_code and surface_type and distance_category:
        key = f"{track_code}_{surface_type}_{distance_category}"
        if key in thresholds.get('by_track_surface_distance', {}):
            logger.info(f"[UPSET-THRESHOLD] {key} の閾値を使用")
            return thresholds['by_track_surface_distance'][key]
    
    # 2. 競馬場_芝ダ
    if track_code and surface_type:
        key = f"{track_code}_{surface_type}"
        if key in thresholds.get('by_track_surface', {}):
            logger.info(f"[UPSET-THRESHOLD] {key} の閾値を使用")
            return thresholds['by_track_surface'][key]
    
    # 3. 競馬場
    if track_code and track_code in thresholds.get('by_track', {}):
        logger.info(f"[UPSET-THRESHOLD] track={track_code} の閾値を使用")
        return thresholds['by_track'][track_code]
    
    # 4. 芝ダ区分
    if surface_type and surface_type in thresholds.get('by_surface', {}):
        logger.info(f"[UPSET-THRESHOLD] surface={surface_type} の閾値を使用")
        return thresholds['by_surface'][surface_type]
    
    # 5. 距離区分
    if distance_category and distance_category in thresholds.get('by_distance', {}):
        logger.info(f"[UPSET-THRESHOLD] distance={distance_category} の閾値を使用")
        return thresholds['by_distance'][distance_category]
    
    # 6. デフォルト
    logger.info(f"[UPSET-THRESHOLD] デフォルト閾値を使用: {default_threshold}")
    return default_threshold


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
    parser.add_argument('--target_date', type=str, default=None,
                        help='テスト用日付（YYYYMMDD形式）。指定時は過去データ範囲と速報日付を制御')
    parser.add_argument('--no_file_output', action='store_true',
                        help='ファイル出力を抑制（バッチテスト用）')
    
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
            model_description=config['description'],
            target_date=args.target_date
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
        return None
    else:
        logger.info(f"\n[SUMMARY] {len(all_results)}個のモデルで予測を実行しました")
        
        # 全モデルの結果を統合
        all_dfs = [result['dataframe'] for result in all_results]
        df_all = pd.concat(all_dfs, ignore_index=True)
        
        # 集約ファイルでは不要な列を削除
        columns_to_drop = ['スコア差', 'スキップ理由', '購入推奨']
        df_all_clean = df_all.drop(columns=[col for col in columns_to_drop if col in df_all.columns])
        
        # ファイル出力が有効な場合のみ保存
        if not args.no_file_output:
            # 個別ファイルとして保存
            for result in all_results:
                save_sokuho_results(result['dataframe'], result['model_name'])
            
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
        
        return df_all_clean


if __name__ == '__main__':
    main()