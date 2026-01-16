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
    
    logger.info(f"[DONE] {model_description} の予測完了")
    
    return output_df


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
