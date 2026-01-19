#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: 穴馬予測パイプライン

二段階予測アプローチ:
1. 既存のRankerモデルで順位予測 (predicted_rank, predicted_score)
2. 穴馬分類器で7-12番人気から穴馬候補を抽出
3. 予測確率でソートしTop-N候補を出力
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import json
from pathlib import Path
import sys
import psycopg2

# 既存モジュールをインポート
from db_query_builder import build_race_data_query
from data_preprocessing import preprocess_race_data
from feature_engineering import create_features, add_advanced_features

def load_models(ranker_path, classifier_path):
    """モデル読み込み"""
    print(f"\nモデル読み込み:")
    print(f"  Ranker: {ranker_path}")
    with open(ranker_path, 'rb') as f:
        ranker = pickle.load(f)
    
    print(f"  Classifier: {classifier_path}")
    with open(classifier_path, 'rb') as f:
        classifiers = pickle.load(f)
    
    if isinstance(classifiers, dict):
        models = classifiers['models']
        features = classifiers['feature_cols']
    else:
        models = classifiers
        features = None
    
    print(f"  Classifierモデル数: {len(models)}個")
    if features:
        print(f"  使用特徴量数: {len(features)}個")
    
    return ranker, models, features


def get_prediction_data(year, track='hanshin', surface='芝', age_group='3歳以上', distance_type='中長距離'):
    """予測用データ取得 (analyze_upset_patterns.pyと同じロジック)"""
    
    # DB接続情報をdb_config.jsonから読み込み
    with open('db_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    db_config = config['database']
    
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        dbname=db_config['dbname']
    )
    
    # トラックコード
    track_codes = {'hanshin': '06', 'kyoto': '09', 'chukyo': '07', 'tokyo': '05', 'niigata': '04'}
    track_code = track_codes.get(track, '06')
    
    # 距離範囲
    if distance_type == '中長距離':
        distance_min, distance_max = 1700, 3600
    else:
        distance_min, distance_max = 1000, 1600
    
    # サーフェスタイプ
    surface_type_map = {'芝': 'turf', 'ダート': 'dirt'}
    surface_type = surface_type_map.get(surface, 'turf')
    
    # クエリ構築
    sql = build_race_data_query(
        track_code=track_code,
        year_start=year,
        year_end=year,
        surface_type=surface_type,
        distance_min=distance_min,
        distance_max=distance_max
    )
    
    # データ取得
    df = pd.read_sql_query(sql, conn)
    conn.close()
    
    # 前処理
    df = preprocess_race_data(df, verbose=False)
    
    # 特徴量生成
    X = create_features(df)
    X = add_advanced_features(
        df=df,
        X=X,
        surface_type=surface_type,
        min_distance=distance_min,
        max_distance=distance_max,
        logger=None,
        inverse_rank=True
    )
    
    # popularity_rank追加
    df['popularity_rank'] = df['tansho_ninkijun_numeric']
    
    # 展開要因特徴量を追加（Classifier用）
    # 1. 推定脚質: 4コーナー位置の平均から推定 (0=逃げ先行, 1=差し, 2=追込)
    if '4corner_position_numeric' in df.columns:
        df['avg_4corner_position'] = df.groupby('bamei')['4corner_position_numeric'].transform('mean')
        df['estimated_running_style'] = pd.cut(
            df['avg_4corner_position'],
            bins=[0, 3, 8, 18],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(float)
    else:
        df['avg_4corner_position'] = 9  # デフォルト値
        df['estimated_running_style'] = 1  # デフォルト（差し）
    
    # 2. 距離変化
    if 'zenso_kyori' in df.columns and 'kyori' in df.columns:
        df['distance_change'] = df['kyori'] - df['zenso_kyori']
    else:
        df['distance_change'] = 0
    
    # 3. 内枠・外枠フラグ
    if 'wakuban' in df.columns:
        df['wakuban_inner'] = (df['wakuban'] <= 3).astype(int)
        df['wakuban_outer'] = (df['wakuban'] >= 6).astype(int)
    else:
        df['wakuban_inner'] = 0
        df['wakuban_outer'] = 0
    
    # 4. 前走着順変化
    if 'zenso_chakujun' in df.columns and 'kakutei_chakujun_numeric' in df.columns:
        df['prev_rank_change'] = df['zenso_chakujun'] - df['kakutei_chakujun_numeric']
    else:
        df['prev_rank_change'] = 0
    
    # 欠損値を0で埋める
    df = df.fillna(0)
    
    print(f"\n{year}年データ取得:")
    print(f"  総データ数: {len(df)}頭")
    print(f"  レース数: {df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']).ngroups}レース")
    
    return df, X


def add_ranker_predictions(df, X, ranker_model):
    """Rankerモデルで順位予測を追加"""
    print("\n[Stage 1] Rankerで順位予測:")
    
    # 予測スコア取得
    predicted_scores = ranker_model.predict(X)
    df['predicted_score'] = predicted_scores
    
    # レース内でスコア順位を計算
    df['predicted_rank'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango'])['predicted_score'].rank(ascending=False, method='min').astype(int)
    
    # value_gap計算
    df['value_gap'] = df['predicted_rank'] - df['popularity_rank']
    
    print(f"  予測完了: {len(df)}頭")
    print(f"  予測順位範囲: {df['predicted_rank'].min()}-{df['predicted_rank'].max()}")
    
    return df


def add_classifier_predictions(df, classifier_models, classifier_features, threshold=0.3):
    """Classifierで穴馬確率予測を追加"""
    print(f"\n[Stage 2] Classifierで穴馬予測 (閾値={threshold}):")
    
    # 7-12番人気のみ対象
    target_mask = (df['popularity_rank'] >= 7) & (df['popularity_rank'] <= 12)
    target_df = df[target_mask].copy()
    
    if len(target_df) == 0:
        print("  対象馬なし (7-12番人気が存在しません)")
        df['upset_probability'] = 0.0
        df['is_upset_candidate'] = False
        return df
    
    print(f"  対象馬: {len(target_df)}頭 (7-12番人気)")
    
    # Classifierに必要な特徴量のみ抽出
    X_classifier = target_df[classifier_features].values
    
    # アンサンブル予測 (5モデルの平均)
    predictions_all = []
    for i, model in enumerate(classifier_models):
        pred = model.predict(X_classifier)
        predictions_all.append(pred)
    
    # 平均確率
    avg_predictions = np.mean(predictions_all, axis=0)
    target_df['upset_probability'] = avg_predictions
    target_df['is_upset_candidate'] = (avg_predictions >= threshold)
    
    # 元のdfにマージ
    df['upset_probability'] = 0.0
    df['is_upset_candidate'] = False
    df.loc[target_mask, 'upset_probability'] = target_df['upset_probability']
    df.loc[target_mask, 'is_upset_candidate'] = target_df['is_upset_candidate']
    
    candidates = df[df['is_upset_candidate']].copy()
    print(f"  穴馬候補: {len(candidates)}頭")
    
    if len(candidates) > 0:
        print(f"  確率範囲: {candidates['upset_probability'].min():.3f} - {candidates['upset_probability'].max():.3f}")
    
    return df


def extract_top_candidates(df, top_n=30):
    """Top-N候補を抽出"""
    candidates = df[df['is_upset_candidate']].copy()
    
    if len(candidates) == 0:
        print("\n候補なし")
        return pd.DataFrame()
    
    # 確率でソート
    candidates = candidates.sort_values('upset_probability', ascending=False).head(top_n)
    
    print(f"\nTop-{top_n}候補抽出: {len(candidates)}頭")
    
    return candidates


def evaluate_predictions(df):
    """予測評価 (正解データがある場合)"""
    candidates = df[df['is_upset_candidate']].copy()
    
    if len(candidates) == 0:
        return {
            'total_candidates': 0,
            'correct_predictions': 0,
            'precision': 0.0,
            'total_races': 0,
            'hit_races': 0,
            'hit_rate': 0.0,
            'total_odds': 0.0,
            'roi': 0.0
        }
    
    # 正解 (7-12番人気 & 3着以内)
    candidates['is_actual_upset'] = (
        (candidates['popularity_rank'] >= 7) & 
        (candidates['popularity_rank'] <= 12) & 
        (candidates['kakutei_chakujun_numeric'] <= 3)
    )
    
    total_candidates = len(candidates)
    correct_predictions = candidates['is_actual_upset'].sum()
    precision = correct_predictions / total_candidates if total_candidates > 0 else 0.0
    
    # レース単位の評価
    race_groups = candidates.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango'])
    total_races = len(race_groups)
    hit_races = sum(group['is_actual_upset'].any() for _, group in race_groups)
    hit_rate = hit_races / total_races if total_races > 0 else 0.0
    
    # ROI計算 (的中馬券のオッズ合計 / 総投資額)
    total_odds = candidates[candidates['is_actual_upset']]['tansho_odds'].sum()
    roi = (total_odds / total_candidates * 100) if total_candidates > 0 else 0.0
    
    return {
        'total_candidates': total_candidates,
        'correct_predictions': correct_predictions,
        'precision': precision * 100,
        'total_races': total_races,
        'hit_races': hit_races,
        'hit_rate': hit_rate * 100,
        'total_odds': total_odds,
        'roi': roi
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: 穴馬予測パイプライン')
    parser.add_argument('--year', type=int, required=True, help='予測対象年')
    parser.add_argument('--ranker', type=str, default='models/hanshin_turf_3ageup_long.sav', help='Rankerモデルパス')
    parser.add_argument('--classifier', type=str, default='models/upset_classifier.sav', help='Classifierモデルパス')
    parser.add_argument('--threshold', type=float, default=0.3, help='穴馬判定閾値 (default: 0.3)')
    parser.add_argument('--top-n', type=int, default=30, help='Top-N候補数 (default: 30)')
    parser.add_argument('--track', type=str, default='hanshin', help='競馬場 (default: hanshin)')
    parser.add_argument('--output', type=str, help='出力ファイルパス (default: results/upset_predictions_{year}.tsv)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 2: 穴馬予測パイプライン")
    print("=" * 80)
    print(f"対象年: {args.year}年")
    print(f"競馬場: {args.track}")
    print(f"閾値: {args.threshold}")
    print(f"Top-N: {args.top_n}")
    
    # モデル読み込み
    ranker, classifiers, classifier_features = load_models(args.ranker, args.classifier)
    
    # データ取得
    df, X_ranker = get_prediction_data(args.year, track=args.track)
    
    # Stage 1: Ranker予測
    df = add_ranker_predictions(df, X_ranker, ranker)
    
    # Stage 2: Classifier予測
    df = add_classifier_predictions(df, classifiers, classifier_features, threshold=args.threshold)
    
    # Top-N抽出
    candidates = extract_top_candidates(df, top_n=args.top_n)
    
    # 評価
    print("\n" + "=" * 80)
    print("予測結果評価")
    print("=" * 80)
    metrics = evaluate_predictions(df)
    
    print(f"総候補数: {metrics['total_candidates']}頭")
    print(f"的中数: {metrics['correct_predictions']}頭")
    print(f"適合率 (Precision): {metrics['precision']:.2f}%")
    print(f"対象レース数: {metrics['total_races']}レース")
    print(f"的中レース数: {metrics['hit_races']}レース")
    print(f"レース的中率: {metrics['hit_rate']:.2f}%")
    print(f"総オッズ: {metrics['total_odds']:.1f}倍")
    print(f"ROI: {metrics['roi']:.1f}%")
    
    # ファイル出力
    if args.output:
        output_path = args.output
    else:
        output_path = f"results/upset_predictions_{args.year}.tsv"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if len(candidates) > 0:
        output_cols = [
            'kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango', 'umaban',
            'popularity_rank', 'predicted_rank', 'predicted_score', 'value_gap',
            'upset_probability', 'kakutei_chakujun', 'tansho_odds'
        ]
        candidates[output_cols].to_csv(output_path, sep='\t', index=False)
        print(f"\n予測結果を {output_path} に保存しました")
    else:
        print(f"\n候補なしのため出力をスキップしました")
    
    print("\n" + "=" * 80)
    print("予測完了!")
    print("=" * 80)


if __name__ == '__main__':
    main()
