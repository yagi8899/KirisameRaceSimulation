#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: 穴馬検出スクリプト（オッズ乖離検出）

既存のランキングモデルを使用して、予測順位と人気順位の乖離から穴馬を検出します。

使用方法:
    python upset_detector.py <model_path> [options]
    
    引数:
        model_path: モデルファイルのパス（.pkl または .sav）
        
    オプション:
        --test-year: テスト年（デフォルト: 2024）
        --track-code: 競馬場コード（デフォルト: '09' 阪神）
        --threshold: 乖離度閾値（デフォルト: -5.0）
        --popularity-min: 最低人気順位（デフォルト: 10）
        --predicted-rank-max: 予測上位何位まで（デフォルト: 3）
        --output: 出力ファイル名（デフォルト: upset_results_<track>_<year>.tsv）
        --show-plot: Precision-Recallカーブを表示

使用例:
    # 基本的な使用
    python upset_detector.py models/hanshin_turf_3ageup_long.pkl
    
    # 閾値を変更して検証
    python upset_detector.py models/hanshin_turf_3ageup_long.pkl --threshold -7.0
    
    # 人気境界を8番人気以下に変更（函館向け）
    python upset_detector.py models/hakodate_turf_3ageup_long.pkl --popularity-min 8
    
    # グラフ表示付き
    python upset_detector.py models/tokyo_turf_3ageup_long.pkl --show-plot
"""

import psycopg2
import pandas as pd
import pickle
import numpy as np
import json
import argparse
from pathlib import Path
from scipy.stats import rankdata
from typing import Tuple, Dict

from db_query_builder import build_race_data_query
from keiba_constants import get_track_name


def load_db_config(config_path: str = 'db_config.json') -> Dict:
    """
    データベース設定を読み込み
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['database']


def load_model(model_path: str):
    """
    モデルファイルを読み込み
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def detect_upsets(
    model_path: str,
    test_year: int = 2024,
    track_code: str = '09',
    surface_type: str = 'turf',
    distance_min: int = 1700,
    distance_max: int = 9999,
    kyoso_shubetsu_code: str = '13',
    threshold: float = -5.0,
    popularity_min: int = 10,
    predicted_rank_max: int = 3
) -> pd.DataFrame:
    """
    穴馬を検出
    
    Args:
        model_path: モデルファイルのパス
        test_year: テスト年
        track_code: 競馬場コード
        surface_type: 馬場タイプ（'turf' or 'dirt'）
        distance_min: 最小距離
        distance_max: 最大距離
        kyoso_shubetsu_code: 競走種別コード
        threshold: 乖離度閾値（負の値 = 過小評価）
        popularity_min: 最低人気順位（例: 10 = 10番人気以下）
        predicted_rank_max: 予測上位何位まで（例: 3 = 1-3位）
    
    Returns:
        DataFrame: 穴馬候補データ
    """
    print("="*60)
    print("Phase 1: 穴馬検出（オッズ乖離検出）")
    print("="*60)
    
    # モデル読み込み
    print(f"\n[1/5] モデル読み込み: {model_path}")
    model = load_model(model_path)
    
    # データベース接続
    print(f"[2/5] データ取得: {test_year}年 {get_track_name(track_code)}")
    db_config = load_db_config()
    
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        dbname=db_config['dbname']
    )
    
    # SQLクエリ生成
    sql = build_race_data_query(
        track_code=track_code,
        year_start=test_year,
        year_end=test_year,
        surface_type=surface_type,
        distance_min=distance_min,
        distance_max=distance_max,
        kyoso_shubetsu_code=kyoso_shubetsu_code,
        include_payout=True  # 複勝配当情報を含む
    )
    
    df_test = pd.read_sql_query(sql, conn)
    conn.close()
    
    # 正しいレース数カウント（kaisai_nen, kaisai_tsukihi, keibajo_code, race_bangoの組み合わせ）
    total_races = df_test.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']).ngroups
    print(f"   総レース数: {total_races}レース")
    print(f"   総出走頭数: {len(df_test)}頭")
    print(f"   平均出走頭数: {len(df_test) / total_races:.1f}頭/レース" if total_races > 0 else "")
    
    # データ前処理（共通化モジュール使用）
    from data_preprocessing import preprocess_race_data
    from feature_engineering import create_features, add_advanced_features
    
    df_test = preprocess_race_data(df_test, verbose=False)
    
    # 基本特徴量を作成
    X_test = create_features(df_test)
    
    # 高度な特徴量を追加
    X_test = add_advanced_features(
        df=df_test,
        X=X_test,
        surface_type=surface_type,
        min_distance=distance_min,
        max_distance=distance_max,
        logger=None,
        inverse_rank=True  # model_creator.pyと同じ設定
    )
    
    # 予測
    print(f"[3/5] 予測実行")
    predictions = model.predict(X_test)
    df_test['predicted_score'] = predictions
    
    # 予測順位を計算
    df_test['predicted_rank'] = df_test.groupby(
        ['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']
    )['predicted_score'].rank(ascending=False, method='first')
    
    # 乖離度を計算
    print(f"[4/5] 乖離度計算")
    df_test['popularity_rank'] = df_test['tansho_ninkijun_numeric']
    df_test['value_gap'] = df_test['predicted_rank'] - df_test['popularity_rank']
    
    # 乖離度の正規化（レース内でZ-score化）
    df_test['value_gap_normalized'] = df_test.groupby(
        ['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']
    )['value_gap'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    
    # 穴馬候補を抽出
    print(f"[5/5] 穴馬候補抽出")
    print(f"   条件1: 予測上位 1-{predicted_rank_max}位")
    print(f"   条件2: 人気薄 {popularity_min}番人気以下")
    print(f"   条件3: 乖離度 < {threshold}")
    print()
    
    # デバッグ用: 各条件での候補数
    print(f"   [デバッグ] 予測3位以内: {len(df_test[df_test['predicted_rank'] <= predicted_rank_max])}頭")
    print(f"   [デバッグ] {popularity_min}番人気以下: {len(df_test[df_test['popularity_rank'] >= popularity_min])}頭")
    cond1_and_2 = df_test[
        (df_test['predicted_rank'] <= predicted_rank_max) &
        (df_test['popularity_rank'] >= popularity_min)
    ]
    print(f"   [デバッグ] 条件1 AND 2: {len(cond1_and_2)}頭")
    if len(cond1_and_2) > 0:
        print(f"   [デバッグ] その乖離度分布: min={cond1_and_2['value_gap'].min():.1f}, max={cond1_and_2['value_gap'].max():.1f}, mean={cond1_and_2['value_gap'].mean():.1f}")
    print()
    
    upset_candidates = df_test[
        (df_test['predicted_rank'] <= predicted_rank_max) &
        (df_test['popularity_rank'] >= popularity_min) &
        (df_test['value_gap'] < threshold)
    ].copy()
    
    # 実際に3着以内に入った穴馬を集計
    upset_hits = upset_candidates[upset_candidates['kakutei_chakujun_numeric'] <= 3]
    
    # 全体での穴馬統計（ベースライン）
    total_unpopular = len(df_test[df_test['popularity_rank'] >= popularity_min])
    total_unpopular_top3 = len(df_test[
        (df_test['popularity_rank'] >= popularity_min) &
        (df_test['kakutei_chakujun_numeric'] <= 3)
    ])
    
    baseline_rate = total_unpopular_top3 / total_unpopular * 100 if total_unpopular > 0 else 0
    
    # Precision/Recall計算
    precision = len(upset_hits) / len(upset_candidates) * 100 if len(upset_candidates) > 0 else 0
    recall = len(upset_hits) / total_unpopular_top3 * 100 if total_unpopular_top3 > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 回収率計算（複勝）
    roi = 0
    if len(upset_candidates) > 0:
        # 複勝配当カラムの存在確認
        payout_columns = ['複勝1着馬番', '複勝1着オッズ', '複勝2着馬番', '複勝2着オッズ', '複勝3着馬番', '複勝3着オッズ']
        has_payout_data = all(col in df_test.columns for col in payout_columns)
        
        if has_payout_data:
            try:
                # 複勝配当を取得
                upset_candidates = upset_candidates.merge(
                    df_test[['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango', 'umaban',
                             '複勝1着馬番', '複勝1着オッズ',
                             '複勝2着馬番', '複勝2着オッズ',
                             '複勝3着馬番', '複勝3着オッズ']],
                    on=['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango', 'umaban'],
                    how='left',
                    suffixes=('', '_payout')
                )
                
                # 複勝配当を計算
                def get_fukusho_payout(row):
                    umaban = row['umaban']
                    if umaban == row.get('複勝1着馬番', -1):
                        return row.get('複勝1着オッズ', 0)
                    elif umaban == row.get('複勝2着馬番', -1):
                        return row.get('複勝2着オッズ', 0)
                    elif umaban == row.get('複勝3着馬番', -1):
                        return row.get('複勝3着オッズ', 0)
                    else:
                        return 0
                
                upset_candidates['fukusho_payout'] = upset_candidates.apply(get_fukusho_payout, axis=1)
                
                total_bet = len(upset_candidates) * 100  # 100円ずつ賭けた想定
                total_return = upset_candidates['fukusho_payout'].sum() * 100
                roi = total_return / total_bet * 100 if total_bet > 0 else 0
            except Exception as e:
                print(f"   ⚠️ 回収率計算エラー: {e}")
                roi = 0
        else:
            print("   ⚠️ 複勝配当データがありません（回収率計算スキップ）")
    
    # 結果表示
    print("\n" + "="*60)
    print("穴馬検出結果")
    print("="*60)
    print(f"候補数: {len(upset_candidates)}頭")
    print(f"的中数: {len(upset_hits)}頭")
    print(f"---")
    print(f"Precision: {precision:.2f}% （検出した穴馬候補のうち実際に3着以内に入った割合）")
    print(f"Recall: {recall:.2f}% （全穴馬的中のうち検出できた割合）")
    print(f"F1 Score: {f1:.2f}%")
    print(f"---")
    print(f"ベースライン（{popularity_min}番人気以下全体の3着以内率）: {baseline_rate:.2f}%")
    print(f"改善率: {precision - baseline_rate:+.2f}pt ({(precision / baseline_rate - 1) * 100:+.1f}%)" if baseline_rate > 0 else "N/A")
    print(f"---")
    print(f"回収率（複勝・100円ずつ購入想定）: {roi:.1f}%")
    print("="*60)
    
    return upset_candidates


def optimize_threshold(
    model_path: str,
    test_year: int,
    track_code: str,
    surface_type: str = 'turf',
    distance_min: int = 1700,
    distance_max: int = 9999,
    kyoso_shubetsu_code: str = '13',
    popularity_min: int = 10,
    predicted_rank_max: int = 3,
    thresholds: list = None
) -> pd.DataFrame:
    """
    複数の閾値で評価して最適値を探索
    
    Returns:
        DataFrame: 閾値ごとの評価結果
    """
    if thresholds is None:
        thresholds = [0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
    
    print("\n" + "="*60)
    print("閾値最適化")
    print("="*60)
    
    # モデル読み込み
    model = load_model(model_path)
    
    # データ取得（1回のみ）
    db_config = load_db_config()
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        dbname=db_config['dbname']
    )
    
    sql = build_race_data_query(
        track_code=track_code,
        year_start=test_year,
        year_end=test_year,
        surface_type=surface_type,
        distance_min=distance_min,
        distance_max=distance_max,
        kyoso_shubetsu_code=kyoso_shubetsu_code,
        include_payout=True
    )
    
    df_test = pd.read_sql_query(sql, conn)
    conn.close()
    df_test = df_test.fillna(0)
    
    # データ前処理（共通化モジュール使用）
    from data_preprocessing import preprocess_race_data
    from feature_engineering import create_features, add_advanced_features
    
    df_test = preprocess_race_data(df_test, verbose=False)
    
    # 基本特徴量を作成
    X_test = create_features(df_test)
    
    # 高度な特徴量を追加
    X_test = add_advanced_features(
        df=df_test,
        X=X_test,
        surface_type=surface_type,
        min_distance=distance_min,
        max_distance=distance_max,
        logger=None,
        inverse_rank=True
    )
    
    # 予測
    predictions = model.predict(X_test)
    df_test['predicted_score'] = predictions
    df_test['predicted_rank'] = df_test.groupby(
        ['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']
    )['predicted_score'].rank(ascending=False, method='first')
    
    df_test['popularity_rank'] = df_test['tansho_ninkijun_numeric']
    df_test['value_gap'] = df_test['predicted_rank'] - df_test['popularity_rank']
    
    # 各閾値で評価
    results = []
    
    for threshold in thresholds:
        upset_candidates = df_test[
            (df_test['predicted_rank'] <= predicted_rank_max) &
            (df_test['popularity_rank'] >= popularity_min) &
            (df_test['value_gap'] < threshold)
        ].copy()
        
        upset_hits = upset_candidates[upset_candidates['kakutei_chakujun_numeric'] <= 3]
        
        total_unpopular_top3 = len(df_test[
            (df_test['popularity_rank'] >= popularity_min) &
            (df_test['kakutei_chakujun_numeric'] <= 3)
        ])
        
        precision = len(upset_hits) / len(upset_candidates) * 100 if len(upset_candidates) > 0 else 0
        recall = len(upset_hits) / total_unpopular_top3 * 100 if total_unpopular_top3 > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            '閾値': threshold,
            '候補数': len(upset_candidates),
            '的中数': len(upset_hits),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    df_results = pd.DataFrame(results)
    
    print("\n閾値別評価結果:")
    print(df_results.to_string(index=False))
    
    # 最適閾値（F1スコア最大）
    best_idx = df_results['F1 Score'].idxmax()
    best_threshold = df_results.loc[best_idx, '閾値']
    
    print(f"\n推奨閾値: {best_threshold} (F1 Score={df_results.loc[best_idx, 'F1 Score']:.2f}%)")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: 穴馬検出（オッズ乖離検出）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本的な使用
    python upset_detector.py models/hanshin_turf_3ageup_long.pkl
    
    # 閾値を変更
    python upset_detector.py models/hanshin_turf_3ageup_long.pkl --threshold -7.0
    
    # 人気境界を8番人気以下に変更
    python upset_detector.py models/hakodate_turf_3ageup_long.pkl --popularity-min 8
    
    # 閾値最適化
    python upset_detector.py models/tokyo_turf_3ageup_long.pkl --optimize
        """
    )
    
    parser.add_argument('model_path', type=str, help='モデルファイルのパス')
    parser.add_argument('--test-year', type=int, default=2024, help='テスト年（デフォルト: 2024）')
    parser.add_argument('--track-code', type=str, default='09', help='競馬場コード（デフォルト: 09=阪神）')
    parser.add_argument('--surface-type', type=str, default='turf', choices=['turf', 'dirt'], help='馬場タイプ（デフォルト: turf）')
    parser.add_argument('--distance-min', type=int, default=1700, help='最小距離（デフォルト: 1700m）')
    parser.add_argument('--distance-max', type=int, default=9999, help='最大距離（デフォルト: 9999=以上）')
    parser.add_argument('--kyoso-shubetsu', type=str, default='13', help='競走種別コード（デフォルト: 13=3歳以上）')
    parser.add_argument('--threshold', type=float, default=-5.0, help='乖離度閾値（デフォルト: -5.0）')
    parser.add_argument('--popularity-min', type=int, default=10, help='最低人気順位（デフォルト: 10）')
    parser.add_argument('--predicted-rank-max', type=int, default=3, help='予測上位何位まで（デフォルト: 3）')
    parser.add_argument('--output', type=str, default=None, help='出力ファイル名')
    parser.add_argument('--optimize', action='store_true', help='閾値最適化を実行')
    
    args = parser.parse_args()
    
    # モデルファイルの存在確認
    if not Path(args.model_path).exists():
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        return
    
    # 閾値最適化モード
    if args.optimize:
        df_results = optimize_threshold(
            model_path=args.model_path,
            test_year=args.test_year,
            track_code=args.track_code,
            surface_type=args.surface_type,
            distance_min=args.distance_min,
            distance_max=args.distance_max,
            kyoso_shubetsu_code=args.kyoso_shubetsu,
            popularity_min=args.popularity_min,
            predicted_rank_max=args.predicted_rank_max
        )
        
        # 結果保存
        output_file = args.output or f'upset_threshold_optimization_{args.track_code}_{args.test_year}.tsv'
        df_results.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
        print(f"\n最適化結果を {output_file} に保存しました")
        return
    
    # 通常モード
    upset_candidates = detect_upsets(
        model_path=args.model_path,
        test_year=args.test_year,
        track_code=args.track_code,
        surface_type=args.surface_type,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        kyoso_shubetsu_code=args.kyoso_shubetsu,
        threshold=args.threshold,
        popularity_min=args.popularity_min,
        predicted_rank_max=args.predicted_rank_max
    )
    
    # 結果保存
    # resultsフォルダを作成（存在しない場合）
    results_dir = Path('upset_results')
    results_dir.mkdir(exist_ok=True)
    
    output_file = args.output or str(results_dir / f'upset_results_{args.track_code}_{args.test_year}.tsv')
    
    # 保存用に列を選択
    output_columns = [
        'kaisai_nen', 'kaisai_tsukihi', 'keibajo_name', 'race_bango',
        'bamei', 'umaban', 'predicted_rank', 'predicted_score',
        'popularity_rank', 'tansho_odds', 'value_gap', 'value_gap_normalized',
        'kakutei_chakujun', 'kakutei_chakujun_numeric'
    ]
    
    upset_candidates[output_columns].to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"\n結果を {output_file} に保存しました")


if __name__ == '__main__':
    main()
