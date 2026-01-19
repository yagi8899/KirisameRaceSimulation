"""
穴馬パターン分析スクリプト

実際に的中した穴馬の特徴を分析し、Phase 2設計の知見を得る

分析項目:
1. 穴馬の基本統計（人気分布、的中率、オッズ分布）
2. モデル予測との関係（予測順位、予測スコア、乖離度）
3. 穴馬特有の特徴量パターン
4. レース条件との関係（距離、クラス、馬場状態など）
"""

import psycopg2
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt

from db_query_builder import build_race_data_query
from data_preprocessing import preprocess_race_data
from feature_engineering import create_features, add_advanced_features, add_upset_features

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False


def load_db_config(config_path: str = 'db_config.json') -> dict:
    """データベース設定を読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['database']


def get_data_with_predictions(
    model_path: str,
    years: list,
    track_codes: list = None,
    surface_type: str = 'turf',
    distance_min: int = 1000,
    distance_max: int = 9999,
    kyoso_shubetsu_code: str = None
) -> pd.DataFrame:
    """
    データ取得 + モデル予測を実行
    
    Args:
        model_path: モデルファイルパス
        years: 対象年リスト
        track_codes: 競馬場コードリスト（Noneの場合は全10競馬場）
        surface_type: 路面タイプ ('turf' or 'dirt' or None)
        distance_min: 最小距離
        distance_max: 最大距離
        kyoso_shubetsu_code: 競争種別コード ('12'=3歳, '13'=3歳以上, None=全年齢)
        
    Returns:
        pd.DataFrame: 予測結果付きデータ
    """
    # 全競馬場対応（Phase 2.5）
    if track_codes is None:
        from keiba_constants import TRACK_CODES
        track_codes = list(TRACK_CODES.keys())
    
    print(f"\n{'='*80}")
    print(f"データ取得 & 予測実行")
    print(f"{'='*80}")
    print(f"対象年: {years}")
    print(f"競馬場: {', '.join(track_codes)} ({len(track_codes)}競馬場)")
    print(f"路面タイプ: {'芝・ダート両方' if surface_type is None else surface_type}")
    print(f"距離: {distance_min}m - {distance_max}m")
    print()
    
    # モデル読み込み
    print(f"モデル読み込み: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # DB接続
    db_config = load_db_config()
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        dbname=db_config['dbname']
    )
    
    all_data = []
    
    # surface_typeがNoneの場合は芝・ダート両方を取得
    surface_types = [surface_type] if surface_type else ['turf', 'dirt']
    
    for track_code in track_codes:
        for year in years:
            for surf_type in surface_types:
                print(f"\n{year}年 - 競馬場コード{track_code} - {surf_type}のデータ取得中...")
                
                sql = build_race_data_query(
                    track_code=track_code,
                    year_start=year,
                    year_end=year,
                    surface_type=surf_type,
                    distance_min=distance_min,
                    distance_max=distance_max,
                    kyoso_shubetsu_code=kyoso_shubetsu_code,
                    include_payout=True
                )
                
                df = pd.read_sql_query(sql, conn)
                
                # 正しいレース数カウント
                total_races = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']).ngroups
                print(f"  レース数: {total_races}, 出走頭数: {len(df)}頭")
                
                # データが0件の場合はスキップ（改修工事等で休止していた競馬場対応）
                if len(df) == 0:
                    print(f"  ⚠ データなし。スキップします。")
                    continue
                
                # データ前処理
                df = preprocess_race_data(df, verbose=False)
                
                # 特徴量生成
                X = create_features(df)
                X = add_advanced_features(
                    df=df,
                    X=X,
                    surface_type=surf_type,
                    min_distance=distance_min,
                    max_distance=distance_max,
                    logger=None,
                    inverse_rank=True
                )
                
                # 予測
                df['predicted_score'] = model.predict(X)
                df['predicted_rank'] = df.groupby(
                    ['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango']
                )['predicted_score'].rank(ascending=False, method='first')
                
                df['popularity_rank'] = df['tansho_ninkijun_numeric']
                df['value_gap'] = df['predicted_rank'] - df['popularity_rank']
                
                all_data.append(df)
    
    conn.close()
    
    # 結合
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\n合計: {len(df_all)}頭のデータ")
    
    return df_all


def analyze_upset_basics(df: pd.DataFrame, popularity_threshold: int = 7):
    """
    穴馬の基本統計を分析
    """
    print(f"\n{'='*80}")
    print(f"1. 穴馬の基本統計（{popularity_threshold}番人気以下）")
    print(f"{'='*80}")
    
    # 穴馬の定義
    df_unpopular = df[df['popularity_rank'] >= popularity_threshold].copy()
    df_upset = df_unpopular[df_unpopular['kakutei_chakujun_numeric'] <= 3].copy()
    
    print(f"\n人気薄馬: {len(df_unpopular)}頭")
    print(f"穴馬（3着以内）: {len(df_upset)}頭")
    print(f"穴馬的中率: {len(df_upset) / len(df_unpopular) * 100:.2f}%")
    
    # 人気別の的中率
    print(f"\n人気別の3着以内率:")
    for pop in range(popularity_threshold, min(df['popularity_rank'].max().astype(int) + 1, 19)):
        pop_horses = df[df['popularity_rank'] == pop]
        if len(pop_horses) > 0:
            hit_rate = len(pop_horses[pop_horses['kakutei_chakujun_numeric'] <= 3]) / len(pop_horses) * 100
            print(f"  {pop:2d}番人気: {hit_rate:5.2f}% ({len(pop_horses):3d}頭)")
    
    # オッズ分布
    print(f"\n穴馬のオッズ分布:")
    print(f"  最小: {df_upset['tansho_odds'].min():.1f}倍")
    print(f"  最大: {df_upset['tansho_odds'].max():.1f}倍")
    print(f"  平均: {df_upset['tansho_odds'].mean():.1f}倍")
    print(f"  中央値: {df_upset['tansho_odds'].median():.1f}倍")
    
    # 着順分布
    print(f"\n穴馬の着順分布:")
    for rank in [1, 2, 3]:
        count = len(df_upset[df_upset['kakutei_chakujun_numeric'] == rank])
        pct = count / len(df_upset) * 100
        print(f"  {rank}着: {count}頭 ({pct:.1f}%)")
    
    return df_upset


def analyze_model_predictions(df: pd.DataFrame, df_upset: pd.DataFrame):
    """
    モデル予測と穴馬の関係を分析
    """
    print(f"\n{'='*80}")
    print(f"2. モデル予測との関係")
    print(f"{'='*80}")
    
    # 予測順位分布
    print(f"\n穴馬の予測順位分布:")
    for rank_range in [(1, 3), (4, 6), (7, 9), (10, 18)]:
        start, end = rank_range
        count = len(df_upset[(df_upset['predicted_rank'] >= start) & (df_upset['predicted_rank'] <= end)])
        pct = count / len(df_upset) * 100
        print(f"  予測{start:2d}-{end:2d}位: {count:3d}頭 ({pct:5.1f}%)")
    
    # 予測3位以内の穴馬
    df_upset_top3 = df_upset[df_upset['predicted_rank'] <= 3]
    print(f"\n予測3位以内の穴馬: {len(df_upset_top3)}頭 ({len(df_upset_top3) / len(df_upset) * 100:.1f}%)")
    
    if len(df_upset_top3) > 0:
        print(f"  的中した穴馬のうち予測3位以内: {len(df_upset_top3) / len(df_upset) * 100:.1f}%")
        print(f"  平均人気順位: {df_upset_top3['popularity_rank'].mean():.1f}番人気")
        print(f"  平均オッズ: {df_upset_top3['tansho_odds'].mean():.1f}倍")
    
    # 乖離度分布
    print(f"\n穴馬の乖離度（predicted_rank - popularity_rank）分布:")
    print(f"  最小: {df_upset['value_gap'].min():.1f}")
    print(f"  最大: {df_upset['value_gap'].max():.1f}")
    print(f"  平均: {df_upset['value_gap'].mean():.1f}")
    print(f"  中央値: {df_upset['value_gap'].median():.1f}")
    
    # 乖離度別の分布
    print(f"\n乖離度別の穴馬分布:")
    for threshold in [0, -2, -4, -6, -8, -10]:
        count = len(df_upset[df_upset['value_gap'] < threshold])
        pct = count / len(df_upset) * 100
        print(f"  乖離度 < {threshold:3d}: {count:3d}頭 ({pct:5.1f}%)")
    
    # Phase 1で検出できた穴馬
    phase1_detected = df_upset[
        (df_upset['predicted_rank'] <= 3) &
        (df_upset['value_gap'] < -5.0)
    ]
    print(f"\nPhase 1で検出できた穴馬（予測3位以内 & 乖離度<-5）:")
    print(f"  {len(phase1_detected)}頭 / {len(df_upset)}頭 ({len(phase1_detected) / len(df_upset) * 100:.1f}%)")


def analyze_feature_patterns(df: pd.DataFrame, df_upset: pd.DataFrame):
    """
    穴馬特有の特徴量パターンを分析
    """
    print(f"\n{'='*80}")
    print(f"3. 穴馬特有の特徴量パターン")
    print(f"{'='*80}")
    
    # 重要特徴量のリスト
    key_features = [
        'past_score', 'past_avg_sotai_chakujun', 'kohan_3f_index',
        'time_index', 'relative_ability', 'current_class_score',
        'class_score_change', 'past_score_mean'
    ]
    
    # 人気馬（1-3番人気）との比較
    df_popular = df[(df['popularity_rank'] <= 3) & (df['kakutei_chakujun_numeric'] <= 3)]
    
    print(f"\n人気馬（1-3番人気で3着以内）vs 穴馬の特徴量比較:")
    print(f"{'特徴量':<30} {'人気馬平均':>12} {'穴馬平均':>12} {'差分':>12}")
    print(f"{'-'*70}")
    
    for feat in key_features:
        if feat in df.columns:
            popular_mean = df_popular[feat].mean()
            upset_mean = df_upset[feat].mean()
            diff = upset_mean - popular_mean
            print(f"{feat:<30} {popular_mean:>12.3f} {upset_mean:>12.3f} {diff:>+12.3f}")


def analyze_race_conditions(df: pd.DataFrame, df_upset: pd.DataFrame):
    """
    レース条件と穴馬の関係を分析
    """
    print(f"\n{'='*80}")
    print(f"4. レース条件と穴馬の関係")
    print(f"{'='*80}")
    
    # 距離別
    print(f"\n距離別の穴馬出現率:")
    distance_ranges = [(1700, 2000), (2001, 2400), (2401, 3000), (3001, 9999)]
    for d_min, d_max in distance_ranges:
        df_range = df[(df['kyori'] >= d_min) & (df['kyori'] <= d_max)]
        df_upset_range = df_upset[(df_upset['kyori'] >= d_min) & (df_upset['kyori'] <= d_max)]
        
        if len(df_range) > 0:
            rate = len(df_upset_range) / len(df_range) * 100
            print(f"  {d_min}-{d_max}m: {len(df_upset_range):3d}頭 / {len(df_range):4d}頭 ({rate:.2f}%)")
    
    # クラス別
    if 'current_class_score' in df.columns:
        print(f"\nクラス別の穴馬出現率:")
        class_ranges = [(0, 50), (51, 100), (101, 150), (151, 200)]
        for c_min, c_max in class_ranges:
            df_class = df[(df['current_class_score'] >= c_min) & (df['current_class_score'] <= c_max)]
            df_upset_class = df_upset[(df_upset['current_class_score'] >= c_min) & (df_upset['current_class_score'] <= c_max)]
            
            if len(df_class) > 0:
                rate = len(df_upset_class) / len(df_class) * 100
                print(f"  クラススコア{c_min:3d}-{c_max:3d}: {len(df_upset_class):3d}頭 / {len(df_class):4d}頭 ({rate:.2f}%)")


def save_upset_data(df_upset: pd.DataFrame, output_dir: str = 'results'):
    """
    穴馬データを保存
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    output_cols = [
        'kaisai_nen', 'kaisai_tsukihi', 'keibajo_name', 'race_bango',
        'bamei', 'umaban', 'kakutei_chakujun', 'kakutei_chakujun_numeric',
        'popularity_rank', 'tansho_odds', 'predicted_rank', 'predicted_score',
        'value_gap', 'kyori', 'past_score', 'relative_ability',
        'current_class_score', 'class_score_change'
    ]
    
    output_cols = [col for col in output_cols if col in df_upset.columns]
    
    output_file = Path(output_dir) / 'upset_horses_analysis.tsv'
    df_upset[output_cols].to_csv(output_file, sep='\t', index=False, encoding='utf-8', float_format='%.8f')
    print(f"\n穴馬データを {output_file} に保存しました")


def create_training_dataset(df: pd.DataFrame, popularity_min: int = 7, popularity_max: int = 12):
    """
    Phase 2用の訓練データセットを作成
    
    Args:
        df: 全データ
        popularity_min: 穴馬定義の最小人気順位
        popularity_max: 穴馬定義の最大人気順位
    
    Returns:
        訓練用データフレーム
    """
    print(f"\n{'='*80}")
    print(f"Phase 2訓練データセット作成")
    print(f"{'='*80}")
    
    # 展開要因特徴量の追加（feature_engineering.pyの共通関数を使用）
    print("\n展開要因特徴量を計算中...")
    df = add_upset_features(df)
    print("  ✓ 展開要因特徴量を追加しました")
    
    # ラベル作成: 7-12番人気 & 3着以内 = 1
    df['is_upset'] = (
        (df['popularity_rank'] >= popularity_min) &
        (df['popularity_rank'] <= popularity_max) &
        (df['kakutei_chakujun_numeric'] <= 3)
    ).astype(int)
    
    # 統計情報
    n_upset = df['is_upset'].sum()
    n_total = len(df)
    upset_rate = n_upset / n_total * 100
    
    print(f"\nデータセット統計:")
    print(f"  総データ数: {n_total}頭")
    print(f"  穴馬（is_upset=1）: {n_upset}頭 ({upset_rate:.2f}%)")
    print(f"  非穴馬（is_upset=0）: {n_total - n_upset}頭 ({100 - upset_rate:.2f}%)")
    print(f"  不均衡比率: 1:{(n_total - n_upset) / n_upset:.1f}")
    
    # 訓練用特徴量を選択
    feature_cols = [
        # ランキングモデルの出力
        'predicted_rank', 'predicted_score',
        
        # 人気・オッズ情報
        'popularity_rank', 'tansho_odds', 'value_gap',
        
        # 既存の重要特徴量
        'past_score', 'past_avg_sotai_chakujun', 'kohan_3f_index',
        'time_index', 'relative_ability', 'current_class_score',
        'class_score_change', 'past_score_mean',
        
        # 展開要因（新規）
        'estimated_running_style', 'avg_4corner_position',
        'distance_change', 'wakuban_inner', 'wakuban_outer',
        'prev_rank_change',
        
        # レース条件
        'kyori', 'baba_jotai_code_numeric', 'tenko_code',
        
        # 競馬場コード（Phase 2.5で追加）
        'keibajo_code_numeric'
    ]
    
    # keibajo_codeを数値化（Phase 2.5）
    if 'keibajo_code' in df.columns:
        df['keibajo_code_numeric'] = df['keibajo_code'].astype(int)
    else:
        df['keibajo_code_numeric'] = 9  # デフォルト（阪神）
    
    # 距離適性スコアなどの追加特徴量（あれば）
    optional_features = [
        'distance_aptitude_score', 'baba_score', 
        'kishu_score', 'chokyoshi_score'
    ]
    
    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # 存在する特徴量のみ選択
    available_features = [col for col in feature_cols if col in df.columns]
    
    print(f"\n使用特徴量数: {len(available_features)}個")
    print(f"特徴量: {', '.join(available_features[:10])}...")
    
    # 訓練データ作成
    training_cols = available_features + [
        'is_upset',
        'kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango',
        'bamei', 'umaban', 'kakutei_chakujun_numeric'
    ]
    
    df_training = df[[col for col in training_cols if col in df.columns]].copy()
    
    # 欠損値を0で埋める
    df_training = df_training.fillna(0)
    
    return df_training, available_features


def main():
    """
    メイン処理
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='穴馬パターン分析＆訓練データ作成')
    parser.add_argument('--all-tracks', action='store_true', help='全10競馬場対象（Phase 2.5）')
    parser.add_argument('--track-code', type=str, default='09', help='競馬場コード（単一競馬場の場合）')
    args = parser.parse_args()
    
    print("="*80)
    print("穴馬パターン分析")
    print("="*80)
    
    # 設定
    model_path = 'models/hanshin_turf_3ageup_long.sav'
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]  # 2020年除外（Phase 1と統一）
    popularity_threshold = 7  # 7番人気以下を穴馬とする
    
    # 競馬場設定（Phase 2.5対応）
    if args.all_tracks:
        print("モード: 全10競馬場統合（Phase 2.5）")
        track_codes = None  # Noneで全競馬場
        output_suffix = '_universal'
    else:
        print(f"モード: 単一競馬場（コード: {args.track_code}）")
        track_codes = [args.track_code]
        output_suffix = f'_{args.track_code}'
    
    # データ取得 & 予測
    df = get_data_with_predictions(
        model_path=model_path,
        years=years,
        track_codes=track_codes
    )
    
    # 分析実行
    df_upset = analyze_upset_basics(df, popularity_threshold)
    analyze_model_predictions(df, df_upset)
    analyze_feature_patterns(df, df_upset)
    analyze_race_conditions(df, df_upset)
    
    # データ保存
    save_upset_data(df_upset)
    
    # Phase 2訓練データセット作成
    df_training, feature_cols = create_training_dataset(df, popularity_min=7, popularity_max=12)
    
    # 訓練データを保存（Phase 2.5対応）
    output_file = Path('results') / f'upset_training_data{output_suffix}.tsv'
    df_training.to_csv(output_file, sep='\t', index=False, encoding='utf-8', float_format='%.8f')
    print(f"\n訓練データを {output_file} に保存しました")
    print(f"サンプル数: {len(df_training):,}件")
    print(f"穴馬サンプル数: {df_training['is_upset'].sum():,}件")
    print(f"特徴量数: {len(feature_cols)}個")
    
    print(f"\n{'='*80}")
    print("分析完了!")
    print(f"{'='*80}")
    
    # Phase 2への提言
    print(f"\n【Phase 2設計への知見】")
    print(f"1. Phase 1で検出できる穴馬の割合を確認")
    print(f"2. 予測順位が低くても的中する穴馬の特徴を特定")
    print(f"3. 人気馬との特徴量の差分を確認")
    print(f"4. レース条件による穴馬出現率の違いを確認")
    print(f"\nこれらの知見を元に、Phase 2（重み付きor二段階モデル）を設計してください。")


if __name__ == '__main__':
    main()
