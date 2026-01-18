"""
特徴量作成の共通化モジュール

model_creator.pyとuniversal_test.pyで共通の特徴量作成ロジックを提供します。
機械学習モデルに入力する特徴量（X）を生成します。
"""

import pandas as pd
import numpy as np


def create_features(df):
    """
    競馬データから機械学習用の特徴量を作成
    
    Args:
        df (pd.DataFrame): 前処理済みのDataFrame
        
    Returns:
        pd.DataFrame: 特徴量DataFrame (X)
    """
    # 基本特徴量を選択
    X = df.loc[:, [
        # "futan_juryo",
        "past_score",
        "kohan_3f_index",
        "past_avg_sotai_chakujun",
        "time_index",
    ]].astype(float)
    
    # 高性能な派生特徴量を追加
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
    
    # 短距離特化特徴量
    # 枠番×距離の相互作用（短距離ほど内枠有利を数値化）
    # 距離が短いほど枠番の影響が大きい: (2000 - 距離) / 1000 で重み付け
    df['wakuban_kyori_interaction'] = df['wakuban'] * (2000 - df['kyori']) / 1000
    X['wakuban_kyori_interaction'] = df['wakuban_kyori_interaction']
    
    # 期待斤量からの差分（年齢別期待斤量との差）
    expected_weight_by_age = {2: 48, 3: 52, 4: 55, 5: 57, 6: 57, 7: 56, 8: 55}
    df['futan_deviation'] = df.apply(
        lambda row: row['futan_juryo'] - expected_weight_by_age.get(row['barei'], 55), 
        axis=1
    )
    X['futan_deviation'] = df['futan_deviation']
        
    return X


def add_advanced_features(
    df: pd.DataFrame, 
    X: pd.DataFrame, 
    surface_type: str, 
    min_distance: int, 
    max_distance: int,
    logger=None,
    inverse_rank: bool = False
) -> pd.DataFrame:
    """
    高度な特徴量を追加（3ファイル共通化版）
    
    Args:
        df: 元データフレーム
        X: 基本特徴量データフレーム
        surface_type: 路面タイプ ('turf' or 'dirt')
        min_distance: 最小距離
        max_distance: 最大距離
        logger: ロガー（Noneの場合はprint使用）
        inverse_rank: 騎手スコア計算で着順を反転するか（model_creator.py用）
    
    Returns:
        pd.DataFrame: 高度特徴量が追加されたデータフレーム
    """
    def log(msg):
        """ログ出力のヘルパー関数"""
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log("[START] 高度な特徴量生成を開始...")
    
    # ========================================
    # 0️⃣ 基本特徴量（SQL側で計算済み）
    # ========================================
    # SQL側で計算済みの特徴量をXに追加
    X['umaban_percentile'] = df['umaban_percentile']
    X['futan_zscore'] = df['futan_zscore']
    X['futan_percentile'] = df['futan_percentile']
    
    # 時系列順にソート（必要な場合のみ使用）
    df_sorted = df.sort_values(['ketto_toroku_bango', 'kaisai_nen', 'kaisai_tsukihi']).copy()
    
    # ========================================
    # 1️⃣ 距離適性スコア
    # ========================================
    log("  [1/7] 距離適性スコアを計算中...")
    
    # 距離帯別スコアを重み付け平均で統合
    # 各距離帯の中心値から現在レースの距離までの差で重み付け
    def get_distance_score_weighted(row):
        kyori = row['kyori']
        
        # 各距離帯の中心値（m）
        centers = {'short': 1200, 'mile': 1600, 'middle': 2100, 'long': 2600}
        
        scores, weights = [], []
        for key, center in centers.items():
            score = row.get(f'past_score_{key}')
            if pd.notna(score):
                # 距離差200mごとに重みを0.8倍に減衰
                distance_diff = abs(kyori - center)
                weight = 0.8 ** (distance_diff / 200)
                scores.append(score)
                weights.append(weight)
        
        # 重み付け平均、実績がない場合は0.5（中立）
        return np.average(scores, weights=weights) if len(scores) > 0 else 0.5
    
    df['similar_distance_score'] = df.apply(get_distance_score_weighted, axis=1)
    X['similar_distance_score'] = df['similar_distance_score']
    
    # SQL側で計算済みの特徴量をXに追加
    X['zenso_kyori_sa'] = df['zenso_kyori_sa']
    X['long_distance_experience_count'] = df['long_distance_experience_count']
    
    # ========================================
    # 2️⃣ スタート指数（SQL側で計算済み）
    # ========================================
    log("  [2/7] スタート指数を計算中...")
    X['start_index'] = df['start_index']
    
    # ========================================
    # 3️⃣ コーナー通過位置スコア（SQL側で計算済み）
    # ========================================
    log("  [3/7] コーナー通過位置スコアを計算中...")
    X['corner_position_score'] = df['corner_position_score']
    
    # ========================================
    # 4️⃣ 馬場適性スコア（SQL側で計算済み）
    # ========================================
    log("  [4/7] 馬場適性スコアを計算中...")
    X['surface_aptitude_score'] = df['surface_aptitude_score']
    
    # ========================================
    # 5️⃣ 騎手スコア（SQL側で計算済み + 人気差スコアのみPython計算）
    # ========================================
    log("  [5/7] 騎手スコアを計算中...")
        
    # SQL側で計算済みの騎手スコアをXに追加
    X['kishu_skill_score'] = df['kishu_skill_score']
    X['kishu_surface_score'] = df['kishu_surface_score']
    
    # ========================================
    # 6️⃣ 調教師スコア（SQL側で計算済み）
    # ========================================
    log("  [6/7] 調教師スコアを計算中...")
    X['chokyoshi_recent_score'] = df['chokyoshi_recent_score']
    
    # ========================================
    # 7️⃣ 路面×距離別特徴量選択
    # ========================================
    log("  [7/7] 路面×距離別特徴量選択を実施中...")
    log(f"    路面: {surface_type}, 距離: {min_distance}m 〜 {max_distance}m")
    
    is_turf = surface_type.lower() == 'turf'
    is_short = max_distance <= 1600
    is_long = min_distance >= 1700
    
    # 短距離専用特徴量の調整
    if is_short:
        log(f"    [短距離モデル] 短距離特化特徴量を使用")
        # 短距離では長距離特化特徴量を削除
        if 'long_distance_experience_count' in X.columns:
            X = X.drop(columns=['long_distance_experience_count'])
            log(f"      削除: long_distance_experience_count")
    else:
        log(f"    [中長距離モデル] 短距離特化特徴量を削除")
        # 中長距離では短距離特化特徴量を削除
        features_to_remove_for_long = ['start_index', 'corner_position_score', 'zenso_kyori_sa']
        for feature in features_to_remove_for_long:
            if feature in X.columns:
                X = X.drop(columns=[feature])
                log(f"      削除: {feature}")
        
        # 長距離(2200m以上)では長距離特化特徴量を残す
        if min_distance >= 2200:
            log(f"    [長距離モデル] 長距離特化特徴量を使用")
        else:
            # 中距離では長距離特化特徴量も削除
            if 'long_distance_experience_count' in X.columns:
                X = X.drop(columns=['long_distance_experience_count'])
                log(f"      削除: long_distance_experience_count")
    
    # 路面×距離別の特徴量削除
    features_to_remove = []
    
    # wakuban_kyori_interactionは短距離モデル専用なので、中長距離では削除
    if not is_short and 'wakuban_kyori_interaction' in X.columns:
        X = X.drop(columns=['wakuban_kyori_interaction'])
        log(f"      削除: wakuban_kyori_interaction（中長距離では不要）")
    
    if is_turf and is_long:
        log("    [芝中長距離] 全特徴量を使用（ベースモデル）")
    elif is_turf and is_short:
        log("    [芝短距離] 不要な特徴量を削除")
        features_to_remove = ['kohan_3f_index', 'surface_aptitude_score', 'wakuban_ratio']
    elif not is_turf and is_long:
        log("    [ダート中長距離] 全特徴量を使用")
    elif not is_turf and is_short:
        log("    [ダート短距離] 不要な特徴量を削除")
        features_to_remove = ['kohan_3f_index', 'surface_aptitude_score', 'wakuban_ratio']
    else:
        log("    [中間距離] 全特徴量を使用")
    
    if features_to_remove:
        for feature in features_to_remove:
            if feature in X.columns:
                X = X.drop(columns=[feature])
                log(f"      削除: {feature}")
    
    log(f"  [DONE] 最終特徴量数: {len(X.columns)}個")
    
    return X
