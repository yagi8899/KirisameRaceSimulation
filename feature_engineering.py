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
    inverse_rank: bool = False,
    include_upset_phase1: bool = False  # 🆕 Phase 1特徴量を含めるか
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
        include_upset_phase1: Phase 1穴馬予測強化特徴量を含めるか（upset_classifier専用）
    
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
    
    # 🔥 Tier S（最優先）: ランキング学習必須特徴量
    log("  [0/7] Tier S特徴量（ランキング学習）を追加中...")
    X['current_class_score'] = df['current_class_score']
    X['class_score_change'] = df['class_score_change']
    X['kyuyo_kikan'] = df['kyuyo_kikan']
    X['past_score_mean'] = df['past_score_mean']
    X['relative_ability'] = df['relative_ability']
    log("    追加: current_class_score, class_score_change, kyuyo_kikan, past_score_mean, relative_ability")
    
    # 🟢 Tier A（優先）: ランキング差別化特徴量
    log("  [0.5/7] Tier A特徴量（ランキング差別化）を追加中...")
    X['left_direction_score'] = df['left_direction_score']
    X['right_direction_score'] = df['right_direction_score']
    X['current_direction_match'] = df['current_direction_match']
    log("    追加: left_direction_score, right_direction_score, current_direction_match")
    
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
    # 🆕 Phase 1: 穴馬予測強化特徴量（2026-01-20 追加）
    # ========================================
    # upset_classifier専用の特徴量なので、フラグがTrueのときのみ追加
    if include_upset_phase1:
        log("  [Phase 1] 穴馬予測強化特徴量を追加中...")
        
        # SQL側で計算済みのPhase 1特徴量をXに追加
        if 'is_turf_bad_condition' in df.columns:
            X['is_turf_bad_condition'] = df['is_turf_bad_condition']
            log("    追加: is_turf_bad_condition (芝不良フラグ)")
        
        if 'is_turf_heavy' in df.columns:
            X['is_turf_heavy'] = df['is_turf_heavy']
            log("    追加: is_turf_heavy (芝重フラグ)")
        
        if 'is_local_track' in df.columns:
            X['is_local_track'] = df['is_local_track']
            log("    追加: is_local_track (ローカル競馬場フラグ)")
        
        if 'is_open_class' in df.columns:
            X['is_open_class'] = df['is_open_class']
            log("    追加: is_open_class (オープンクラスフラグ)")
        
        if 'is_3win_class' in df.columns:
            X['is_3win_class'] = df['is_3win_class']
            log("    追加: is_3win_class (3勝クラスフラグ)")
        
        if 'is_age_prime' in df.columns:
            X['is_age_prime'] = df['is_age_prime']
            log("    追加: is_age_prime (最盛期年齢フラグ)")
        
        if 'zenso_top6' in df.columns:
            X['zenso_top6'] = df['zenso_top6']
            log("    追加: zenso_top6 (前走6着以内フラグ)")
        
        if 'rest_days_fresh' in df.columns:
            X['rest_days_fresh'] = df['rest_days_fresh']
            log("    追加: rest_days_fresh (休養1-3週フラグ)")
    else:
        log("  [Phase 1] 穴馬予測強化特徴量はスキップ (Universal Ranker用)")
    
    # ========================================
    # 7️⃣ 路面×距離別特徴量選択
    # ========================================
    log("  [7/7] 路面×距離別特徴量選択を実施中...")
    log(f"    路面: {surface_type}, 距離: {min_distance}m 〜 {max_distance}m")
    
    is_turf = surface_type.lower() == 'turf' if surface_type else False
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


def add_upset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    穴馬予測用の展開要因特徴量を追加
    
    Phase 2.5で追加: universal_test.py、batch_model_creator.py、walk_forward_validation.pyで共通利用
    
    Args:
        df: 元データフレーム（前処理済み、基本特徴量あり）
        
    Returns:
        pd.DataFrame: 展開要因特徴量が追加されたDataFrame
        
    追加される特徴量:
        - estimated_running_style: 推定脚質 (0=逃げ先行, 1=差し, 2=追込)
        - avg_4corner_position: 4コーナー平均位置
        - distance_change: 距離変化 (今回距離 - 前走距離)
        - wakuban_inner: 内枠フラグ (1-3枠=1)
        - wakuban_outer: 外枠フラグ (6-8枠=1)
    """
    # 1. 推定脚質: 4コーナー位置の平均から推定
    if 'corner_4_numeric' in df.columns and 'bamei' in df.columns:
        df['avg_4corner_position'] = df.groupby('bamei')['corner_4_numeric'].transform('mean')
        # 0-3位=逃げ先行, 4-8位=差し, 9位以降=追込
        df['estimated_running_style'] = pd.cut(
            df['avg_4corner_position'],
            bins=[0, 3, 8, 18],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(float)
    else:
        df['avg_4corner_position'] = 9  # デフォルト値（中団）
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
    
    # 欠損値を0で埋める
    upset_feature_cols = [
        'estimated_running_style', 'avg_4corner_position', 'distance_change',
        'wakuban_inner', 'wakuban_outer'
    ]
    for col in upset_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def add_upset_specific_features(X: pd.DataFrame, df: pd.DataFrame, log=print):
    """
    穴馬予測専用の特徴量を追加（Phase 3 & Phase 3.5）
    
    通常のランキングモデル（model_creator.py）では使用せず、
    穴馬分類モデル（upset_classifier_creator.py）でのみ使用する
    
    Args:
        X: 既存の特徴量DataFrame
        df: 元データのDataFrame
        log: ログ出力関数（デフォルトはprint）
        
    Returns:
        pd.DataFrame: 穴馬特化特徴量を追加したX
    """
    log("  [Phase 3] 穴馬特化特徴量（フェーズ1・SQL実装）を追加中...")
    
    # Phase 3: 実装済み4特徴量
    if 'past_score_std' in df.columns:
        X['past_score_std'] = df['past_score_std'].fillna(-1)  # 欠損値を-1に（過去データなしを明示）
        log("    追加: past_score_std（成績スコア標準偏差）※欠損値=-1")
    
    if 'past_chakujun_variance' in df.columns:
        X['past_chakujun_variance'] = df['past_chakujun_variance'].fillna(-1)  # 欠損値を-1に
        log("    追加: past_chakujun_variance（着順分散）※欠損値=-1")
    
    if 'zenso_oikomi_power' in df.columns:
        X['zenso_oikomi_power'] = df['zenso_oikomi_power'].fillna(0.0)
        log("    追加: zenso_oikomi_power（前走追い込み力）")
    
    if 'zenso_kakoi_komon' in df.columns:
        X['zenso_kakoi_komon'] = df['zenso_kakoi_komon'].fillna(0.0)
        log("    追加: zenso_kakoi_komon（前走包まれ度）")
    
    log("  [Phase 3.5] 穴馬特化特徴量（追加実装 2026-01-20）を追加中...")
    
    # Phase 3.5: 新規5特徴量
    if 'zenso_ninki_gap' in df.columns:
        X['zenso_ninki_gap'] = df['zenso_ninki_gap'].fillna(-1)  # 欠損値を-1に（前走データなし）
        log("    追加: zenso_ninki_gap（前走人気着順ギャップ）※欠損値=-1")
    
    if 'zenso_nigeba' in df.columns:
        X['zenso_nigeba'] = df['zenso_nigeba'].fillna(0)  # 欠損値を0に（逃げではない）
        log("    追加: zenso_nigeba（前走逃げ成功フラグ）※欠損値=0")
    
    if 'zenso_taihai' in df.columns:
        X['zenso_taihai'] = df['zenso_taihai'].fillna(0)  # 欠損値を0に（大敗ではない）
        log("    追加: zenso_taihai（前走大敗フラグ）※欠損値=0")
    
    if 'zenso_agari_rank' in df.columns:
        X['zenso_agari_rank'] = df['zenso_agari_rank'].fillna(-1)  # 欠損値を-1に（データなし）
        log("    追加: zenso_agari_rank（前走上がり順位）※欠損値=-1")
    
    if 'saikin_kaikakuritsu' in df.columns:
        X['saikin_kaikakuritsu'] = df['saikin_kaikakuritsu'].fillna(0.5)  # 欠損値を0.5に（中立）
        log("    追加: saikin_kaikakuritsu（直近3走改善率）※欠損値=0.5")
    
    log("  [Phase 3.5.1] 騎手・調教師・馬統計特徴量（2026-01-20 追加）を追加中...")
    
    # Phase 3.5.1: SQL実装済み10特徴量
    if 'jockey_win_rate' in df.columns:
        X['jockey_win_rate'] = df['jockey_win_rate'].fillna(0.5)  # 欠損値を0.5に（中立）
        log("    追加: jockey_win_rate（騎手勝率）※欠損値=0.5")
    
    if 'jockey_place_rate' in df.columns:
        X['jockey_place_rate'] = df['jockey_place_rate'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: jockey_place_rate（騎手連対率）※欠損値=0.5")
    
    if 'jockey_recent_form' in df.columns:
        X['jockey_recent_form'] = df['jockey_recent_form'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: jockey_recent_form（騎手最近成績）※欠損値=0.5")
    
    if 'trainer_win_rate' in df.columns:
        X['trainer_win_rate'] = df['trainer_win_rate'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: trainer_win_rate（調教師勝率）※欠損値=0.5")
    
    if 'trainer_place_rate' in df.columns:
        X['trainer_place_rate'] = df['trainer_place_rate'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: trainer_place_rate（調教師連対率）※欠損値=0.5")
    
    if 'trainer_recent_form' in df.columns:
        X['trainer_recent_form'] = df['trainer_recent_form'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: trainer_recent_form（調教師最近成績）※欠損値=0.5")
    
    if 'horse_career_win_rate' in df.columns:
        X['horse_career_win_rate'] = df['horse_career_win_rate'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: horse_career_win_rate（馬通算勝率）※欠損値=0.5")
    
    if 'horse_career_place_rate' in df.columns:
        X['horse_career_place_rate'] = df['horse_career_place_rate'].fillna(0.5)  # 欠損値を0.5に
        log("    追加: horse_career_place_rate（馬通算連対率）※欠損値=0.5")
    
    if 'rest_weeks' in df.columns:
        X['rest_weeks'] = df['rest_weeks'].fillna(4.0)  # 欠損値を4週（中立値）に
        log("    追加: rest_weeks（休養週数）※欠損値=4.0")
    
    # 削除予定の特徴量を除外（Phase 3.5）
    drop_cols = ['wakuban_inner', 'wakuban_outer', 'estimated_running_style', 
                 'tenko_code', 'distance_change']
    existing_drop_cols = [col for col in drop_cols if col in X.columns]
    if existing_drop_cols:
        X = X.drop(columns=existing_drop_cols)
        log(f"    削除: {', '.join(existing_drop_cols)}（Phase 3.5で削除）")
    
    return X
