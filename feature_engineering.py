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
    
    # コメントアウトされた特徴量（将来的に追加可能）
    # futan_per_bareiの非線形変換
    # df['futan_per_barei_log'] = np.log(df['futan_per_barei'].clip(lower=0.1))
    # X['futan_per_barei_log'] = df['futan_per_barei_log']
    
    # 複数のピーク年齢パターン
    # df['barei_peak_distance'] = abs(df['barei'] - 4)  # 4歳をピークと仮定
    # X['barei_peak_distance'] = df['barei_peak_distance']
    
    # 3歳短距離ピーク（早熟型）
    # df['barei_peak_short'] = abs(df['barei'] - 3)
    # X['barei_peak_short'] = df['barei_peak_short']
    
    # 5歳長距離ピーク（晩成型）
    # df['barei_peak_long'] = abs(df['barei'] - 5)
    # X['barei_peak_long'] = df['barei_peak_long']
    
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
    log("  [1/7] 距離適性スコアを計算中...")
    
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
    log("  [2/7] スタート指数を計算中...")
    
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
    log("  [3/7] コーナー通過位置スコアを計算中...")
    
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
    log("  [4/7] 馬場適性スコアを計算中...")
    
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
    log("  [5/7] 騎手スコアを計算中...")
    
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
                    if inverse_rank:
                        # model_creator.py用: 着順を反転（1着=高スコア）
                        recent_races['rank_score'] = 1.0 - ((18 - recent_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                    else:
                        # universal_test.py/sokuho_prediction.py用: 着順そのまま（1着=低スコア）
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
                        
                        if inverse_rank:
                            # model_creator.py用
                            valid_races['actual_score'] = 1.0 - ((18 - valid_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                        else:
                            # universal_test.py/sokuho_prediction.py用
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
                    if inverse_rank:
                        # model_creator.py用
                        avg_score = (1.0 - ((18 - recent_same_surface['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                    else:
                        # universal_test.py/sokuho_prediction.py用
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
    log("  [6/7] 調教師スコアを計算中...")
    
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
                    if inverse_rank:
                        # model_creator.py用
                        avg_score = (1.0 - ((18 - recent_races['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                    else:
                        # universal_test.py/sokuho_prediction.py用
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
