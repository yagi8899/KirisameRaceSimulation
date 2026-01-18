"""
データ前処理の共通化モジュール

model_creator.pyとuniversal_test.pyで共通のデータ前処理ロジックを提供します。
騎手コード・調教師コード・馬名などの文字列列を保持しながら、
数値列のみを適切に処理します。
"""

import pandas as pd


def preprocess_race_data(df, verbose=True):
    """
    競馬データの前処理を実行
    
    Args:
        df (pd.DataFrame): 前処理対象のDataFrame
        verbose (bool): デバッグ情報を表示するか（デフォルト: True）
        
    Returns:
        pd.DataFrame: 前処理済みのDataFrame
    """
    if verbose:
        print("[TEST] データ型確認...")
        print(f"  kishu_code型（修正前）: {df['kishu_code'].dtype}")
        print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
        print(f"  kishu_codeユニーク数: {df['kishu_code'].nunique()}")
    
    # 数値化する列を明示的に指定（文字列列は除外）
    numeric_columns = [
        'wakuban', 'umaban_numeric', 'barei', 'futan_juryo', 'tansho_odds',
        'kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'kyori', 'shusso_tosu',
        'tenko_code', 'babajotai_code', 'grade_code', 'kyoso_joken_code',
        'kyoso_shubetsu_code', 'track_code', 'seibetsu_code',
        'kakutei_chakujun_numeric', 'chakujun_score', 'past_avg_sotai_chakujun',
        'time_index', 'past_score', 'kohan_3f_index', 'corner_1', 'corner_2',
        'corner_3', 'corner_4', 'kyakushitsu_hantei',
        'kishu_skill_score', 'kishu_surface_score', 'chokyoshi_recent_score',
        'umaban_percentile', 'futan_zscore', 'futan_percentile',
        'past_score_short', 'past_score_mile', 'past_score_middle', 'past_score_long',
        'zenso_kyori_sa', 'long_distance_experience_count',
        'start_index', 'corner_position_score', 'surface_aptitude_score',
        # 🔥 Tier S: ランキング学習必須特徴量
        'current_class_score', 'previous_class_score', 'class_score_change',
        'kyuyo_kikan', 'past_score_mean', 'relative_ability',
        # 🟢 Tier A: ランキング差別化特徴量
        'distance_gap', 'track_code_change', 'left_direction_score',
        'right_direction_score', 'current_direction_match'
    ]
    
    # 数値化する列のみ処理（文字列列は保持）
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値を特徴量ごとに適切な中立値で補完
    # ⚠️ NULL=「データなし/未経験」の場合、0にすると「極めて悪い成績」になってしまう特徴量が多数存在
    fill_values = {
        # ========== 基本特徴量 ==========
        # past_score系: SQLのAVGでNULLが返る場合がある（実績0走）→ 中立値50（5着×OP1.0倍相当）
        'past_score': 50.0,                   # NULL = 実績なし → 5着×OP1.0倍 = 30×1.0 ≒ 50
        'past_avg_sotai_chakujun': 0.5,      # NULL = 実績なし → 中間順位相当
        'time_index': 15.0,                  # NULL = 実績なし → 中央値的な速度（15m/s）
        'kohan_3f_index': 0.0,               # NULL = 実績なし → 標準タイム相当（差分0）
        
        # ========== 距離適性特徴量 ==========
        # SQL側でNULLが返る場合（該当距離帯の実績なし）
        'past_score_short': 0.5,             # 短距離未経験 → 中立
        'past_score_mile': 0.5,              # マイル未経験 → 中立
        'past_score_middle': 0.5,            # 中距離未経験 → 中立
        'past_score_long': 0.5,              # 長距離未経験 → 中立
        'similar_distance_score': 0.5,       # 全距離帯未経験（デビュー戦）→ 中立
        
        # ========== 馬場適性特徴量 ==========
        'surface_aptitude_score': 0.5,       # 同路面未経験 → 中立
        
        # ========== 騎手・調教師特徴量 ==========
        # SQL側で既に0.5設定済みだが、念のため
        'kishu_skill_score': 0.5,            # SQL側で実装済み
        'kishu_surface_score': 0.5,          # SQL側で実装済み
        'chokyoshi_recent_score': 0.5,       # SQL側で実装済み
        
        # ========== 短距離特化特徴量 ==========
        'start_index': 0.0,                  # デビュー戦 → 平均的な位置取り（補正なし）
        'corner_position_score': 0.5,        # デビュー戦 → 中間的な位置取り
        
        # ========== Tier S: ランキング学習必須特徴量 ==========
        'current_class_score': 0.5,          # 不明なクラス → 中間クラス（念のため）
        'class_score_change': 0.0,           # デビュー戦 → 変化なし
        'kyuyo_kikan': 60,                   # デビュー戦 → 中央値的な休養期間（約2ヶ月）
        'past_score_mean': 50.0,             # 実績なし → 中立値（5着×OP1.0倍相当）
        'relative_ability': 0.0,             # 計算不可 → 平均的（z-score=0）
        
        # ========== Tier A: ランキング差別化特徴量 ==========
        'left_direction_score': 0.5,         # 左回り未経験 → 中立
        'right_direction_score': 0.5,        # 右回り未経験 → 中立
        'current_direction_match': 0.5,      # 未経験 → 中立（SQL側でも0.5設定済み）
    }
    
    # 特徴量ごとに中立値で補完
    for col, fill_val in fill_values.items():
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                df[col] = df[col].fillna(fill_val)
                if verbose:
                    print(f"  {col}: {null_count}件のNULLを{fill_val}で補完")
    
    # カウント系・フラグ系の特徴量は0で埋める（既存の動作を維持）
    zero_fill_features = [
        'long_distance_experience_count',  # カウント系: 0回が正しい
        'zenso_kyori_sa',                 # 変化系: 0=変化なし
        'umaban_percentile',              # SQL側で計算済みのはず
        'futan_zscore',                   # SQL側で計算済みのはず
        'futan_percentile',               # SQL側で計算済みのはず
    ]
    
    for col in zero_fill_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # その他の数値列も0で埋める（後方互換性のため）
    existing_numeric_columns = [col for col in numeric_columns 
                               if col in df.columns 
                               and col not in fill_values 
                               and col not in zero_fill_features]
    if existing_numeric_columns:
        df[existing_numeric_columns] = df[existing_numeric_columns].fillna(0)
    
    # 文字列型の列はそのまま保持（kishu_code, chokyoshi_code, bamei など）
    if verbose:
        print(f"  kishu_code型（修正後）: {df['kishu_code'].dtype}")
        print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
        print("[OK] データ前処理完了（特徴量ごとに適切な中立値で補完）")
    
    return df
