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
        'corner_3', 'corner_4', 'kyakushitsu_hantei'
    ]
    
    # 数値化する列のみ処理（文字列列は保持）
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値を0で埋める（数値列のみ、存在する列のみ処理）
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
    df[existing_numeric_columns] = df[existing_numeric_columns].fillna(0)
    
    # 文字列型の列はそのまま保持（kishu_code, chokyoshi_code, bamei など）
    if verbose:
        print(f"  kishu_code型（修正後）: {df['kishu_code'].dtype}")
        print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
        print("[OK] データ前処理完了（文字列列を保持）")
    
    return df
