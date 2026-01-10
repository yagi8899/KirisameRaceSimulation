"""
特徴量作成の共通化モジュール

model_creator.pyとuniversal_test.pyで共通の特徴量作成ロジックを提供します。
機械学習モデルに入力する特徴量（X）を生成します。
"""

import pandas as pd


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
