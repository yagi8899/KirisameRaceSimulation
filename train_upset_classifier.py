"""
Phase 3.5: 穴馬分類モデル訓練スクリプト（年指定版）

指定された年範囲でUPSET分類モデルを訓練する

使用方法:
    python train_upset_classifier.py --years 2015-2024
"""

import argparse
import sys
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import psycopg2
from sklearn.metrics import precision_score, recall_score, f1_score

# プロジェクトのルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from db_query_builder import build_race_data_query
from data_preprocessing import preprocess_race_data
from feature_engineering import create_features, add_advanced_features, add_upset_features


def load_db_config(config_path='db_config.json'):
    """DB設定ファイルを読み込み"""
    with open(config_path, 'r') as f:
        config = json.load(f)
        return config['database']  # databaseキー配下を取得


def get_db_connection():
    """PostgreSQL接続を取得"""
    config = load_db_config()
    return psycopg2.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        dbname=config['dbname']
    )


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description='UPSET分類モデルを指定年範囲で訓練')
    parser.add_argument('--years', type=str, required=True, help='訓練年範囲（例: 2015-2024）')
    return parser.parse_args()


def load_data_for_years(year_start, year_end):
    """指定年範囲のデータをDBから取得"""
    print(f"\n{'='*80}")
    print(f"データ取得: {year_start}-{year_end}年")
    print(f"競馬場: 全場統合（Universal Model）")
    print(f"{'='*80}")
    
    conn = get_db_connection()
    all_data = []
    
    # 全競馬場のデータを取得
    track_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    
    for year in range(year_start, year_end + 1):
        for track_code in track_codes:
            print(f"\n{year}年 - 競馬場コード{track_code}のデータ取得中...")
            
            query = build_race_data_query(
                track_code=track_code,
                year_start=year,
                year_end=year,
                surface_type='turf',  # 芝のみ
                distance_min=1000,
                distance_max=9999,
                include_payout=True
            )
            
            df = pd.read_sql_query(query, conn)
            print(f"  取得: {len(df):,}頭")
            all_data.append(df)
    
    conn.close()
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ データ取得完了: {len(df):,}頭")
    
    return df


def prepare_data_for_upset(df):
    """UPSET訓練用データを準備"""
    print(f"\n{'='*80}")
    print("データ前処理")
    print(f"{'='*80}")
    
    # データ前処理
    df = preprocess_race_data(df)
    
    # 特徴量作成
    print("\n特徴量作成中...")
    df = create_features(df)
    df = add_advanced_features(df)
    df = add_upset_features(df)
    
    print(f"✅ 特徴量作成完了")
    
    return df


def create_upset_label(df):
    """is_upsetラベルを作成"""
    print(f"\n{'='*80}")
    print("穴馬ラベル作成")
    print(f"{'='*80}")
    
    df['popularity_rank'] = pd.to_numeric(df['tansho_ninkijun'], errors='coerce')
    df['final_rank'] = pd.to_numeric(df['kakutei_chakujun'], errors='coerce')
    
    # 穴馬: 7-12番人気 & 3着以内
    df['is_upset'] = (
        (df['popularity_rank'] >= 7) & 
        (df['popularity_rank'] <= 12) & 
        (df['final_rank'] <= 3)
    ).astype(int)
    
    n_upset = df['is_upset'].sum()
    n_total = len(df)
    upset_rate = n_upset / n_total * 100
    
    print(f"  総データ数: {n_total:,}頭")
    print(f"  穴馬（is_upset=1）: {n_upset:,}頭 ({upset_rate:.2f}%)")
    print(f"  不均衡比率: 1:{(n_total - n_upset) / n_upset:.1f}")
    
    return df


def prepare_features(df):
    """特徴量とラベルを準備"""
    print(f"\n{'='*80}")
    print("特徴量準備")
    print(f"{'='*80}")
    
    # 特徴量カラム選択
    exclude_cols = [
        'is_upset', 'kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango',
        'bamei', 'umaban', 'kakutei_chakujun', 'kakutei_chakujun_numeric',
        'tansho_ninkijun', 'final_rank', 'popularity_rank'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
    
    X = df[feature_cols].copy()
    y = df['is_upset'].copy()
    
    # 欠損値・無限大処理
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"✅ 特徴量数: {X.shape[1]}個")
    
    return X, y


def train_upset_classifier(X, y, n_estimators=5):
    """UPSET分類モデルを訓練"""
    print(f"\n{'='*80}")
    print(f"UPSET分類モデル訓練（{n_estimators}モデルアンサンブル）")
    print(f"{'='*80}")
    
    # クラスウェイト計算
    n_upset = y.sum()
    n_normal = len(y) - n_upset
    scale_pos_weight = n_normal / n_upset
    
    print(f"\nクラスウェイト: scale_pos_weight={scale_pos_weight:.2f}")
    
    # LightGBMパラメータ
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'seed': 42
    }
    
    models = []
    
    for i in range(n_estimators):
        print(f"\n[Model {i+1}/{n_estimators}]")
        
        # シード変更
        params['seed'] = 42 + i
        
        # データセット作成
        train_data = lgb.Dataset(X, label=y)
        
        # 訓練
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        models.append(model)
        
        # 訓練データで評価
        y_pred_proba = model.predict(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1: {f1:.4f}")
    
    print(f"\n✅ 訓練完了: {n_estimators}モデル")
    
    return models


def save_models(models, year_start, year_end):
    """モデルを保存"""
    output_dir = Path('walk_forward_results_custom2/period_10/models/2025')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"upset_classifier_{year_start}-{year_end}.sav"
    model_path = output_dir / model_name
    
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n{'='*80}")
    print(f"モデル保存完了!")
    print(f"{'='*80}")
    print(f"保存先: {model_path}")
    print(f"モデル数: {len(models)}個")
    
    return model_path


def main():
    """メイン処理"""
    args = parse_arguments()
    
    # 年範囲をパース
    year_range = args.years.split('-')
    if len(year_range) != 2:
        print("エラー: --yearsは 'YYYY-YYYY' 形式で指定してください")
        sys.exit(1)
    
    year_start = int(year_range[0])
    year_end = int(year_range[1])
    
    print(f"\n{'='*80}")
    print(f"Phase 3.5: UPSET分類モデル訓練")
    print(f"{'='*80}")
    print(f"訓練期間: {year_start}-{year_end}年")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: データ取得
    df = load_data_for_years(year_start, year_end)
    
    # Step 2: データ前処理・特徴量作成
    df = prepare_data_for_upset(df)
    
    # Step 3: 穴馬ラベル作成
    df = create_upset_label(df)
    
    # Step 4: 特徴量準備
    X, y = prepare_features(df)
    
    # Step 5: モデル訓練
    models = train_upset_classifier(X, y, n_estimators=5)
    
    # Step 6: モデル保存
    model_path = save_models(models, year_start, year_end)
    
    print(f"\n{'='*80}")
    print("訓練完了!")
    print(f"{'='*80}")
    print("\n次のステップ:")
    print(f"  1. 特徴量重要度分析:")
    print(f"     python analyze_upset_model_features.py \"{model_path}\"")
    print(f"  2. 閾値最適化:")
    print(f"     python analyze_upset_threshold.py \"{model_path}\"")
    print(f"  3. 有効性検証:")
    print(f"     python validate_feature_effectiveness.py")


if __name__ == '__main__':
    main()
