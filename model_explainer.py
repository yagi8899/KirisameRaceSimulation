#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP分析による競馬予測モデル説明スクリプト

学習済みモデルの予測理由をSHAPで可視化・分析します。
- 個別レースの予測理由を詳細表示
- 特徴量の全体的な影響度を可視化
- 特徴量間の相互作用を分析
"""

import psycopg2
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
import os
import json
import argparse
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from model_config_loader import get_all_models
from keiba_constants import format_model_description
from db_query_builder import build_race_data_query
from feature_engineering import create_features, add_advanced_features

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

# プロット保存用ディレクトリ
PLOT_DIR = Path('shap_analysis')
PLOT_DIR.mkdir(exist_ok=True)


def load_model_and_data(model_filename, track_code, kyoso_shubetsu_code, surface_type, 
                        min_distance, max_distance, test_year=2022, sample_size=None):
    """
    モデルとテストデータを読み込む
    
    Args:
        model_filename (str): モデルファイル名
        track_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競争種別コード
        surface_type (str): 'turf' or 'dirt'
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        test_year (int): テスト対象年 (デフォルト: 2022)
        sample_size (int): サンプル数制限 (None=全件)
        
    Returns:
        tuple: (model, X_test, y_test, test_df_full)
    """
    
    # モデル読み込み
    model_path = Path('models') / model_filename
    if not model_path.exists():
        model_path = Path(model_filename)
    
    print(f"📦 モデル読み込み: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # PostgreSQL接続（db_config.jsonから読み込み）
    with open('db_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    conn = psycopg2.connect(**config['database'])
    
    # SQLクエリ（db_query_builder.pyを使用）
    print(f"🔍 データ取得中: {track_code}競馬場 {test_year}年")
    sql = build_race_data_query(
        track_code=track_code,
        year_start=test_year,
        year_end=test_year,
        surface_type=surface_type,
        distance_min=min_distance,
        distance_max=max_distance,
        kyoso_shubetsu_code=kyoso_shubetsu_code,
        include_payout=False
    )
    
    print(f"[+] データ取得: {test_year}年")
    df_raw = pd.read_sql(sql, conn)
    conn.close()
    
    print(f"取得レコード数: {len(df_raw)}")
    
    if len(df_raw) == 0:
        print("[ERROR] データが取得できませんでした")
        return None, None, None, None
    
    # データ前処理
    df = df_raw.copy()
    
    # 文字列として保持すべきカラム
    string_columns = ['kishu_code', 'chokyoshi_code', 'bamei']
    
    # 数値カラムを明示的に定義（string_columnsを除く）
    numeric_columns = [col for col in df.columns if col not in string_columns + 
                      ['kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango', 
                       'keibajo_name', 'ketto_toroku_bango', 'seibetsu_code', 
                       'kyoso_joken_code', 'kyoso_shubetsu_code', 
                       'grade_code', 'track_code']]
    
    # 数値カラムのみを数値型に変換
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # グループキー作成
    df['group_key'] = (df['kaisai_nen'].astype(str) + '_' + 
                       df['kaisai_tsukihi'].astype(str) + '_' + 
                       df['keibajo_code'].astype(str) + '_' + 
                       df['race_bango'].astype(str))
    
    # 特徴量計算（feature_engineering.pyを使用）
    print("🔄 feature_engineering.pyで特徴量を計算中...")
    
    # 基本特徴量を作成
    X = create_features(df)
    
    # 高度な特徴量を追加
    X = add_advanced_features(
        df=df,
        X=X,
        surface_type=surface_type,
        min_distance=min_distance,
        max_distance=max_distance,
        logger=None,
        inverse_rank=False  # SHAP分析では反転不要
    )
    
    print(f"[OK] 特徴量計算完了: {len(X.columns)}個")
    
    # モデルの実際の特徴量名を取得して順序を合わせる
    if hasattr(model, 'feature_name'):
        actual_features = model.feature_name()
        print(f"[LIST] モデルの実際の特徴量: {len(actual_features)}個")
        
        # 不足している特徴量をチェック
        missing = [f for f in actual_features if f not in X.columns]
        if missing:
            print(f"[WARNING] 一部特徴量が不足しています: {missing}")
            print(f"[INFO] 不足特徴量を中立値(0.5)で補完します")
            # 不足特徴量を0.5(中立値)で埋める
            for feat in missing:
                X[feat] = 0.5
        
        # 特徴量の順序をモデルと合わせる
        X = X[actual_features]
    else:
        print("[ERROR] モデルから特徴量名を取得できませんでした")
        return None, None, None, None
    
    # 着順データを取得（SQL側では kakutei_chakujun_numeric として計算済み）
    y = df['kakutei_chakujun_numeric'].values
    
    # サンプリング
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[indices]
        y = y[indices]
        df = df.iloc[indices]
    
    print(f"[OK] データ準備完了: {len(X)}件")
    
    return model, X, y, df


def analyze_shap_global(model, X, feature_names, output_prefix):
    """
    SHAP全体分析（特徴量重要度、依存性プロット）
    
    Args:
        model: LightGBMモデル
        X: 特徴量データ
        feature_names: 特徴量名リスト
        output_prefix: 出力ファイル名プレフィックス
    """
    print("\n[+] SHAP全体分析を実行中...")
    
    # SHAP値計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. Summary Plot（特徴量重要度と分布）
    print("  - Summary Plot作成中...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - 特徴量の影響度と分布', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_prefix}_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 保存: {PLOT_DIR / f'{output_prefix}_summary.png'}")
    
    # 2. Bar Plot（平均絶対SHAP値）
    print("  - Bar Plot作成中...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Bar Plot - 特徴量の平均影響度', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_prefix}_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 保存: {PLOT_DIR / f'{output_prefix}_bar.png'}")
    
    # 3. 上位5特徴量の依存性プロット
    print("  - Dependence Plot作成中...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_features_idx):
        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, 
                            ax=axes[i], show=False)
        axes[i].set_title(f'{feature_names[idx]} の依存性', fontsize=12)
    
    # 最後のサブプロットを非表示
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_prefix}_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 保存: {PLOT_DIR / f'{output_prefix}_dependence.png'}")
    
    # 4. 特徴量重要度をCSV出力
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'lgb_gain': model.feature_importance(importance_type='gain')
    }).sort_values('mean_abs_shap', ascending=False)
    
    csv_path = PLOT_DIR / f'{output_prefix}_importance.csv'
    feature_importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    [OK] 特徴量重要度保存: {csv_path}")
    
    print("\n[LIST] 特徴量重要度トップ10:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    return shap_values, explainer


def analyze_shap_individual(shap_values, explainer, X, df_full, feature_names, 
                            output_prefix, num_samples=5):
    """
    個別レースのSHAP分析
    
    Args:
        shap_values: SHAP値配列
        explainer: SHAPExplainer
        X: 特徴量データ
        df_full: 元データフレーム（馬名などの情報含む）
        feature_names: 特徴量名リスト
        output_prefix: 出力ファイル名プレフィックス
        num_samples: 分析するサンプル数
    """
    print(f"\n[TEST] 個別レース分析（サンプル{num_samples}件）...")
    
    # ランダムにサンプル選択
    sample_indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n--- サンプル {i+1}/{num_samples} ---")
        
        # レース情報
        race_info = df_full.iloc[idx]
        print(f"日付: {race_info['kaisai_nen']}/{race_info['kaisai_tsukihi']}")
        print(f"競馬場: {race_info['keibajo_name']} R{race_info['race_bango']}")
        print(f"馬名: {race_info['bamei']}")
        print(f"実際の着順: {race_info['kakutei_chakujun']:.0f}着")
        print(f"人気: {race_info['tansho_ninkijun_numeric']:.0f}番人気")
        
        # Force Plot
        shap.force_plot(
            explainer.expected_value, 
            shap_values[idx], 
            X.iloc[idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"{race_info['bamei']} - SHAP Force Plot", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f'{output_prefix}_force_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 貢献度トップ10を表示
        shap_contributions = pd.DataFrame({
            'feature': feature_names,
            'value': X.iloc[idx].values,
            'shap_value': shap_values[idx]
        })
        shap_contributions['abs_shap'] = np.abs(shap_contributions['shap_value'])
        shap_contributions = shap_contributions.sort_values('abs_shap', ascending=False)
        
        print("\n貢献度トップ10:")
        for _, row in shap_contributions.head(10).iterrows():
            direction = "↑" if row['shap_value'] > 0 else "↓"
            print(f"  {row['feature']:30s}: {row['value']:8.2f} → SHAP={row['shap_value']:+8.4f} {direction}")
        
        print(f"  [OK] Force Plot保存: {PLOT_DIR / f'{output_prefix}_force_{i+1}.png'}")


def main():
    """
    メイン処理
    """
    # argparseでコマンドライン引数を解析
    parser = argparse.ArgumentParser(
        description='SHAP分析による競馬予測モデル説明',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='モデルファイルパス（例: models/tokyo_turf_3ageup_long.sav）'
    )
    
    parser.add_argument(
        '--test-year',
        type=int,
        default=2023,
        help='テスト対象年（デフォルト: 2023）'
    )
    
    parser.add_argument(
        '--track-code',
        type=str,
        help='競馬場コード（例: 05=東京）'
    )
    
    parser.add_argument(
        '--surface-type',
        type=str,
        choices=['turf', 'dirt'],
        help='路面タイプ（turf or dirt）'
    )
    
    parser.add_argument(
        '--min-distance',
        type=int,
        help='最小距離（例: 1000）'
    )
    
    parser.add_argument(
        '--max-distance',
        type=int,
        help='最大距離（例: 1600）'
    )
    
    parser.add_argument(
        '--kyoso-shubetsu-code',
        type=str,
        help='競争種別コード（例: 13=3歳以上）'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='SHAP分析のサンプル数（デフォルト: 500）'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("[TARGET] SHAP分析による競馬予測モデル説明")
    print("=" * 80)
    
    # 分析対象モデルを選択
    models = get_all_models()
    
    if not models:
        print("[ERROR] model_configs.jsonにモデルが定義されていません")
        return
    
    print("\n利用可能なモデル:")
    for i, model_info in enumerate(models, 1):
        desc = format_model_description(
            model_info['track_code'],
            model_info['kyoso_shubetsu_code'],
            model_info['surface_type'],
            model_info['min_distance'],
            model_info['max_distance']
        )
        print(f"  {i}. {desc}")
    
    # モデル情報を決定
    model_info = None
    
    if args.model:
        # --modelが指定された場合
        model_path = Path(args.model)
        model_filename = model_path.name
        
        # model_configs.jsonから該当モデルを検索
        for m in models:
            if m['model_filename'] == model_filename:
                model_info = m.copy()
                break
        
        if not model_info:
            print(f"[WARNING] モデル {model_filename} がmodel_configs.jsonに見つかりません")
            
            # コマンドライン引数から直接パラメータを取得
            if all([args.track_code, args.surface_type, args.min_distance, args.max_distance, args.kyoso_shubetsu_code]):
                model_info = {
                    'model_filename': model_filename,
                    'track_code': args.track_code,
                    'surface_type': args.surface_type,
                    'min_distance': args.min_distance,
                    'max_distance': args.max_distance,
                    'kyoso_shubetsu_code': args.kyoso_shubetsu_code
                }
                print("[INFO] コマンドライン引数からモデル情報を構築しました")
            else:
                print("[ERROR] モデル情報が不足しています。--track-code, --surface-type, --min-distance, --max-distance, --kyoso-shubetsu-codeを指定してください")
                return
    else:
        # --modelが指定されていない場合、最初のモデルを使用
        model_info = models[0]
        print("[INFO] モデルが指定されていないため、最初のモデルを使用します")
    
    print(f"\n[PIN] 分析対象: {format_model_description(model_info['track_code'], model_info['kyoso_shubetsu_code'], model_info['surface_type'], model_info['min_distance'], model_info['max_distance'])}")
    print(f"[PIN] 対象年: {args.test_year}年")
    
    # モデルとデータ読み込み
    model, X, y, df_full = load_model_and_data(
        model_filename=model_info['model_filename'],
        track_code=model_info['track_code'],
        kyoso_shubetsu_code=model_info['kyoso_shubetsu_code'],
        surface_type=model_info['surface_type'],
        min_distance=model_info['min_distance'],
        max_distance=model_info['max_distance'],
        test_year=args.test_year,
        sample_size=args.sample_size
    )
    
    if model is None:
        print("[ERROR] データ取得に失敗しました")
        return
    
    # 出力ファイル名のプレフィックス
    output_prefix = Path(model_info['model_filename']).stem
    
    # SHAP全体分析
    shap_values, explainer = analyze_shap_global(
        model=model,
        X=X,
        feature_names=X.columns.tolist(),
        output_prefix=output_prefix
    )
    
    # 個別レース分析
    analyze_shap_individual(
        shap_values=shap_values,
        explainer=explainer,
        X=X,
        df_full=df_full,
        feature_names=X.columns.tolist(),
        output_prefix=output_prefix,
        num_samples=5
    )
    
    print("\n" + "=" * 80)
    print("[OK] SHAP分析完了!")
    print(f"[FILE] 結果保存先: {PLOT_DIR.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
