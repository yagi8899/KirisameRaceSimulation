"""
Phase 2.5: 穴馬モデルクイックテスト

SMOTE削除後のモデルで確率分布を確認
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def fetch_test_data():
    """
    阪神芝短距離2023のテストデータを取得（学習データから抽出）
    """
    # 学習データを読み込み
    df = pd.read_csv('results/upset_training_data_universal.tsv', sep='\t')
    
    # 阪神芝短距離2023でフィルタリング
    df_test = df[
        (df['kaisai_nen'] == 2023) &
        (df['keibajo_code'] == 6) &  # 阪神
        (df['kyori'] >= 1200) & (df['kyori'] <= 1400) &  # 短距離
        (df['popularity_rank'] >= 7) & (df['popularity_rank'] <= 12)  # 7-12番人気
    ].copy()
    
    return df_test


def quick_test():
    """
    阪神芝短距離2023で穴馬モデルをテスト
    """
    print("="*80)
    print("Phase 2.5: 穴馬モデルクイックテスト（SMOTE削除版）")
    print("="*80)
    print()
    
    # モデル読み込み
    model_path = Path('models/upset_classifier_universal.sav')
    if not model_path.exists():
        print(f"❌ モデルが見つかりません: {model_path}")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    models = model_data['models']
    feature_cols = model_data['feature_cols']  # 'features' → 'feature_cols'
    
    print(f"✅ モデル読み込み完了")
    print(f"  モデル数: {len(models)}個")
    print(f"  特徴量数: {len(feature_cols)}個")
    print(f"  特徴量: {', '.join(feature_cols)}")
    print()
    
    # テストデータ取得
    print("テストデータ取得中...")
    df = fetch_test_data()
    
    print(f"  データ数: {len(df)}頭")
    print(f"  競馬場: 阪神（芝短距離1200-1400m）")
    print(f"  人気範囲: 7-12番人気")
    print(f"  年: 2023")
    print()
    
    # 穴馬確率予測
    X = df[feature_cols]
    
    # アンサンブル予測（5モデルの平均）
    predictions = np.zeros(len(X))
    for model in models:
        predictions += model.predict(X, num_iteration=model.best_iteration)
    predictions /= len(models)
    
    df['upset_probability'] = predictions
    df['is_upset'] = df['is_upset'].astype(int)  # 既にカラムが存在
    df['chakujun_numeric'] = df['kakutei_chakujun_numeric']  # カラム名マッピング
    
    # 確率分布を確認
    print("="*80)
    print("確率分布")
    print("="*80)
    print(f"最小値: {predictions.min():.6f}")
    print(f"最大値: {predictions.max():.6f}")
    print(f"平均値: {predictions.mean():.6f}")
    print(f"中央値: {np.median(predictions):.6f}")
    print(f"標準偏差: {predictions.std():.6f}")
    print()
    
    # パーセンタイル
    percentiles = [50, 75, 90, 95, 99]
    print("パーセンタイル:")
    for p in percentiles:
        value = np.percentile(predictions, p)
        print(f"  {p}%: {value:.6f}")
    print()
    
    # 実際の結果
    actual_upsets = df[df['is_upset'] == 1]
    
    print(f"実際の穴馬（3着以内）: {len(actual_upsets)}頭")
    if len(actual_upsets) > 0:
        print(f"穴馬の確率分布:")
        print(f"  平均: {actual_upsets['upset_probability'].mean():.6f}")
        print(f"  中央値: {actual_upsets['upset_probability'].median():.6f}")
        print(f"  最小: {actual_upsets['upset_probability'].min():.6f}")
        print(f"  最大: {actual_upsets['upset_probability'].max():.6f}")
        print()
        print("穴馬の詳細:")
        upset_details = actual_upsets[['bamei', 'popularity_rank', 'chakujun_numeric', 'upset_probability', 'tansho_odds']].copy()
        upset_details = upset_details.sort_values('upset_probability', ascending=False)
        print(upset_details.to_string(index=False))
    print()
    
    # 各閾値での評価
    print("="*80)
    print("閾値別評価")
    print("="*80)
    
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    results = []
    
    for threshold in thresholds:
        candidates = df[df['upset_probability'] > threshold]
        hits = len(candidates[candidates['is_upset'] == 1])
        
        if len(candidates) > 0:
            precision = hits / len(candidates)
            roi = (candidates[candidates['is_upset'] == 1]['tansho_odds'].sum() * 100 - len(candidates) * 100) / (len(candidates) * 100) if len(candidates) > 0 else 0
        else:
            precision = 0
            roi = 0
        
        recall = hits / len(actual_upsets) if len(actual_upsets) > 0 else 0
        
        results.append({
            '閾値': threshold,
            '候補数': len(candidates),
            '的中数': hits,
            '適合率': f"{precision:.2%}",
            '再現率': f"{recall:.2%}",
            'ROI': f"{roi:.1%}"
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print()
    
    print("="*80)
    print("テスト完了")
    print("="*80)


if __name__ == '__main__':
    quick_test()
