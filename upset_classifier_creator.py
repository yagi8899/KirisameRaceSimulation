"""
Phase 2: 穴馬分類モデル作成スクリプト

7-12番人気で3着以内に入る穴馬を検出する二値分類モデルを学習

実装内容:
- LightGBM Classifierで二値分類
- クラスウェイト調整で不均衡データに対応（SMOTEなし）
- 5-fold Cross Validationで評価
- モデルをmodels/upset_classifier.savに保存
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb


def load_training_data(file_path: str = 'results/upset_training_data_universal.tsv'):
    """
    訓練データを読み込み（Phase 2.5: 全10競馬場統合データ）
    """
    print(f"訓練データ読み込み: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(f"  データ数: {len(df)}頭")
    print(f"  穴馬: {df['is_upset'].sum()}頭 ({df['is_upset'].mean() * 100:.2f}%)")
    return df


def prepare_features(df: pd.DataFrame):
    """
    特徴量とラベルを準備
    """
    print(f"\n[DEBUG] prepare_features開始: df.shape={df.shape}, df.index範囲=[{df.index.min()}, {df.index.max()}]")
    
    # 特徴量カラム（is_upset, メタ情報以外）
    exclude_cols = [
        'is_upset',
        'kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'race_bango',
        'bamei', 'umaban', 'kakutei_chakujun_numeric'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['is_upset'].copy()
    
    print(f"[DEBUG] X, y作成後: X.shape={X.shape}, y.shape={y.shape}, X.index範囲=[{X.index.min()}, {X.index.max()}]")
    
    # インデックスをリセット（TimeSeriesSplitで正しく動作するように）
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    print(f"[DEBUG] reset_index後: X.shape={X.shape}, y.shape={y.shape}, X.index範囲=[{X.index.min()}, {X.index.max()}]")
    
    # 欠損値を0で埋める
    X = X.fillna(0)
    
    # 無限大を最大値/最小値で置換
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\n特徴量数: {len(feature_cols)}個")
    print(f"特徴量: {', '.join(feature_cols)}")
    
    return X, y, feature_cols


def train_with_class_weights(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    n_splits: int = 5,
    random_state: int = 42,
    use_timeseries: bool = True
):
    """
    クラスウェイトで不均衡データを調整して学習（SMOTEなし）
    
    Args:
        X: 特徴量
        y: ラベル
        feature_cols: 特徴量名リスト
        n_splits: CVのfold数
        random_state: 乱数シード
        use_timeseries: TimeSeriesSplitを使用するか（デフォルト: True）
    
    Returns:
        学習済みモデルのリスト、評価結果
    """
    print(f"\n{'='*80}")
    print(f"クラスウェイトを使った学習開始（SMOTEなし）")
    print(f"{'='*80}")
    if use_timeseries:
        print(f"Cross Validation: TimeSeriesSplit {n_splits}-fold (時系列対応)")
    else:
        print(f"Cross Validation: StratifiedKFold {n_splits}-fold")
    
    # 不均衡比率を計算
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count
    print(f"不均衡比率: 1:{scale_pos_weight:.1f}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    print()
    
    # LightGBMパラメータ（クラスウェイト調整）
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 10,
        'verbose': -1,
        'random_state': random_state,
        'scale_pos_weight': scale_pos_weight  # 不均衡データ対策（自動計算）
        # is_unbalanceとscale_pos_weightは同時に設定できないため、scale_pos_weightのみ使用
    }
    
    # Cross Validation設定（時系列データリーク防止）
    if use_timeseries:
        cv_splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    models = []
    cv_results = []
    
    print(f"[DEBUG] CV開始: X.shape={X.shape}, y.shape={y.shape}, X.index範囲=[{X.index.min()}, {X.index.max()}]")
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y), 1):
        print(f"Fold {fold}/{n_splits}")
        print(f"[DEBUG] train_idx: min={train_idx.min()}, max={train_idx.max()}, len={len(train_idx)}")
        print(f"[DEBUG] val_idx: min={val_idx.min()}, max={val_idx.max()}, len={len(val_idx)}")
        print(f"[DEBUG] X.shape={X.shape}, len(X)={len(X)}")
        
        # インデックス範囲チェック（デバッグ用）
        max_idx = len(X) - 1
        if train_idx.max() > max_idx or val_idx.max() > max_idx:
            print(f"  ⚠ インデックスエラー検出: train_idx.max()={train_idx.max()}, val_idx.max()={val_idx.max()}, max_idx={max_idx}")
            raise IndexError(f"train_idx.max()={train_idx.max()} or val_idx.max()={val_idx.max()} exceeds max_idx={max_idx}")
        
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        
        print(f"  訓練データ: {len(X_train)}頭 (穴馬: {y_train.sum()}頭 = {y_train.mean()*100:.2f}%)")
        print(f"  検証データ: {len(X_val)}頭 (穴馬: {y_val.sum()}頭 = {y_val.mean()*100:.2f}%)")
        
        # LightGBM Dataset作成（SMOTEなし）
        train_data = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data, feature_name=feature_cols)
        
        # 学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 検証データで予測
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        
        # 確率分布を確認
        print(f"  確率分布: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}, median={np.median(y_pred_proba):.4f}")
        
        # 動的閾値で評価（実データ不均衡比率に基づく）
        optimal_threshold = y_train.mean()  # 訓練データの穴馬比率
        y_pred = (y_pred_proba > optimal_threshold).astype(int)
        
        # 評価
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba) if y_val.sum() > 0 else 0
        
        print(f"  閾値: {optimal_threshold:.4f} (訓練データ穴馬比率)")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")
        print(f"  AUC: {auc:.4f}")
        print()
        
        models.append(model)
        cv_results.append({
            'fold': fold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })
    
    # CV結果サマリー
    df_cv = pd.DataFrame(cv_results)
    print(f"{'='*80}")
    print(f"Cross Validation結果サマリー")
    print(f"{'='*80}")
    print(df_cv.to_string(index=False))
    print()
    print(f"平均 Precision: {df_cv['precision'].mean():.2%} (±{df_cv['precision'].std():.2%})")
    print(f"平均 Recall: {df_cv['recall'].mean():.2%} (±{df_cv['recall'].std():.2%})")
    print(f"平均 F1 Score: {df_cv['f1'].mean():.2%} (±{df_cv['f1'].std():.2%})")
    print(f"平均 AUC: {df_cv['auc'].mean():.4f} (±{df_cv['auc'].std():.4f})")
    
    return models, df_cv


def save_models(models: list, feature_cols: list, output_dir: str = 'models'):
    """
    学習済みモデルを保存（Phase 2.5: 全10競馬場統合モデル）
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # アンサンブルモデル（全foldのモデル）を保存
    model_data = {
        'models': models,
        'feature_cols': feature_cols,
        'n_models': len(models)
    }
    
    output_file = Path(output_dir) / 'upset_classifier_universal.sav'
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nモデルを {output_file} に保存しました")
    print(f"  モデル数: {len(models)}個 (アンサンブル)")
    print(f"  特徴量数: {len(feature_cols)}個")


def analyze_feature_importance(models: list, feature_cols: list, top_n: int = 20):
    """
    特徴量重要度を分析
    """
    print(f"\n{'='*80}")
    print(f"特徴量重要度 (Top {top_n})")
    print(f"{'='*80}")
    
    # 全モデルの特徴量重要度を平均
    importance_dict = {feat: [] for feat in feature_cols}
    
    for model in models:
        importances = model.feature_importance(importance_type='gain')
        for feat, imp in zip(feature_cols, importances):
            importance_dict[feat].append(imp)
    
    # 平均と標準偏差を計算
    importance_summary = []
    for feat, imps in importance_dict.items():
        importance_summary.append({
            'feature': feat,
            'importance_mean': np.mean(imps),
            'importance_std': np.std(imps)
        })
    
    df_importance = pd.DataFrame(importance_summary)
    df_importance = df_importance.sort_values('importance_mean', ascending=False)
    
    print(df_importance.head(top_n).to_string(index=False))
    
    # CSVに保存
    output_file = Path('results') / 'upset_classifier_feature_importance.tsv'
    df_importance.to_csv(output_file, sep='\t', index=False)
    print(f"\n特徴量重要度を {output_file} に保存しました")


def main():
    """
    メイン処理
    """
    print("="*80)
    print("Phase 2: 穴馬分類モデル作成（クラスウェイト方式）")
    print("="*80)
    print()
    
    # データ読み込み
    df = load_training_data()
    
    # 特徴量準備
    X, y, feature_cols = prepare_features(df)
    
    # 学習（クラスウェイトのみ、SMOTEなし）
    models, cv_results = train_with_class_weights(
        X, y, feature_cols,
        n_splits=5,
        random_state=42
    )
    
    # モデル保存
    save_models(models, feature_cols)
    
    # 特徴量重要度分析
    analyze_feature_importance(models, feature_cols, top_n=20)
    
    print(f"\n{'='*80}")
    print("学習完了!")
    print(f"{'='*80}")
    print("\n次のステップ:")
    print("  1. upset_predictor.py で予測パイプラインを構築")
    print("  2. 2019-2023年データで評価")
    print("  3. Phase 1と比較")


if __name__ == '__main__':
    main()
