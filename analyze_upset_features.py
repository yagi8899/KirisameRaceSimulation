"""
UPSET分類器の特徴量重要度分析スクリプト

使用方法:
    python analyze_upset_features.py
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_feature_importance():
    """UPSET分類器の特徴量重要度を分析"""
    print("=" * 80)
    print("[ANALYZE] UPSET分類器 特徴量重要度分析")
    print("=" * 80)
    
    # モデルファイルを探す
    model_paths = []
    
    # 1. modelsディレクトリ
    model_dir = Path("models")
    if model_dir.exists():
        model_paths.extend(model_dir.glob("upset_classifier_*.sav"))
    
    # 2. walk_forward結果ディレクトリ
    wf_dirs = ["walk_forward_results_custom2", "walk_forward_results"]
    for wf_dir in wf_dirs:
        wf_path = Path(wf_dir)
        if wf_path.exists():
            model_paths.extend(wf_path.glob("**/upset_classifier_*.sav"))
    
    if not model_paths:
        print("[ERROR] UPSET分類器モデルが見つかりません")
        print(f"[INFO] 探索場所:")
        print(f"  - models/")
        for wf_dir in wf_dirs:
            print(f"  - {wf_dir}/**/")
        return
    
    # 最新のモデルを使用
    model_file = sorted(model_paths, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\n[MODEL] 使用モデル: {model_file.name}")
    print(f"[MODEL] 作成日時: {pd.Timestamp.fromtimestamp(model_file.stat().st_mtime)}")
    
    # モデル読み込み
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # モデルリストを取得
        if isinstance(model_data, dict):
            # walk-forward形式（辞書）
            models = model_data.get('models', [])
            feature_names = model_data.get('feature_cols', [])
            print(f"[MODEL] CVモデル数: {len(models)}")
            print(f"[MODEL] 特徴量数: {len(feature_names)}")
        elif isinstance(model_data, list):
            # 旧形式（リスト）
            models = model_data
            feature_names = None
            print(f"[MODEL] CVモデル数: {len(models)}")
        else:
            # 単一モデル
            models = [model_data]
            feature_names = None
            print(f"[MODEL] 単一モデル")
        
        # 最初のモデルから特徴量重要度を取得
        model = models[0]
        
        # LightGBMモデルから特徴量重要度取得
        if hasattr(model, 'feature_importance'):
            importances = model.feature_importance(importance_type='gain')
            if feature_names is None:
                feature_names = model.feature_name()
        else:
            print("[ERROR] モデルに特徴量重要度情報がありません")
            return
        
        # DataFrameに変換
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n[INFO] 総特徴量数: {len(df_importance)}")
        print(f"[INFO] 非ゼロ重要度: {(df_importance['importance'] > 0).sum()}")
        print(f"[INFO] ゼロ重要度: {(df_importance['importance'] == 0).sum()}")
        
        # Top 30表示
        print(f"\n{'='*80}")
        print("[TOP 30] 重要度の高い特徴量")
        print(f"{'='*80}")
        print(f"{'順位':>4s} {'特徴量名':<50s} {'重要度':>12s} {'累積割合':>10s}")
        print("-" * 80)
        
        total_importance = df_importance['importance'].sum()
        cumsum = 0
        
        for i, (idx, row) in enumerate(df_importance.head(30).iterrows(), 1):
            cumsum += row['importance']
            cumsum_pct = cumsum / total_importance * 100 if total_importance > 0 else 0
            print(f"{i:4d} {row['feature']:<50s} {row['importance']:>12.1f} {cumsum_pct:>9.1f}%")
        
        # Bottom 20表示（重要度ゼロまたは極小）
        print(f"\n{'='*80}")
        print("[BOTTOM 20] 重要度の低い特徴量")
        print(f"{'='*80}")
        print(f"{'順位':>4s} {'特徴量名':<50s} {'重要度':>12s}")
        print("-" * 80)
        
        for i, (idx, row) in enumerate(df_importance.tail(20).iterrows(), 1):
            print(f"{i:4d} {row['feature']:<50s} {row['importance']:>12.1f}")
        
        # Phase 3.5特徴量の重要度確認
        print(f"\n{'='*80}")
        print("[PHASE 3.5] 追加特徴量の重要度")
        print(f"{'='*80}")
        
        phase35_keywords = [
            'jockey_win_rate', 'jockey_place_rate', 'jockey_recent_form',
            'trainer_win_rate', 'trainer_place_rate', 'trainer_recent_form',
            'horse_career_win_rate', 'horse_career_place_rate',
            'weight_change', 'rest_weeks'
        ]
        
        phase35_features = df_importance[
            df_importance['feature'].str.contains('|'.join(phase35_keywords), case=False, na=False)
        ]
        
        if len(phase35_features) > 0:
            print(f"{'特徴量名':<50s} {'重要度':>12s} {'順位':>6s}")
            print("-" * 70)
            for idx, row in phase35_features.iterrows():
                rank = df_importance.index.get_loc(idx) + 1
                print(f"{row['feature']:<50s} {row['importance']:>12.1f} {rank:>6d}")
        else:
            print("⚠️  Phase 3.5特徴量が見つかりませんでした")
        
        # グラフ作成
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top 30のグラフ
        top30 = df_importance.head(30)
        axes[0].barh(range(len(top30)), top30['importance'])
        axes[0].set_yticks(range(len(top30)))
        axes[0].set_yticklabels(top30['feature'], fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('重要度 (Gain)', fontsize=10)
        axes[0].set_title('Top 30 重要度の高い特徴量', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 重要度累積グラフ
        cumsum_importance = df_importance['importance'].cumsum() / total_importance * 100
        axes[1].plot(range(len(cumsum_importance)), cumsum_importance, linewidth=2)
        axes[1].axhline(y=80, color='r', linestyle='--', label='80%ライン')
        axes[1].axhline(y=95, color='orange', linestyle='--', label='95%ライン')
        axes[1].set_xlabel('特徴量数', fontsize=10)
        axes[1].set_ylabel('累積重要度 (%)', fontsize=10)
        axes[1].set_title('特徴量の累積重要度', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        # 80%/95%達成に必要な特徴量数
        n_80 = (cumsum_importance >= 80).idxmax() + 1 if (cumsum_importance >= 80).any() else len(df_importance)
        n_95 = (cumsum_importance >= 95).idxmax() + 1 if (cumsum_importance >= 95).any() else len(df_importance)
        axes[1].axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
        axes[1].axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)
        axes[1].text(n_80, 40, f'{n_80}特徴量', rotation=90, va='bottom', fontsize=9)
        axes[1].text(n_95, 40, f'{n_95}特徴量', rotation=90, va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_file = Path("check_results/upset_feature_importance.png")
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"\n[SAVE] グラフ保存: {output_file}")
        
        # CSVエクスポート
        csv_file = Path("check_results/upset_feature_importance.csv")
        df_importance.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"[SAVE] CSV保存: {csv_file}")
        
        print(f"\n{'='*80}")
        print("[SUMMARY] 分析サマリー")
        print(f"{'='*80}")
        print(f"✓ 上位30特徴量で全体の {cumsum_importance.iloc[29]:.1f}% を説明")
        print(f"✓ 80%達成に必要な特徴量数: {n_80}")
        print(f"✓ 95%達成に必要な特徴量数: {n_95}")
        
        # ゼロ重要度特徴量の割合
        zero_count = (df_importance['importance'] == 0).sum()
        zero_pct = zero_count / len(df_importance) * 100
        print(f"⚠️  重要度ゼロ特徴量: {zero_count}/{len(df_importance)} ({zero_pct:.1f}%)")
        
        if zero_pct > 30:
            print(f"\n[RECOMMEND] 重要度ゼロ特徴量が多いため、特徴量選択を検討してください")
        
    except Exception as e:
        print(f"[ERROR] モデル読み込みエラー: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_feature_importance()
