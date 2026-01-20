"""
穴馬分類器モデルの特徴量を分析
- モデルに含まれる特徴量リスト
- 特徴量重要度（gain-based）
- Phase 3特徴量が含まれているか確認
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_upset_model(model_path: str):
    """穴馬分類器モデルの特徴量を分析"""
    print("=" * 80)
    print(f"[ANALYZE] 穴馬分類器モデル分析")
    print("=" * 80)
    print(f"モデルパス: {model_path}")
    
    # モデル読み込み
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n[MODEL INFO]")
    print(f"  モデル数: {model_data['n_models']}個（アンサンブル）")
    print(f"  学習期間: {model_data['train_period']}")
    
    # 特徴量リスト
    feature_cols = model_data['feature_cols']
    print(f"\n[FEATURES] 特徴量数: {len(feature_cols)}個")
    print(f"\n特徴量リスト:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Phase 3特徴量チェック
    phase3_features = [
        'past_score_std',
        'past_chakujun_variance', 
        'zenso_oikomi_power',
        'kishu_changed',
        'class_downgrade',
        'zenso_kakoi_komon'
    ]
    
    print(f"\n[PHASE 3 CHECK] Phase 3特徴量の含有状況:")
    phase3_exists = []
    for feat in phase3_features:
        exists = feat in feature_cols
        phase3_exists.append(exists)
        status = "✅ 含まれる" if exists else "❌ 含まれない"
        print(f"  {feat}: {status}")
    
    if all(phase3_exists):
        print(f"\n✅ Phase 3特徴量は全て含まれています")
    elif any(phase3_exists):
        print(f"\n⚠️  Phase 3特徴量の一部のみ含まれています")
    else:
        print(f"\n❌ Phase 3特徴量が全く含まれていません（モデル再作成が必要）")
    
    # 特徴量重要度（全モデルの平均）
    models = model_data['models']
    
    print(f"\n[IMPORTANCE] 特徴量重要度（gain-based、{len(models)}モデル平均）")
    
    importance_list = []
    for i, model in enumerate(models):
        if hasattr(model, 'feature_importances_'):
            importance_list.append(model.feature_importances_)
        elif hasattr(model, 'feature_importance'):
            importance_list.append(model.feature_importance(importance_type='gain'))
    
    if len(importance_list) > 0:
        importance_mean = np.mean(importance_list, axis=0)
        importance_std = np.std(importance_list, axis=0)
        
        # DataFrame作成
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance_mean': importance_mean,
            'importance_std': importance_std
        }).sort_values('importance_mean', ascending=False)
        
        # Top 20表示
        print(f"\nTop 20特徴量:")
        print(importance_df.head(20).to_string(index=False))
        
        # Phase 3特徴量の重要度
        print(f"\n[PHASE 3 IMPORTANCE] Phase 3特徴量の重要度:")
        for feat in phase3_features:
            if feat in feature_cols:
                row = importance_df[importance_df['feature'] == feat].iloc[0]
                rank = importance_df[importance_df['feature'] == feat].index[0] + 1
                print(f"  {feat}: {row['importance_mean']:.2f} (±{row['importance_std']:.2f}) - 順位{rank}/{len(feature_cols)}")
        
        # グラフ作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top 20特徴量重要度
        top20 = importance_df.head(20)
        ax1.barh(range(len(top20)), top20['importance_mean'], xerr=top20['importance_std'])
        ax1.set_yticks(range(len(top20)))
        ax1.set_yticklabels(top20['feature'])
        ax1.invert_yaxis()
        ax1.set_xlabel('重要度 (gain)', fontsize=12)
        ax1.set_title('特徴量重要度 Top 20', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Phase 3特徴量のみ
        phase3_df = importance_df[importance_df['feature'].isin(phase3_features)]
        if len(phase3_df) > 0:
            ax2.barh(range(len(phase3_df)), phase3_df['importance_mean'], xerr=phase3_df['importance_std'], color='orange')
            ax2.set_yticks(range(len(phase3_df)))
            ax2.set_yticklabels(phase3_df['feature'])
            ax2.invert_yaxis()
            ax2.set_xlabel('重要度 (gain)', fontsize=12)
            ax2.set_title('Phase 3特徴量の重要度', fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Phase 3特徴量なし', ha='center', va='center', fontsize=16)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        output_file = Path('results') / 'upset_model_feature_importance.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[FILE] グラフ保存: {output_file}")
        plt.close()
        
        # CSVも保存
        csv_file = Path('results') / 'upset_model_feature_importance.csv'
        importance_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"[FILE] CSV保存: {csv_file}")
        
        return importance_df
    else:
        print("⚠️  特徴量重要度を取得できませんでした")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # デフォルト: 2025年テスト用モデル (2021-2024学習)
        model_path = "walk_forward_models/period_4/models/2025/upset_classifier_2021-2024.sav"
    
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        print(f"\n利用可能なモデル:")
        
        wfv_dir = Path("walk_forward_models")
        if wfv_dir.exists():
            for sav_file in wfv_dir.rglob("upset_classifier_*.sav"):
                print(f"  {sav_file}")
        
        sys.exit(1)
    
    importance_df = analyze_upset_model(str(model_file))
    
    print("\n" + "=" * 80)
    print("[DONE] 分析完了！")
    print("=" * 80)
