"""
SHAP分析結果の詳細統計分析スクリプト

実行:
    python analyze_shap_results.py --input shap_analysis/tokyo_turf_3ageup_long/2023/tokyo_turf_3ageup_long_importance.csv --model-name tokyo_turf_3ageup_long
    python analyze_shap_results.py --input shap_analysis/hanshin_turf_3ageup_long/2023/hanshin_turf_3ageup_long_importance.csv --model-name hanshin_turf_3ageup_long
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature_importance(input_csv, model_name, output_dir):
    """特徴量重要度の詳細分析
    
    Args:
        input_csv (str): SHAP重要度CSVファイルパス
        model_name (str): モデル名（出力ファイル名に使用）
        output_dir (str): 出力ディレクトリ（デフォルト: shap_analysis）
    """
    print("="*80)
    print(f"[TEST] SHAP特徴量重要度の詳細分析: {model_name}")
    print("="*80)
    
    # CSVファイル読み込み
    if not Path(input_csv).exists():
        print(f"[ERROR] ファイルが見つかりません: {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    
    print(f"\n[+] 全特徴量数: {len(df)}個\n")
    
    # 基本統計
    print("=" * 80)
    print("【基本統計量】")
    print("=" * 80)
    print(f"SHAP値の合計: {df['mean_abs_shap'].sum():.4f}")
    print(f"SHAP値の平均: {df['mean_abs_shap'].mean():.4f}")
    print(f"SHAP値の中央値: {df['mean_abs_shap'].median():.4f}")
    print(f"SHAP値の標準偏差: {df['mean_abs_shap'].std():.4f}")
    print(f"SHAP値の最大値: {df['mean_abs_shap'].max():.4f}")
    print(f"SHAP値の最小値: {df['mean_abs_shap'].min():.4f}")
    
    # 累積寄与率
    df['cumsum_ratio'] = df['mean_abs_shap'].cumsum() / df['mean_abs_shap'].sum()
    
    print("\n" + "=" * 80)
    print("【累積寄与率分析】")
    print("=" * 80)
    
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        n_features = (df['cumsum_ratio'] <= threshold).sum() + 1
        print(f"累積寄与率 {threshold*100:.0f}% に必要な特徴量数: {n_features}個")
        print(f"  → Top{n_features}: {', '.join(df.head(n_features)['feature'].tolist())}")
    
    # カテゴリ別分析
    print("\n" + "=" * 80)
    print("【特徴量カテゴリ別分析】")
    print("=" * 80)
    
    categories = {
        '過去成績系': ['past_avg_sotai_chakujun', 'past_score', 'time_index'],
        '斤量系': ['futan_per_barei', 'futan_zscore', 'futan_percentile', 'futan_deviation', 'futan_juryo', 'futan_per_barei_log'],
        '騎手系': ['kishu_surface_score', 'kishu_skill_score', 'kishu_popularity_score'],
        '調教師系': ['chokyoshi_recent_score'],
        '馬番・枠番系': ['umaban_kyori_interaction', 'umaban_percentile', 'wakuban_ratio', 'wakuban_bias_score'],
        '距離適性系': ['similar_distance_score', 'distance_category_score', 'distance_change_adaptability'],
        '馬場適性系': ['surface_aptitude_score', 'baba_condition_score', 'baba_change_adaptability', 'kohan_3f_index'],
        '年齢系': ['barei_peak_distance', 'barei_peak_short']
    }
    
    category_stats = []
    for category, features in categories.items():
        category_df = df[df['feature'].isin(features)]
        total_shap = category_df['mean_abs_shap'].sum()
        avg_shap = category_df['mean_abs_shap'].mean()
        n_features = len(category_df)
        category_stats.append({
            'カテゴリ': category,
            '特徴量数': n_features,
            'SHAP合計': total_shap,
            'SHAP平均': avg_shap,
            '寄与率(%)': total_shap / df['mean_abs_shap'].sum() * 100
        })
    
    category_df = pd.DataFrame(category_stats).sort_values('SHAP合計', ascending=False)
    print(category_df.to_string(index=False))
    
    # 削除推奨特徴量
    print("\n" + "=" * 80)
    print("【削除推奨特徴量(SHAP < 0.005)】")
    print("=" * 80)
    
    low_impact = df[df['mean_abs_shap'] < 0.005].sort_values('mean_abs_shap', ascending=False)
    if len(low_impact) > 0:
        print(f"削除候補: {len(low_impact)}個\n")
        for idx, row in low_impact.iterrows():
            print(f"  [ERROR] {row['feature']:30s} SHAP={row['mean_abs_shap']:.6f}")
        
        print(f"\n削除することで:")
        print(f"  - 特徴量数: {len(df)}個 → {len(df) - len(low_impact)}個")
        print(f"  - 削減率: {len(low_impact)/len(df)*100:.1f}%")
        print(f"  - 失われる情報量: {low_impact['mean_abs_shap'].sum()/df['mean_abs_shap'].sum()*100:.2f}%")
    else:
        print("削除推奨の特徴量はありません")
    
    # LightGBM GainとSHAPの相関
    print("\n" + "=" * 80)
    print("【LightGBM Gain vs SHAP値の相関】")
    print("=" * 80)
    
    correlation = df['mean_abs_shap'].corr(df['lgb_gain'])
    print(f"ピアソン相関係数: {correlation:.4f}")
    
    # 乖離が大きい特徴量
    df['gain_shap_ratio'] = df['lgb_gain'] / (df['mean_abs_shap'] * 1000)
    df_sorted = df.sort_values('gain_shap_ratio', ascending=False)
    
    print("\nGainが高いのにSHAPが低い特徴量(モデルが過剰に使用):")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"  {row['feature']:30s} Gain={row['lgb_gain']:8.2f} SHAP={row['mean_abs_shap']:.4f} 比率={row['gain_shap_ratio']:.2f}")
    
    print("\nSHAPが高いのにGainが低い特徴量(効率的な特徴量):")
    for idx, row in df_sorted.tail(5).iterrows():
        print(f"  {row['feature']:30s} Gain={row['lgb_gain']:8.2f} SHAP={row['mean_abs_shap']:.4f} 比率={row['gain_shap_ratio']:.2f}")
    
    # 可視化
    create_visualizations(df, category_df)
    
    return df, category_df


def create_visualizations(df, category_df):
    """SHAP分析結果の追加可視化"""
    print("\n" + "=" * 80)
    print("[+] 追加グラフを作成中...")
    print("=" * 80)
    
    # 1. 累積寄与率グラフ
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1-1. 累積寄与率
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(df)+1), df['cumsum_ratio'], 'b-', linewidth=2)
    ax1.axhline(y=0.8, color='r', linestyle='--', label='80%ライン')
    ax1.axhline(y=0.9, color='orange', linestyle='--', label='90%ライン')
    ax1.set_xlabel('特徴量数', fontsize=12)
    ax1.set_ylabel('累積寄与率', fontsize=12)
    ax1.set_title('特徴量の累積寄与率', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 1-2. カテゴリ別寄与率
    ax2 = axes[0, 1]
    colors = plt.cm.Set3(range(len(category_df)))
    ax2.bar(range(len(category_df)), category_df['寄与率(%)'], color=colors)
    ax2.set_xticks(range(len(category_df)))
    ax2.set_xticklabels(category_df['カテゴリ'], rotation=45, ha='right')
    ax2.set_ylabel('寄与率 (%)', fontsize=12)
    ax2.set_title('特徴量カテゴリ別寄与率', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 1-3. SHAP値の分布
    ax3 = axes[1, 0]
    ax3.hist(df['mean_abs_shap'], bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(df['mean_abs_shap'].median(), color='r', linestyle='--', label=f'中央値={df["mean_abs_shap"].median():.4f}')
    ax3.axvline(df['mean_abs_shap'].mean(), color='g', linestyle='--', label=f'平均値={df["mean_abs_shap"].mean():.4f}')
    ax3.set_xlabel('SHAP値', fontsize=12)
    ax3.set_ylabel('特徴量数', fontsize=12)
    ax3.set_title('SHAP値の分布', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 1-4. LightGBM Gain vs SHAP散布図
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['lgb_gain'], df['mean_abs_shap'], alpha=0.6, s=100)
    ax4.set_xlabel('LightGBM Gain', fontsize=12)
    ax4.set_ylabel('SHAP値', fontsize=12)
    ax4.set_title('LightGBM Gain vs SHAP値', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # トップ3にラベル付け
    for idx, row in df.head(3).iterrows():
        ax4.annotate(row['feature'], 
                    (row['lgb_gain'], row['mean_abs_shap']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'detailed_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    
    # 2. パレート図
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = range(len(df))
    ax1.bar(x, df['mean_abs_shap'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('特徴量', fontsize=12)
    ax1.set_ylabel('SHAP値', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['feature'], rotation=90, fontsize=9)
    
    ax2 = ax1.twinx()
    ax2.plot(x, df['cumsum_ratio'] * 100, 'r-', marker='o', linewidth=2, markersize=4)
    ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylabel('累積寄与率 (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 105])
    
    plt.title('特徴量重要度のパレート図', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = Path(output_dir) / 'pareto_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    
    plt.close('all')


def suggest_improvements(df):
    """改善提案を生成"""
    print("\n" + "=" * 80)
    print("[TIP] 具体的な改善提案")
    print("=" * 80)
    
    # Top3特徴量の強化案
    print("\n【1. Top3特徴量の強化】")
    top3 = df.head(3)
    for idx, row in top3.iterrows():
        print(f"\n {row['feature']} (SHAP={row['mean_abs_shap']:.4f})")
        
        if 'past_avg_sotai_chakujun' in row['feature']:
            print("  改善案:")
            print("    - 現在: 単純平均(直近3走)")
            print("    - 提案: 指数加重平均(最新レースを重視)")
            print("    - コード例:")
            print("      weights = [0.5, 0.3, 0.2]  # 最新、2走前、3走前")
            print("      past_avg_sotai_chakujun = np.average(past_3_races, weights=weights)")
            
        elif 'umaban_kyori_interaction' in row['feature']:
            print("  改善案:")
            print("    - 現在: umaban * kyori / 1000")
            print("    - 提案: 非線形変換で長距離×外枠のペナルティ強化")
            print("    - コード例:")
            print("      if kyori >= 2400 and umaban >= 13:")
            print("          penalty = 1.5")
            print("      elif kyori <= 1800 and umaban <= 3:")
            print("          bonus = 0.7")
            
        elif 'past_score' in row['feature']:
            print("  改善案:")
            print("    - 現在: グレード別固定倍率")
            print("    - 提案: 賞金額ベースの動的重み付け")
            print("    - コード例:")
            print("      weight = prize_money / 10000000  # 賞金1億円で10.0")
    
    # 削除推奨
    print("\n【2. 不要特徴量の削除】")
    low_features = df[df['mean_abs_shap'] < 0.005]['feature'].tolist()
    if low_features:
        print(f"削除推奨: {len(low_features)}個")
        for feat in low_features:
            print(f"  [ERROR] {feat}")
        print("\n削除による期待効果:")
        print("  - 過学習リスク減少")
        print("  - 学習時間短縮")
        print("  - モデル解釈性向上")
    
    # 中位特徴量の改善
    print("\n【3. 中位特徴量の改善可能性】")
    mid_features = df[(df['mean_abs_shap'] >= 0.01) & (df['mean_abs_shap'] < 0.05)]
    print(f"改善候補: {len(mid_features)}個")
    for idx, row in mid_features.iterrows():
        print(f"  [TOOL] {row['feature']:30s} SHAP={row['mean_abs_shap']:.4f}")
    
    print("\n改善アプローチ:")
    print("  - 非線形変換の追加")
    print("  - 他の特徴量との相互作用")
    print("  - 時間窓の調整(3ヶ月→6ヶ月など)")


def generate_markdown_report(df, category_df, model_name, output_dir):
    """Markdownレポート自動生成
    
    Args:
        df: 特徴量重要度DataFrame
        category_df: カテゴリ別集計DataFrame
        model_name: モデル名
        output_dir: 出力ディレクトリ
    """
    print("\n" + "=" * 80)
    print(f"[+] Markdownレポートを生成中: {model_name}")
    print("=" * 80)
    
    # 現在日時
    current_date = datetime.now().strftime('%Y年%m月%d日')
    
    # 削除推奨特徴量
    low_impact = df[df['mean_abs_shap'] < 0.005].sort_values('mean_abs_shap', ascending=False)
    
    # Top3の寄与率
    total_shap = df['mean_abs_shap'].sum()
    top3_ratio = df.head(3)['mean_abs_shap'].sum() / total_shap * 100
    
    # レポート本文生成
    report = f"""# SHAP分析レポート - {model_name}

## 📊 実行日: {current_date}

---

## 🎯 重要な発見

### 1️⃣ **過去成績系の特徴量が圧倒的に重要**

**Top 3の特徴量:**
"""
    
    # Top3詳細
    for i, (idx, row) in enumerate(df.head(3).iterrows(), 1):
        feature_ratio = row['mean_abs_shap'] / total_shap * 100
        report += f"{i}. **{row['feature']}** ({row['mean_abs_shap']:.3f}) - "
        
        if 'past_avg_sotai_chakujun' in row['feature']:
            report += "過去の相対着順\n"
            report += f"   - SHAP値: {row['mean_abs_shap']:.3f} (ぶっちぎり1位)\n"
            report += f"   - LightGBM Gain: {row['lgb_gain']:.1f}\n"
            report += "   - 意味: 直近3走の相対着順(1-(着順/出走頭数))の平均\n"
            report += "   - **結論**: 馬の直近パフォーマンスが最も重要!\n\n"
        elif 'umaban_kyori_interaction' in row['feature']:
            report += "馬番×距離の相互作用\n"
            report += f"   - SHAP値: {row['mean_abs_shap']:.3f}\n"
            report += f"   - LightGBM Gain: {row['lgb_gain']:.1f}\n"
            report += "   - 意味: 馬番と距離の組み合わせ効果\n"
            report += "   - **結論**: 内枠/外枠と長距離の組み合わせが重要\n\n"
        elif 'past_score' in row['feature']:
            report += "グレード別過去成績スコア\n"
            report += f"   - SHAP値: {row['mean_abs_shap']:.3f}\n"
            report += f"   - LightGBM Gain: {row['lgb_gain']:.1f}\n"
            report += "   - 意味: レースグレードを考慮した過去3走の重み付けスコア\n"
            report += "   - **結論**: G1で1着は重く評価される\n\n"
        else:
            report += f"{row['feature']}\n"
            report += f"   - SHAP値: {row['mean_abs_shap']:.3f}\n"
            report += f"   - LightGBM Gain: {row['lgb_gain']:.1f}\n\n"
    
    report += f"**Top3だけで全体影響の{top3_ratio:.1f}%を占める!**\n"
    for i, (idx, row) in enumerate(df.head(3).iterrows(), 1):
        feature_ratio = row['mean_abs_shap'] / total_shap * 100
        report += f"- {row['feature']}: {row['mean_abs_shap']:.3f} / {total_shap:.3f} = {feature_ratio:.1f}%\n"
    
    report += "\n---\n\n"
    
    # カテゴリ別分析
    report += "### 2️⃣ **カテゴリ別特徴量の重要度**\n\n"
    report += "**特徴量カテゴリ別寄与率:**\n"
    for idx, row in category_df.head(5).iterrows():
        report += f"- **{row['カテゴリ']}** ({row['寄与率(%)']:.1f}%) - {row['特徴量数']}個の特徴量\n"
    
    report += "\n**分析:**\n"
    top_category = category_df.iloc[0]
    report += f"- {top_category['カテゴリ']}が{top_category['寄与率(%)']:.1f}%でトップ\n"
    report += f"- モデルは馬の基本能力を最も重視している\n"
    
    report += "\n---\n\n"
    
    # 削除推奨特徴量
    report += "### 3️⃣ **削除推奨特徴量の分析**\n\n"
    
    if len(low_impact) > 0:
        report += f"**削除候補(SHAP < 0.005): {len(low_impact)}個**\n\n"
        for idx, row in low_impact.iterrows():
            report += f"- `{row['feature']}` (SHAP={row['mean_abs_shap']:.6f}) ❌\n"
        
        info_loss = low_impact['mean_abs_shap'].sum() / total_shap * 100
        report += f"\n**削除による影響:**\n"
        report += f"- 特徴量数: {len(df)}個 → {len(df) - len(low_impact)}個\n"
        report += f"- 削減率: {len(low_impact)/len(df)*100:.1f}%\n"
        report += f"- 失われる情報量: {info_loss:.2f}%\n\n"
        report += "**期待効果:**\n"
        report += "- 過学習リスク減少\n"
        report += "- 学習速度向上\n"
        report += "- モデルの解釈性向上\n"
    else:
        report += "**削除推奨の特徴量はありません ✅**\n\n"
        bottom3 = df.tail(3)
        report += "最下位3つの特徴量でも一定の貢献度があります:\n"
        for idx, row in bottom3.iterrows():
            report += f"- `{row['feature']}` (SHAP={row['mean_abs_shap']:.4f})\n"
        report += "\nすべての特徴量が意味のある貢献をしています！\n"
    
    report += "\n---\n\n"
    
    # 累積寄与率
    report += "### 4️⃣ **累積寄与率分析**\n\n"
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        n_features = (df['cumsum_ratio'] <= threshold).sum() + 1
        report += f"- **累積寄与率 {threshold*100:.0f}%**: Top{n_features}個の特徴量\n"
    
    report += "\n**パレートの法則:**\n"
    n_50 = (df['cumsum_ratio'] <= 0.5).sum() + 1
    report += f"- 上位{n_50}個（全体の{n_50/len(df)*100:.1f}%）で全体の50%を説明\n"
    report += "- 理想的な重要度分布を実現！\n"
    
    report += "\n---\n\n"
    
    # 改善提案
    report += "## 🔥 改善提案\n\n"
    report += "### ✅ すぐできる改善\n\n"
    
    # 削除提案
    if len(low_impact) > 0:
        report += "#### 1. **不要な特徴量を削除(次元削減)**\n"
        report += "削除候補(SHAP < 0.005):\n"
        for idx, row in low_impact.iterrows():
            report += f"- `{row['feature']}` ({row['mean_abs_shap']:.6f}) ❌\n"
        report += "\n"
    
    # Top3強化
    report += "#### 2. **Top3特徴量の強化**\n\n"
    
    if 'past_avg_sotai_chakujun' in df.head(3)['feature'].values:
        report += "**past_avg_sotai_chakujun強化案:**\n"
        report += "- 現在: 直近3走の平均\n"
        report += "- 改善: **指数加重平均**(最新レースを重視)\n"
        report += "  - 3走前: 重み0.2\n"
        report += "  - 2走前: 重み0.3\n"
        report += "  - 1走前: 重み0.5\n\n"
    
    if 'umaban_kyori_interaction' in df.head(3)['feature'].values:
        report += "**umaban_kyori_interaction強化案:**\n"
        report += "- 現在: umaban × kyori / 1000\n"
        report += "- 改善: **非線形変換**\n"
        report += "  - 長距離(2400m+) × 外枠(13番+) → ペナルティ大\n"
        report += "  - 短距離(1800m-) × 内枠(1-3番) → ボーナス\n\n"
    
    if 'past_score' in df.head(3)['feature'].values:
        report += "**past_score強化案:**\n"
        report += "- 現在: G1=1.0, G2=0.8, G3=0.6...\n"
        report += "- 改善: **賞金ベース**の重み付け\n"
        report += "  - 1着賞金が高いレース = より高評価\n\n"
    
    report += "---\n\n"
    
    # 統計サマリー
    report += "## 📈 統計サマリー\n\n"
    report += f"- **全特徴量数**: {len(df)}個\n"
    report += f"- **SHAP値合計**: {total_shap:.4f}\n"
    report += f"- **SHAP値平均**: {df['mean_abs_shap'].mean():.4f}\n"
    report += f"- **SHAP値中央値**: {df['mean_abs_shap'].median():.4f}\n"
    report += f"- **SHAP値標準偏差**: {df['mean_abs_shap'].std():.4f}\n"
    report += f"- **LightGBM Gain相関**: {df['mean_abs_shap'].corr(df['lgb_gain']):.4f}\n"
    
    report += "\n---\n\n"
    
    # 次のアクション
    report += "## 🎲 次のアクション\n\n"
    
    if len(low_impact) > 0:
        report += "### 優先度高(すぐやる)\n"
        report += f"1. ✅ **{len(low_impact)}個の不要特徴量を削除**\n"
        report += "2. ✅ **Top3特徴量を強化**\n"
        report += "3. ✅ **モデル再学習**\n\n"
    else:
        report += "### 優先度高(すぐやる)\n"
        report += "1. ✅ **Top3特徴量を強化**（指数加重平均、非線形変換）\n"
        report += "2. ✅ **バックテストで効果検証**\n"
        report += "3. ⏳ **閾値の再調整**\n\n"
    
    report += "### 優先度中(検証後に実施)\n"
    report += "4. ⏳ **中位特徴量の改善**（非線形変換、相互作用追加）\n"
    report += "5. ⏳ **過去成績参照期間の調整**（3走→5走など）\n"
    report += "6. ⏳ **騎手特徴量の精緻化**（競馬場別に分割）\n\n"
    
    report += "### 優先度低(余裕があれば)\n"
    report += "7. 🔮 **騎手×馬の相性特徴量を追加**\n"
    report += "8. 🔮 **賞金額ベースの特徴量を追加**\n"
    
    report += "\n---\n\n"
    
    # 結論
    report += "## 💡 結論\n\n"
    report += "**SHAP分析から得られた最大の知見:**\n\n"
    
    top1 = df.iloc[0]
    top1_ratio = top1['mean_abs_shap'] / total_shap * 100
    report += f"> **「{top1['feature']}が全体の{top1_ratio:.1f}%を占め、他のすべてを圧倒している」**\n\n"
    
    report += "現在のモデルは:\n"
    report += "- ✅ 馬の過去成績を正しく評価できている\n"
    
    kishu_count = len([f for f in df['feature'].values if 'kishu' in f])
    if kishu_count > 0:
        report += "- ✅ 騎手の能力も適切に考慮している\n"
    
    futan_count = len([f for f in df['feature'].values if 'futan' in f])
    if futan_count > 0:
        report += "- ✅ 斤量の影響も捉えている\n"
    
    if len(low_impact) > 0:
        report += f"- ❌ ノイズ特徴量が多すぎる({len(df)}個中{len(low_impact)}個は不要)\n"
    else:
        report += "- ✅ すべての特徴量が意味のある貢献をしている\n"
    
    report += "- ❌ Top特徴量の作り方に改善余地あり\n\n"
    
    if len(low_impact) > 0:
        report += "**次のステップ:**\n"
        report += f"1. 不要特徴量を削除して{len(df) - len(low_impact)}個に減らす\n"
        report += "2. Top3特徴量を強化（指数加重平均など）\n"
        report += "3. モデルを再学習して的中率を確認\n"
    else:
        report += "**次のステップ:**\n"
        report += "1. Top3特徴量を強化（指数加重平均、非線形変換）\n"
        report += "2. バックテストで実際の的中率改善を確認\n"
        report += "3. さらなる特徴量エンジニアリング\n"
    
    # ファイル書き出し
    output_path = Path(output_dir) / f'{model_name}_analysis_report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  [OK] {output_path}")
    
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHAP分析結果の詳細統計分析')
    parser.add_argument('--input', type=str, required=True,
                        help='SHAP重要度CSVファイルパス (例: shap_analysis/tokyo_turf_3ageup_long/2023/tokyo_turf_3ageup_long_importance.csv)')
    parser.add_argument('--model-name', type=str, required=True,
                        help='モデル名 (例: tokyo_turf_3ageup_long)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力ディレクトリ (デフォルト: 入力ファイルと同じディレクトリ)')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 入力ファイルと同じディレクトリ
        output_dir = str(Path(args.input).parent)
    
    df, category_df = analyze_feature_importance(args.input, args.model_name, output_dir)
    if df is not None:
        suggest_improvements(df)
        generate_markdown_report(df, category_df, args.model_name, output_dir)
        
        print("\n" + "=" * 80)
        print("[OK] 分析完了!")
        print("=" * 80)
        print("\n生成ファイル:")
        print(f"  - {Path(output_dir) / 'detailed_analysis.png'}")
        print(f"  - {Path(output_dir) / 'pareto_chart.png'}")
        print(f"  - {Path(output_dir) / f'{args.model_name}_analysis_report.md'}")
        print("\n次のステップ:")
        print("  1. レポートを読んで改善内容を確認")
        print("  2. 不要特徴量を削除")
        print("  3. Top3特徴量を強化")
        print("  4. モデル再学習")
        print("  5. 的中率の変化を確認")
