"""
SHAP分析結果の詳細統計分析スクリプト

実行:
    python analyze_shap_results.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature_importance():
    """特徴量重要度の詳細分析"""
    print("="*80)
    print("🔍 SHAP特徴量重要度の詳細分析")
    print("="*80)
    
    # CSVファイル読み込み
    df = pd.read_csv('shap_analysis/tokyo_turf_3ageup_long_importance.csv')
    
    print(f"\n📊 全特徴量数: {len(df)}個\n")
    
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
            print(f"  ❌ {row['feature']:30s} SHAP={row['mean_abs_shap']:.6f}")
        
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
    print("📊 追加グラフを作成中...")
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
    plt.savefig('shap_analysis/detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✅ shap_analysis/detailed_analysis.png")
    
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
    plt.savefig('shap_analysis/pareto_chart.png', dpi=300, bbox_inches='tight')
    print("  ✅ shap_analysis/pareto_chart.png")
    
    plt.close('all')


def suggest_improvements(df):
    """改善提案を生成"""
    print("\n" + "=" * 80)
    print("💡 具体的な改善提案")
    print("=" * 80)
    
    # Top3特徴量の強化案
    print("\n【1. Top3特徴量の強化】")
    top3 = df.head(3)
    for idx, row in top3.iterrows():
        print(f"\n🔥 {row['feature']} (SHAP={row['mean_abs_shap']:.4f})")
        
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
            print(f"  ❌ {feat}")
        print("\n削除による期待効果:")
        print("  - 過学習リスク減少")
        print("  - 学習時間短縮")
        print("  - モデル解釈性向上")
    
    # 中位特徴量の改善
    print("\n【3. 中位特徴量の改善可能性】")
    mid_features = df[(df['mean_abs_shap'] >= 0.01) & (df['mean_abs_shap'] < 0.05)]
    print(f"改善候補: {len(mid_features)}個")
    for idx, row in mid_features.iterrows():
        print(f"  🔧 {row['feature']:30s} SHAP={row['mean_abs_shap']:.4f}")
    
    print("\n改善アプローチ:")
    print("  - 非線形変換の追加")
    print("  - 他の特徴量との相互作用")
    print("  - 時間窓の調整(3ヶ月→6ヶ月など)")


if __name__ == '__main__':
    df, category_df = analyze_feature_importance()
    suggest_improvements(df)
    
    print("\n" + "=" * 80)
    print("✅ 分析完了!")
    print("=" * 80)
    print("\n生成ファイル:")
    print("  - shap_analysis/detailed_analysis.png")
    print("  - shap_analysis/pareto_chart.png")
    print("  - shap_analysis_report.md")
    print("\n次のステップ:")
    print("  1. レポートを読んで改善内容を確認")
    print("  2. 不要特徴量を削除")
    print("  3. Top3特徴量を強化")
    print("  4. モデル再学習")
    print("  5. 的中率の変化を確認")
