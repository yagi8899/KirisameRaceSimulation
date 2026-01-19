import pandas as pd
import numpy as np

# 阪神短距離の結果読み込み
df = pd.read_csv('results/predicted_results_hanshin_turf_3ageup_short_trainunknown_test2023_all.tsv', sep='\t')

print("=" * 80)
print("穴馬予測確率の分布確認")
print("=" * 80)

# 列名確認
print(f"\n列名（upset関連）:")
upset_cols = [c for c in df.columns if 'upset' in c.lower()]
print(f"  {upset_cols}")

print(f"\n列名（全{len(df.columns)}列）の最後10個:")
print(f"  {list(df.columns[-10:])}")

# 基本統計
print(f"\n総データ数: {len(df)}頭")

# upset_probabilityの統計
print(f"\nupset_probability統計:")
print(f"  最小値: {df['upset_probability'].min():.6f}")
print(f"  最大値: {df['upset_probability'].max():.6f}")
print(f"  平均値: {df['upset_probability'].mean():.6f}")
print(f"  中央値: {df['upset_probability'].median():.6f}")

# 確率分布
print(f"\n確率分布:")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(bins)-1):
    count = ((df['upset_probability'] >= bins[i]) & (df['upset_probability'] < bins[i+1])).sum()
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {count}頭")

# 実際の穴馬の確率分布
upset_horses = df[df['is_actual_upset'] == 1]
if len(upset_horses) > 0:
    print(f"\n実際の穴馬の確率分布:")
    print(f"  平均: {upset_horses['upset_probability'].mean():.6f}")
    print(f"  中央値: {upset_horses['upset_probability'].median():.6f}")
    print(f"  最大: {upset_horses['upset_probability'].max():.6f}")
    print(f"  最小: {upset_horses['upset_probability'].min():.6f}")
    
    # 確率>0.4の穴馬
    high_prob_upsets = upset_horses[upset_horses['upset_probability'] > 0.4]
    print(f"\n確率>0.4の穴馬: {len(high_prob_upsets)}頭 / {len(upset_horses)}頭")
    
    if len(high_prob_upsets) > 0:
        print("\n確率>0.4の穴馬リスト:")
        for idx, row in high_prob_upsets.iterrows():
            print(f"  {row['bamei']}: 人気{row['tansho_ninkijun_numeric']:.0f} 着順{row['kakutei_chakujun_numeric']:.0f} 確率{row['upset_probability']:.4f}")
    else:
        print("\n⚠ 確率>0.4の穴馬は0頭！")
        print("\n確率Top5の穴馬:")
        top5 = upset_horses.nlargest(5, 'upset_probability')
        for idx, row in top5.iterrows():
            print(f"  {row['bamei']}: 人気{row['tansho_ninkijun_numeric']:.0f} 着順{row['kakutei_chakujun_numeric']:.0f} 確率{row['upset_probability']:.4f}")
