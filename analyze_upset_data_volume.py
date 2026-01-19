import pandas as pd
import numpy as np

# データ読み込み
df = pd.read_csv('results/upset_training_data_universal.tsv', sep='\t')

print("="*80)
print("## データ量調査結果")
print("="*80)

# 1. 全体統計
print("\n### 全体統計")
total_records = len(df)
upset_count = df['is_upset'].sum()
non_upset_count = len(df) - upset_count
upset_ratio = (upset_count / total_records) * 100
imbalance_ratio = non_upset_count / upset_count if upset_count > 0 else 0

print(f"- 総レコード数: {total_records:,}件")
print(f"- 穴馬サンプル (is_upset=1): {upset_count:,}件 ({upset_ratio:.2f}%)")
print(f"- 非穴馬サンプル (is_upset=0): {non_upset_count:,}件 ({100-upset_ratio:.2f}%)")
print(f"- 不均衡比率: 1:{imbalance_ratio:.2f}")

# 2. 距離帯別のデータ分布
print("\n### 距離帯別分布")

# 距離カテゴリを定義
def categorize_distance(kyori):
    if kyori < 1400:
        return '短距離(1000-1399)'
    elif kyori <= 1600:
        return 'マイル(1400-1600)'
    elif kyori <= 2200:
        return '中距離(1700-2200)'
    else:
        return '長距離(2201-)'

df['distance_category'] = df['kyori'].apply(categorize_distance)

# 大まかな分類
def categorize_distance_simple(kyori):
    if kyori <= 1600:
        return '短距離(1000-1600)'
    else:
        return '中長距離(1700-)'

df['distance_category_simple'] = df['kyori'].apply(categorize_distance_simple)

print("\n#### 詳細な距離帯別分布")
print(f"{'距離帯':<20} {'総数':>10} {'穴馬数':>10} {'穴馬率':>10}")
print("-" * 55)

for category in ['短距離(1000-1399)', 'マイル(1400-1600)', '中距離(1700-2200)', '長距離(2201-)']:
    cat_df = df[df['distance_category'] == category]
    total = len(cat_df)
    upset = cat_df['is_upset'].sum()
    upset_pct = (upset / total * 100) if total > 0 else 0
    print(f"{category:<20} {total:>10,} {upset:>10,} {upset_pct:>9.2f}%")

print("\n#### シンプルな距離帯別分布")
print(f"{'距離帯':<20} {'総数':>10} {'穴馬数':>10} {'穴馬率':>10}")
print("-" * 55)

for category in ['短距離(1000-1600)', '中長距離(1700-)']:
    cat_df = df[df['distance_category_simple'] == category]
    total = len(cat_df)
    upset = cat_df['is_upset'].sum()
    upset_pct = (upset / total * 100) if total > 0 else 0
    print(f"{category:<20} {total:>10,} {upset:>10,} {upset_pct:>9.2f}%")

# 3. 競馬場別のデータ分布
print("\n### 競馬場別分布")

# 競馬場コードのマッピング（一般的な値）
keibajo_map = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉'
}

# 競馬場別集計
keibajo_stats = df.groupby('keibajo_code').agg({
    'is_upset': ['count', 'sum']
}).reset_index()
keibajo_stats.columns = ['keibajo_code', 'total', 'upset_count']
keibajo_stats['upset_rate'] = (keibajo_stats['upset_count'] / keibajo_stats['total'] * 100)
keibajo_stats['keibajo_name'] = keibajo_stats['keibajo_code'].map(keibajo_map)
keibajo_stats = keibajo_stats.sort_values('total', ascending=False)

print(f"\n{'競馬場':<10} {'コード':<8} {'総数':>10} {'穴馬数':>10} {'穴馬率':>10}")
print("-" * 60)
for _, row in keibajo_stats.iterrows():
    name = row['keibajo_name'] if pd.notna(row['keibajo_name']) else row['keibajo_code']
    print(f"{name:<10} {row['keibajo_code']:<8} {int(row['total']):>10,} {int(row['upset_count']):>10,} {row['upset_rate']:>9.2f}%")

# 4. 年度別分布
print("\n### 年度別分布")
year_stats = df.groupby('kaisai_nen').agg({
    'is_upset': ['count', 'sum']
}).reset_index()
year_stats.columns = ['year', 'total', 'upset_count']
year_stats['upset_rate'] = (year_stats['upset_count'] / year_stats['total'] * 100)
year_stats = year_stats.sort_values('year')

print(f"\n{'年度':<8} {'総数':>10} {'穴馬数':>10} {'穴馬率':>10}")
print("-" * 45)
for _, row in year_stats.iterrows():
    print(f"{int(row['year']):<8} {int(row['total']):>10,} {int(row['upset_count']):>10,} {row['upset_rate']:>9.2f}%")

# 5. 天候別分布
print("\n### 天候別分布")
tenko_map = {1: '晴', 2: '曇', 3: '雨', 4: '小雨', 5: '小雪', 6: '雪'}
tenko_stats = df.groupby('tenko_code').agg({
    'is_upset': ['count', 'sum']
}).reset_index()
tenko_stats.columns = ['tenko_code', 'total', 'upset_count']
tenko_stats['upset_rate'] = (tenko_stats['upset_count'] / tenko_stats['total'] * 100)
tenko_stats['tenko_name'] = tenko_stats['tenko_code'].map(tenko_map)
tenko_stats = tenko_stats.sort_values('total', ascending=False)

print(f"\n{'天候':<10} {'総数':>10} {'穴馬数':>10} {'穴馬率':>10}")
print("-" * 45)
for _, row in tenko_stats.iterrows():
    name = row['tenko_name'] if pd.notna(row['tenko_name']) else str(int(row['tenko_code']))
    print(f"{name:<10} {int(row['total']):>10,} {int(row['upset_count']):>10,} {row['upset_rate']:>9.2f}%")

# 6. 主要競馬場×距離帯のクロス集計
print("\n### 主要競馬場×距離帯クロス集計")
major_keibajo = ['05', '06', '08', '09', '01']  # 東京、中山、京都、阪神、札幌
major_df = df[df['keibajo_code'].isin(major_keibajo)]

print(f"\n{'競馬場×距離帯':<25} {'総数':>10} {'穴馬数':>10} {'穴馬率':>10}")
print("-" * 60)

for keibajo in major_keibajo:
    keibajo_name = keibajo_map.get(keibajo, keibajo)
    for category in ['短距離(1000-1600)', '中長距離(1700-)']:
        cross_df = df[(df['keibajo_code'] == keibajo) & (df['distance_category_simple'] == category)]
        total = len(cross_df)
        upset = cross_df['is_upset'].sum()
        upset_pct = (upset / total * 100) if total > 0 else 0
        label = f"{keibajo_name}・{category}"
        print(f"{label:<25} {total:>10,} {upset:>10,} {upset_pct:>9.2f}%")

# 7. 判断
print("\n" + "="*80)
print("### 判断・推奨事項")
print("="*80)

print("\n#### 距離帯別モデル分割の妥当性")
for category in ['短距離(1000-1600)', '中長距離(1700-)']:
    cat_df = df[df['distance_category_simple'] == category]
    total = len(cat_df)
    upset = cat_df['is_upset'].sum()
    
    print(f"\n【{category}】")
    print(f"  - 総サンプル数: {total:,}件")
    print(f"  - 穴馬サンプル数: {upset:,}件")
    
    # 一般的な推奨基準
    if upset >= 1000:
        print(f"  ✓ 十分なデータ量（穴馬1000件以上）")
    elif upset >= 500:
        print(f"  △ やや少ないが学習可能（穴馬500-1000件）")
    else:
        print(f"  ✗ データ不足の可能性（穴馬500件未満）")

print("\n#### 更に細かい分割の可能性")
for category in ['短距離(1000-1399)', 'マイル(1400-1600)', '中距離(1700-2200)', '長距離(2201-)']:
    cat_df = df[df['distance_category'] == category]
    total = len(cat_df)
    upset = cat_df['is_upset'].sum()
    
    print(f"\n【{category}】")
    print(f"  - 総サンプル数: {total:,}件")
    print(f"  - 穴馬サンプル数: {upset:,}件")
    
    if upset >= 1000:
        print(f"  ✓ 個別モデル化推奨")
    elif upset >= 500:
        print(f"  △ 可能だが慎重に検証")
    else:
        print(f"  ✗ 他カテゴリと統合推奨")

print("\n#### 最終推奨")
print("機械学習の一般的な基準:")
print("  - 最低限必要: 穴馬サンプル300-500件")
print("  - 推奨: 穴馬サンプル1000件以上")
print("  - 理想: 穴馬サンプル3000件以上")
print("\n上記の各セグメントのデータ量と照らし合わせてモデル分割を決定してください。")

print("\n" + "="*80)
