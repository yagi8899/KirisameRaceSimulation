import pandas as pd
import numpy as np
import psycopg2
import json

def load_db_config(config_path: str = 'db_config.json') -> dict:
    """データベース設定を読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['database']

# DB接続
db_config = load_db_config()
conn = psycopg2.connect(
    host=db_config['host'],
    port=db_config['port'],
    user=db_config['user'],
    password=db_config['password'],
    dbname=db_config['dbname']
)

# 全距離・全競馬場のデータを取得（2013-2023年、3歳以上）
sql = """
SELECT 
    nullif(cast(se.tansho_ninkijun as integer), 0) as popularity,
    cast(se.kakutei_chakujun as integer) as chakujun,
    nullif(cast(se.tansho_odds as float), 0) / 10 as odds,
    cast(ra.kyori as integer) as kyori
FROM jvd_ra as ra
INNER JOIN jvd_se as se
    ON ra.kaisai_nen = se.kaisai_nen
    AND ra.kaisai_tsukihi = se.kaisai_tsukihi
    AND ra.keibajo_code = se.keibajo_code
    AND ra.race_bango = se.race_bango
WHERE ra.kaisai_nen IN ('2013', '2014', '2015', '2016', '2017', '2018', '2019', '2021', '2022', '2023')
  AND ra.kyoso_shubetsu_code = '13'
  AND se.tansho_ninkijun IS NOT NULL
  AND se.kakutei_chakujun IS NOT NULL
  AND se.tansho_odds IS NOT NULL
  AND cast(ra.kyori as integer) >= 1000
ORDER BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
"""

print("データ取得中...")
df = pd.read_sql_query(sql, conn)
conn.close()

print(f"総データ数: {len(df):,}件\n")

# 人気順位別の分析
print("="*80)
print("人気順位別の的中率・配当分析")
print("="*80)

popularity_stats = []

for pop in range(1, 19):  # 1-18番人気
    pop_df = df[df['popularity'] == pop]
    
    if len(pop_df) == 0:
        continue
    
    # 3着以内率
    hit_rate_3 = (pop_df['chakujun'] <= 3).sum() / len(pop_df) * 100
    
    # 1着率
    win_rate = (pop_df['chakujun'] == 1).sum() / len(pop_df) * 100
    
    # 平均オッズ
    avg_odds = pop_df['odds'].mean()
    
    # 期待値（単勝購入想定）
    expected_value = (pop_df['chakujun'] == 1).sum() * pop_df[pop_df['chakujun'] == 1]['odds'].mean() if (pop_df['chakujun'] == 1).sum() > 0 else 0
    expected_value_per_bet = expected_value / len(pop_df) if len(pop_df) > 0 else 0
    roi = expected_value_per_bet * 100  # 回収率
    
    popularity_stats.append({
        'popularity': pop,
        'count': len(pop_df),
        'win_rate': win_rate,
        'hit_rate_3': hit_rate_3,
        'avg_odds': avg_odds,
        'roi': roi
    })

stats_df = pd.DataFrame(popularity_stats)

print(f"\n{'人気':>4} {'出走数':>8} {'勝率':>8} {'複勝率':>8} {'平均オッズ':>10} {'単勝ROI':>10}")
print("-" * 65)

for _, row in stats_df.iterrows():
    print(f"{int(row['popularity']):>4} {int(row['count']):>8,} {row['win_rate']:>7.2f}% {row['hit_rate_3']:>7.2f}% {row['avg_odds']:>9.1f}倍 {row['roi']:>9.1f}%")

# 穴馬候補範囲の評価
print("\n" + "="*80)
print("穴馬候補範囲の評価")
print("="*80)

def evaluate_range(min_pop, max_pop, min_chakujun=1, max_chakujun=3):
    """指定人気範囲での穴馬パフォーマンスを評価"""
    range_df = df[(df['popularity'] >= min_pop) & (df['popularity'] <= max_pop)]
    
    if len(range_df) == 0:
        return None
    
    # 3着以内の馬
    upset_df = range_df[(range_df['chakujun'] >= min_chakujun) & (range_df['chakujun'] <= max_chakujun)]
    
    upset_count = len(upset_df)
    total_count = len(range_df)
    upset_rate = upset_count / total_count * 100
    
    # 平均配当
    avg_return = upset_df['odds'].mean() if len(upset_df) > 0 else 0
    
    # 期待値（全頭購入想定）
    total_bet = len(range_df) * 100  # 1頭100円
    total_return = (upset_df['odds'] * 100).sum()
    roi = (total_return / total_bet * 100) if total_bet > 0 else 0
    
    return {
        'range': f"{min_pop}-{max_pop}番人気",
        'total': total_count,
        'upset_count': upset_count,
        'upset_rate': upset_rate,
        'avg_return': avg_return,
        'roi': roi
    }

# 様々な人気範囲を評価
ranges = [
    (5, 10),
    (6, 10),
    (7, 10),
    (7, 12),
    (7, 15),
    (8, 12),
    (8, 15),
    (10, 15),
]

print(f"\n{'人気範囲':<15} {'総出走':>10} {'3着以内':>10} {'的中率':>8} {'平均配当':>10} {'ROI':>8}")
print("-" * 75)

for min_pop, max_pop in ranges:
    result = evaluate_range(min_pop, max_pop)
    if result:
        print(f"{result['range']:<15} {result['total']:>10,} {result['upset_count']:>10,} {result['upset_rate']:>7.2f}% {result['avg_return']:>9.1f}倍 {result['roi']:>7.1f}%")

# 距離帯別の分析
print("\n" + "="*80)
print("距離帯別×人気範囲の分析（7-12番人気、3着以内）")
print("="*80)

def categorize_distance(kyori):
    if kyori <= 1600:
        return '短距離(1000-1600)'
    else:
        return '中長距離(1700-)'

df['distance_cat'] = df['kyori'].apply(categorize_distance)

print(f"\n{'距離帯':<20} {'総出走':>10} {'3着以内':>10} {'的中率':>8}")
print("-" * 55)

for dist_cat in ['短距離(1000-1600)', '中長距離(1700-)']:
    dist_df = df[(df['distance_cat'] == dist_cat) & 
                 (df['popularity'] >= 7) & 
                 (df['popularity'] <= 12)]
    
    upset_count = ((dist_df['chakujun'] >= 1) & (dist_df['chakujun'] <= 3)).sum()
    total = len(dist_df)
    rate = (upset_count / total * 100) if total > 0 else 0
    
    print(f"{dist_cat:<20} {total:>10,} {upset_count:>10,} {rate:>7.2f}%")

# 推奨事項
print("\n" + "="*80)
print("推奨事項")
print("="*80)
print("""
【人気範囲の選定基準】
1. 的中率: 0.5%-2.0%程度が理想（高すぎると配当が低く、低すぎると当たらない）
2. 平均配当: 15倍以上が望ましい（7番人気で10-20倍、12番人気で30-50倍）
3. ROI: 70%以上あれば実用的（100%超えは難しい）
4. サンプル数: 穴馬サンプルが最低300-500件は欲しい

【上記データから判断】
- 7-10番人気: 中穴狙い、的中率高め、配当やや低め
- 7-12番人気: バランス型（現在の設定）
- 7-15番人気: 大穴含む、データ量確保、的中率やや下がる

短距離データを含めて再学習する場合、データ量確保のため「7-15番人気」も検討価値あり。
ただし15番人気以降は実力不足の馬も多いため、特徴量で見極められるかが鍵。
""")

print("\n" + "="*80)
