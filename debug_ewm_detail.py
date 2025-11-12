#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EWM実装の詳細検証スクリプト
実際のデータでSQL平均とEWMの違いを詳しく調べる
"""
import psycopg2
import pandas as pd
import numpy as np

conn = psycopg2.connect(
    host='localhost', port='5432', user='postgres',
    password='ahtaht88', dbname='keiba'
)

print("="*80)
print("[*] EWM実装の詳細検証")
print("="*80)

# 2013-2022年の学習データで検証
sql = """
SELECT 
    ra.kaisai_nen,
    ra.kaisai_tsukihi,
    ra.race_bango,
    ra.kyori,
    ra.shusso_tosu,
    seum.ketto_toroku_bango,
    trim(seum.bamei) as bamei,
    seum.kakutei_chakujun,
    AVG(
        1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float))
    ) OVER (
        PARTITION BY seum.ketto_toroku_bango
        ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS past_avg_sotai_chakujun_sql
FROM jvd_ra ra
INNER JOIN jvd_se seum ON
    ra.kaisai_nen = seum.kaisai_nen AND
    ra.kaisai_tsukihi = seum.kaisai_tsukihi AND
    ra.keibajo_code = seum.keibajo_code AND
    ra.race_bango = seum.race_bango
WHERE cast(ra.kaisai_nen as integer) BETWEEN 2013 AND 2022
    AND ra.keibajo_code = '05'
    AND ra.kyoso_shubetsu_code = '13'
    AND ra.track_code IN ('11', '14', '17', '20', '23', '25', '28')
    AND cast(ra.kyori as integer) >= 1700
    AND seum.kakutei_chakujun <> '00'
ORDER BY seum.ketto_toroku_bango, cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
LIMIT 5000
"""

print("\n[+] データ読み込み中...")
df = pd.read_sql(sql, conn)
print(f"取得件数: {len(df)}件")

# 数値化
df['kakutei_chakujun'] = pd.to_numeric(df['kakutei_chakujun'], errors='coerce')
df['shusso_tosu'] = pd.to_numeric(df['shusso_tosu'], errors='coerce')
df['kaisai_nen'] = pd.to_numeric(df['kaisai_nen'], errors='coerce')
df['kaisai_tsukihi'] = pd.to_numeric(df['kaisai_tsukihi'], errors='coerce')

# ソート
df = df.sort_values(['ketto_toroku_bango', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()

print("\n🔄 EWM計算中...")

# EWM計算
def calc_ewm_past_avg(group):
    group['sotai_chakujun'] = 1 - (group['kakutei_chakujun'] / group['shusso_tosu'])
    group['past_avg_sotai_chakujun_ewm'] = group['sotai_chakujun'].shift(1).ewm(
        span=3, 
        adjust=False,
        min_periods=1
    ).mean()
    return group

df = df.groupby('ketto_toroku_bango', group_keys=False).apply(calc_ewm_past_avg)

# 両方が存在する行のみで比較
df_compare = df[df['past_avg_sotai_chakujun_sql'].notna() & df['past_avg_sotai_chakujun_ewm'].notna()].copy()

print(f"\n比較対象件数: {len(df_compare)}件")

# 統計情報
print("\n" + "="*80)
print("【統計比較】")
print("="*80)

print(f"\nSQL平均版:")
print(f"  平均: {df_compare['past_avg_sotai_chakujun_sql'].mean():.6f}")
print(f"  中央値: {df_compare['past_avg_sotai_chakujun_sql'].median():.6f}")
print(f"  標準偏差: {df_compare['past_avg_sotai_chakujun_sql'].std():.6f}")
print(f"  最小値: {df_compare['past_avg_sotai_chakujun_sql'].min():.6f}")
print(f"  最大値: {df_compare['past_avg_sotai_chakujun_sql'].max():.6f}")

print(f"\nEWM版:")
print(f"  平均: {df_compare['past_avg_sotai_chakujun_ewm'].mean():.6f}")
print(f"  中央値: {df_compare['past_avg_sotai_chakujun_ewm'].median():.6f}")
print(f"  標準偏差: {df_compare['past_avg_sotai_chakujun_ewm'].std():.6f}")
print(f"  最小値: {df_compare['past_avg_sotai_chakujun_ewm'].min():.6f}")
print(f"  最大値: {df_compare['past_avg_sotai_chakujun_ewm'].max():.6f}")

# 差分分析
df_compare['diff'] = df_compare['past_avg_sotai_chakujun_ewm'] - df_compare['past_avg_sotai_chakujun_sql']
df_compare['abs_diff'] = df_compare['diff'].abs()

print(f"\n差分統計:")
print(f"  平均差分: {df_compare['diff'].mean():.6f}")
print(f"  絶対差分平均: {df_compare['abs_diff'].mean():.6f}")
print(f"  標準偏差: {df_compare['diff'].std():.6f}")

# ヒストグラム的な分析
print(f"\n差分の分布:")
print(f"  差分 < -0.1: {(df_compare['diff'] < -0.1).sum()}件 ({(df_compare['diff'] < -0.1).sum()/len(df_compare)*100:.1f}%)")
print(f"  -0.1 <= 差分 < -0.05: {((df_compare['diff'] >= -0.1) & (df_compare['diff'] < -0.05)).sum()}件")
print(f"  -0.05 <= 差分 < 0.05: {((df_compare['diff'] >= -0.05) & (df_compare['diff'] < 0.05)).sum()}件 ({((df_compare['diff'] >= -0.05) & (df_compare['diff'] < 0.05)).sum()/len(df_compare)*100:.1f}%)")
print(f"  0.05 <= 差分 < 0.1: {((df_compare['diff'] >= 0.05) & (df_compare['diff'] < 0.1)).sum()}件")
print(f"  差分 >= 0.1: {(df_compare['diff'] >= 0.1).sum()}件 ({(df_compare['diff'] >= 0.1).sum()/len(df_compare)*100:.1f}%)")

# 大きく差が出るケースを調査
print("\n" + "="*80)
print("【差分が大きいケース Top20】")
print("="*80)

large_diff = df_compare.nlargest(20, 'abs_diff')[
    ['ketto_toroku_bango', 'bamei', 'kaisai_nen', 'kaisai_tsukihi', 
     'past_avg_sotai_chakujun_sql', 'past_avg_sotai_chakujun_ewm', 'diff']
]
print(large_diff.to_string(index=False))

# 特定の馬の時系列推移を見る
print("\n" + "="*80)
print("【サンプル馬の時系列推移】")
print("="*80)

sample_horse = df_compare['ketto_toroku_bango'].value_counts().head(1).index[0]
horse_df = df_compare[df_compare['ketto_toroku_bango'] == sample_horse].head(10)
print(f"\n馬ID: {sample_horse}")
print(f"馬名: {horse_df.iloc[0]['bamei']}")
print(f"\n時系列データ:")
print(horse_df[['kaisai_nen', 'kaisai_tsukihi', 'kakutei_chakujun', 
               'past_avg_sotai_chakujun_sql', 'past_avg_sotai_chakujun_ewm', 'diff']].to_string(index=False))

conn.close()

print("\n" + "="*80)
print("【考察】")
print("="*80)
print("1. 平均差分が負 → EWMの方が低く出てる")
print("2. 差分が±0.05以内が大多数 → 大きな違いはない")
print("3. 一部で大きな差 → これが問題の可能性")
print("\n次の調査:")
print("  - 差分が大きいレースの着順を見る")
print("  - 1-3走目のデータでEWMが不安定になってないか")
print("  - min_periods=1が原因で精度が落ちてないか")
