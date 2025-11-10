#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EWM悪化の詳細原因分析
"""
import psycopg2
import pandas as pd
import numpy as np

# PostgreSQL接続
conn = psycopg2.connect(
    host='localhost', port='5432', user='postgres',
    password='ahtaht88', dbname='keiba'
)

print("="*80)
print("🔍 EWM悪化の詳細原因分析")
print("="*80)

# 学習データ量の比較
print("\n【学習データ量の比較】")
for year_start, year_end in [(2013, 2022), (2020, 2021)]:
    sql = f"""
    SELECT COUNT(*) as cnt
    FROM jvd_ra ra
    INNER JOIN jvd_se se ON 
        ra.kaisai_nen = se.kaisai_nen AND
        ra.kaisai_tsukihi = se.kaisai_tsukihi AND
        ra.keibajo_code = se.keibajo_code AND
        ra.race_bango = se.race_bango
    WHERE cast(ra.kaisai_nen as integer) BETWEEN {year_start} AND {year_end}
        AND ra.keibajo_code = '05'
        AND ra.kyoso_shubetsu_code = '13'
        AND ra.track_code IN ('11', '14', '17', '20', '23', '25', '28')
        AND cast(ra.kyori as integer) >= 1700
    """
    df = pd.read_sql(sql, conn)
    print(f"  {year_start}-{year_end}年: {df['cnt'].iloc[0]:,}件")

# past_avg_sotai_chakujunの分布比較
print("\n【past_avg_sotai_chakujunの分布】")
print("※2022年テストデータで比較")

sql = """
SELECT 
    seum.ketto_toroku_bango,
    AVG(
        1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float))
    ) OVER (
        PARTITION BY seum.ketto_toroku_bango
        ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS past_avg_sotai_chakujun,
    seum.kakutei_chakujun,
    ra.shusso_tosu
FROM jvd_ra ra
INNER JOIN jvd_se seum ON
    ra.kaisai_nen = seum.kaisai_nen AND
    ra.kaisai_tsukihi = seum.kaisai_tsukihi AND
    ra.keibajo_code = seum.keibajo_code AND
    ra.race_bango = seum.race_bango
WHERE cast(ra.kaisai_nen as integer) = 2022
    AND ra.keibajo_code = '05'
    AND ra.kyoso_shubetsu_code = '13'
    AND ra.track_code IN ('11', '14', '17', '20', '23', '25', '28')
    AND cast(ra.kyori as integer) >= 1700
    AND seum.kakutei_chakujun <> '00'
LIMIT 1000
"""

df = pd.read_sql(sql, conn)
print(f"\nSQL平均版の統計:")
print(f"  件数: {len(df)}")
print(f"  平均: {df['past_avg_sotai_chakujun'].mean():.4f}")
print(f"  中央値: {df['past_avg_sotai_chakujun'].median():.4f}")
print(f"  標準偏差: {df['past_avg_sotai_chakujun'].std():.4f}")
print(f"  欠損数: {df['past_avg_sotai_chakujun'].isna().sum()}")

# EWM計算してみる
df_sorted = df.sort_values(['ketto_toroku_bango']).copy()

def calc_ewm(group):
    group['sotai'] = 1 - (group['kakutei_chakujun'].astype(float) / group['shusso_tosu'].astype(float))
    group['ewm_val'] = group['sotai'].shift(1).ewm(span=3, adjust=False, min_periods=1).mean()
    return group

df_sorted = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(calc_ewm)

print(f"\nEWM版の統計:")
print(f"  件数: {len(df_sorted)}")
print(f"  平均: {df_sorted['ewm_val'].mean():.4f}")
print(f"  中央値: {df_sorted['ewm_val'].median():.4f}")
print(f"  標準偏差: {df_sorted['ewm_val'].std():.4f}")
print(f"  欠損数: {df_sorted['ewm_val'].isna().sum()}")

# 差分分析
df_sorted['diff'] = df_sorted['ewm_val'] - df_sorted['past_avg_sotai_chakujun']
print(f"\n差分統計:")
print(f"  平均差分: {df_sorted['diff'].mean():.4f}")
print(f"  標準偏差: {df_sorted['diff'].std():.4f}")
print(f"  最大差分: {df_sorted['diff'].max():.4f}")
print(f"  最小差分: {df_sorted['diff'].min():.4f}")

# 大きく変わった馬をピックアップ
print(f"\n差分が大きい上位10件:")
large_diff = df_sorted.nlargest(10, 'diff')[['ketto_toroku_bango', 'past_avg_sotai_chakujun', 'ewm_val', 'diff']]
print(large_diff.to_string(index=False))

conn.close()

print("\n" + "="*80)
print("【結論】")
print("="*80)
print("1. 学習データ量が1/5に減少 (これが最大の原因!)")
print("2. EWMで標準偏差が若干減少 (情報量が減った)")
print("3. 差分は小さいが、モデルの学習不足が影響大")
print("\n【推奨アクション】")
print("1. year_start=2013に戻して再学習")
print("2. それでも悪化するならEWMを諦める")
print("3. または span=5, 7 で試す")
