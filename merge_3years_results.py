#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3年間のテスト結果を統合してpredicted_results.tsvに出力
"""
import pandas as pd
import os

# 統合するファイルリスト（2016-2022, 2017-2023, 2018-2024の7年学習モデル）
result_files = [
    # 2023年テスト (2016-2022学習)
    'results/predicted_results_tokyo_turf_3ageup_long_train2016-2022_test2023.tsv',
    'results/predicted_results_tokyo_turf_3ageup_short_train2016-2022_test2023.tsv',
    
    # 2024年テスト (2017-2023学習)
    'results/predicted_results_tokyo_turf_3ageup_long_train2017-2023_test2024.tsv',
    'results/predicted_results_tokyo_turf_3ageup_short_train2017-2023_test2024.tsv',
    
    # 2025年テスト (2018-2024学習)
    'results/predicted_results_tokyo_turf_3ageup_long_train2018-2024_test2025.tsv',
    'results/predicted_results_tokyo_turf_3ageup_short_train2018-2024_test2025.tsv',
]

print("=" * 60)
print("3年間テスト結果の統合")
print("=" * 60)

# データフレームのリスト
dfs = []

for file_path in result_files:
    if os.path.exists(file_path):
        print(f"[LOAD] {file_path}")
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        print(f"  レコード数: {len(df)}")
        dfs.append(df)
    else:
        print(f"[SKIP] ファイルが見つかりません: {file_path}")

if not dfs:
    print("\n[ERROR] 統合するファイルが見つかりませんでした")
    exit(1)

# 統合
print("\n" + "=" * 60)
print("データを統合中...")
merged_df = pd.concat(dfs, ignore_index=True)

# 年でソート
if '開催年' in merged_df.columns:
    merged_df = merged_df.sort_values(['開催年', '開催日', 'レース番号', '馬番'], ignore_index=True)

print(f"統合後レコード数: {len(merged_df)}")

# 年別の内訳を表示
if '開催年' in merged_df.columns:
    print("\n年別レコード数:")
    year_counts = merged_df['開催年'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}年: {count}件")

# 出力
output_file = 'results/predicted_results.tsv'
merged_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
print(f"\n[OK] 統合結果を {output_file} に保存しました")

print("=" * 60)
print("完了!")
print("=" * 60)
