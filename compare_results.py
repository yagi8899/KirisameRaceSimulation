#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数テスト結果比較スクリプト

results/配下の全betting_summaryファイルを読み込んで、
統合比較表を作成します。
"""

import pandas as pd
from pathlib import Path
import re


def compare_all_results(output_file='results/all_results_comparison.tsv'):
    """
    results/配下の全betting_summaryファイルを比較
    
    Args:
        output_file (str): 出力ファイルパス
    
    Returns:
        DataFrame: 比較結果
    """
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("[ERROR] resultsディレクトリが見つかりません")
        return None
    
    # betting_summary_*.tsvを全部読み込み
    summary_files = list(results_dir.glob('betting_summary_*.tsv'))
    
    if not summary_files:
        print("[!]  結果ファイルが見つかりません")
        return None
    
    print(f"[FILE] {len(summary_files)}個の結果ファイルを発見")
    print("=" * 60)
    
    all_results = []
    
    for file in summary_files:
        # ファイル名から情報を抽出
        # 例: betting_summary_tokyo_turf_3ageup_long_train2020-2022_test2023.tsv
        filename = file.stem  # 拡張子なし
        
        # 学習期間を抽出
        train_match = re.search(r'_train(\d{4})-(\d{4})', filename)
        if train_match:
            train_start = train_match.group(1)
            train_end = train_match.group(2)
            train_period = f"{train_start}-{train_end}"
        else:
            train_period = "unknown"
        
        # テスト年を抽出
        test_match = re.search(r'_test(\d{4})(?:-(\d{4}))?', filename)
        if test_match:
            test_start = test_match.group(1)
            test_end = test_match.group(2) if test_match.group(2) else test_start
            test_period = f"{test_start}-{test_end}" if test_start != test_end else test_start
        else:
            test_period = "unknown"
        
        # モデル名を抽出（betting_summary_の後、_trainの前）
        model_match = re.search(r'betting_summary_(.+?)_train', filename)
        if model_match:
            model_name = model_match.group(1)
        else:
            # _trainがない場合
            model_name = filename.replace('betting_summary_', '')
        
        try:
            # ファイルを読み込み
            df = pd.read_csv(file, sep='\t', index_col=0)
            
            # 主要指標を抽出
            result_row = {
                'モデル': model_name,
                '学習期間': train_period,
                'テスト期間': test_period,
                '単勝的中率': df.loc['単勝', '的中率(%)'],
                '単勝回収率': df.loc['単勝', '回収率(%)'],
                '単勝的中数': int(df.loc['単勝', '的中数']),
                '複勝的中率': df.loc['複勝', '的中率(%)'],
                '複勝回収率': df.loc['複勝', '回収率(%)'],
                '馬連的中率': df.loc['馬連', '的中率(%)'],
                '馬連回収率': df.loc['馬連', '回収率(%)'],
                'ワイド的中率': df.loc['ワイド', '的中率(%)'],
                'ワイド回収率': df.loc['ワイド', '回収率(%)'],
                '三連複的中率': df.loc['３連複', '的中率(%)'],
                '三連複回収率': df.loc['３連複', '回収率(%)'],
            }
            
            all_results.append(result_row)
            
            print(f"[OK] {file.name}")
            print(f"   学習: {train_period}, テスト: {test_period}")
            print(f"   単勝: {result_row['単勝的中率']:.1f}%, 回収率: {result_row['単勝回収率']:.1f}%")
            
        except Exception as e:
            print(f"[!]  {file.name} 読み込みエラー: {e}")
    
    if not all_results:
        print("[ERROR] 有効な結果がありません")
        return None
    
    # DataFrameに変換
    comparison_df = pd.DataFrame(all_results)
    
    # ソート（学習期間→テスト期間→モデル名）
    comparison_df = comparison_df.sort_values(['学習期間', 'テスト期間', 'モデル'])
    
    # 保存
    comparison_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("[+] 比較結果サマリー")
    print("=" * 60)
    
    # 学習期間ごとの平均を表示
    if '学習期間' in comparison_df.columns and comparison_df['学習期間'].nunique() > 1:
        print("\n【学習期間別の平均的中率】")
        grouped = comparison_df.groupby('学習期間')[['単勝的中率', '複勝的中率', '三連複的中率']].mean()
        print(grouped.to_string())
    
    # テスト期間ごとの平均
    if 'テスト期間' in comparison_df.columns and comparison_df['テスト期間'].nunique() > 1:
        print("\n【テスト期間別の平均的中率】")
        grouped = comparison_df.groupby('テスト期間')[['単勝的中率', '複勝的中率', '三連複的中率']].mean()
        print(grouped.to_string())
    
    print(f"\n[LIST] 詳細結果を {output_file} に保存しました")
    
    return comparison_df


def analyze_year_trends(comparison_df):
    """
    年ごとのトレンド分析
    
    Args:
        comparison_df (DataFrame): 比較結果
    """
    if comparison_df is None or len(comparison_df) == 0:
        return
    
    print("\n" + "=" * 60)
    print("[STATS] 年次トレンド分析")
    print("=" * 60)
    
    # テスト年を抽出
    comparison_df['テスト年'] = comparison_df['テスト期間'].apply(lambda x: x.split('-')[0] if '-' in str(x) else str(x))
    
    # 年ごとの統計
    yearly_stats = comparison_df.groupby('テスト年').agg({
        '単勝的中率': ['mean', 'std', 'min', 'max'],
        '複勝的中率': ['mean', 'std'],
        '単勝回収率': ['mean', 'std']
    })
    
    print("\n【テスト年別統計】")
    print(yearly_stats.to_string())
    
    # トレンドの判定
    years = sorted(comparison_df['テスト年'].unique())
    if len(years) >= 2:
        yearly_mean = comparison_df.groupby('テスト年')['単勝的中率'].mean()
        
        print("\n【トレンド判定】")
        for i in range(len(years) - 1):
            year1, year2 = years[i], years[i+1]
            change = yearly_mean[year2] - yearly_mean[year1]
            pct_change = (change / yearly_mean[year1]) * 100
            
            if change > 0:
                trend = "[STATS] 改善"
            elif change < 0:
                trend = "[-] 悪化"
            else:
                trend = "➡️  横ばい"
            
            print(f"{year1}→{year2}: {trend} ({change:+.1f}%, {pct_change:+.1f}%変化)")


if __name__ == '__main__':
    import sys
    
    output_file = 'results/all_results_comparison.tsv'
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    # 全結果を比較
    comparison_df = compare_all_results(output_file)
    
    if comparison_df is not None:
        # トレンド分析
        analyze_year_trends(comparison_df)
        
        print(f"\n{'='*60}")
        print("[DONE] 比較分析完了!")
        print(f"{'='*60}")
