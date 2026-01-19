#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: 穴馬予測閾値最適化

複数の閾値で予測を実行し、最適なバランスを見つける
目標: 候補数20-50頭/年、適合率5%以上、ROI150%以上
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

def run_prediction_with_threshold(year, threshold):
    """指定した閾値で予測実行"""
    cmd = f"python upset_predictor.py --year {year} --threshold {threshold}"
    
    # 出力をキャプチャ
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=Path(__file__).parent
    )
    
    # 結果をパース
    output = result.stdout
    
    # メトリクスを抽出
    metrics = {}
    for line in output.split('\n'):
        if '総候補数:' in line:
            metrics['candidates'] = int(line.split(':')[1].strip().replace('頭', ''))
        elif '的中数:' in line:
            metrics['hits'] = int(line.split(':')[1].strip().replace('頭', ''))
        elif '適合率 (Precision):' in line:
            metrics['precision'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'レース的中率:' in line:
            metrics['hit_rate'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'ROI:' in line:
            metrics['roi'] = float(line.split(':')[1].strip().replace('%', ''))
    
    return metrics


def optimize_thresholds(years, thresholds):
    """複数年・複数閾値で最適化"""
    print("=" * 80)
    print("Phase 2: 閾値最適化実験")
    print("=" * 80)
    print(f"対象年: {years}")
    print(f"テスト閾値: {thresholds}")
    print()
    
    results = []
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"閾値 {threshold} でテスト中...")
        print(f"{'='*80}")
        
        threshold_results = {
            'threshold': threshold,
            'total_candidates': 0,
            'total_hits': 0,
            'total_races': 0,
            'hit_races': 0,
            'yearly_results': []
        }
        
        for year in years:
            print(f"\n{year}年...")
            metrics = run_prediction_with_threshold(year, threshold)
            
            if metrics:
                threshold_results['total_candidates'] += metrics.get('candidates', 0)
                threshold_results['total_hits'] += metrics.get('hits', 0)
                threshold_results['yearly_results'].append({
                    'year': year,
                    **metrics
                })
                
                print(f"  候補: {metrics.get('candidates', 0)}頭")
                print(f"  的中: {metrics.get('hits', 0)}頭")
                print(f"  適合率: {metrics.get('precision', 0):.2f}%")
                print(f"  ROI: {metrics.get('roi', 0):.1f}%")
        
        # 集計
        avg_candidates = threshold_results['total_candidates'] / len(years)
        avg_precision = (threshold_results['total_hits'] / threshold_results['total_candidates'] * 100) if threshold_results['total_candidates'] > 0 else 0
        avg_roi = np.mean([r.get('roi', 0) for r in threshold_results['yearly_results']])
        
        threshold_results['avg_candidates_per_year'] = avg_candidates
        threshold_results['overall_precision'] = avg_precision
        threshold_results['avg_roi'] = avg_roi
        
        results.append(threshold_results)
        
        print(f"\n【閾値 {threshold} 集計】")
        print(f"  平均候補数/年: {avg_candidates:.1f}頭")
        print(f"  全体適合率: {avg_precision:.2f}%")
        print(f"  平均ROI: {avg_roi:.1f}%")
    
    return results


def display_summary(results):
    """結果サマリー表示"""
    print("\n" + "=" * 80)
    print("閾値最適化結果サマリー")
    print("=" * 80)
    
    # DataFrameで整形
    summary_data = []
    for r in results:
        summary_data.append({
            '閾値': r['threshold'],
            '平均候補数/年': f"{r['avg_candidates_per_year']:.1f}頭",
            '全体適合率': f"{r['overall_precision']:.2f}%",
            '平均ROI': f"{r['avg_roi']:.1f}%",
            '総的中数': r['total_hits']
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # 推奨閾値を提案
    print("\n" + "=" * 80)
    print("推奨閾値")
    print("=" * 80)
    
    # 条件1: 候補数20-50頭/年
    candidates_ok = [r for r in results if 20 <= r['avg_candidates_per_year'] <= 50]
    
    if candidates_ok:
        # 条件2: ROIが最大
        best = max(candidates_ok, key=lambda x: x['avg_roi'])
        print(f"✅ 閾値 {best['threshold']} を推奨")
        print(f"   平均候補数: {best['avg_candidates_per_year']:.1f}頭/年")
        print(f"   全体適合率: {best['overall_precision']:.2f}%")
        print(f"   平均ROI: {best['avg_roi']:.1f}%")
    else:
        # 候補数に合う閾値がない場合、ROI最大を選択
        best = max(results, key=lambda x: x['avg_roi'])
        print(f"⚠️ 理想的な候補数範囲(20-50頭)に該当なし")
        print(f"   ROI最大の閾値 {best['threshold']} を推奨")
        print(f"   平均候補数: {best['avg_candidates_per_year']:.1f}頭/年")
        print(f"   全体適合率: {best['overall_precision']:.2f}%")
        print(f"   平均ROI: {best['avg_roi']:.1f}%")
    
    # 詳細結果をファイル出力
    output_file = Path('results') / 'threshold_optimization_summary.tsv'
    
    detailed_results = []
    for r in results:
        for yearly in r['yearly_results']:
            detailed_results.append({
                '閾値': r['threshold'],
                '年': yearly['year'],
                '候補数': yearly.get('candidates', 0),
                '的中数': yearly.get('hits', 0),
                '適合率': yearly.get('precision', 0),
                'レース的中率': yearly.get('hit_rate', 0),
                'ROI': yearly.get('roi', 0)
            })
    
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"\n詳細結果を {output_file} に保存しました")


def main():
    """メイン処理"""
    # テスト対象年
    years = [2019, 2021, 2022, 2023]
    
    # テスト閾値（0.3は既に実行済みなので0.4から）
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
    
    # 最適化実行
    results = optimize_thresholds(years, thresholds)
    
    # サマリー表示
    display_summary(results)
    
    print("\n" + "=" * 80)
    print("最適化完了!")
    print("=" * 80)


if __name__ == '__main__':
    main()
