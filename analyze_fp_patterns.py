#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FP（偽陽性）分析: Upset候補なのに3着以内に来なかった馬の特徴を分析

目的:
- FPに共通するパターンを見つける
- 欠けている特徴量のヒントを得る
"""

import argparse
import pandas as pd
import numpy as np
from collections import Counter


def load_results(file_path: str) -> pd.DataFrame:
    """結果ファイルを読み込む"""
    df = pd.read_csv(file_path, sep='\t')
    return df


def analyze_fp(df: pd.DataFrame, upset_threshold: float = 0.15):
    """FP（偽陽性）を分析"""
    
    # Upset候補を抽出（7-12番人気 & 閾値以上）
    candidates = df[
        (df['穴馬確率'] >= upset_threshold) & 
        (df['人気順'] >= 7) & 
        (df['人気順'] <= 12)
    ].copy()
    
    # TP（的中）と FP（外れ）に分類
    candidates['is_hit'] = candidates['確定着順'] <= 3
    tp = candidates[candidates['is_hit'] == True]
    fp = candidates[candidates['is_hit'] == False]
    
    print("=" * 70)
    print("FP（偽陽性）分析")
    print("=" * 70)
    print(f"\n全候補: {len(candidates)}")
    print(f"  TP（的中）: {len(tp)} ({len(tp)/len(candidates)*100:.1f}%)")
    print(f"  FP（外れ）: {len(fp)} ({len(fp)/len(candidates)*100:.1f}%)")
    
    # 1. 人気順の分布
    print("\n" + "-" * 50)
    print("【1】人気順の分布")
    print("-" * 50)
    print(f"{'人気':<6} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for pop in range(7, 13):
        tp_count = len(tp[tp['人気順'] == pop])
        fp_count = len(fp[fp['人気順'] == pop])
        total = tp_count + fp_count
        tp_rate = tp_count / total * 100 if total > 0 else 0
        print(f"{pop}番人気 {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 2. Ranker予測順位の分布
    print("\n" + "-" * 50)
    print("【2】Ranker予測順位の分布")
    print("-" * 50)
    print(f"{'予測順位':<10} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for rank_range in [(1, 3), (4, 6), (7, 9), (10, 12), (13, 18)]:
        tp_count = len(tp[(tp['予測順位'] >= rank_range[0]) & (tp['予測順位'] <= rank_range[1])])
        fp_count = len(fp[(fp['予測順位'] >= rank_range[0]) & (fp['予測順位'] <= rank_range[1])])
        total = tp_count + fp_count
        tp_rate = tp_count / total * 100 if total > 0 else 0
        print(f"{rank_range[0]}-{rank_range[1]}位    {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 3. 穴馬確率の分布
    print("\n" + "-" * 50)
    print("【3】穴馬確率の分布")
    print("-" * 50)
    print(f"{'確率帯':<12} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for prob_range in [(0.15, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 1.0)]:
        tp_count = len(tp[(tp['穴馬確率'] >= prob_range[0]) & (tp['穴馬確率'] < prob_range[1])])
        fp_count = len(fp[(fp['穴馬確率'] >= prob_range[0]) & (fp['穴馬確率'] < prob_range[1])])
        total = tp_count + fp_count
        tp_rate = tp_count / total * 100 if total > 0 else 0
        print(f"{prob_range[0]:.2f}-{prob_range[1]:.2f}   {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 4. オッズの分布
    print("\n" + "-" * 50)
    print("【4】単勝オッズの分布")
    print("-" * 50)
    print(f"{'オッズ帯':<12} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for odds_range in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 1000)]:
        tp_count = len(tp[(tp['単勝オッズ'] >= odds_range[0]) & (tp['単勝オッズ'] < odds_range[1])])
        fp_count = len(fp[(fp['単勝オッズ'] >= odds_range[0]) & (fp['単勝オッズ'] < odds_range[1])])
        total = tp_count + fp_count
        tp_rate = tp_count / total * 100 if total > 0 else 0
        print(f"{odds_range[0]}-{odds_range[1]}倍   {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 5. 芝/ダートの分布
    print("\n" + "-" * 50)
    print("【5】芝/ダートの分布")
    print("-" * 50)
    print(f"{'区分':<8} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for surface in df['芝ダ区分'].unique():
        tp_count = len(tp[tp['芝ダ区分'] == surface])
        fp_count = len(fp[fp['芝ダ区分'] == surface])
        total = tp_count + fp_count
        tp_rate = tp_count / total * 100 if total > 0 else 0
        print(f"{surface:<8} {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 6. 距離の分布
    print("\n" + "-" * 50)
    print("【6】距離の分布")
    print("-" * 50)
    print(f"{'距離帯':<12} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for dist_range in [(0, 1400), (1400, 1800), (1800, 2200), (2200, 4000)]:
        tp_count = len(tp[(tp['距離'] >= dist_range[0]) & (tp['距離'] < dist_range[1])])
        fp_count = len(fp[(fp['距離'] >= dist_range[0]) & (fp['距離'] < dist_range[1])])
        total = tp_count + fp_count
        tp_rate = tp_count / total * 100 if total > 0 else 0
        dist_name = f"{dist_range[0]}-{dist_range[1]}m"
        print(f"{dist_name:<12} {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 7. 競馬場の分布
    print("\n" + "-" * 50)
    print("【7】競馬場の分布")
    print("-" * 50)
    track_names = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    print(f"{'競馬場':<8} {'TP':>6} {'FP':>6} {'TP率':>8}")
    for track in sorted(df['競馬場'].unique()):
        tp_count = len(tp[tp['競馬場'] == track])
        fp_count = len(fp[fp['競馬場'] == track])
        total = tp_count + fp_count
        if total > 0:
            tp_rate = tp_count / total * 100
            name = track_names.get(track, track)
            print(f"{name:<8} {tp_count:>6} {fp_count:>6} {tp_rate:>7.1f}%")
    
    # 8. 実際の着順分布（FPのみ）
    print("\n" + "-" * 50)
    print("【8】FPの実際の着順分布（惜しい外れの分析）")
    print("-" * 50)
    print(f"{'着順':<8} {'件数':>6} {'割合':>8}")
    for finish in range(4, 19):
        count = len(fp[fp['確定着順'] == finish])
        rate = count / len(fp) * 100 if len(fp) > 0 else 0
        if count > 0:
            bar = "█" * int(rate / 2)
            print(f"{finish}着     {count:>6} {rate:>7.1f}% {bar}")
    
    # 9. TP vs FP の統計比較
    print("\n" + "-" * 50)
    print("【9】TP vs FP の特徴量比較")
    print("-" * 50)
    print(f"{'特徴量':<16} {'TP平均':>10} {'FP平均':>10} {'差':>10}")
    print("-" * 50)
    
    for col in ['人気順', '予測順位', '予測スコア', '穴馬確率', '単勝オッズ', '距離']:
        if col in candidates.columns:
            tp_mean = tp[col].mean()
            fp_mean = fp[col].mean()
            diff = tp_mean - fp_mean
            print(f"{col:<16} {tp_mean:>10.2f} {fp_mean:>10.2f} {diff:>+10.2f}")
    
    # 10. 最も信頼性の高い条件を探す
    print("\n" + "=" * 70)
    print("【10】有望な条件の組み合わせ")
    print("=" * 70)
    
    conditions = []
    
    # Ranker上位 + 各条件
    for ranker_filter in [3, 5]:
        for surface in df['芝ダ区分'].unique():
            filtered = candidates[
                (candidates['予測順位'] <= ranker_filter) & 
                (candidates['芝ダ区分'] == surface)
            ]
            if len(filtered) >= 10:
                tp_rate = filtered['is_hit'].mean() * 100
                conditions.append({
                    'condition': f"Ranker上位{ranker_filter} & {surface}",
                    'count': len(filtered),
                    'tp': filtered['is_hit'].sum(),
                    'tp_rate': tp_rate
                })
    
    # 人気 + Ranker
    for pop_range in [(7, 8), (9, 10), (11, 12)]:
        for ranker_filter in [3, 5, 8]:
            filtered = candidates[
                (candidates['人気順'] >= pop_range[0]) & 
                (candidates['人気順'] <= pop_range[1]) &
                (candidates['予測順位'] <= ranker_filter)
            ]
            if len(filtered) >= 10:
                tp_rate = filtered['is_hit'].mean() * 100
                conditions.append({
                    'condition': f"{pop_range[0]}-{pop_range[1]}番人気 & Ranker上位{ranker_filter}",
                    'count': len(filtered),
                    'tp': filtered['is_hit'].sum(),
                    'tp_rate': tp_rate
                })
    
    # ソートして表示
    conditions.sort(key=lambda x: x['tp_rate'], reverse=True)
    print(f"{'条件':<35} {'候補数':>8} {'TP':>6} {'TP率':>8}")
    print("-" * 60)
    for cond in conditions[:15]:
        print(f"{cond['condition']:<35} {cond['count']:>8} {cond['tp']:>6} {cond['tp_rate']:>7.1f}%")
    
    # 11. FPの共通パターンを探す
    print("\n" + "=" * 70)
    print("【11】FPの共通パターン分析")
    print("=" * 70)
    
    # FPの中で特に多いパターン
    fp_patterns = fp.groupby(['芝ダ区分', '競馬場']).size().sort_values(ascending=False)
    print("\nFPが多い条件（上位10）:")
    for (surface, track), count in fp_patterns.head(10).items():
        total_in_cond = len(candidates[(candidates['芝ダ区分'] == surface) & (candidates['競馬場'] == track)])
        fp_rate = count / total_in_cond * 100 if total_in_cond > 0 else 0
        track_name = track_names.get(track, track)
        print(f"  {track_name} {surface}: FP {count}件 / 全{total_in_cond}件 (FP率 {fp_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='FP分析')
    parser.add_argument('input_file', help='予測結果ファイル（TSV）')
    parser.add_argument('--threshold', type=float, default=0.15, help='Upset閾値')
    args = parser.parse_args()
    
    df = load_results(args.input_file)
    analyze_fp(df, args.threshold)


if __name__ == '__main__':
    main()
