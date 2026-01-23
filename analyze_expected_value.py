#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期待値分析スクリプト

穴馬候補の期待値（オッズ × 的中率）を条件別に分析し、
プラス収支が見込める条件を特定する。

Usage:
    python analyze_expected_value.py check_results/predicted_results_all.tsv
    python analyze_expected_value.py check_results/predicted_results_all.tsv --threshold 0.20
    python analyze_expected_value.py check_results/predicted_results_all.tsv --use-actual-odds
"""

import argparse
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_results(file_path: str) -> pd.DataFrame:
    """結果ファイルを読み込む"""
    df = pd.read_csv(file_path, sep='\t')
    return df


def get_fukusho_odds(row: pd.Series, uma_ban: int) -> float:
    """馬番に対応する複勝オッズを取得"""
    for i, col in enumerate(['複勝1着馬番', '複勝2着馬番', '複勝3着馬番']):
        if row[col] == uma_ban:
            odds_col = f'複勝{i+1}着オッズ'
            return row[odds_col] if pd.notna(row[odds_col]) else 0
    return 0


def calculate_expected_value(
    candidates: pd.DataFrame,
    use_actual_odds: bool = False
) -> Tuple[float, float, float, float]:
    """
    期待値を計算
    
    Returns:
        (候補数, TP率, 平均オッズ, 期待値)
    """
    if len(candidates) == 0:
        return 0, 0, 0, 0
    
    tp_rate = candidates['is_hit'].mean()
    avg_tansho_odds = candidates['単勝オッズ'].mean()
    
    if use_actual_odds:
        # 実際の複勝オッズを使用（的中した馬のみ）
        hits = candidates[candidates['is_hit'] == True]
        if len(hits) > 0:
            total_return = hits['fukusho_odds'].sum()
            total_bet = len(candidates)
            expected_value = total_return / total_bet if total_bet > 0 else 0
        else:
            expected_value = 0
    else:
        # 推定複勝オッズ（単勝の約25%）を使用
        estimated_fukusho_odds = avg_tansho_odds * 0.25
        expected_value = tp_rate * estimated_fukusho_odds
    
    return len(candidates), tp_rate, avg_tansho_odds, expected_value


def analyze_by_odds_band(
    candidates: pd.DataFrame,
    use_actual_odds: bool = False
) -> pd.DataFrame:
    """オッズ帯別の期待値分析"""
    results = []
    
    odds_bands = [
        (0, 10, '0-10倍'),
        (10, 20, '10-20倍'),
        (20, 30, '20-30倍'),
        (30, 50, '30-50倍'),
        (50, 100, '50-100倍'),
        (100, 500, '100-500倍'),
    ]
    
    for odds_min, odds_max, label in odds_bands:
        subset = candidates[
            (candidates['単勝オッズ'] >= odds_min) & 
            (candidates['単勝オッズ'] < odds_max)
        ]
        
        count, tp_rate, avg_odds, ev = calculate_expected_value(subset, use_actual_odds)
        
        if count >= 5:
            results.append({
                '条件': label,
                '候補数': count,
                'TP率': tp_rate,
                '平均オッズ': avg_odds,
                '期待値': ev,
                'ROI': (ev - 1) * 100 if ev > 0 else -100
            })
    
    return pd.DataFrame(results)


def analyze_by_ranker_odds(
    candidates: pd.DataFrame,
    use_actual_odds: bool = False
) -> pd.DataFrame:
    """Ranker上位 × オッズ帯の期待値分析"""
    results = []
    
    ranker_filters = [3, 5, 8]
    odds_bands = [(10, 30), (30, 50), (50, 100)]
    
    for ranker_max in ranker_filters:
        for odds_min, odds_max in odds_bands:
            subset = candidates[
                (candidates['予測順位'] <= ranker_max) & 
                (candidates['単勝オッズ'] >= odds_min) & 
                (candidates['単勝オッズ'] < odds_max)
            ]
            
            count, tp_rate, avg_odds, ev = calculate_expected_value(subset, use_actual_odds)
            
            if count >= 5:
                results.append({
                    '条件': f'Ranker上位{ranker_max} & {odds_min}-{odds_max}倍',
                    '候補数': count,
                    'TP率': tp_rate,
                    '平均オッズ': avg_odds,
                    '期待値': ev,
                    'ROI': (ev - 1) * 100 if ev > 0 else -100
                })
    
    return pd.DataFrame(results)


def analyze_by_surface_ranker_odds(
    candidates: pd.DataFrame,
    use_actual_odds: bool = False
) -> pd.DataFrame:
    """芝/ダート × Ranker上位 × オッズ帯の期待値分析"""
    results = []
    
    surfaces = ['芝', 'ダート']
    ranker_filters = [3, 5, 8]
    odds_bands = [(10, 30), (30, 60), (60, 100)]
    
    for surface in surfaces:
        for ranker_max in ranker_filters:
            for odds_min, odds_max in odds_bands:
                subset = candidates[
                    (candidates['芝ダ区分'] == surface) &
                    (candidates['予測順位'] <= ranker_max) & 
                    (candidates['単勝オッズ'] >= odds_min) & 
                    (candidates['単勝オッズ'] < odds_max)
                ]
                
                count, tp_rate, avg_odds, ev = calculate_expected_value(subset, use_actual_odds)
                
                if count >= 5:
                    results.append({
                        '条件': f'{surface} & Ranker上位{ranker_max} & {odds_min}-{odds_max}倍',
                        '候補数': count,
                        'TP率': tp_rate,
                        '平均オッズ': avg_odds,
                        '期待値': ev,
                        'ROI': (ev - 1) * 100 if ev > 0 else -100
                    })
    
    return pd.DataFrame(results)


def analyze_by_popularity_ranker(
    candidates: pd.DataFrame,
    use_actual_odds: bool = False
) -> pd.DataFrame:
    """人気順 × Ranker上位の期待値分析"""
    results = []
    
    pop_ranges = [(7, 8), (9, 10), (11, 12)]
    ranker_filters = [3, 5, 8]
    
    for pop_min, pop_max in pop_ranges:
        for ranker_max in ranker_filters:
            subset = candidates[
                (candidates['人気順'] >= pop_min) & 
                (candidates['人気順'] <= pop_max) &
                (candidates['予測順位'] <= ranker_max)
            ]
            
            count, tp_rate, avg_odds, ev = calculate_expected_value(subset, use_actual_odds)
            
            if count >= 5:
                results.append({
                    '条件': f'{pop_min}-{pop_max}番人気 & Ranker上位{ranker_max}',
                    '候補数': count,
                    'TP率': tp_rate,
                    '平均オッズ': avg_odds,
                    '期待値': ev,
                    'ROI': (ev - 1) * 100 if ev > 0 else -100
                })
    
    return pd.DataFrame(results)


def print_analysis_table(df: pd.DataFrame, title: str):
    """分析結果テーブルを表示"""
    if len(df) == 0:
        print(f"\n{title}: データなし")
        return
    
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print('=' * 80)
    
    # 期待値でソート
    df_sorted = df.sort_values('期待値', ascending=False)
    
    print(f"{'条件':<40} {'候補数':>8} {'TP率':>8} {'平均ｵｯｽﾞ':>10} {'期待値':>8} {'ROI':>10}")
    print('-' * 86)
    
    for _, row in df_sorted.iterrows():
        ev_marker = '✅' if row['期待値'] >= 1.0 else '  '
        print(f"{row['条件']:<38} {row['候補数']:>8} {row['TP率']:>7.1%} "
              f"{row['平均オッズ']:>10.1f} {row['期待値']:>7.2f} {ev_marker} {row['ROI']:>+8.1f}%")


def find_profitable_conditions(all_results: list) -> pd.DataFrame:
    """期待値 > 1.0 の条件を抽出"""
    profitable = []
    
    for df in all_results:
        if len(df) > 0:
            for _, row in df.iterrows():
                if row['期待値'] >= 1.0:
                    profitable.append(row)
    
    if profitable:
        return pd.DataFrame(profitable).sort_values('期待値', ascending=False)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='穴馬候補の期待値分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
    python analyze_expected_value.py check_results/predicted_results_all.tsv
    python analyze_expected_value.py check_results/predicted_results_all.tsv --threshold 0.20
    python analyze_expected_value.py check_results/predicted_results_all.tsv --use-actual-odds
        """
    )
    parser.add_argument('input_file', help='予測結果ファイル（TSV）')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Upset閾値（デフォルト: 0.15）')
    parser.add_argument('--use-actual-odds', action='store_true',
                        help='実際の複勝オッズを使用（デフォルト: 推定値を使用）')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='最小サンプル数（デフォルト: 5）')
    args = parser.parse_args()
    
    print("=" * 80)
    print("期待値分析: 穴馬候補の条件別期待値")
    print("=" * 80)
    
    # データ読み込み
    df = load_results(args.input_file)
    print(f"\n読み込みデータ: {len(df)} 行")
    print(f"Upset閾値: {args.threshold}")
    print(f"複勝オッズ: {'実際の値' if args.use_actual_odds else '推定値（単勝×0.25）'}")
    
    # 穴馬候補を抽出
    candidates = df[
        (df['穴馬確率'] >= args.threshold) & 
        (df['人気順'] >= 7) & 
        (df['人気順'] <= 12)
    ].copy()
    
    candidates['is_hit'] = candidates['確定着順'] <= 3
    
    # 実際の複勝オッズを取得（use_actual_oddsの場合）
    if args.use_actual_odds:
        fukusho_odds_list = []
        for idx, row in candidates.iterrows():
            race_df = df[
                (df['競馬場'] == row['競馬場']) & 
                (df['開催年'] == row['開催年']) &
                (df['開催日'] == row['開催日']) &
                (df['レース番号'] == row['レース番号'])
            ]
            if len(race_df) > 0:
                odds = get_fukusho_odds(race_df.iloc[0], row['馬番'])
            else:
                odds = 0
            fukusho_odds_list.append(odds)
        candidates['fukusho_odds'] = fukusho_odds_list
    
    print(f"\n穴馬候補: {len(candidates)} 頭")
    print(f"  TP（的中）: {candidates['is_hit'].sum()} ({candidates['is_hit'].mean()*100:.1f}%)")
    
    # 全体の期待値
    _, tp_rate, avg_odds, ev = calculate_expected_value(candidates, args.use_actual_odds)
    print(f"\n【全体】")
    print(f"  TP率: {tp_rate:.1%}, 平均オッズ: {avg_odds:.1f}, 期待値: {ev:.2f}")
    
    # 各条件での分析
    all_results = []
    
    # 1. オッズ帯別
    df_odds = analyze_by_odds_band(candidates, args.use_actual_odds)
    all_results.append(df_odds)
    print_analysis_table(df_odds, "【1】オッズ帯別")
    
    # 2. Ranker上位 × オッズ帯
    df_ranker_odds = analyze_by_ranker_odds(candidates, args.use_actual_odds)
    all_results.append(df_ranker_odds)
    print_analysis_table(df_ranker_odds, "【2】Ranker上位 × オッズ帯")
    
    # 3. 芝/ダート × Ranker × オッズ帯
    df_surface = analyze_by_surface_ranker_odds(candidates, args.use_actual_odds)
    all_results.append(df_surface)
    print_analysis_table(df_surface, "【3】芝/ダート × Ranker上位 × オッズ帯")
    
    # 4. 人気順 × Ranker上位
    df_pop = analyze_by_popularity_ranker(candidates, args.use_actual_odds)
    all_results.append(df_pop)
    print_analysis_table(df_pop, "【4】人気順 × Ranker上位")
    
    # プラス期待値の条件まとめ
    df_profitable = find_profitable_conditions(all_results)
    
    print("\n" + "=" * 80)
    print("【まとめ】期待値 >= 1.0 の条件（プラス収支が見込める）")
    print("=" * 80)
    
    if len(df_profitable) > 0:
        print(f"\n{'条件':<40} {'候補数':>8} {'TP率':>8} {'期待値':>8} {'ROI':>10}")
        print('-' * 78)
        for _, row in df_profitable.iterrows():
            print(f"{row['条件']:<38} {row['候補数']:>8} {row['TP率']:>7.1%} "
                  f"{row['期待値']:>7.2f} ✅ {row['ROI']:>+8.1f}%")
        
        print(f"\n※ 期待値 > 1.0 = 理論上プラス収支")
        print(f"※ サンプル数が少ない条件は統計的に不安定な可能性あり")
    else:
        print("\n期待値 >= 1.0 の条件は見つかりませんでした。")
    
    # 推奨購入条件
    print("\n" + "=" * 80)
    print("【推奨】購入条件")
    print("=" * 80)
    
    if len(df_profitable) > 0:
        # サンプル数20以上かつ期待値1.0以上の条件
        reliable = df_profitable[df_profitable['候補数'] >= 20]
        if len(reliable) > 0:
            print("\n信頼性の高い条件（サンプル20件以上）:")
            for _, row in reliable.iterrows():
                print(f"  ✅ {row['条件']}: 期待値 {row['期待値']:.2f}, ROI {row['ROI']:+.1f}%")
        
        # サンプル数少ないが高期待値の条件
        high_ev = df_profitable[(df_profitable['候補数'] < 20) & (df_profitable['期待値'] >= 1.2)]
        if len(high_ev) > 0:
            print("\n要検証（高期待値だがサンプル少）:")
            for _, row in high_ev.iterrows():
                print(f"  ⚠️ {row['条件']}: 期待値 {row['期待値']:.2f} (N={row['候補数']})")


if __name__ == '__main__':
    main()
