#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
実際の複勝オッズに基づくROI分析スクリプト

予測結果ファイルから実際の複勝オッズを取得し、
条件別（競馬場、芝/ダート、Ranker順位、オッズ帯）のROIを計算する。

使用方法:
    python analyze_actual_roi.py <予測結果ファイル> [オプション]

例:
    python analyze_actual_roi.py check_results/predicted_results_all.tsv
    python analyze_actual_roi.py check_results/predicted_results_all.tsv --threshold 0.20
    python analyze_actual_roi.py check_results/predicted_results_all.tsv --min-samples 10
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def get_fukusho_odds(row: pd.Series) -> float:
    """
    的中した馬の実際の複勝オッズを取得する
    
    Args:
        row: 予測結果の1行（的中した馬のみ）
    
    Returns:
        複勝オッズ（的中していない場合は0）
    """
    uma_ban = row['馬番']
    for i, col in enumerate(['複勝1着馬番', '複勝2着馬番', '複勝3着馬番']):
        if col in row.index and row[col] == uma_ban:
            odds_col = f'複勝{i+1}着オッズ'
            if odds_col in row.index and pd.notna(row[odds_col]):
                return row[odds_col]
    return 0


def calculate_roi(candidates: pd.DataFrame, hits: pd.DataFrame) -> dict:
    """
    ROIを計算する
    
    Args:
        candidates: 候補馬のDataFrame
        hits: 的中馬のDataFrame（fukusho_oddsカラム付き）
    
    Returns:
        ROI情報の辞書
    """
    count = len(candidates)
    hit_count = len(hits)
    
    if count == 0:
        return {
            'count': 0,
            'hits': 0,
            'tp_rate': 0,
            'invest': 0,
            'returns': 0,
            'roi': 0
        }
    
    tp_rate = hit_count / count
    invest = count * 100
    returns = hits['fukusho_odds'].sum() * 100 if hit_count > 0 else 0
    roi = (returns - invest) / invest * 100 if invest > 0 else 0
    
    return {
        'count': count,
        'hits': hit_count,
        'tp_rate': tp_rate,
        'invest': invest,
        'returns': returns,
        'roi': roi
    }


def analyze_by_condition(candidates: pd.DataFrame, hits: pd.DataFrame, 
                         condition_cols: list, condition_values: dict,
                         min_samples: int = 5) -> dict:
    """
    指定条件でフィルタリングしてROIを計算
    
    Args:
        candidates: 候補馬のDataFrame
        hits: 的中馬のDataFrame
        condition_cols: 条件カラム名のリスト
        condition_values: 条件値の辞書
        min_samples: 最小サンプル数
    
    Returns:
        ROI情報の辞書（サンプル不足の場合はNone）
    """
    mask_cand = pd.Series([True] * len(candidates), index=candidates.index)
    mask_hits = pd.Series([True] * len(hits), index=hits.index)
    
    for col, value in condition_values.items():
        if isinstance(value, tuple):
            # 範囲指定 (min, max)
            mask_cand &= (candidates[col] >= value[0]) & (candidates[col] < value[1])
            mask_hits &= (hits[col] >= value[0]) & (hits[col] < value[1])
        elif isinstance(value, list):
            # リスト指定
            mask_cand &= candidates[col].isin(value)
            mask_hits &= hits[col].isin(value)
        else:
            # 単一値
            mask_cand &= candidates[col] == value
            mask_hits &= hits[col] == value
    
    subset_cand = candidates[mask_cand]
    subset_hits = hits[mask_hits]
    
    if len(subset_cand) < min_samples:
        return None
    
    return calculate_roi(subset_cand, subset_hits)


def print_separator(char='=', length=90):
    print(char * length)


def print_header(title: str):
    print_separator()
    print(title)
    print_separator()


def main():
    parser = argparse.ArgumentParser(description='実際の複勝オッズに基づくROI分析')
    parser.add_argument('input_file', help='予測結果ファイル（TSV形式）')
    parser.add_argument('--threshold', type=float, default=0.15, help='穴馬確率の閾値（デフォルト: 0.15）')
    parser.add_argument('--min-pop', type=int, default=7, help='最小人気順（デフォルト: 7）')
    parser.add_argument('--max-pop', type=int, default=12, help='最大人気順（デフォルト: 12）')
    parser.add_argument('--min-samples', type=int, default=5, help='最小サンプル数（デフォルト: 5）')
    parser.add_argument('--output', type=str, default=None, help='結果出力ファイル（省略時は標準出力）')
    args = parser.parse_args()
    
    # データ読み込み
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {args.input_file}")
        return 1
    
    df = pd.read_csv(input_path, sep='\t')
    print(f"データ読み込み完了: {len(df)}件")
    
    # 候補馬の抽出
    candidates = df[
        (df['穴馬確率'] >= args.threshold) & 
        (df['人気順'] >= args.min_pop) & 
        (df['人気順'] <= args.max_pop)
    ].copy()
    candidates['is_hit'] = candidates['確定着順'] <= 3
    
    # 的中馬の抽出と複勝オッズ取得
    hits = candidates[candidates['is_hit'] == True].copy()
    hits['fukusho_odds'] = hits.apply(get_fukusho_odds, axis=1)
    
    # 競馬場リストを取得
    tracks = candidates['競馬場'].unique().tolist()
    
    # ========================================
    # 1. 全体統計
    # ========================================
    print_header(f'【全体統計】閾値={args.threshold}, {args.min_pop}-{args.max_pop}番人気')
    
    overall = calculate_roi(candidates, hits)
    print(f"全候補数: {overall['count']:,}件")
    print(f"的中数: {overall['hits']:,}件")
    print(f"全体TP率: {overall['tp_rate']:.1%}")
    print(f"投資額: {overall['invest']:,.0f}円")
    print(f"回収額: {overall['returns']:,.0f}円")
    print(f"全体ROI: {overall['roi']:+.1f}%")
    
    # 複勝オッズの統計
    if len(hits) > 0:
        print(f"\n複勝オッズ統計:")
        print(f"  平均: {hits['fukusho_odds'].mean():.2f}倍")
        print(f"  中央値: {hits['fukusho_odds'].median():.2f}倍")
        print(f"  最大: {hits['fukusho_odds'].max():.2f}倍")
        print(f"  最小: {hits['fukusho_odds'].min():.2f}倍")
    
    # ========================================
    # 2. 競馬場 × 芝/ダート 別ROI
    # ========================================
    print_header('【競馬場 × 芝/ダート 別ROI】')
    print(f"{'競馬場':<8} {'芝/ダート':<8} {'候補数':>8} {'的中':>6} {'TP率':>8} {'ROI':>10}")
    print('-' * 60)
    
    track_results = []
    for track in tracks:
        for surface in ['芝', 'ダート']:
            result = analyze_by_condition(
                candidates, hits, 
                ['競馬場', '芝ダ区分'],
                {'競馬場': track, '芝ダ区分': surface},
                min_samples=args.min_samples
            )
            if result:
                result['track'] = track
                result['surface'] = surface
                track_results.append(result)
                marker = '✅' if result['roi'] >= 0 else '❌'
                print(f"{track:<8} {surface:<8} {result['count']:>8} {result['hits']:>6} {result['tp_rate']:>7.1%} {result['roi']:>+9.1f}% {marker}")
    
    # ========================================
    # 3. 芝/ダート × Ranker × オッズ帯 別ROI
    # ========================================
    print_header('【芝/ダート × Ranker × オッズ帯 別ROI】')
    print(f"{'条件':<40} {'候補数':>8} {'的中':>6} {'TP率':>8} {'ROI':>10}")
    print('-' * 80)
    
    surface_results = []
    for surface in ['芝', 'ダート']:
        for ranker_max in [3, 5, 8]:
            for odds_min, odds_max in [(10, 30), (30, 60), (60, 100), (100, 500)]:
                result = analyze_by_condition(
                    candidates, hits,
                    ['芝ダ区分', '予測順位', '単勝オッズ'],
                    {
                        '芝ダ区分': surface,
                        '予測順位': (1, ranker_max + 1),  # <= ranker_max
                        '単勝オッズ': (odds_min, odds_max)
                    },
                    min_samples=args.min_samples
                )
                if result:
                    cond_name = f"{surface} & Ranker上位{ranker_max} & {odds_min}-{odds_max}倍"
                    result['condition'] = cond_name
                    result['surface'] = surface
                    result['ranker_max'] = ranker_max
                    result['odds_range'] = f"{odds_min}-{odds_max}"
                    surface_results.append(result)
    
    # ROIでソート
    for r in sorted(surface_results, key=lambda x: x['roi'], reverse=True):
        marker = '✅' if r['roi'] >= 0 else '❌'
        print(f"{r['condition']:<38} {r['count']:>8} {r['hits']:>6} {r['tp_rate']:>7.1%} {r['roi']:>+9.1f}% {marker}")
    
    # ========================================
    # 4. 競馬場 × 芝/ダート × Ranker × オッズ帯 詳細
    # ========================================
    print_header('【競馬場 × 芝/ダート × Ranker × オッズ帯 詳細ROI】')
    print(f"{'条件':<50} {'候補数':>6} {'的中':>4} {'TP率':>7} {'ROI':>10}")
    print('-' * 85)
    
    detailed_results = []
    for track in tracks:
        for surface in ['芝', 'ダート']:
            for ranker_max in [5, 8]:
                for odds_min, odds_max in [(10, 30), (60, 100)]:
                    result = analyze_by_condition(
                        candidates, hits,
                        ['競馬場', '芝ダ区分', '予測順位', '単勝オッズ'],
                        {
                            '競馬場': track,
                            '芝ダ区分': surface,
                            '予測順位': (1, ranker_max + 1),
                            '単勝オッズ': (odds_min, odds_max)
                        },
                        min_samples=3  # 詳細分析は少し緩く
                    )
                    if result:
                        cond_name = f"{track} {surface} Ranker上位{ranker_max} {odds_min}-{odds_max}倍"
                        result['condition'] = cond_name
                        result['track'] = track
                        result['surface'] = surface
                        result['ranker_max'] = ranker_max
                        result['odds_range'] = f"{odds_min}-{odds_max}"
                        detailed_results.append(result)
    
    for r in sorted(detailed_results, key=lambda x: x['roi'], reverse=True):
        marker = '✅' if r['roi'] >= 0 else '❌'
        print(f"{r['condition']:<48} {r['count']:>6} {r['hits']:>4} {r['tp_rate']:>6.1%} {r['roi']:>+9.1f}% {marker}")
    
    # ========================================
    # 5. プラスROI条件のサマリー
    # ========================================
    print_header('【プラスROI条件サマリー】')
    
    # 芝/ダート × Ranker × オッズ帯 でプラスの条件
    positive_surface = [r for r in surface_results if r['roi'] > 0]
    if positive_surface:
        print("\n■ 芝/ダート × Ranker × オッズ帯:")
        for r in sorted(positive_surface, key=lambda x: x['roi'], reverse=True):
            print(f"  ✅ {r['condition']}: {r['count']}件, TP率{r['tp_rate']:.1%}, ROI {r['roi']:+.1f}%")
    
    # 競馬場別でプラスの条件
    positive_detailed = [r for r in detailed_results if r['roi'] > 0]
    if positive_detailed:
        print("\n■ 競馬場別詳細:")
        for r in sorted(positive_detailed, key=lambda x: x['roi'], reverse=True):
            note = "⚠️少" if r['count'] < 10 else ""
            print(f"  ✅ {r['condition']}: {r['count']}件, TP率{r['tp_rate']:.1%}, ROI {r['roi']:+.1f}% {note}")
    
    # ========================================
    # 6. 戦略パターン分析
    # ========================================
    print_header('【戦略パターン分析】')
    
    # 芝の最適条件
    turf_positive = [r for r in surface_results if r['surface'] == '芝' and r['roi'] > 0]
    if turf_positive:
        best_turf = max(turf_positive, key=lambda x: x['roi'])
        print(f"\n■ 芝の最適条件:")
        print(f"  {best_turf['condition']}")
        print(f"  → {best_turf['count']}件, TP率{best_turf['tp_rate']:.1%}, ROI {best_turf['roi']:+.1f}%")
    
    # ダートの最適条件
    dirt_positive = [r for r in surface_results if r['surface'] == 'ダート' and r['roi'] > 0]
    if dirt_positive:
        best_dirt = max(dirt_positive, key=lambda x: x['roi'])
        print(f"\n■ ダートの最適条件:")
        print(f"  {best_dirt['condition']}")
        print(f"  → {best_dirt['count']}件, TP率{best_dirt['tp_rate']:.1%}, ROI {best_dirt['roi']:+.1f}%")
    
    # パターン比較
    if turf_positive and dirt_positive:
        print("\n■ 芝/ダート戦略比較:")
        print("  | 芝/ダート | 狙い目オッズ帯 | Ranker条件 | 戦略 |")
        print("  |-----------|---------------|-----------|------|")
        
        # 芝のパターン抽出
        turf_odds = set([r['odds_range'] for r in turf_positive])
        turf_ranker = min([r['ranker_max'] for r in turf_positive])
        print(f"  | 芝 | {', '.join(turf_odds)} | 上位{turf_ranker}以内 | 高TP率狙い |")
        
        # ダートのパターン抽出
        dirt_odds = set([r['odds_range'] for r in dirt_positive])
        dirt_ranker = min([r['ranker_max'] for r in dirt_positive])
        print(f"  | ダート | {', '.join(dirt_odds)} | 上位{dirt_ranker}以内 | 高配当狙い |")
    
    # ========================================
    # 7. 推奨購入フィルタ
    # ========================================
    print_header('【推奨購入フィルタ（コード例）】')
    
    if turf_positive or dirt_positive:
        print("""
def should_buy(row):
    \"\"\"プラスROI条件に基づく購入判断\"\"\"
    surface = row['芝ダ区分']
    odds = row['単勝オッズ']
    ranker = row['予測順位']
    """)
        
        if turf_positive:
            best = max(turf_positive, key=lambda x: x['count'])  # サンプル数が多いものを採用
            odds_parts = best['odds_range'].split('-')
            print(f"""
    if surface == '芝':
        # 芝: {best['odds_range']}倍 × Ranker上位{best['ranker_max']}
        return odds >= {odds_parts[0]} and odds < {odds_parts[1]} and ranker <= {best['ranker_max']}""")
        
        if dirt_positive:
            best = max(dirt_positive, key=lambda x: x['count'])
            odds_parts = best['odds_range'].split('-')
            print(f"""
    elif surface == 'ダート':
        # ダート: {best['odds_range']}倍 × Ranker上位{best['ranker_max']}
        return odds >= {odds_parts[0]} and odds < {odds_parts[1]} and ranker <= {best['ranker_max']}""")
        
        print("""
    return False
""")
    
    # ========================================
    # 8. フィルタ適用後の期待ROI
    # ========================================
    if positive_surface:
        print_header('【フィルタ適用後の期待効果】')
        
        total_count = sum([r['count'] for r in positive_surface])
        total_hits = sum([r['hits'] for r in positive_surface])
        total_invest = total_count * 100
        # 各条件の回収額を計算
        total_returns = sum([r['invest'] * (1 + r['roi']/100) for r in positive_surface])
        filtered_roi = (total_returns - total_invest) / total_invest * 100 if total_invest > 0 else 0
        
        print(f"フィルタ前: {overall['count']}件, ROI {overall['roi']:+.1f}%")
        print(f"フィルタ後: {total_count}件 ({total_count/overall['count']*100:.1f}%), 期待ROI {filtered_roi:+.1f}%")
        print(f"的中期待: {total_hits}件, TP率 {total_hits/total_count*100:.1f}%")
    
    print_separator()
    print("分析完了")
    
    return 0


if __name__ == '__main__':
    exit(main())
