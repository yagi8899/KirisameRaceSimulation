"""
相対ランクベースの穴馬候補選定と評価

確率の絶対値ではなく、レース内の相対順位で候補を選定する方式。
これにより確率校正の問題を回避できる。

使い方:
  # デフォルト（各レース上位2頭）
  python evaluate_rank_based_selection.py check_results/predicted_results_all.tsv
  
  # 上位N頭を指定
  python evaluate_rank_based_selection.py check_results/predicted_results_all.tsv --top-n 3
  
  # 上位X%を指定
  python evaluate_rank_based_selection.py check_results/predicted_results_all.tsv --top-pct 30
  
  # 競馬場別に分析
  python evaluate_rank_based_selection.py check_results/predicted_results_all.tsv --by-track
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_race_id(df):
    """レースIDを作成"""
    df = df.copy()
    df['race_id'] = df['競馬場'].astype(str) + '_' + \
                    df['開催年'].astype(str) + '_' + \
                    df['開催日'].astype(str) + '_' + \
                    df['レース番号'].astype(str)
    return df


def evaluate_rank_based(df, top_n=None, top_pct=None, label="全体"):
    """
    相対ランクベースの評価
    
    Args:
        df: 7-12番人気のデータ
        top_n: 各レースで上位N頭を候補とする
        top_pct: 各レースで上位X%を候補とする
        label: 出力ラベル
    """
    df = df.copy()
    df = create_race_id(df)
    df['is_upset'] = (df['確定着順'] <= 3).astype(int)
    
    # レース内での穴馬確率順位
    df['rank_in_race'] = df.groupby('race_id')['穴馬確率'].rank(ascending=False, method='first')
    
    # 各レースの7-12番人気頭数
    race_counts = df.groupby('race_id').size().reset_index(name='n_horses')
    df = df.merge(race_counts, on='race_id')
    
    print(f"\n{'='*70}")
    print(f"[{label}] 相対ランクベース評価")
    print(f"{'='*70}")
    print(f"総レコード: {len(df)}頭")
    print(f"レース数: {df['race_id'].nunique()}")
    print(f"実際の穴馬: {df['is_upset'].sum()}頭 ({df['is_upset'].mean()*100:.2f}%)")
    
    results = []
    
    # 評価するパターン
    if top_n is not None:
        patterns = [('top_n', top_n, None)]
    elif top_pct is not None:
        patterns = [('top_pct', None, top_pct)]
    else:
        # デフォルト: 複数パターンを評価
        patterns = [
            ('top_n', 1, None),
            ('top_n', 2, None),
            ('top_n', 3, None),
            ('top_pct', None, 20),
            ('top_pct', None, 30),
            ('top_pct', None, 50),
        ]
    
    print(f"\n{'方式':<15} {'候補数':>8} {'TP':>6} {'Precision':>10} {'Recall':>8} {'カバー率':>10}")
    print("-" * 65)
    
    for pattern_type, n, pct in patterns:
        if pattern_type == 'top_n':
            df['is_candidate'] = df['rank_in_race'] <= n
            desc = f"上位{n}頭/レース"
        else:
            # 各レースで上位X%
            df['pct_rank'] = df['rank_in_race'] / df['n_horses'] * 100
            df['is_candidate'] = df['pct_rank'] <= pct
            desc = f"上位{pct}%/レース"
        
        candidates = df[df['is_candidate']]
        tp = candidates['is_upset'].sum()
        fp = len(candidates) - tp
        fn = df['is_upset'].sum() - tp
        
        precision = tp / len(candidates) * 100 if len(candidates) > 0 else 0
        recall = tp / df['is_upset'].sum() * 100 if df['is_upset'].sum() > 0 else 0
        
        # カバー率: 候補がいるレースの割合
        races_with_candidates = candidates['race_id'].nunique()
        total_races = df['race_id'].nunique()
        coverage = races_with_candidates / total_races * 100
        
        print(f"{desc:<15} {len(candidates):>8} {tp:>6} {precision:>9.2f}% {recall:>7.2f}% {coverage:>9.2f}%")
        
        results.append({
            'pattern': desc,
            'candidates': len(candidates),
            'tp': tp,
            'precision': precision,
            'recall': recall,
            'coverage': coverage
        })
    
    return pd.DataFrame(results)


def compare_with_threshold(df, label="全体"):
    """閾値方式と相対ランク方式を比較"""
    df = df.copy()
    df = create_race_id(df)
    df['is_upset'] = (df['確定着順'] <= 3).astype(int)
    df['rank_in_race'] = df.groupby('race_id')['穴馬確率'].rank(ascending=False, method='first')
    
    print(f"\n{'='*70}")
    print(f"[{label}] 閾値方式 vs 相対ランク方式 比較")
    print(f"{'='*70}")
    
    print(f"\n{'方式':<20} {'候補数':>8} {'TP':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    
    # 閾値方式
    for threshold in [0.10, 0.15, 0.20, 0.30]:
        df['is_candidate'] = df['穴馬確率'] >= threshold
        candidates = df[df['is_candidate']]
        tp = candidates['is_upset'].sum()
        precision = tp / len(candidates) * 100 if len(candidates) > 0 else 0
        recall = tp / df['is_upset'].sum() * 100
        print(f"閾値 >= {threshold:.2f}      {len(candidates):>8} {tp:>6} {precision:>9.2f}% {recall:>7.2f}%")
    
    print("-" * 60)
    
    # 相対ランク方式
    for top_n in [1, 2, 3]:
        df['is_candidate'] = df['rank_in_race'] <= top_n
        candidates = df[df['is_candidate']]
        tp = candidates['is_upset'].sum()
        precision = tp / len(candidates) * 100 if len(candidates) > 0 else 0
        recall = tp / df['is_upset'].sum() * 100
        print(f"上位{top_n}頭/レース      {len(candidates):>8} {tp:>6} {precision:>9.2f}% {recall:>7.2f}%")


def calculate_roi_rank_based(df, label="全体"):
    """相対ランク方式でのROI計算"""
    df = df.copy()
    df = create_race_id(df)
    df['is_upset'] = (df['確定着順'] <= 3).astype(int)
    df['rank_in_race'] = df.groupby('race_id')['穴馬確率'].rank(ascending=False, method='first')
    
    # 複勝配当を取得
    def get_fukusho_odds(row):
        if row['確定着順'] > 3:
            return 0
        umaban = row['馬番']
        for i in [1, 2, 3]:
            col_umaban = f'複勝{i}着馬番'
            col_odds = f'複勝{i}着オッズ'
            if col_umaban in row.index and col_odds in row.index:
                try:
                    if float(row[col_umaban]) == float(umaban):
                        odds = row[col_odds]
                        if pd.notna(odds):
                            return float(odds)
                except:
                    pass
        return 0
    
    df['fukusho_odds'] = df.apply(get_fukusho_odds, axis=1)
    
    print(f"\n{'='*70}")
    print(f"[{label}] 相対ランク方式でのROI分析")
    print(f"{'='*70}")
    
    print(f"\n{'方式':<15} {'候補':>6} {'的中':>6} {'投資額':>12} {'払戻額':>12} {'収支':>12} {'ROI':>8} {'Prec':>8}")
    print("-" * 90)
    
    for top_n in [1, 2, 3, 4, 5]:
        df['is_candidate'] = df['rank_in_race'] <= top_n
        candidates = df[df['is_candidate']]
        
        hits = candidates[candidates['is_upset'] == 1]
        investment = len(candidates) * 100
        payout = hits['fukusho_odds'].sum() * 100
        profit = payout - investment
        roi = (profit / investment) * 100 if investment > 0 else 0
        precision = len(hits) / len(candidates) * 100 if len(candidates) > 0 else 0
        
        print(f"上位{top_n}頭/レース  {len(candidates):>6} {len(hits):>6} {investment:>11,} {payout:>11,.0f} {profit:>+11,.0f} {roi:>7.1f}% {precision:>7.2f}%")


def main():
    parser = argparse.ArgumentParser(description='相対ランクベースの穴馬候補評価')
    parser.add_argument('file_path', nargs='?', default='check_results/predicted_results_all.tsv')
    parser.add_argument('--top-n', type=int, help='各レースで上位N頭を候補とする')
    parser.add_argument('--top-pct', type=float, help='各レースで上位X%%を候補とする')
    parser.add_argument('--by-track', action='store_true', help='競馬場別に分析')
    
    args = parser.parse_args()
    
    # データ読み込み
    df = pd.read_csv(args.file_path, sep='\t')
    print(f"[FILE] {args.file_path}")
    print(f"[DATA] 総レコード: {len(df)}")
    
    # 7-12番人気のみ
    df_target = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    print(f"[FILTER] 7-12番人気: {len(df_target)}")
    
    # 全体評価
    evaluate_rank_based(df_target, args.top_n, args.top_pct, "全体")
    compare_with_threshold(df_target, "全体")
    calculate_roi_rank_based(df_target, "全体")
    
    # 競馬場別
    if args.by_track:
        for track in df_target['競馬場'].unique():
            df_track = df_target[df_target['競馬場'] == track]
            if len(df_track) >= 100:
                evaluate_rank_based(df_track, args.top_n, args.top_pct, track)
                calculate_roi_rank_based(df_track, track)
    
    print("\n" + "=" * 70)
    print("[結論]")
    print("=" * 70)
    print("""
相対ランク方式のメリット:
1. 確率の絶対値に依存しない → 校正不要
2. 各レースで一定数の候補が出る → 安定性
3. 「候補なし」レースがない → カバー率100%

推奨設定:
- 保守的: 上位1頭/レース（Precision重視）
- バランス: 上位2頭/レース（推奨）
- 積極的: 上位3頭/レース（Recall重視）
""")


if __name__ == '__main__':
    main()
