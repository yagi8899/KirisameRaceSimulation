#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
穴馬予測閾値最適化

predicted_results_all.tsv を読み込み、
複数の閾値でPrecision/Recall/ROIを計算して最適閾値を探索

使用方法:
    # デフォルト（2023-2025年、閾値0.10-0.35）
    python optimize_upset_threshold.py check_results/predicted_results_all.tsv
    
    # 年度範囲指定
    python optimize_upset_threshold.py check_results/predicted_results_all.tsv --year-start 2023 --year-end 2025
    
    # 閾値範囲指定
    python optimize_upset_threshold.py check_results/predicted_results_all.tsv --threshold-start 0.10 --threshold-end 0.40 --threshold-step 0.05
    
    # ファイル出力
    python optimize_upset_threshold.py check_results/predicted_results_all.tsv --output check_results/threshold_optimization.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime
from contextlib import redirect_stdout


def load_data(file_path: str) -> pd.DataFrame:
    """TSVファイルを読み込む"""
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
    return df


def calculate_roi_for_threshold(df: pd.DataFrame, threshold: float, year: int = None) -> dict:
    """
    指定した閾値でROI等を計算
    
    Args:
        df: 予測結果DataFrame
        threshold: 穴馬確率の閾値
        year: 対象年（Noneなら全年）
    
    Returns:
        dict: 計算結果
    """
    # 年でフィルタ
    if year is not None and '開催年' in df.columns:
        df = df[df['開催年'] == year].copy()
    
    if len(df) == 0:
        return {
            'candidates': 0,
            'hits': 0,
            'precision': 0.0,
            'recall': 0.0,
            'investment': 0,
            'return': 0,
            'roi': 0.0,
            'avg_odds': 0.0
        }
    
    # 必要な列の確認
    required_cols = ['穴馬確率', '人気順', '確定着順']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ 列 '{col}' がありません")
            return None
    
    # 7-12番人気でフィルタ
    df_filtered = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    
    # 閾値を適用して穴馬候補を選定
    candidates = df_filtered[df_filtered['穴馬確率'] >= threshold].copy()
    
    # 穴馬の正解（7-12番人気で3着以内）
    actual_upset = df_filtered[df_filtered['確定着順'] <= 3]
    
    # 候補数
    n_candidates = len(candidates)
    
    # 的中数（候補のうち3着以内）
    n_hits = len(candidates[candidates['確定着順'] <= 3])
    
    # Precision（適合率）
    precision = (n_hits / n_candidates * 100) if n_candidates > 0 else 0.0
    
    # Recall（再現率）
    n_actual = len(actual_upset)
    recall = (n_hits / n_actual * 100) if n_actual > 0 else 0.0
    
    # ROI計算（複勝オッズを使用）
    investment = n_candidates * 100  # 100円/点
    
    # 的中馬の払戻を計算
    # ヘッダ構造: 複勝1着馬番, 複勝1着オッズ, 複勝2着馬番, 複勝2着オッズ, 複勝3着馬番, 複勝3着オッズ
    hits_df = candidates[candidates['確定着順'] <= 3].copy()
    
    if len(hits_df) > 0 and '馬番' in hits_df.columns:
        total_return = 0
        odds_list = []
        
        for _, row in hits_df.iterrows():
            uma_ban = row['馬番']
            payout = 0
            
            # 複勝1着、2着、3着の馬番と照合してオッズを取得
            for i in [1, 2, 3]:
                col_ban = f'複勝{i}着馬番'
                col_odds = f'複勝{i}着オッズ'
                
                if col_ban in row.index and col_odds in row.index:
                    if pd.notna(row[col_ban]) and row[col_ban] == uma_ban:
                        if pd.notna(row[col_odds]):
                            payout = row[col_odds] * 100  # オッズ × 100円
                            odds_list.append(row[col_odds])
                        break
            
            total_return += payout
        
        roi = ((total_return - investment) / investment * 100) if investment > 0 else 0.0
        avg_odds = sum(odds_list) / len(odds_list) if odds_list else 0.0
    else:
        total_return = 0
        roi = 0.0
        avg_odds = 0.0
    
    return {
        'candidates': n_candidates,
        'hits': n_hits,
        'precision': precision,
        'recall': recall,
        'investment': investment,
        'return': total_return,
        'roi': roi,
        'avg_odds': avg_odds,
        'total_upset': n_actual
    }


def optimize_thresholds(df: pd.DataFrame, years: list, thresholds: list) -> list:
    """
    複数年・複数閾値で最適化
    
    Args:
        df: 予測結果DataFrame
        years: 対象年のリスト
        thresholds: テスト閾値のリスト
    
    Returns:
        list: 各閾値の結果
    """
    print("=" * 80)
    print("🎯 穴馬予測 閾値最適化")
    print("=" * 80)
    print(f"\n対象年: {years}")
    print(f"テスト閾値: {[f'{t:.2f}' for t in thresholds]}")
    print(f"データ件数: {len(df):,}件")
    
    results = []
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"閾値 {threshold:.2f} でテスト中...")
        print(f"{'='*80}")
        
        threshold_results = {
            'threshold': threshold,
            'total_candidates': 0,
            'total_hits': 0,
            'total_investment': 0,
            'total_return': 0,
            'total_upset': 0,
            'yearly_results': []
        }
        
        for year in years:
            print(f"\n  {year}年...")
            metrics = calculate_roi_for_threshold(df, threshold, year)
            
            if metrics:
                threshold_results['total_candidates'] += metrics['candidates']
                threshold_results['total_hits'] += metrics['hits']
                threshold_results['total_investment'] += metrics['investment']
                threshold_results['total_return'] += metrics['return']
                threshold_results['total_upset'] += metrics.get('total_upset', 0)
                threshold_results['yearly_results'].append({
                    'year': year,
                    **metrics
                })
                
                print(f"    候補: {metrics['candidates']:,}頭")
                print(f"    的中: {metrics['hits']:,}頭")
                print(f"    Precision: {metrics['precision']:.2f}%")
                print(f"    Recall: {metrics['recall']:.2f}%")
                print(f"    ROI: {metrics['roi']:.1f}%")
        
        # 全体集計
        total_candidates = threshold_results['total_candidates']
        total_hits = threshold_results['total_hits']
        total_investment = threshold_results['total_investment']
        total_return = threshold_results['total_return']
        total_upset = threshold_results['total_upset']
        
        avg_candidates = total_candidates / len(years) if years else 0
        overall_precision = (total_hits / total_candidates * 100) if total_candidates > 0 else 0
        overall_recall = (total_hits / total_upset * 100) if total_upset > 0 else 0
        overall_roi = ((total_return - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        threshold_results['avg_candidates_per_year'] = avg_candidates
        threshold_results['overall_precision'] = overall_precision
        threshold_results['overall_recall'] = overall_recall
        threshold_results['overall_roi'] = overall_roi
        
        print(f"\n  【閾値 {threshold:.2f} 集計】")
        print(f"    平均候補数/年: {avg_candidates:.1f}頭")
        print(f"    全体Precision: {overall_precision:.2f}%")
        print(f"    全体Recall: {overall_recall:.2f}%")
        print(f"    全体ROI: {overall_roi:.1f}%")
        
        results.append(threshold_results)
    
    return results


def display_summary(results: list):
    """結果サマリーを表示"""
    print("\n" + "=" * 80)
    print("📊 閾値最適化結果サマリー")
    print("=" * 80)
    
    # テーブル形式で表示
    print(f"\n{'閾値':>6} {'候補数/年':>10} {'Precision':>10} {'Recall':>10} {'ROI':>10} {'総的中数':>8}")
    print("-" * 60)
    
    for r in results:
        threshold = r['threshold']
        avg_candidates = r['avg_candidates_per_year']
        precision = r['overall_precision']
        recall = r['overall_recall']
        roi = r['overall_roi']
        total_hits = r['total_hits']
        
        roi_str = f"{roi:.1f}%" if not np.isnan(roi) else "N/A"
        
        print(f"{threshold:>6.2f} {avg_candidates:>10.1f} {precision:>9.2f}% {recall:>9.2f}% {roi_str:>10} {total_hits:>8}")
    
    # 推奨閾値を提案
    print("\n" + "=" * 80)
    print("💡 推奨閾値")
    print("=" * 80)
    
    # 有効な結果のみ（候補数 > 0）
    valid_results = [r for r in results if r['total_candidates'] > 0]
    
    if not valid_results:
        print("⚠️ すべての閾値で候補数が0でした。閾値を下げてください。")
        return
    
    # 条件1: 候補数が適度（年間100-500頭程度）
    candidates_ok = [r for r in valid_results if 100 <= r['avg_candidates_per_year'] * len(r['yearly_results']) <= 2000]
    
    if candidates_ok:
        # 条件2: Precisionが10%以上
        precision_ok = [r for r in candidates_ok if r['overall_precision'] >= 10]
        
        if precision_ok:
            # 条件3: ROIが最大
            best = max(precision_ok, key=lambda x: x['overall_roi'])
            print(f"✅ 閾値 {best['threshold']:.2f} を推奨")
        else:
            # Precision条件を満たすものがない場合、ROI最大を選択
            best = max(candidates_ok, key=lambda x: x['overall_roi'])
            print(f"⚠️ Precision 10%以上の閾値がありません")
            print(f"   ROI最大の閾値 {best['threshold']:.2f} を推奨")
    else:
        # 候補数条件を満たすものがない場合
        best = max(valid_results, key=lambda x: x['overall_roi'])
        print(f"⚠️ 適切な候補数範囲の閾値がありません")
        print(f"   ROI最大の閾値 {best['threshold']:.2f} を推奨")
    
    print(f"\n   平均候補数: {best['avg_candidates_per_year']:.1f}頭/年")
    print(f"   総候補数: {best['total_candidates']:,}頭")
    print(f"   総的中数: {best['total_hits']:,}頭")
    print(f"   Precision: {best['overall_precision']:.2f}%")
    print(f"   Recall: {best['overall_recall']:.2f}%")
    print(f"   ROI: {best['overall_roi']:.1f}%")
    
    # 年別詳細
    print("\n  【年別詳細】")
    for yr in best['yearly_results']:
        print(f"    {yr['year']}年: 候補{yr['candidates']:,}頭, 的中{yr['hits']:,}頭, "
              f"Prec {yr['precision']:.1f}%, ROI {yr['roi']:.1f}%")


def save_detailed_results(results: list, output_path: Path):
    """詳細結果をTSVファイルに保存"""
    detailed_results = []
    for r in results:
        for yearly in r['yearly_results']:
            detailed_results.append({
                '閾値': r['threshold'],
                '年': yearly['year'],
                '候補数': yearly.get('candidates', 0),
                '的中数': yearly.get('hits', 0),
                'Precision': yearly.get('precision', 0),
                'Recall': yearly.get('recall', 0),
                'ROI': yearly.get('roi', 0),
                '投資額': yearly.get('investment', 0),
                '回収額': yearly.get('return', 0)
            })
    
    df_detailed = pd.DataFrame(detailed_results)
    tsv_path = output_path.with_suffix('.tsv')
    df_detailed.to_csv(tsv_path, sep='\t', index=False, encoding='utf-8')
    print(f"\n📁 詳細結果を {tsv_path} に保存しました")


def main():
    parser = argparse.ArgumentParser(
        description='穴馬予測の閾値最適化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # デフォルト
  python optimize_upset_threshold.py check_results/predicted_results_all.tsv
  
  # 年度範囲指定
  python optimize_upset_threshold.py check_results/predicted_results_all.tsv --year-start 2023 --year-end 2025
  
  # 閾値範囲指定
  python optimize_upset_threshold.py check_results/predicted_results_all.tsv --threshold-start 0.10 --threshold-end 0.40
  
  # ファイル出力
  python optimize_upset_threshold.py check_results/predicted_results_all.tsv -o check_results/threshold_opt.txt
        """
    )
    
    parser.add_argument('file', nargs='?', default='check_results/predicted_results_all.tsv',
                        help='分析対象のTSVファイルパス')
    parser.add_argument('--year-start', type=int, default=2023,
                        help='対象開始年（デフォルト: 2023）')
    parser.add_argument('--year-end', type=int, default=2025,
                        help='対象終了年（デフォルト: 2025）')
    parser.add_argument('--threshold-start', type=float, default=0.10,
                        help='閾値開始値（デフォルト: 0.10）')
    parser.add_argument('--threshold-end', type=float, default=0.35,
                        help='閾値終了値（デフォルト: 0.35）')
    parser.add_argument('--threshold-step', type=float, default=0.05,
                        help='閾値ステップ（デフォルト: 0.05）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='出力ファイルパス（省略時: コンソール出力）')
    
    args = parser.parse_args()
    
    # ファイル存在確認
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ ファイルが見つかりません: {file_path}")
        sys.exit(1)
    
    # 年度リスト作成
    years = list(range(args.year_start, args.year_end + 1))
    
    # 閾値リスト作成
    thresholds = []
    t = args.threshold_start
    while t <= args.threshold_end + 0.001:  # 浮動小数点誤差対策
        thresholds.append(round(t, 2))
        t += args.threshold_step
    
    def run_optimization():
        # データ読み込み
        print(f"📂 ファイル読み込み: {file_path}")
        df = load_data(str(file_path))
        print(f"✅ {len(df):,}件読み込み完了")
        
        # 最適化実行
        results = optimize_thresholds(df, years, thresholds)
        
        # サマリー表示
        display_summary(results)
        
        print("\n" + "=" * 80)
        print("✅ 最適化完了!")
        print("=" * 80)
        
        return results
    
    # ファイル出力の場合
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print(f"# 閾値最適化結果")
                print(f"# 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"# 入力ファイル: {file_path}")
                print()
                results = run_optimization()
        
        # TSV形式でも保存
        save_detailed_results(results, output_path)
        
        # コンソールにも完了メッセージ
        print(f"✅ 結果を {output_path} に保存しました")
    else:
        results = run_optimization()
        
        # TSV形式でも保存（デフォルト出力先）
        default_output = Path('check_results/threshold_optimization_summary.tsv')
        default_output.parent.mkdir(parents=True, exist_ok=True)
        save_detailed_results(results, default_output)


if __name__ == '__main__':
    main()
