"""
穴馬検出モデルの閾値最適化スクリプト
Precision 8%以上を達成する最適閾値を見つける

使い方:
  # デフォルト（check_results/predicted_results_all.tsv を分析）
  python analyze_upset_threshold.py
  
  # ファイル指定
  python analyze_upset_threshold.py path/to/file.tsv
  
  # 競馬場別に分析
  python analyze_upset_threshold.py --by-track
  
  # 年度別に分析
  python analyze_upset_threshold.py --by-year
  
  # 特定の競馬場のみ
  python analyze_upset_threshold.py --track 函館
  
  # 特定の年度のみ
  python analyze_upset_threshold.py --year 2024
  
  # 組み合わせ
  python analyze_upset_threshold.py path/to/file.tsv --by-track --by-year
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_single_dataset(df_target: pd.DataFrame, label: str = "全体", output_prefix: str = "upset_threshold_optimization"):
    """
    単一データセットの閾値最適化分析
    
    Args:
        df_target: 7-12番人気のデータ
        label: 出力ラベル（例: "函館", "京都"）
        output_prefix: グラフファイル名のプレフィックス
    
    Returns:
        results_df: 閾値別の結果DataFrame
    """
    print(f"\n{'=' * 60}")
    print(f"[ANALYZE] {label}")
    print(f"{'=' * 60}")
    
    # 実際の穴馬（7-12番人気で3着以内）
    df_target = df_target.copy()
    df_target['is_upset'] = (df_target['確定着順'] <= 3).astype(int)
    
    total_records = len(df_target)
    total_upsets = df_target['is_upset'].sum()
    upset_rate = total_upsets / total_records * 100 if total_records > 0 else 0
    
    print(f"[DATA] 7-12番人気: {total_records}頭")
    print(f"[DATA] 実際の穴馬数: {total_upsets}頭")
    print(f"[DATA] 穴馬率: {upset_rate:.2f}%")
    
    if total_records == 0:
        print(f"[WARN] データがありません")
        return None
    
    # 穴馬確率の分布
    probs = df_target['穴馬確率'].dropna()
    if len(probs) == 0:
        print(f"[WARN] 穴馬確率データがありません")
        return None
    
    print(f"\n[STATS] 穴馬確率の基本統計")
    print(f"  - 平均: {probs.mean():.4f}")
    print(f"  - 中央値: {probs.median():.4f}")
    print(f"  - 標準偏差: {probs.std():.4f}")
    print(f"  - 最小値: {probs.min():.4f}")
    print(f"  - 最大値: {probs.max():.4f}")
    
    # パーセンタイル
    print(f"\n[PERCENTILE] 穴馬確率の分位点")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = probs.quantile(p/100)
        print(f"  - {p:2d}%点: {val:.4f}")
    
    # 各閾値でPrecision/Recall/F1を計算
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    print(f"\n[SIMULATE] 各閾値でのPrecision/Recall/F1")
    print(f"{'閾値':>8s} {'候補数':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} {'判定':>12s}")
    print("-" * 90)
    
    for threshold in thresholds:
        # 閾値以上を穴馬候補として予測
        df_target['predicted'] = (df_target['穴馬確率'] >= threshold).astype(int)
        
        # TP, FP, FN, TNを計算
        tp = ((df_target['predicted'] == 1) & (df_target['is_upset'] == 1)).sum()
        fp = ((df_target['predicted'] == 1) & (df_target['is_upset'] == 0)).sum()
        fn = ((df_target['predicted'] == 0) & (df_target['is_upset'] == 1)).sum()
        tn = ((df_target['predicted'] == 0) & (df_target['is_upset'] == 0)).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        candidates = tp + fp
        
        # 判定
        if precision >= 8.0:
            judgment = "✅ 目標達成"
        elif precision >= 6.0:
            judgment = "⭕ 良好"
        else:
            judgment = "⚠️  未達成"
        
        print(f"{threshold:8.2f} {candidates:8d} {tp:6d} {fp:6d} {fn:6d} {precision:9.2f}% {recall:9.2f}% {f1:8.2f} {judgment}")
        
        results.append({
            'threshold': threshold,
            'candidates': candidates,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # グラフ作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'穴馬検出閾値の最適化分析: {label}', fontsize=16, fontweight='bold')
    
    # 1. Precision/Recall曲線
    ax1.plot(results_df['threshold'], results_df['precision'], marker='o', linewidth=2, markersize=6, label='Precision', color='blue')
    ax1.plot(results_df['threshold'], results_df['recall'], marker='s', linewidth=2, markersize=6, label='Recall', color='green')
    ax1.axhline(y=8.0, color='red', linestyle='--', linewidth=2, label='目標Precision: 8%')
    ax1.axhline(y=70.0, color='orange', linestyle='--', linewidth=1, label='理想Recall: 70%')
    ax1.set_xlabel('閾値', fontsize=12)
    ax1.set_ylabel('% (Precision/Recall)', fontsize=12)
    ax1.set_title('閾値とPrecision/Recallの関係', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. F1スコア
    ax2.plot(results_df['threshold'], results_df['f1'], marker='^', linewidth=2, markersize=6, color='purple')
    ax2.set_xlabel('閾値', fontsize=12)
    ax2.set_ylabel('F1スコア', fontsize=12)
    ax2.set_title('閾値とF1スコアの関係', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 候補数
    ax3.plot(results_df['threshold'], results_df['candidates'], marker='D', linewidth=2, markersize=6, color='orange')
    ax3.set_xlabel('閾値', fontsize=12)
    ax3.set_ylabel('候補数（頭）', fontsize=12)
    ax3.set_title('閾値と候補数の関係', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. TP/FP/FN
    ax4.plot(results_df['threshold'], results_df['tp'], marker='o', linewidth=2, markersize=6, label='TP (True Positive)', color='green')
    ax4.plot(results_df['threshold'], results_df['fp'], marker='s', linewidth=2, markersize=6, label='FP (False Positive)', color='red')
    ax4.plot(results_df['threshold'], results_df['fn'], marker='^', linewidth=2, markersize=6, label='FN (False Negative)', color='blue')
    ax4.set_xlabel('閾値', fontsize=12)
    ax4.set_ylabel('件数（頭）', fontsize=12)
    ax4.set_title('閾値とTP/FP/FNの関係', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    output_dir = Path('check_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{output_prefix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[FILE] 最適化グラフを保存: {output_file}")
    plt.close()
    
    # 推奨閾値を提案
    print_recommendations(results_df, label)
    
    return results_df


def print_recommendations(results_df: pd.DataFrame, label: str = "全体"):
    """推奨閾値を表示"""
    print(f"\n{'=' * 60}")
    print(f"[RECOMMEND] {label} - 推奨閾値")
    print(f"{'=' * 60}")
    
    # Precision 8%以上 かつ Recall 50-80%の範囲で最もバランスの良い閾値
    balanced_results = results_df[
        (results_df['precision'] >= 8.0) & 
        (results_df['recall'] >= 50.0) & 
        (results_df['recall'] <= 80.0)
    ]
    
    if len(balanced_results) > 0:
        # F1スコア最大
        best_idx = balanced_results['f1'].idxmax()
        best = balanced_results.loc[best_idx]
        
        print(f"\n✅ 推奨閾値（バランス重視）: {best['threshold']:.2f}")
        print(f"   - Precision: {best['precision']:.2f}%")
        print(f"   - Recall: {best['recall']:.2f}%")
        print(f"   - F1スコア: {best['f1']:.2f}")
        print(f"   - 候補数: {int(best['candidates'])}頭")
    
    # Precision 8%以上で最もRecallが高い閾値
    good_results = results_df[results_df['precision'] >= 8.0]
    
    if len(good_results) > 0:
        best_recall_idx = good_results['recall'].idxmax()
        best = good_results.loc[best_recall_idx]
        
        print(f"\n📊 Recall重視（Precision 8%以上で最大Recall）: {best['threshold']:.2f}")
        print(f"   - Precision: {best['precision']:.2f}%")
        print(f"   - Recall: {best['recall']:.2f}%")
        print(f"   - F1スコア: {best['f1']:.2f}")
        print(f"   - 候補数: {int(best['candidates'])}頭")
    else:
        # Precision 8%未達成の場合
        best_precision_idx = results_df['precision'].idxmax()
        best = results_df.loc[best_precision_idx]
        
        print(f"\n⚠️  Precision 8%を達成できる閾値が見つかりませんでした")
        print(f"\n📊 最もPrecisionが高い閾値: {best['threshold']:.2f}")
        print(f"   - Precision: {best['precision']:.2f}%")
        print(f"   - Recall: {best['recall']:.2f}%")
        print(f"   - F1スコア: {best['f1']:.2f}")
        print(f"   - 候補数: {int(best['candidates'])}頭")
    
    # F1スコア最大の閾値
    best_f1_idx = results_df['f1'].idxmax()
    best_f1 = results_df.loc[best_f1_idx]
    
    print(f"\n📈 F1スコア最大: {best_f1['threshold']:.2f}")
    print(f"   - Precision: {best_f1['precision']:.2f}%")
    print(f"   - Recall: {best_f1['recall']:.2f}%")
    print(f"   - F1スコア: {best_f1['f1']:.2f}")
    print(f"   - 候補数: {int(best_f1['candidates'])}頭")


def analyze_upset_threshold_optimization(file_path: str = None, by_track: bool = False, track_filter: str = None, by_year: bool = False, year_filter: int = None):
    """
    穴馬検出閾値とPrecision/Recallの関係を分析
    
    Args:
        file_path: 分析対象のTSVファイルパス
        by_track: 競馬場別に分析するか
        track_filter: 特定の競馬場のみ分析（例: "函館"）
        by_year: 年度別に分析するか
        year_filter: 特定の年度のみ分析（例: 2024）
    """
    print("=" * 80)
    print("[ANALYZE] 穴馬検出閾値の最適化分析")
    print("=" * 80)
    
    # ファイルパスの決定
    if file_path is None:
        results_file = Path("check_results/predicted_results_all.tsv")
    else:
        results_file = Path(file_path)
    
    print(f"\n[FILE] 対象ファイル: {results_file}")
    
    if not results_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {results_file}")
        print("[INFO] 先にwalk_forward_validation.pyを実行してください")
        return None
    
    df = pd.read_csv(results_file, sep='\t', encoding='utf-8-sig')
    
    print(f"[DATA] 総レコード数: {len(df)}")
    print(f"[DATA] カラム: {list(df.columns)}")
    
    # 必要な列の確認
    required_cols = ['人気順', '確定着順', '穴馬確率']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[ERROR] 必要な列が見つかりません: {missing_cols}")
        print(f"[INFO] 利用可能な列: {list(df.columns)}")
        return None
    
    # 7-12番人気のみを対象
    df_target = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    print(f"[FILTER] 7-12番人気: {len(df_target)}頭")
    
    # 競馬場の一覧を取得
    if '競馬場' in df.columns:
        tracks = df_target['競馬場'].unique()
        print(f"[TRACKS] 含まれる競馬場: {', '.join(sorted(tracks))}")
    else:
        tracks = []
        by_track = False
        print(f"[WARN] '競馬場'列がないため、競馬場別分析はスキップ")
    
    # 年度の一覧を取得
    if '開催年' in df.columns:
        years = sorted(df_target['開催年'].unique())
        print(f"[YEARS] 含まれる年度: {', '.join(map(str, years))}")
    else:
        years = []
        by_year = False
        print(f"[WARN] '開催年'列がないため、年度別分析はスキップ")
    
    results = {}
    
    # 特定の年度のみ
    if year_filter:
        if year_filter in years:
            df_year = df_target[df_target['開催年'] == year_filter]
            results[f"{year_filter}年"] = analyze_single_dataset(
                df_year, 
                label=f"{year_filter}年",
                output_prefix=f"upset_threshold_{year_filter}"
            )
        else:
            print(f"[ERROR] 年度 '{year_filter}' が見つかりません")
            print(f"[INFO] 利用可能な年度: {', '.join(map(str, years))}")
            return None
    
    # 特定の競馬場のみ
    elif track_filter:
        if track_filter in tracks:
            df_track = df_target[df_target['競馬場'] == track_filter]
            results[track_filter] = analyze_single_dataset(
                df_track, 
                label=track_filter,
                output_prefix=f"upset_threshold_{track_filter}"
            )
        else:
            print(f"[ERROR] 競馬場 '{track_filter}' が見つかりません")
            print(f"[INFO] 利用可能な競馬場: {', '.join(sorted(tracks))}")
            return None
    
    # 競馬場別と年度別の両方
    elif by_track and by_year and len(tracks) > 0 and len(years) > 0:
        # まず全体の分析
        results['全体'] = analyze_single_dataset(
            df_target,
            label="全体",
            output_prefix="upset_threshold_all"
        )
        
        # 競馬場別の分析
        print("\n" + "=" * 80)
        print("[SECTION] 競馬場別分析")
        print("=" * 80)
        track_results = {'全体': results['全体']}
        for track in sorted(tracks):
            df_track = df_target[df_target['競馬場'] == track]
            if len(df_track) > 0:
                r = analyze_single_dataset(
                    df_track,
                    label=track,
                    output_prefix=f"upset_threshold_{track}"
                )
                results[track] = r
                track_results[track] = r
        print_summary(track_results)
        
        # 年度別の分析
        print("\n" + "=" * 80)
        print("[SECTION] 年度別分析")
        print("=" * 80)
        year_results = {'全体': results['全体']}
        for year in years:
            df_year = df_target[df_target['開催年'] == year]
            if len(df_year) > 0:
                r = analyze_single_dataset(
                    df_year,
                    label=f"{year}年",
                    output_prefix=f"upset_threshold_{year}"
                )
                results[f"{year}年"] = r
                year_results[f"{year}年"] = r
        print_summary(year_results)
    
    # 競馬場別のみ
    elif by_track and len(tracks) > 0:
        # まず全体の分析
        results['全体'] = analyze_single_dataset(
            df_target,
            label="全体",
            output_prefix="upset_threshold_all"
        )
        
        # 競馬場別の分析
        for track in sorted(tracks):
            df_track = df_target[df_target['競馬場'] == track]
            if len(df_track) > 0:
                results[track] = analyze_single_dataset(
                    df_track,
                    label=track,
                    output_prefix=f"upset_threshold_{track}"
                )
        
        # サマリー表示
        print_summary(results)
    
    # 年度別のみ
    elif by_year and len(years) > 0:
        # まず全体の分析
        results['全体'] = analyze_single_dataset(
            df_target,
            label="全体",
            output_prefix="upset_threshold_all"
        )
        
        # 年度別の分析
        for year in years:
            df_year = df_target[df_target['開催年'] == year]
            if len(df_year) > 0:
                results[f"{year}年"] = analyze_single_dataset(
                    df_year,
                    label=f"{year}年",
                    output_prefix=f"upset_threshold_{year}"
                )
        
        # サマリー表示
        print_summary(results)
    
    # 全体のみ分析
    else:
        results['全体'] = analyze_single_dataset(
            df_target,
            label="全体",
            output_prefix="upset_threshold_optimization"
        )
    
    return results


def print_summary(results: dict):
    """競馬場別のサマリーを表示"""
    print("\n" + "=" * 80)
    print("[SUMMARY] 競馬場別サマリー")
    print("=" * 80)
    
    print(f"\n{'競馬場':<10} {'推奨閾値':>10} {'Precision':>12} {'Recall':>10} {'F1':>8} {'候補数':>8}")
    print("-" * 70)
    
    for track, df in results.items():
        if df is None:
            continue
        
        # Precision 8%以上 かつ Recall 50-80%の範囲で最もF1が高い閾値
        balanced = df[
            (df['precision'] >= 8.0) & 
            (df['recall'] >= 50.0) & 
            (df['recall'] <= 80.0)
        ]
        
        if len(balanced) > 0:
            best_idx = balanced['f1'].idxmax()
            best = balanced.loc[best_idx]
        else:
            # なければPrecision 8%以上でRecall最大
            good = df[df['precision'] >= 8.0]
            if len(good) > 0:
                best_idx = good['recall'].idxmax()
                best = good.loc[best_idx]
            else:
                # なければF1最大
                best_idx = df['f1'].idxmax()
                best = df.loc[best_idx]
        
        print(f"{track:<10} {best['threshold']:>10.2f} {best['precision']:>11.2f}% {best['recall']:>9.2f}% {best['f1']:>8.2f} {int(best['candidates']):>8}")
    
    print("\n💡 設定ファイル（upset_threshold_config.json）への反映例:")
    print('```json')
    print('{')
    print('  "thresholds_by_condition": {')
    print('    "by_track": {')
    
    track_codes = {
        "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
        "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"
    }
    
    for track, df in results.items():
        if df is None or track == "全体":
            continue
        
        balanced = df[
            (df['precision'] >= 8.0) & 
            (df['recall'] >= 50.0) & 
            (df['recall'] <= 80.0)
        ]
        
        if len(balanced) > 0:
            best_idx = balanced['f1'].idxmax()
        else:
            good = df[df['precision'] >= 8.0]
            if len(good) > 0:
                best_idx = good['recall'].idxmax()
            else:
                best_idx = df['f1'].idxmax()
        
        threshold = df.loc[best_idx, 'threshold']
        code = track_codes.get(track, "??")
        print(f'      "{code}": {threshold:.2f},  // {track}')
    
    print('    }')
    print('  }')
    print('}')
    print('```')


def main():
    parser = argparse.ArgumentParser(
        description='穴馬検出閾値の最適化分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # デフォルト（check_results/predicted_results_all.tsv を分析）
  python analyze_upset_threshold.py
  
  # ファイル指定
  python analyze_upset_threshold.py path/to/file.tsv
  
  # 競馬場別に分析
  python analyze_upset_threshold.py --by-track
  
  # 年度別に分析
  python analyze_upset_threshold.py --by-year
  
  # 特定の競馬場のみ
  python analyze_upset_threshold.py --track 函館
  
  # 特定の年度のみ
  python analyze_upset_threshold.py --year 2024
  
  # 組み合わせ
  python analyze_upset_threshold.py path/to/file.tsv --by-track
        """
    )
    
    parser.add_argument('file', nargs='?', default=None,
                        help='分析対象のTSVファイルパス（省略時: check_results/predicted_results_all.tsv）')
    parser.add_argument('--by-track', '-b', action='store_true',
                        help='競馬場別に分析する')
    parser.add_argument('--by-year', '-y', action='store_true',
                        help='年度別に分析する')
    parser.add_argument('--track', '-t', type=str, default=None,
                        help='特定の競馬場のみ分析（例: 函館）')
    parser.add_argument('--year', type=int, default=None,
                        help='特定の年度のみ分析（例: 2024）')
    
    args = parser.parse_args()
    
    results = analyze_upset_threshold_optimization(
        file_path=args.file,
        by_track=args.by_track,
        track_filter=args.track,
        by_year=args.by_year,
        year_filter=args.year
    )
    
    if results is not None:
        print("\n" + "=" * 80)
        print("[DONE] 分析完了！")
        print("=" * 80)
        print("\n次のステップ:")
        print("1. check_results/upset_threshold_*.png を確認")
        print("2. 推奨閾値を upset_threshold_config.json に設定")
        print("3. 再度テストを実行してPrecision/Recallを確認")


if __name__ == "__main__":
    main()
