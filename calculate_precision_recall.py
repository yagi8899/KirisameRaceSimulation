"""
穴馬予測の正確なPrecision/Recallを計算

使用方法:
    # デフォルト（check_results/predicted_results_all.tsv を分析）
    python calculate_precision_recall.py
    
    # ファイル指定
    python calculate_precision_recall.py path/to/file.tsv
    
    # 競馬場別に分析
    python calculate_precision_recall.py --by-track
    
    # 年度別に分析
    python calculate_precision_recall.py --by-year
    
    # 特定の競馬場のみ
    python calculate_precision_recall.py --track 函館
    
    # 特定の年度のみ
    python calculate_precision_recall.py --year 2024
    
    # 組み合わせ
    python calculate_precision_recall.py path/to/file.tsv --by-track --by-year
"""
import pandas as pd
from pathlib import Path
import argparse
import json


# 競馬場名からコードへのマッピング
TRACK_NAME_TO_CODE = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
    '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10'
}


def load_threshold_config() -> dict:
    """upset_threshold_config.json から閾値設定を読み込む"""
    config_path = Path(__file__).parent / "upset_threshold_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"default_threshold": 0.20, "thresholds_by_condition": {}}


def get_threshold_for_track(config: dict, track_name: str) -> float:
    """競馬場名に対応する閾値を取得"""
    default = config.get("default_threshold", 0.20)
    track_code = TRACK_NAME_TO_CODE.get(track_name)
    
    if track_code:
        by_track = config.get("thresholds_by_condition", {}).get("by_track", {})
        return by_track.get(track_code, default)
    
    return default


def apply_threshold_to_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    DataFrameに閾値を適用して穴馬候補を再計算
    競馬場ごとに異なる閾値を適用
    """
    df = df.copy()
    
    if '穴馬確率' not in df.columns:
        print("⚠️ '穴馬確率'列がないため、既存の'穴馬候補'を使用")
        return df
    
    if '競馬場' not in df.columns:
        # 競馬場列がない場合はデフォルト閾値を使用
        threshold = config.get("default_threshold", 0.20)
        df['穴馬候補'] = (df['穴馬確率'] >= threshold).astype(int)
        print(f"📊 閾値 {threshold} を全体に適用")
        return df
    
    # 競馬場ごとに閾値を適用
    df['穴馬候補'] = 0
    applied_thresholds = {}
    
    for track_name in df['競馬場'].unique():
        threshold = get_threshold_for_track(config, track_name)
        mask = df['競馬場'] == track_name
        df.loc[mask, '穴馬候補'] = (df.loc[mask, '穴馬確率'] >= threshold).astype(int)
        applied_thresholds[track_name] = threshold
    
    print(f"📊 適用した閾値: {applied_thresholds}")
    
    return df


def calculate_single_metrics(df: pd.DataFrame, label: str = "全体") -> dict:
    """
    単一データセットのPrecision/Recallを計算
    
    Args:
        df: 分析対象のDataFrame
        label: 出力ラベル（例: "函館", "京都"）
    
    Returns:
        dict: 計算結果
    """
    print(f"\n{'='*70}")
    print(f"🎯 {label} - 穴馬予測の評価結果（7-12番人気で3着以内）")
    print(f"{'='*70}")
    
    total_records = len(df)
    if total_records == 0:
        print(f"⚠️ データがありません")
        return None
    
    # 実際の穴馬を定義（7-12番人気で3着以内）
    df = df.copy()
    df['実際の穴馬'] = (
        (df['人気順'] >= 7) & 
        (df['人気順'] <= 12) & 
        (df['確定着順'] <= 3)
    ).astype(int)
    
    # True Positive: 穴馬候補かつ実際の穴馬
    TP = ((df['穴馬候補'] == 1) & (df['実際の穴馬'] == 1)).sum()
    
    # False Positive: 穴馬候補だが実際は穴馬ではない
    FP = ((df['穴馬候補'] == 1) & (df['実際の穴馬'] == 0)).sum()
    
    # False Negative: 穴馬候補ではないが実際は穴馬
    FN = ((df['穴馬候補'] == 0) & (df['実際の穴馬'] == 1)).sum()
    
    # True Negative: 穴馬候補でなく実際も穴馬ではない
    TN = ((df['穴馬候補'] == 0) & (df['実際の穴馬'] == 0)).sum()
    
    # 集計
    穴馬候補総数 = TP + FP
    実際の穴馬総数 = TP + FN
    
    # Precision（適合率）
    Precision = (TP / 穴馬候補総数 * 100) if 穴馬候補総数 > 0 else 0
    
    # Recall（再現率）
    Recall = (TP / 実際の穴馬総数 * 100) if 実際の穴馬総数 > 0 else 0
    
    # F1 Score
    F1 = (2 * Precision * Recall / (Precision + Recall)) if (Precision + Recall) > 0 else 0
    
    # 結果表示
    print(f"\n📊 データ概要:")
    print(f"  総レコード数: {total_records:,}頭")
    print(f"  穴馬候補総数: {穴馬候補総数:,}頭 (TP + FP)")
    print(f"  実際の穴馬数: {実際の穴馬総数:,}頭 (TP + FN)")
    
    print(f"\n📊 混同行列:")
    print(f"  True Positive (TP):  {TP:,}頭  ← 穴馬候補かつ実際の穴馬")
    print(f"  False Positive (FP): {FP:,}頭  ← 穴馬候補だが外れ")
    print(f"  False Negative (FN): {FN:,}頭  ← 見逃した穴馬")
    print(f"  True Negative (TN):  {TN:,}頭  ← 正しく除外")
    
    print(f"\n📈 評価指標:")
    print(f"  🎯 Precision（適合率）: {Precision:.2f}%")
    print(f"     → 穴馬候補のうち{Precision:.2f}%が実際に好走")
    print(f"  🔍 Recall（再現率）: {Recall:.2f}%")
    print(f"     → 実際の穴馬の{Recall:.2f}%を検出")
    print(f"  ⚖️ F1 Score: {F1:.2f}")
    
    # Phase評価
    phase1 = Precision >= 8.0
    phase2 = Precision >= 10.0
    phase3 = Precision >= 12.0
    
    print(f"\n📋 Phase目標:")
    print(f"  Phase 1 (8%以上):  {'✅ 達成' if phase1 else '❌ 未達成'}")
    print(f"  Phase 2 (10%以上): {'✅ 達成' if phase2 else '⚠️ 未達成'}")
    print(f"  Phase 3 (12%以上): {'✅ 達成' if phase3 else '⚠️ 未達成'}")
    
    return {
        'label': label,
        'total': total_records,
        'candidates': 穴馬候補総数,
        'actual_upsets': 実際の穴馬総数,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': Precision,
        'recall': Recall,
        'f1': F1,
        'phase1': phase1,
        'phase2': phase2,
        'phase3': phase3
    }


def print_summary(results: list):
    """競馬場別のサマリーを表示"""
    print("\n" + "=" * 80)
    print("📊 競馬場別サマリー")
    print("=" * 80)
    
    print(f"\n{'競馬場':<10} {'レコード':>10} {'候補数':>8} {'穴馬数':>8} {'TP':>6} {'Precision':>12} {'Recall':>10} {'F1':>8} {'Phase1':>8}")
    print("-" * 100)
    
    for r in results:
        if r is None:
            continue
        phase1_mark = "✅" if r['phase1'] else "❌"
        print(f"{r['label']:<10} {r['total']:>10,} {r['candidates']:>8} {r['actual_upsets']:>8} {r['TP']:>6} {r['precision']:>11.2f}% {r['recall']:>9.2f}% {r['f1']:>8.2f} {phase1_mark:>8}")
    
    # Phase達成率
    phase1_count = sum(1 for r in results if r and r['phase1'])
    phase2_count = sum(1 for r in results if r and r['phase2'])
    phase3_count = sum(1 for r in results if r and r['phase3'])
    total_count = sum(1 for r in results if r)
    
    print(f"\n📋 Phase達成状況:")
    print(f"  Phase 1 (8%以上):  {phase1_count}/{total_count} で達成")
    print(f"  Phase 2 (10%以上): {phase2_count}/{total_count} で達成")
    print(f"  Phase 3 (12%以上): {phase3_count}/{total_count} で達成")


def print_track_year_summary(track: str, results: list):
    """特定競馬場の年度別サマリーを表示"""
    print(f"\n  📅 {track} 年度別サマリー:")
    print(f"  {'年度':<12} {'レコード':>8} {'候補数':>6} {'TP':>5} {'Precision':>10} {'Recall':>8} {'Phase1':>7}")
    print("  " + "-" * 70)
    
    for r in results:
        if r is None:
            continue
        # ラベルから年度部分を抽出（"  └ 中山 2022年" → "2022年"）
        label = r['label'].split()[-1] if r['label'] else ""
        phase1_mark = "✅" if r['phase1'] else "❌"
        print(f"  {label:<12} {r['total']:>8,} {r['candidates']:>6} {r['TP']:>5} {r['precision']:>9.2f}% {r['recall']:>7.2f}% {phase1_mark:>7}")


def calculate_metrics(file_path: str = None, by_track: bool = False, track_filter: str = None, by_year: bool = False, year_filter: int = None):
    """
    Precision/Recallを正確に計算
    
    Args:
        file_path: 分析対象のTSVファイルパス
        by_track: 競馬場別に分析するか
        track_filter: 特定の競馬場のみ分析（例: "函館"）
        by_year: 年度別に分析するか
        year_filter: 特定の年度のみ分析（例: 2024）
    """
    print("=" * 80)
    print("🎯 穴馬予測 Precision/Recall 計算")
    print("=" * 80)
    
    # ファイルパスの決定
    if file_path is None:
        results_file = Path("check_results/predicted_results_all.tsv")
    else:
        results_file = Path(file_path)
    
    print(f"\n📂 対象ファイル: {results_file}")
    
    if not results_file.exists():
        print(f"❌ ファイルが見つかりません: {results_file}")
        return None
    
    df = pd.read_csv(results_file, sep='\t', encoding='utf-8-sig')
    
    print(f"✅ {len(df):,}レコード読み込み完了")
    print(f"📋 列一覧: {df.columns.tolist()}")
    
    # upset_threshold_config.json から閾値を読み込んで適用
    config = load_threshold_config()
    print(f"\n📂 閾値設定ファイル読み込み完了")
    print(f"   デフォルト閾値: {config.get('default_threshold', 0.20)}")
    df = apply_threshold_to_df(df, config)
    
    # 必要な列があるか確認
    required_cols = ['穴馬候補', '人気順', '確定着順']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n❌ 必要な列がありません: {missing}")
        return None
    
    # データクリーニング
    df = df.dropna(subset=['穴馬候補', '人気順', '確定着順'])
    print(f"📊 NaN除外後: {len(df):,}レコード")
    
    # 競馬場の一覧を取得
    if '競馬場' in df.columns:
        tracks = df['競馬場'].unique()
        print(f"🏇 含まれる競馬場: {', '.join(sorted(tracks))}")
    else:
        tracks = []
        by_track = False
        print(f"⚠️ '競馬場'列がないため、競馬場別分析はスキップ")
    
    # 年度の一覧を取得
    if '開催年' in df.columns:
        years = sorted(df['開催年'].unique())
        print(f"📅 含まれる年度: {', '.join(map(str, years))}")
    else:
        years = []
        by_year = False
        print(f"⚠️ '開催年'列がないため、年度別分析はスキップ")
    
    results = []
    
    # 特定の年度のみ
    if year_filter:
        if year_filter in years:
            df_year = df[df['開催年'] == year_filter]
            result = calculate_single_metrics(df_year, label=f"{year_filter}年")
            results.append(result)
        else:
            print(f"❌ 年度 '{year_filter}' が見つかりません")
            print(f"📋 利用可能な年度: {', '.join(map(str, years))}")
            return None
    
    # 特定の競馬場のみ
    elif track_filter:
        if track_filter in tracks:
            df_track = df[df['競馬場'] == track_filter]
            result = calculate_single_metrics(df_track, label=track_filter)
            results.append(result)
        else:
            print(f"❌ 競馬場 '{track_filter}' が見つかりません")
            print(f"📋 利用可能な競馬場: {', '.join(sorted(tracks))}")
            return None
    
    # 競馬場別と年度別の両方
    elif by_track and by_year and len(tracks) > 0 and len(years) > 0:
        # まず全体の分析
        result = calculate_single_metrics(df, label="全体")
        results.append(result)
        
        # 競馬場別の分析 + 各競馬場の年度別内訳
        print("\n" + "=" * 80)
        print("📊 競馬場別分析（年度別内訳付き）")
        print("=" * 80)
        track_results = [result]  # 全体を含む
        for track in sorted(tracks):
            df_track = df[df['競馬場'] == track]
            if len(df_track) > 0:
                # 競馬場全体
                r = calculate_single_metrics(df_track, label=track)
                results.append(r)
                track_results.append(r)
                
                # この競馬場の年度別内訳
                track_year_results = []
                for year in years:
                    df_track_year = df_track[df_track['開催年'] == year]
                    if len(df_track_year) > 0:
                        r_year = calculate_single_metrics(df_track_year, label=f"  └ {track} {year}年")
                        results.append(r_year)
                        track_year_results.append(r_year)
                
                # 競馬場内の年度サマリー
                if len(track_year_results) > 1:
                    print_track_year_summary(track, track_year_results)
        
        # 競馬場サマリー
        print_summary(track_results)
    
    # 競馬場別のみ
    elif by_track and len(tracks) > 0:
        # まず全体の分析
        result = calculate_single_metrics(df, label="全体")
        results.append(result)
        
        # 競馬場別の分析
        for track in sorted(tracks):
            df_track = df[df['競馬場'] == track]
            if len(df_track) > 0:
                result = calculate_single_metrics(df_track, label=track)
                results.append(result)
        
        # サマリー表示
        print_summary(results)
    
    # 年度別のみ
    elif by_year and len(years) > 0:
        # まず全体の分析
        result = calculate_single_metrics(df, label="全体")
        results.append(result)
        
        # 年度別の分析
        for year in years:
            df_year = df[df['開催年'] == year]
            if len(df_year) > 0:
                result = calculate_single_metrics(df_year, label=f"{year}年")
                results.append(result)
        
        # サマリー表示
        print_summary(results)
    
    # 全体のみ分析
    else:
        result = calculate_single_metrics(df, label="全体")
        results.append(result)
        
        # 詳細な評価を表示（全体のみの場合）
        if result:
            print_detailed_evaluation(result)
    
    return results


def print_detailed_evaluation(result: dict):
    """詳細な評価を表示（全体分析時のみ）"""
    print(f"\n{'='*70}")
    print(f"📋 詳細評価")
    print(f"{'='*70}")
    
    # 候補数評価
    candidates = result['candidates']
    print(f"\n【候補数の評価】")
    if candidates <= 500:
        print(f"  ✅ 少なめ ({candidates:,}頭) - 絞り込み効いている")
    elif candidates <= 1500:
        print(f"  ✅ 適正 ({candidates:,}頭)")
    elif candidates <= 3000:
        print(f"  ⚠️ やや多め ({candidates:,}頭)")
    else:
        print(f"  ❌ 多すぎ ({candidates:,}頭) - 閾値を上げることを推奨")
    
    # Recall評価
    recall = result['recall']
    print(f"\n【Recall（再現率）の評価】")
    if recall >= 90:
        print(f"  ⚠️ 高すぎ ({recall:.2f}%) - 閾値が低すぎる可能性")
    elif recall >= 60:
        print(f"  ✅ 適正 ({recall:.2f}%)")
    elif recall >= 40:
        print(f"  ⚠️ やや低め ({recall:.2f}%) - 見逃しが多い")
    else:
        print(f"  ❌ 低すぎ ({recall:.2f}%) - 閾値を下げることを推奨")
    
    # バランス評価
    precision = result['precision']
    print(f"\n【バランス評価】")
    if precision >= 12 and 50 <= recall <= 80:
        print(f"  ✅ 理想的なバランス")
    elif precision >= 8 and recall >= 50:
        print(f"  ✅ 良好なバランス")
    elif precision >= 8:
        print(f"  ⚠️ Precisionは達成、Recallが低め")
    elif recall >= 80:
        print(f"  ⚠️ Recallは高いが、Precisionが低い → 閾値を上げる")
    else:
        print(f"  ❌ 改善が必要")


def main():
    parser = argparse.ArgumentParser(
        description='穴馬予測のPrecision/Recall計算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # デフォルト（check_results/predicted_results_all.tsv を分析）
  python calculate_precision_recall.py
  
  # ファイル指定
  python calculate_precision_recall.py path/to/file.tsv
  
  # 競馬場別に分析
  python calculate_precision_recall.py --by-track
  
  # 特定の競馬場のみ
  python calculate_precision_recall.py --track 函館
  
  # 組み合わせ
  # 年度別に分析
  python calculate_precision_recall.py --by-year
  
  # 特定の年度のみ
  python calculate_precision_recall.py --year 2024
  
  # 組み合わせ
  python calculate_precision_recall.py path/to/file.tsv --by-track
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
    
    results = calculate_metrics(
        file_path=args.file,
        by_track=args.by_track,
        track_filter=args.track,
        by_year=args.by_year,
        year_filter=args.year
    )
    
    if results is not None:
        print("\n" + "=" * 80)
        print("✅ 計算完了！")
        print("=" * 80)


if __name__ == '__main__':
    main()
