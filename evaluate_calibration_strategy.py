"""
確率校正＋フォールバック戦略の実装と評価

校正を実施しつつ、候補が少なすぎる場合のフォールバックを用意。
以前の失敗（候補がほぼ0になる）を防ぐ。

使い方:
  # 校正の効果を検証（モデルは変更しない）
  python evaluate_calibration_strategy.py check_results/predicted_results_all.tsv
  
  # 競馬場別に分析
  python evaluate_calibration_strategy.py check_results/predicted_results_all.tsv --by-track
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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


def calibrate_with_isotonic(proba, y_true):
    """
    Isotonic Regressionで確率を校正
    訓練データで校正器を学習し、同じデータに適用（検証用）
    """
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibrator.fit(proba, y_true)
    calibrated = calibrator.predict(proba)
    return calibrated, calibrator


def calibrate_with_platt(proba, y_true):
    """
    Platt Scaling（シグモイド校正）で確率を校正
    """
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(proba.reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
    return calibrated, calibrator


def apply_fallback(df, threshold, min_per_race=1):
    """
    フォールバック戦略: 候補が0のレースでは最高確率の馬を候補に
    
    Args:
        df: DataFrameに'is_candidate'列が必要
        threshold: 適用された閾値
        min_per_race: レースあたりの最低候補数
    
    Returns:
        更新されたDataFrame
    """
    df = df.copy()
    
    for race_id in df['race_id'].unique():
        race_mask = df['race_id'] == race_id
        race_df = df[race_mask]
        
        current_candidates = race_df['is_candidate'].sum()
        
        if current_candidates < min_per_race:
            # 最高確率の馬を候補に追加
            need = min_per_race - current_candidates
            non_candidates = race_df[~race_df['is_candidate']].nlargest(need, 'calibrated_prob')
            if len(non_candidates) > 0:
                df.loc[non_candidates.index, 'is_candidate'] = True
                df.loc[non_candidates.index, 'fallback'] = True
    
    return df


def analyze_calibration(df, label="全体"):
    """校正前後の比較分析"""
    df = df.copy()
    df = create_race_id(df)
    df['is_upset'] = (df['確定着順'] <= 3).astype(int)
    
    proba = df['穴馬確率'].values
    y_true = df['is_upset'].values
    
    print(f"\n{'='*80}")
    print(f"[{label}] 確率校正の効果分析")
    print(f"{'='*80}")
    print(f"総レコード: {len(df)}頭")
    print(f"実際の穴馬: {df['is_upset'].sum()}頭 ({df['is_upset'].mean()*100:.2f}%)")
    
    # --- 校正前の分析 ---
    print(f"\n--- 校正前 ---")
    print(f"穴馬確率 平均: {proba.mean():.4f}, 中央値: {np.median(proba):.4f}")
    
    # 確率帯ごとの実際の的中率
    print(f"\n確率区間     件数    穴馬   実際の率   期待値    乖離")
    print("-" * 55)
    
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (proba >= low) & (proba < high)
        n = mask.sum()
        if n > 0:
            actual = y_true[mask].mean() * 100
            expected = (low + high) / 2 * 100
            diff = actual - expected
            print(f"{low:.1f}-{high:.1f}     {n:>5}   {y_true[mask].sum():>4}   {actual:>6.2f}%   {expected:>5.1f}%   {diff:>+6.1f}%")
    
    # --- Isotonic校正 ---
    print(f"\n--- Isotonic校正後 ---")
    calibrated_iso, _ = calibrate_with_isotonic(proba, y_true)
    df['calibrated_prob'] = calibrated_iso
    
    print(f"校正後確率 平均: {calibrated_iso.mean():.4f}, 中央値: {np.median(calibrated_iso):.4f}")
    
    print(f"\n校正後確率区間  件数    穴馬   実際の率")
    print("-" * 45)
    
    # 校正後は低い値に集中するので、細かい区間で見る
    bins_calibrated = [0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.0]
    for i in range(len(bins_calibrated)-1):
        low, high = bins_calibrated[i], bins_calibrated[i+1]
        mask = (calibrated_iso >= low) & (calibrated_iso < high)
        n = mask.sum()
        if n > 0:
            actual = y_true[mask].mean() * 100
            print(f"{low:.2f}-{high:.2f}       {n:>5}   {y_true[mask].sum():>4}   {actual:>6.2f}%")
    
    # --- 閾値別評価（校正前 vs 校正後） ---
    print(f"\n{'='*80}")
    print(f"[{label}] 閾値別評価（校正前 vs 校正後）")
    print(f"{'='*80}")
    
    print(f"\n{'閾値':<8} {'校正前':^30} {'校正後(Isotonic)':^30}")
    print(f"{'':8} {'候補':>8} {'TP':>6} {'Prec':>8} {'候補':>8} {'TP':>6} {'Prec':>8}")
    print("-" * 75)
    
    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        # 校正前
        mask_before = proba >= threshold
        tp_before = (mask_before & (y_true == 1)).sum()
        prec_before = tp_before / mask_before.sum() * 100 if mask_before.sum() > 0 else 0
        
        # 校正後
        mask_after = calibrated_iso >= threshold
        tp_after = (mask_after & (y_true == 1)).sum()
        prec_after = tp_after / mask_after.sum() * 100 if mask_after.sum() > 0 else 0
        
        print(f"{threshold:.2f}     {mask_before.sum():>8} {tp_before:>6} {prec_before:>7.2f}% {mask_after.sum():>8} {tp_after:>6} {prec_after:>7.2f}%")
    
    # --- フォールバック戦略の評価 ---
    print(f"\n{'='*80}")
    print(f"[{label}] フォールバック戦略の評価")
    print(f"{'='*80}")
    
    print(f"\n校正後、閾値0.10で候補がないレースにフォールバック適用:")
    
    df['is_candidate'] = calibrated_iso >= 0.10
    df['fallback'] = False
    
    # フォールバック適用前の状況
    races_with_candidates = df.groupby('race_id')['is_candidate'].sum()
    empty_races = (races_with_candidates == 0).sum()
    total_races = len(races_with_candidates)
    
    print(f"  フォールバック前: 候補あり {total_races - empty_races}レース, 候補なし {empty_races}レース ({empty_races/total_races*100:.1f}%)")
    
    # フォールバック適用
    df = apply_fallback(df, 0.10, min_per_race=1)
    
    candidates = df[df['is_candidate']]
    fallback_candidates = df[df['fallback']]
    
    tp = candidates['is_upset'].sum()
    precision = tp / len(candidates) * 100 if len(candidates) > 0 else 0
    
    tp_fallback = fallback_candidates['is_upset'].sum()
    prec_fallback = tp_fallback / len(fallback_candidates) * 100 if len(fallback_candidates) > 0 else 0
    
    print(f"  フォールバック後: 総候補 {len(candidates)}頭, TP {tp}, Precision {precision:.2f}%")
    print(f"  うちフォールバック分: {len(fallback_candidates)}頭, TP {tp_fallback}, Precision {prec_fallback:.2f}%")
    
    return df


def plot_calibration_curve(df, output_path="check_results/calibration_curve.png"):
    """校正曲線をプロット"""
    df = df.copy()
    df['is_upset'] = (df['確定着順'] <= 3).astype(int)
    
    proba = df['穴馬確率'].values
    y_true = df['is_upset'].values
    
    # 校正
    calibrated_iso, _ = calibrate_with_isotonic(proba, y_true)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 校正曲線（校正前）
    ax1 = axes[0]
    
    # 確率を区間ごとに分けて実際の的中率を計算
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_rates = []
    counts = []
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i+1])
        if mask.sum() > 0:
            actual_rates.append(y_true[mask].mean())
            counts.append(mask.sum())
        else:
            actual_rates.append(np.nan)
            counts.append(0)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='完璧な校正')
    ax1.scatter(bin_centers, actual_rates, s=[max(10, c/5) for c in counts], alpha=0.7, label='実際の的中率')
    ax1.set_xlabel('予測確率', fontsize=12)
    ax1.set_ylabel('実際の的中率', fontsize=12)
    ax1.set_title('校正曲線（校正前）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(0.3, max([r for r in actual_rates if not np.isnan(r)]) * 1.2))
    
    # 2. 確率分布（校正前 vs 校正後）
    ax2 = axes[1]
    ax2.hist(proba, bins=30, alpha=0.5, label='校正前', density=True)
    ax2.hist(calibrated_iso, bins=30, alpha=0.5, label='校正後(Isotonic)', density=True)
    ax2.axvline(x=0.10, color='r', linestyle='--', label='閾値 0.10')
    ax2.set_xlabel('確率', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title('確率分布の変化', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 校正曲線（校正後）
    ax3 = axes[2]
    
    actual_rates_cal = []
    counts_cal = []
    for i in range(n_bins):
        mask = (calibrated_iso >= bin_edges[i]) & (calibrated_iso < bin_edges[i+1])
        if mask.sum() > 0:
            actual_rates_cal.append(y_true[mask].mean())
            counts_cal.append(mask.sum())
        else:
            actual_rates_cal.append(np.nan)
            counts_cal.append(0)
    
    ax3.plot([0, 1], [0, 1], 'k--', label='完璧な校正')
    ax3.scatter(bin_centers, actual_rates_cal, s=[max(10, c/5) for c in counts_cal], alpha=0.7, color='orange', label='実際の的中率')
    ax3.set_xlabel('校正後確率', fontsize=12)
    ax3.set_ylabel('実際の的中率', fontsize=12)
    ax3.set_title('校正曲線（校正後）', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 0.3)
    ax3.set_ylim(0, 0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[FILE] 校正曲線を保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='確率校正＋フォールバック戦略の評価')
    parser.add_argument('file_path', nargs='?', default='check_results/predicted_results_all.tsv')
    parser.add_argument('--by-track', action='store_true', help='競馬場別に分析')
    
    args = parser.parse_args()
    
    # データ読み込み
    df = pd.read_csv(args.file_path, sep='\t')
    print(f"[FILE] {args.file_path}")
    print(f"[DATA] 総レコード: {len(df)}")
    
    # 7-12番人気のみ
    df_target = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    print(f"[FILTER] 7-12番人気: {len(df_target)}")
    
    # 全体分析
    analyze_calibration(df_target, "全体")
    
    # 校正曲線をプロット
    plot_calibration_curve(df_target)
    
    # 競馬場別
    if args.by_track:
        for track in df_target['競馬場'].unique():
            df_track = df_target[df_target['競馬場'] == track]
            if len(df_track) >= 100:
                analyze_calibration(df_track, track)
    
    print("\n" + "=" * 80)
    print("[結論と推奨]")
    print("=" * 80)
    print("""
確率校正の効果:
1. 校正後の確率は実際の的中率に近づく（校正曲線が対角線に近づく）
2. ただし、ほとんどの馬が低確率帯（0.05-0.15）に集中する
3. 閾値0.10以上の候補が大幅に減少する可能性あり

フォールバック戦略:
- 候補が0のレースでは、最高確率の馬を候補に追加
- これにより「候補なし」レースを防止
- ただし、フォールバック候補のPrecisionは低い可能性

推奨アプローチ:
1. 現状のモデル（校正なし）で閾値0.15を使用（ROI最適化済み）
2. 校正は「確率の解釈性」が必要な場合のみ適用
3. フォールバックは積極的な運用時のみ使用
""")


if __name__ == '__main__':
    main()
