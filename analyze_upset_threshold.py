"""
穴馬検出モデルの閾値最適化スクリプト
Precision 8%以上を達成する最適閾値を見つける
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_upset_threshold_optimization():
    """穴馬検出閾値とPrecision/Recallの関係を分析"""
    print("=" * 80)
    print("[ANALYZE] 穴馬検出閾値の最適化分析")
    print("=" * 80)
    
    # check_results/predicted_results_all.tsvから予測結果を取得
    results_file = Path("check_results/predicted_results_all.tsv")
    
    if not results_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {results_file}")
        print("[INFO] 先にwalk_forward_validation.pyを実行してください")
        return None
    
    df = pd.read_csv(results_file, sep='\t', encoding='utf-8-sig')
    
    print(f"\n[DATA] 総レコード数: {len(df)}")
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
    print(f"\n[FILTER] 7-12番人気: {len(df_target)}頭")
    
    # 実際の穴馬（7-12番人気で3着以内）
    df_target['is_upset'] = (df_target['確定着順'] <= 3).astype(int)
    
    total_upsets = df_target['is_upset'].sum()
    print(f"[GROUND TRUTH] 実際の穴馬数: {total_upsets}頭")
    print(f"[GROUND TRUTH] 穴馬率: {total_upsets / len(df_target) * 100:.2f}%")
    
    # 穴馬確率の分布
    probs = df_target['穴馬確率'].dropna()
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
    
    # 1. Precision/Recall曲線
    ax1.plot(results_df['threshold'], results_df['precision'], marker='o', linewidth=2, markersize=6, label='Precision', color='blue')
    ax1.plot(results_df['threshold'], results_df['recall'], marker='s', linewidth=2, markersize=6, label='Recall', color='green')
    ax1.axhline(y=8.0, color='red', linestyle='--', linewidth=2, label='目標Precision: 8%')
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
    output_file = output_dir / 'upset_threshold_optimization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[FILE] 最適化グラフを保存: {output_file}")
    plt.close()
    
    # 推奨閾値を提案
    print("\n" + "=" * 80)
    print("[RECOMMEND] 推奨閾値")
    print("=" * 80)
    
    # Precision 8%以上で最もRecallが高い閾値
    good_results = results_df[results_df['precision'] >= 8.0]
    
    if len(good_results) > 0:
        best_recall_idx = good_results['recall'].idxmax()
        best_threshold = good_results.loc[best_recall_idx, 'threshold']
        best_precision = good_results.loc[best_recall_idx, 'precision']
        best_recall = good_results.loc[best_recall_idx, 'recall']
        best_f1 = good_results.loc[best_recall_idx, 'f1']
        best_candidates = good_results.loc[best_recall_idx, 'candidates']
        
        print(f"\n✅ 推奨閾値: {best_threshold:.2f}")
        print(f"   - Precision: {best_precision:.2f}% (目標8%以上を達成)")
        print(f"   - Recall: {best_recall:.2f}%")
        print(f"   - F1スコア: {best_f1:.2f}")
        print(f"   - 候補数: {best_candidates}頭")
        
        print(f"\n📝 設定方法:")
        print(f"   walk_forward_validation.py の upset_threshold を {best_threshold:.2f} に変更してください")
        
    else:
        # Precision 8%未達成の場合、最もPrecisionが高い閾値を提案
        best_precision_idx = results_df['precision'].idxmax()
        best_threshold = results_df.loc[best_precision_idx, 'threshold']
        best_precision = results_df.loc[best_precision_idx, 'precision']
        best_recall = results_df.loc[best_precision_idx, 'recall']
        best_f1 = results_df.loc[best_precision_idx, 'f1']
        best_candidates = results_df.loc[best_precision_idx, 'candidates']
        
        print(f"\n⚠️  Precision 8%を達成できる閾値が見つかりませんでした")
        print(f"\n📊 最もPrecisionが高い閾値: {best_threshold:.2f}")
        print(f"   - Precision: {best_precision:.2f}% (目標8%未達成)")
        print(f"   - Recall: {best_recall:.2f}%")
        print(f"   - F1スコア: {best_f1:.2f}")
        print(f"   - 候補数: {best_candidates}頭")
        
        print(f"\n⚠️  次のステップ:")
        print(f"   1. Phase 3特徴量の効果を検証（特徴量重要度分析）")
        print(f"   2. 特徴量が効いていない場合、NULL処理をfillna(0)に戻す")
        print(f"   3. Phase 1 Feature Set 2（追加4特徴量）の実装を検討")
    
    # F1スコア最大の閾値も参考情報として表示
    best_f1_idx = results_df['f1'].idxmax()
    f1_threshold = results_df.loc[best_f1_idx, 'threshold']
    f1_precision = results_df.loc[best_f1_idx, 'precision']
    f1_recall = results_df.loc[best_f1_idx, 'recall']
    f1_f1 = results_df.loc[best_f1_idx, 'f1']
    f1_candidates = results_df.loc[best_f1_idx, 'candidates']
    
    print(f"\n📊 参考: F1スコア最大の閾値: {f1_threshold:.2f}")
    print(f"   - Precision: {f1_precision:.2f}%")
    print(f"   - Recall: {f1_recall:.2f}%")
    print(f"   - F1スコア: {f1_f1:.2f}")
    print(f"   - 候補数: {f1_candidates}頭")
    
    return results_df


if __name__ == "__main__":
    results_df = analyze_upset_threshold_optimization()
    
    if results_df is not None:
        print("\n" + "=" * 80)
        print("[DONE] 分析完了！")
        print("=" * 80)
        print("\n次のステップ:")
        print("1. check_results/upset_threshold_optimization.png を確認")
        print("2. 推奨閾値をwalk_forward_validation.pyのupset_thresholdに設定")
        print("3. 再度python walk_forward_validation.py --start_year 2025 --end_year 2025 を実行")
        print("4. Precision 8%以上を達成できているか確認")
