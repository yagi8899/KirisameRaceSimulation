"""
スコア差閾値の最適化分析スクリプト
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_score_diff_distribution():
    """スコア差の分布を分析"""
    print("=" * 80)
    print("[ANALYZE] スコア差の分布分析")
    print("=" * 80)
    
    # predicted_results_skipped.tsvからスコア差を取得
    skipped_file = Path("results/predicted_results_skipped.tsv")
    
    if not skipped_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {skipped_file}")
        return
    
    df = pd.read_csv(skipped_file, sep='\t', encoding='utf-8-sig')
    
    print(f"\n[DATA] 総レコード数: {len(df)}")
    print(f"[DATA] カラム数: {len(df.columns)}")
    
    # スコア差列の確認
    score_diff_col = None
    for col in ['スコア差', 'score_diff']:
        if col in df.columns:
            score_diff_col = col
            break
    
    if score_diff_col is None:
        print("[ERROR] スコア差列が見つかりません")
        return
    
    # レース単位でユニークなスコア差を取得
    # （同じレースの全馬が同じスコア差を持つので、レースIDでグループ化）
    race_id_cols = ['競馬場', '開催年', '開催日', 'レース番号']
    df_races = df.groupby(race_id_cols)[score_diff_col].first().reset_index()
    
    score_diffs = df_races[score_diff_col].dropna()
    
    print(f"\n[STATS] スコア差の基本統計")
    print(f"  - レース数: {len(score_diffs)}")
    print(f"  - 平均: {score_diffs.mean():.6f}")
    print(f"  - 中央値: {score_diffs.median():.6f}")
    print(f"  - 標準偏差: {score_diffs.std():.6f}")
    print(f"  - 最小値: {score_diffs.min():.6f}")
    print(f"  - 最大値: {score_diffs.max():.6f}")
    
    # パーセンタイル
    print(f"\n[PERCENTILE] スコア差の分位点")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = score_diffs.quantile(p/100)
        print(f"  - {p:2d}%点: {val:.6f}")
    
    # 現在の閾値でどれだけフィルタリングされるか
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    print(f"\n[FILTER] 各閾値でのフィルタリング率")
    print(f"{'閾値':>8s} {'スキップ数':>10s} {'残存数':>10s} {'スキップ率':>10s}")
    print("-" * 45)
    for threshold in thresholds:
        skipped = (score_diffs < threshold).sum()
        remained = (score_diffs >= threshold).sum()
        skip_rate = skipped / len(score_diffs) * 100
        marker = " ← 現在" if threshold == 0.05 else ""
        print(f"{threshold:8.2f} {skipped:10d} {remained:10d} {skip_rate:9.1f}%{marker}")
    
    # ヒストグラム作成
    plt.figure(figsize=(12, 6))
    plt.hist(score_diffs, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='現在の閾値 (0.05)')
    plt.xlabel('スコア差（1位 - 2位）', fontsize=12)
    plt.ylabel('レース数', fontsize=12)
    plt.title('予測スコア差の分布', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'score_diff_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[FILE] ヒストグラムを保存: {output_file}")
    plt.close()
    
    return df_races


def analyze_threshold_vs_accuracy(all_results_file="results/predicted_results_all.tsv"):
    """各閾値での的中率・回収率をシミュレーション"""
    print("\n" + "=" * 80)
    print("[OPTIMIZE] 閾値と的中率の関係分析")
    print("=" * 80)
    
    results_file = Path(all_results_file)
    
    if not results_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {results_file}")
        print("[INFO] 先にpython universal_test.py multi 2023を実行してください")
        return
    
    df = pd.read_csv(results_file, sep='\t', encoding='utf-8-sig')
    
    print(f"\n[DATA] 総レコード数: {len(df)}")
    
    # 必要な列の確認
    required_cols = ['競馬場', '開催年', '開催日', 'レース番号', '予測順位', '確定着順', '予測スコア']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[ERROR] 必要な列が見つかりません: {missing_cols}")
        return
    
    # レースごとにスコア差を計算
    race_id_cols = ['競馬場', '開催年', '開催日', 'レース番号']
    
    def calc_score_diff(race_df):
        sorted_df = race_df.sort_values('予測スコア', ascending=False)
        if len(sorted_df) >= 2:
            return sorted_df.iloc[0]['予測スコア'] - sorted_df.iloc[1]['予測スコア']
        return 0.0
    
    df['スコア差'] = df.groupby(race_id_cols, group_keys=False).apply(
        lambda x: pd.Series(calc_score_diff(x), index=x.index)
    )
    
    # 予測1位の馬のみを抽出
    df_top1 = df[df['予測順位'] == 1].copy()
    
    print(f"\n[DATA] 予測1位の馬: {len(df_top1)}レース")
    
    # 各閾値でシミュレーション
    thresholds = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    results = []
    
    print(f"\n[SIMULATE] 各閾値での的中率シミュレーション")
    print(f"{'閾値':>8s} {'対象数':>8s} {'的中数':>8s} {'的中率':>10s} {'判定':>10s}")
    print("-" * 55)
    
    for threshold in thresholds:
        # 閾値以上のレースのみを対象
        df_filtered = df_top1[df_top1['スコア差'] >= threshold]
        
        if len(df_filtered) == 0:
            continue
        
        # 的中判定（予測1位が実際に1着）
        hits = (df_filtered['確定着順'] == 1).sum()
        total = len(df_filtered)
        accuracy = hits / total * 100 if total > 0 else 0
        
        # 判定
        if accuracy >= 30:
            judgment = "✅ 優秀"
        elif accuracy >= 25:
            judgment = "⭕ 良好"
        else:
            judgment = "⚠️  要改善"
        
        marker = " ← 現在" if threshold == 0.05 else ""
        print(f"{threshold:8.2f} {total:8d} {hits:8d} {accuracy:9.1f}% {judgment}{marker}")
        
        results.append({
            'threshold': threshold,
            'total_races': total,
            'hits': hits,
            'accuracy': accuracy
        })
    
    # グラフ作成
    results_df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 的中率のグラフ
    ax1.plot(results_df['threshold'], results_df['accuracy'], marker='o', linewidth=2, markersize=8)
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='現在の閾値 (0.05)')
    ax1.axhline(y=25, color='green', linestyle=':', linewidth=1, label='目標: 25%')
    ax1.set_xlabel('スコア差閾値', fontsize=12)
    ax1.set_ylabel('的中率 (%)', fontsize=12)
    ax1.set_title('閾値と的中率の関係', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 対象レース数のグラフ
    ax2.plot(results_df['threshold'], results_df['total_races'], marker='s', linewidth=2, markersize=8, color='orange')
    ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='現在の閾値 (0.05)')
    ax2.set_xlabel('スコア差閾値', fontsize=12)
    ax2.set_ylabel('対象レース数', fontsize=12)
    ax2.set_title('閾値と対象レース数の関係', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    output_dir = Path('results')
    output_file = output_dir / 'threshold_optimization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[FILE] 最適化グラフを保存: {output_file}")
    plt.close()
    
    # 推奨閾値を提案
    print("\n" + "=" * 80)
    print("[RECOMMEND] 推奨閾値")
    print("=" * 80)
    
    # 的中率が最大の閾値
    best_accuracy_idx = results_df['accuracy'].idxmax()
    best_threshold = results_df.loc[best_accuracy_idx, 'threshold']
    best_accuracy = results_df.loc[best_accuracy_idx, 'accuracy']
    best_races = results_df.loc[best_accuracy_idx, 'total_races']
    
    print(f"\n✅ 的中率最大: 閾値={best_threshold:.2f}, 的中率={best_accuracy:.1f}%, 対象={best_races}レース")
    
    # 的中率25%以上で最もレース数が多い閾値
    good_results = results_df[results_df['accuracy'] >= 25]
    if len(good_results) > 0:
        best_balance_idx = good_results['total_races'].idxmax()
        balance_threshold = good_results.loc[best_balance_idx, 'threshold']
        balance_accuracy = good_results.loc[best_balance_idx, 'accuracy']
        balance_races = good_results.loc[best_balance_idx, 'total_races']
        
        print(f"⭕ バランス型: 閾値={balance_threshold:.2f}, 的中率={balance_accuracy:.1f}%, 対象={balance_races}レース")
    
    # 現在の閾値(0.05)の評価
    current_result = results_df[results_df['threshold'] == 0.05]
    if len(current_result) > 0:
        current_accuracy = current_result.iloc[0]['accuracy']
        current_races = current_result.iloc[0]['total_races']
        
        print(f"\n📊 現在の閾値(0.05): 的中率={current_accuracy:.1f}%, 対象={current_races}レース")
        
        if current_accuracy < best_accuracy - 2:
            print(f"⚠️  現在の閾値は最適ではありません。閾値を{best_threshold:.2f}に変更することで的中率が{best_accuracy - current_accuracy:.1f}%向上します。")
        else:
            print(f"✅ 現在の閾値は概ね適切です。")
    
    return results_df


if __name__ == "__main__":
    # Phase 1: スコア差の分布を分析
    df_races = analyze_score_diff_distribution()
    
    # Phase 2: 閾値と的中率の関係を分析
    results_df = analyze_threshold_vs_accuracy()
    
    print("\n" + "=" * 80)
    print("[DONE] 分析完了！")
    print("=" * 80)
    print("\n次のステップ:")
    print("1. results/score_diff_distribution.png を確認")
    print("2. results/threshold_optimization.png を確認")
    print("3. 推奨閾値をuniversal_test.pyのmin_score_diffに設定")
    print("4. 再度python universal_test.py multi 2023を実行して効果を検証")
