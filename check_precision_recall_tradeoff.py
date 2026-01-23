"""
Precision/Recall トレードオフの確認スクリプト
閾値を上げた時にPrecision↑、Recall↓となるのが正常
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    df = pd.read_csv('check_results/predicted_results_all.tsv', sep='\t')
    df_target = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    df_target['is_upset'] = (df_target['確定着順'] <= 3).astype(int)

    total_upsets = df_target['is_upset'].sum()
    total_records = len(df_target)
    base_rate = total_upsets / total_records * 100
    
    print(f'7-12番人気: {total_records}頭')
    print(f'実際の穴馬（3着以内）: {total_upsets}頭')
    print(f'穴馬率（ベースライン）: {base_rate:.2f}%')
    print()
    print('=' * 80)
    print('Precision/Recall トレードオフの確認')
    print('=' * 80)
    print()
    print('【正常なトレードオフ】')
    print('  - 閾値↑ → 候補数↓ → Precision↑、Recall↓')
    print('  - 閾値↓ → 候補数↑ → Precision↓、Recall↑')
    print()

    print(f'{"閾値":>6} {"候補":>6} {"TP":>5} {"FP":>5} {"Prec":>8} {"Recall":>8} {"判定":>20}')
    print('-' * 75)

    results = []
    prev_prec = None
    prev_recall = None
    
    for threshold in np.arange(0.05, 0.75, 0.05):
        predicted = df_target['穴馬確率'] >= threshold
        tp = ((predicted) & (df_target['is_upset'] == 1)).sum()
        fp = ((predicted) & (df_target['is_upset'] == 0)).sum()
        
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / total_upsets * 100
        
        # トレードオフ確認
        if prev_prec is not None:
            if prec > prev_prec and recall < prev_recall:
                status = 'OK: Prec UP, Recall DOWN'
            elif prec < prev_prec and recall < prev_recall:
                status = 'WARN: Both DOWN'
            elif prec > prev_prec and recall > prev_recall:
                status = 'GREAT: Both UP'
            elif abs(prec - prev_prec) < 0.5:
                status = 'OK: Prec stable'
            else:
                status = 'WARN: Prec DOWN, Recall DOWN'
        else:
            status = '-'
        
        print(f'{threshold:6.2f} {tp+fp:6d} {tp:5d} {fp:5d} {prec:7.2f}% {recall:7.2f}% {status}')
        
        results.append({
            'threshold': threshold,
            'candidates': tp + fp,
            'tp': tp,
            'fp': fp,
            'precision': prec,
            'recall': recall
        })
        
        prev_prec = prec
        prev_recall = recall
    
    results_df = pd.DataFrame(results)
    
    # 理論値との比較
    print()
    print('=' * 80)
    print('【分析】モデルの識別能力')
    print('=' * 80)
    
    # ベースライン（ランダム）との比較
    print(f'\nベースライン（ランダム予測時のPrecision）: {base_rate:.2f}%')
    print(f'現在のモデル（閾値0.10）: Precision {results_df[results_df["threshold"]==0.10]["precision"].values[0]:.2f}%')
    
    lift = results_df[results_df["threshold"]==0.10]["precision"].values[0] / base_rate
    print(f'リフト値: {lift:.2f}x （ランダムより{lift:.2f}倍良い）')
    
    # PR曲線の面積（AUPRC）簡易計算
    auprc = np.trapz(results_df['precision'], results_df['recall']) / 100
    print(f'\nPR曲線下面積（AUPRC近似）: {abs(auprc):.4f}')
    print(f'（参考: ランダムなら {base_rate/100:.4f}）')
    
    # グラフ作成
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Precision-Recall曲線
    ax1 = axes[0]
    ax1.plot(results_df['recall'], results_df['precision'], 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=base_rate, color='r', linestyle='--', label=f'ベースライン: {base_rate:.1f}%')
    ax1.set_xlabel('Recall (%)', fontsize=12)
    ax1.set_ylabel('Precision (%)', fontsize=12)
    ax1.set_title('Precision-Recall曲線', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 閾値とPrecision/Recallの関係
    ax2 = axes[1]
    ax2.plot(results_df['threshold'], results_df['precision'], 'b-o', linewidth=2, markersize=6, label='Precision')
    ax2.plot(results_df['threshold'], results_df['recall'], 'g-s', linewidth=2, markersize=6, label='Recall')
    ax2.axhline(y=base_rate, color='r', linestyle='--', label=f'ベースライン: {base_rate:.1f}%')
    ax2.set_xlabel('閾値', fontsize=12)
    ax2.set_ylabel('%', fontsize=12)
    ax2.set_title('閾値とPrecision/Recallの関係', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 穴馬確率の分布（穴馬vs非穴馬）
    ax3 = axes[2]
    upsets = df_target[df_target['is_upset'] == 1]['穴馬確率']
    non_upsets = df_target[df_target['is_upset'] == 0]['穴馬確率']
    
    ax3.hist(non_upsets, bins=30, alpha=0.5, label=f'非穴馬 (n={len(non_upsets)})', density=True)
    ax3.hist(upsets, bins=30, alpha=0.5, label=f'穴馬 (n={len(upsets)})', density=True)
    ax3.set_xlabel('穴馬確率', fontsize=12)
    ax3.set_ylabel('密度', fontsize=12)
    ax3.set_title('穴馬確率の分布比較', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('check_results/precision_recall_tradeoff.png', dpi=150, bbox_inches='tight')
    print(f'\n[FILE] グラフ保存: check_results/precision_recall_tradeoff.png')
    
    # 結論
    print()
    print('=' * 80)
    print('【結論】')
    print('=' * 80)
    
    # Precisionが閾値で単調増加しているか確認
    prec_increasing = all(
        results_df.iloc[i]['precision'] <= results_df.iloc[i+1]['precision'] + 1  # 1%の誤差許容
        for i in range(len(results_df)-1)
    )
    
    # Recallが閾値で単調減少しているか確認
    recall_decreasing = all(
        results_df.iloc[i]['recall'] >= results_df.iloc[i+1]['recall'] - 1  # 1%の誤差許容
        for i in range(len(results_df)-1)
    )
    
    if prec_increasing and recall_decreasing:
        print('OK: Precision/Recallのトレードオフは正常です')
        print('   → 閾値を上げるとPrecision↑、Recall↓ の関係が成立')
    else:
        print('WARN: トレードオフに異常があります')
        if not prec_increasing:
            print('   → 閾値を上げてもPrecisionが上がらない区間あり')
        if not recall_decreasing:
            print('   → 閾値を上げてもRecallが下がらない区間あり')
    
    # モデルの品質評価
    print()
    if lift >= 1.5:
        print(f'MODEL: リフト値 {lift:.2f}x - 良好なモデル')
    elif lift >= 1.2:
        print(f'MODEL: リフト値 {lift:.2f}x - 改善の余地あり')
    else:
        print(f'MODEL: リフト値 {lift:.2f}x - モデルの見直しが必要')

if __name__ == '__main__':
    main()
