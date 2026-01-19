import pandas as pd
from pathlib import Path

# 最新の予測結果を読み込み
results_dir = Path('results')
result_files = list(results_dir.glob('predicted_results_hanshin_turf_3ageup_short_*.tsv'))

if not result_files:
    print("予測結果ファイルが見つかりません")
    exit()

# 最新ファイルを使用
latest_file = sorted(result_files)[-1]
print(f"読み込むファイル: {latest_file}")

df = pd.read_csv(latest_file, sep='\t')

# 穴馬候補を抽出
candidates = df[df['穴馬候補'] == 1].copy()

print(f"\n{'='*80}")
print(f"穴馬候補の詳細分析")
print(f"{'='*80}")
print(f"\n総候補数: {len(candidates)}頭")

if len(candidates) > 0:
    # 人気分布
    print(f"\n【人気分布】")
    print(candidates['人気順'].value_counts().sort_index())
    
    # 着順分布
    print(f"\n【着順分布】")
    print(candidates['確定着順'].value_counts().sort_index())
    
    # 3着以内の候補
    hit_candidates = candidates[candidates['確定着順'].isin([1, 2, 3])]
    print(f"\n【3着以内の候補】: {len(hit_candidates)}頭")
    if len(hit_candidates) > 0:
        print(hit_candidates[['競馬場', 'レース番号', '馬名', '人気順', '確定着順', '単勝オッズ', '穴馬確率']].to_string(index=False))
    
    # 実際の穴馬（7-12番人気で3着以内）
    actual_upsets = df[(df['人気順'] >= 7) & (df['人気順'] <= 12) & (df['確定着順'].isin([1, 2, 3]))]
    print(f"\n【実際の穴馬（7-12番人気・3着以内）】: {len(actual_upsets)}頭")
    if len(actual_upsets) > 0:
        print(actual_upsets[['競馬場', 'レース番号', '馬名', '人気順', '確定着順', '単勝オッズ', '穴馬確率', '穴馬候補']].to_string(index=False))
    
    # 確率分布
    print(f"\n【候補の確率分布】")
    print(f"  最小: {candidates['穴馬確率'].min():.4f}")
    print(f"  最大: {candidates['穴馬確率'].max():.4f}")
    print(f"  平均: {candidates['穴馬確率'].mean():.4f}")
    print(f"  中央値: {candidates['穴馬確率'].median():.4f}")
    
    # 確率帯別の的中状況
    print(f"\n【確率帯別の候補数と的中数】")
    candidates['prob_range'] = pd.cut(candidates['穴馬確率'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    prob_analysis = candidates.groupby('prob_range').agg({
        '穴馬確率': 'count',
        '実際の穴馬': 'sum'
    }).rename(columns={'穴馬確率': '候補数', '実際の穴馬': '的中数'})
    print(prob_analysis)
    
    # 見逃した穴馬（穴馬候補 == 0だが実際は穴馬）
    missed_upsets = df[(df['実際の穴馬'] == 1) & (df['穴馬候補'] == 0)]
    print(f"\n【見逃した穴馬】: {len(missed_upsets)}頭")
    if len(missed_upsets) > 0:
        print(f"\n確率分布:")
        print(f"  最小: {missed_upsets['穴馬確率'].min():.4f}")
        print(f"  最大: {missed_upsets['穴馬確率'].max():.4f}")
        print(f"  平均: {missed_upsets['穴馬確率'].mean():.4f}")
        print(f"  中央値: {missed_upsets['穴馬確率'].median():.4f}")
        
        print(f"\nサンプル（上位5頭）:")
        print(missed_upsets.nlargest(5, '穴馬確率')[['競馬場', 'レース番号', '馬名', '人気順', '確定着順', '単勝オッズ', '穴馬確率']].to_string(index=False))
    
    # 推奨閾値の計算
    print(f"\n{'='*80}")
    print(f"閾値最適化分析")
    print(f"{'='*80}")
    
    all_horses = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    print(f"\n{'閾値':>6} {'候補数':>8} {'的中数':>8} {'適合率':>8} {'再現率':>8} {'F1':>8}")
    print("-" * 60)
    
    for threshold in thresholds:
        pred_upsets = all_horses[all_horses['穴馬確率'] > threshold]
        true_upsets = all_horses[all_horses['実際の穴馬'] == 1]
        
        tp = len(pred_upsets[pred_upsets['実際の穴馬'] == 1])
        fp = len(pred_upsets) - tp
        fn = len(true_upsets) - tp
        
        precision = (tp / len(pred_upsets) * 100) if len(pred_upsets) > 0 else 0
        recall = (tp / len(true_upsets) * 100) if len(true_upsets) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>6.2f} {len(pred_upsets):>8} {tp:>8} {precision:>7.2f}% {recall:>7.2f}% {f1:>7.2f}")

else:
    print("候補馬が見つかりませんでした")

print(f"\n{'='*80}")
