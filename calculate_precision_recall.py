"""
穴馬予測の正確なPrecision/Recallを計算

使用方法:
    python calculate_precision_recall.py
"""
import pandas as pd
from pathlib import Path

def calculate_metrics():
    """Precision/Recallを正確に計算"""
    
    # データ読み込み
    file_path = Path('check_results/predicted_results_all.tsv')
    
    if not file_path.exists():
        print(f"❌ ファイルが見つかりません: {file_path}")
        return
    
    print(f"📂 ファイル読み込み中: {file_path}")
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
    
    print(f"✅ {len(df):,}レコード読み込み完了")
    print(f"\n📋 列一覧: {df.columns.tolist()}")
    
    # 必要な列があるか確認
    required_cols = ['穴馬候補', '人気順', '確定着順']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n❌ 必要な列がありません: {missing}")
        return
    
    # データクリーニング
    df = df.dropna(subset=['穴馬候補', '人気順', '確定着順'])
    print(f"\n📊 NaN除外後: {len(df):,}レコード")
    
    # 実際の穴馬を定義（7-12番人気で3着以内）
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
    print(f"\n{'='*80}")
    print(f"🎯 穴馬予測の評価結果（7-12番人気で3着以内）")
    print(f"{'='*80}")
    
    print(f"\n📊 混同行列:")
    print(f"  True Positive (TP):  {TP:,}頭  ← 穴馬候補かつ実際の穴馬")
    print(f"  False Positive (FP): {FP:,}頭  ← 穴馬候補だが外れ")
    print(f"  False Negative (FN): {FN:,}頭  ← 見逃した穴馬")
    print(f"  True Negative (TN):  {TN:,}頭  ← 正しく除外")
    
    print(f"\n📈 評価指標:")
    print(f"  穴馬候補総数: {穴馬候補総数:,}頭 (TP + FP)")
    print(f"  実際の穴馬数: {実際の穴馬総数:,}頭 (TP + FN)")
    print(f"  予測的中数:   {TP:,}頭 (TP)")
    
    print(f"\n🎯 Precision（適合率）: {Precision:.2f}%")
    print(f"   = 予測的中数 / 穴馬候補総数")
    print(f"   = {TP:,} / {穴馬候補総数:,}")
    print(f"   → 穴馬候補のうち{Precision:.2f}%が実際に好走")
    
    print(f"\n🔍 Recall（再現率）: {Recall:.2f}%")
    print(f"   = 予測的中数 / 実際の穴馬総数")
    print(f"   = {TP:,} / {実際の穴馬総数:,}")
    print(f"   → 実際の穴馬の{Recall:.2f}%を検出")
    
    print(f"\n⚖️ F1 Score: {F1:.2f}")
    print(f"   = 2 × Precision × Recall / (Precision + Recall)")
    
    # Phase評価
    print(f"\n{'='*80}")
    print(f"📋 Phase目標との比較")
    print(f"{'='*80}")
    
    print(f"\n【Phase 1目標: Precision 8%以上】")
    if Precision >= 8.0:
        print(f"  ✅ 達成！ (Precision {Precision:.2f}% >= 8.0%)")
    else:
        print(f"  ❌ 未達成 (Precision {Precision:.2f}% < 8.0%)")
    
    print(f"\n【Phase 2目標: Precision 10%以上】")
    if Precision >= 10.0:
        print(f"  ✅ 達成！ (Precision {Precision:.2f}% >= 10.0%)")
    else:
        print(f"  ⚠️ 未達成 (Precision {Precision:.2f}% < 10.0%)")
    
    print(f"\n【Phase 3目標: Precision 12%以上】")
    if Precision >= 12.0:
        print(f"  ✅ 達成！ (Precision {Precision:.2f}% >= 12.0%)")
    else:
        print(f"  ⚠️ 未達成 (Precision {Precision:.2f}% < 12.0%)")
    
    # 候補数評価
    print(f"\n【候補数の評価】")
    if 穴馬候補総数 <= 3000:
        print(f"  ✅ 実用的 ({穴馬候補総数:,}頭 <= 3,000頭)")
    elif 穴馬候補総数 <= 5000:
        print(f"  ⚠️ やや多い ({穴馬候補総数:,}頭)")
    else:
        print(f"  ❌ 多すぎ ({穴馬候補総数:,}頭 > 5,000頭)")
    
    # 追加分析：3着以内率
    候補で3着以内 = (df['穴馬候補'] == 1) & (df['確定着順'] <= 3)
    候補3着以内数 = 候補で3着以内.sum()
    候補3着以内率 = (候補3着以内数 / 穴馬候補総数 * 100) if 穴馬候補総数 > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"📊 追加分析")
    print(f"{'='*80}")
    print(f"\n穴馬候補のうち3着以内（人気順位問わず）:")
    print(f"  {候補3着以内数:,}頭 / {穴馬候補総数:,}頭 = {候補3着以内率:.2f}%")
    print(f"\n  ※ これが「穴馬候補的中率」として表示されていた可能性")
    print(f"  ※ 本来のPrecision（7-12番人気かつ3着以内）は {Precision:.2f}%")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    calculate_metrics()
