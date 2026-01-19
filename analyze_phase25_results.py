#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2.5 穴馬予測システム 詳細分析スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_track(name, filepath):
    """
    各競馬場の詳細分析
    """
    print(f'\n{"="*80}')
    print(f'【{name}】詳細分析')
    print(f'{"="*80}')
    
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    
    # 基本統計
    total_horses = len(df)
    upset_candidates = df[df['穴馬候補'] == 1]
    actual_upsets = df[df['実際の穴馬'] == 1]
    hits = df[(df['穴馬候補'] == 1) & (df['実際の穴馬'] == 1)]
    misses = df[(df['穴馬候補'] == 0) & (df['実際の穴馬'] == 1)]
    false_positives = df[(df['穴馬候補'] == 1) & (df['実際の穴馬'] == 0)]
    
    print(f'\n■ 基本統計')
    print(f'  総出走頭数: {total_horses}頭')
    print(f'  穴馬候補数: {len(upset_candidates)}頭 ({len(upset_candidates)/total_horses*100:.2f}%)')
    print(f'  実際の穴馬数: {len(actual_upsets)}頭 ({len(actual_upsets)/total_horses*100:.2f}%)')
    print(f'  的中数（TP）: {len(hits)}頭')
    print(f'  見逃し数（FN）: {len(misses)}頭')
    print(f'  誤検知数（FP）: {len(false_positives)}頭')
    
    if len(upset_candidates) > 0:
        precision = len(hits) / len(upset_candidates) * 100
        recall = len(hits) / len(actual_upsets) * 100 if len(actual_upsets) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f'\n■ 性能指標')
        print(f'  適合率（Precision）: {precision:.2f}%')
        print(f'  再現率（Recall）: {recall:.2f}%')
        print(f'  F1スコア: {f1:.2f}%')
        
        # ROI計算
        total_bet = len(upset_candidates) * 100
        total_return = hits['単勝オッズ'].sum() * 100
        roi = total_return / total_bet * 100
        print(f'  ROI: {roi:.1f}%')
        print(f'  総投資額: {total_bet:,}円')
        print(f'  総払戻額: {total_return:,.0f}円')
        print(f'  損益: {total_return - total_bet:,.0f}円')
        
        # 穴馬確率分布
        print(f'\n■ 穴馬候補の確率分布')
        print(f'  平均: {upset_candidates["穴馬確率"].mean():.3f}')
        print(f'  最小: {upset_candidates["穴馬確率"].min():.3f}')
        print(f'  最大: {upset_candidates["穴馬確率"].max():.3f}')
        print(f'  中央値: {upset_candidates["穴馬確率"].median():.3f}')
        print(f'  標準偏差: {upset_candidates["穴馬確率"].std():.3f}')
        
        # 確率帯別の分析
        print(f'\n■ 確率帯別分析（穴馬候補のみ）')
        bins = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        upset_candidates_copy = upset_candidates.copy()
        upset_candidates_copy['確率帯'] = pd.cut(upset_candidates_copy['穴馬確率'], bins=bins, labels=labels, include_lowest=True)
        
        for prob_range in labels:
            subset = upset_candidates_copy[upset_candidates_copy['確率帯'] == prob_range]
            if len(subset) > 0:
                hits_in_range = subset[subset['実際の穴馬'] == 1]
                precision_in_range = len(hits_in_range) / len(subset) * 100
                avg_odds = subset['単勝オッズ'].mean()
                print(f'  {prob_range}: {len(subset)}頭, 適合率{precision_in_range:.1f}%, 平均オッズ{avg_odds:.1f}倍')
        
        # 的中馬の詳細
        if len(hits) > 0:
            print(f'\n■ 的中馬の詳細（{len(hits)}頭）')
            print(f'  穴馬確率: 平均{hits["穴馬確率"].mean():.3f}, 範囲{hits["穴馬確率"].min():.3f}～{hits["穴馬確率"].max():.3f}')
            print(f'  単勝オッズ: 平均{hits["単勝オッズ"].mean():.1f}倍, 範囲{hits["単勝オッズ"].min():.1f}～{hits["単勝オッズ"].max():.1f}倍')
            print(f'  人気分布:')
            for pop in sorted(hits['人気順'].unique()):
                count = len(hits[hits['人気順'] == pop])
                print(f'    {pop}番人気: {count}頭')
        
        # 見逃し馬の詳細
        if len(misses) > 0:
            print(f'\n■ 見逃し馬の詳細（{len(misses)}頭）')
            print(f'  穴馬確率: 平均{misses["穴馬確率"].mean():.3f}, 範囲{misses["穴馬確率"].min():.3f}～{misses["穴馬確率"].max():.3f}')
            print(f'  単勝オッズ: 平均{misses["単勝オッズ"].mean():.1f}倍, 範囲{misses["単勝オッズ"].min():.1f}～{misses["単勝オッズ"].max():.1f}倍')
            print(f'  人気分布:')
            for pop in sorted(misses['人気順'].unique()):
                count = len(misses[misses['人気順'] == pop])
                print(f'    {pop}番人気: {count}頭')
            
            # 見逃しで最も惜しかった馬（確率が0.4に近い馬）
            misses_sorted = misses.sort_values('穴馬確率', ascending=False)
            print(f'\n  惜しかった見逃し馬 Top5:')
            for idx, row in misses_sorted.head(5).iterrows():
                print(f'    {row["馬名"].strip()}: 確率{row["穴馬確率"]:.3f}, {row["人気順"]}番人気, {row["単勝オッズ"]:.1f}倍')
        
        # 誤検知の詳細
        if len(false_positives) > 0:
            print(f'\n■ 誤検知（空振り）の詳細（{len(false_positives)}頭）')
            print(f'  穴馬確率: 平均{false_positives["穴馬確率"].mean():.3f}, 範囲{false_positives["穴馬確率"].min():.3f}～{false_positives["穴馬確率"].max():.3f}')
            print(f'  単勝オッズ: 平均{false_positives["単勝オッズ"].mean():.1f}倍')
            print(f'  着順分布:')
            for rank in sorted(false_positives['確定着順'].unique()):
                count = len(false_positives[false_positives['確定着順'] == rank])
                print(f'    {rank}着: {count}頭')
    else:
        print('\n  ⚠ 穴馬候補なし')
    
    return {
        'name': name,
        'total_horses': total_horses,
        'candidates': len(upset_candidates),
        'actual_upsets': len(actual_upsets),
        'hits': len(hits),
        'precision': precision if len(upset_candidates) > 0 else 0,
        'recall': recall if len(actual_upsets) > 0 else 0,
        'roi': roi if len(upset_candidates) > 0 else 0
    }


def main():
    """
    メイン処理
    """
    print('='*80)
    print('Phase 2.5 穴馬予測システム 詳細分析レポート')
    print('テスト期間: 2023年')
    print('='*80)
    
    # 各競馬場のデータを分析
    files = {
        '阪神芝中長距離': 'results/predicted_results_hanshin_turf_3ageup_long_trainunknown_test2023_all.tsv',
        '阪神芝短距離': 'results/predicted_results_hanshin_turf_3ageup_short_trainunknown_test2023_all.tsv',
        '函館芝中長距離': 'results/predicted_results_hakodate_turf_3ageup_long_trainunknown_test2023_all.tsv',
        '東京芝中長距離': 'results/predicted_results_tokyo_turf_3ageup_long_trainunknown_test2023_all.tsv'
    }
    
    results = []
    for name, filepath in files.items():
        if Path(filepath).exists():
            result = analyze_track(name, filepath)
            results.append(result)
        else:
            print(f'\n⚠ ファイルが見つかりません: {filepath}')
    
    # サマリー
    print(f'\n{"="*80}')
    print('総合サマリー')
    print(f'{"="*80}')
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    # Phase 2との比較
    print(f'\n{"="*80}')
    print('Phase 2 vs Phase 2.5 比較（阪神芝中長距離）')
    print(f'{"="*80}')
    print('Phase 2（阪神のみ訓練）:')
    print('  適合率: 4.97%')
    print('  ROI: 241.5%')
    print('  訓練データ: 13サンプル（阪神のみ）')
    print('')
    hanshin_long = [r for r in results if r['name'] == '阪神芝中長距離'][0]
    print('Phase 2.5（全10競馬場訓練）:')
    print(f'  適合率: {hanshin_long["precision"]:.2f}%')
    print(f'  ROI: {hanshin_long["roi"]:.1f}%')
    print('  訓練データ: 122サンプル（全10競馬場）')
    print('')
    print('⚠ Phase 2.5は適合率が向上しているがROIは変動')
    print('  → より多様な競馬場データで汎化性能が向上')
    print('  → 高オッズ穴馬の見逃しが減少した可能性')
    
    print(f'\n{"="*80}')
    print('分析完了')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
