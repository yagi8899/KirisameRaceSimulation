"""
Phase 2.5: 全モデル統合テストスクリプト

既存の全モデルでPhase 1+2.5統合テストを実行
"""

import sys
from pathlib import Path
from universal_test import predict_with_model
import pandas as pd


def test_all_models(year_start: int = 2023, year_end: int = 2023):
    """
    全モデルでテスト実行
    
    Args:
        year_start: テスト開始年
        year_end: テスト終了年
    """
    print("="*80)
    print(f"Phase 2.5: 全モデル統合テスト（{year_start}-{year_end}年）")
    print("="*80)
    print()
    
    # モデル定義（ファイル名, 競馬場コード, 競争種別, 路面, 最小距離, 最大距離, 説明）
    models = [
        ('hakodate_turf_3ageup_long.sav', '02', '13', 'turf', 1700, 9999, '函館芝中長距離'),
        ('hanshin_turf_3ageup_long.sav', '09', '13', 'turf', 1700, 9999, '阪神芝中長距離'),
        ('hanshin_turf_3ageup_short.sav', '06', '13', 'turf', 1000, 1600, '阪神芝短距離'),
        ('tokyo_turf_3ageup_long.sav', '05', '13', 'turf', 1700, 9999, '東京芝中長距離'),
    ]
    
    results = []
    
    for model_file, track, kyoso, surface, min_dist, max_dist, desc in models:
        model_path = Path('models') / model_file
        
        if not model_path.exists():
            print(f"[SKIP] スキップ: {desc}（モデルファイルなし）")
            print()
            continue
        
        print(f"{'='*80}")
        print(f"テスト: {desc}")
        print(f"{'='*80}")
        
        try:
            output_df, summary_df, race_count = predict_with_model(
                str(model_path),
                track,
                kyoso,
                surface,
                min_dist,
                max_dist,
                year_start,
                year_end
            )
            
            if output_df is not None and len(output_df) > 0:
                # 穴馬分析
                upset = output_df[output_df['穴馬候補'] == 1]
                actual = output_df[(output_df['人気順'] >= 7) & (output_df['人気順'] <= 12) & (output_df['確定着順'].isin([1, 2, 3]))]
                hits = upset[upset['実際の穴馬'] == 1]
                
                # Phase 1統計
                phase1_hitrate = summary_df.loc['単勝', '的中率(%)'] if '単勝' in summary_df.index else 0
                phase1_roi = summary_df.loc['単勝', '回収率(%)'] if '単勝' in summary_df.index else 0
                
                # Phase 2.5統計
                upset_count = len(upset)
                upset_hits = len(hits)
                upset_precision = (upset_hits / upset_count * 100) if upset_count > 0 else 0
                upset_recall = (upset_hits / len(actual) * 100) if len(actual) > 0 else 0
                
                total_bet = upset_count * 100
                total_return = hits['単勝オッズ'].sum() * 100 if len(hits) > 0 else 0
                upset_roi = (total_return / total_bet * 100) if total_bet > 0 else 0
                
                results.append({
                    'モデル': desc,
                    'レース数': race_count,
                    '全頭数': len(output_df),
                    '実穴馬数': len(actual),
                    '穴馬候補': upset_count,
                    '穴馬的中': upset_hits,
                    '穴馬適合率(%)': f"{upset_precision:.2f}",
                    '穴馬再現率(%)': f"{upset_recall:.2f}",
                    '穴馬ROI(%)': f"{upset_roi:.1f}",
                    'Phase1的中率(%)': f"{phase1_hitrate:.2f}",
                    'Phase1回収率(%)': f"{phase1_roi:.2f}"
                })
                
                print(f"\n[OK] 完了")
                print(f"  レース数: {race_count}")
                print(f"  全頭数: {len(output_df)}頭")
                print(f"  実穴馬数: {len(actual)}頭")
                print(f"  穴馬候補: {upset_count}頭")
                print(f"  穴馬的中: {upset_hits}頭")
                print(f"  穴馬適合率: {upset_precision:.2f}%")
                print(f"  穴馬ROI: {upset_roi:.1f}%")
                print(f"  Phase1的中率: {phase1_hitrate:.2f}%")
                print(f"  Phase1回収率: {phase1_roi:.2f}%")
            else:
                print(f"[WARN] データなし")
                
        except Exception as e:
            print(f"[ERROR] エラー: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 結果まとめ
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        
        print("="*80)
        print("全モデルテスト結果サマリー")
        print("="*80)
        print(df_results.to_string(index=False))
        print()
        
        # 結果保存
        output_file = Path('results') / f'all_models_test_{year_start}-{year_end}.tsv'
        df_results.to_csv(output_file, sep='\t', index=False, encoding='utf-8-sig')
        print(f"結果保存: {output_file}")
        
        # 統計サマリー
        print("\n【統計サマリー】")
        print(f"テストモデル数: {len(results)}個")
        print(f"総レース数: {df_results['レース数'].sum()}")
        print(f"総頭数: {df_results['全頭数'].sum()}頭")
        print(f"総実穴馬数: {df_results['実穴馬数'].sum()}頭")
        print(f"総穴馬候補: {df_results['穴馬候補'].sum()}頭")
        print(f"総穴馬的中: {df_results['穴馬的中'].sum()}頭")
        
        total_candidates = df_results['穴馬候補'].sum()
        total_hits = df_results['穴馬的中'].sum()
        total_actual = df_results['実穴馬数'].sum()
        
        if total_candidates > 0:
            avg_precision = total_hits / total_candidates * 100
            avg_recall = total_hits / total_actual * 100 if total_actual > 0 else 0
            print(f"平均適合率: {avg_precision:.2f}%")
            print(f"平均再現率: {avg_recall:.2f}%")
    
    print("\n" + "="*80)
    print("テスト完了")
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # python test_all_models.py 2023
        year_start = year_end = int(sys.argv[1])
    elif len(sys.argv) == 3:
        # python test_all_models.py 2020 2023
        year_start = int(sys.argv[1])
        year_end = int(sys.argv[2])
    else:
        year_start = year_end = 2023
    
    test_all_models(year_start, year_end)
