#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
穴馬予測分析スクリプト

オッズ10倍以上の穴馬に対するモデルの予測性能を診断する。
- 穴馬の予測順位分布
- 人気順との相関
- スキップ理由の分析
- 的中パターンの分析

Usage:
    python analyze_longshot_predictions.py
    python analyze_longshot_predictions.py --odds_threshold 15.0
    python analyze_longshot_predictions.py --file results/predicted_results_tokyo_turf_3ageup_long_trainunknown_test2023_all.tsv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def analyze_longshot_predictions(
    all_file: str = 'results/predicted_results_all.tsv',
    skipped_file: str = 'results/predicted_results_skipped.tsv',
    odds_threshold: float = 10.0
):
    """
    穴馬の予測分析を実行
    
    Args:
        all_file (str): 全レース結果ファイル
        skipped_file (str): スキップレース結果ファイル
        odds_threshold (float): 穴馬の基準オッズ（デフォルト: 10倍）
    """
    
    print("=" * 80)
    print(f"🔍 穴馬予測分析レポート（オッズ{odds_threshold}倍以上）")
    print("=" * 80)
    
    # ファイル存在チェック
    if not Path(all_file).exists():
        print(f"❌ ファイルが見つかりません: {all_file}")
        return
    
    # 全レースデータ読み込み
    df_all = pd.read_csv(all_file, sep='\t', encoding='utf-8-sig')
    
    # スキップデータ読み込み（存在する場合）
    df_skipped = None
    if Path(skipped_file).exists():
        df_skipped = pd.read_csv(skipped_file, sep='\t', encoding='utf-8-sig')
    
    # カラム名の統一
    column_mapping = {
        '単勝オッズ': 'tansho_odds',
        '予測順位': 'predicted_rank',
        '人気順': 'popularity_rank',
        '確定着順': 'actual_chakujun',
        'スキップ理由': 'skip_reason',
        '購入推奨': 'should_buy'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df_all.columns:
            df_all[new_col] = df_all[old_col]
    
    # 穴馬を抽出
    longshots = df_all[df_all['tansho_odds'] >= odds_threshold].copy()
    
    print(f"\n📊 データ概要:")
    print(f"  - 全馬数: {len(df_all)}頭")
    print(f"  - 穴馬数（オッズ{odds_threshold}倍以上）: {len(longshots)}頭 ({len(longshots)/len(df_all)*100:.1f}%)")
    
    if len(longshots) == 0:
        print(f"\n⚠️ オッズ{odds_threshold}倍以上の馬が見つかりませんでした。")
        return
    
    # ========================================
    # 1. 予測順位の分布
    # ========================================
    print("\n" + "=" * 80)
    print("📈 1. 穴馬の予測順位分布")
    print("=" * 80)
    
    rank_dist = longshots['predicted_rank'].value_counts().sort_index()
    print("\n予測順位 | 頭数 | 割合")
    print("-" * 40)
    for rank, count in rank_dist.items():
        pct = count / len(longshots) * 100
        bar = "█" * int(pct / 2)
        print(f"{int(rank):3d}位   | {count:4d}頭 | {pct:5.1f}% {bar}")
    
    # 上位予測の統計
    top3_count = len(longshots[longshots['predicted_rank'] <= 3])
    top5_count = len(longshots[longshots['predicted_rank'] <= 5])
    
    print(f"\n✅ 重要指標:")
    print(f"  - 予測1-3位の穴馬: {top3_count}頭 ({top3_count/len(longshots)*100:.1f}%)")
    print(f"  - 予測1-5位の穴馬: {top5_count}頭 ({top5_count/len(longshots)*100:.1f}%)")
    
    if top3_count / len(longshots) > 0.15:
        print("  💡 判定: モデルは穴馬をある程度捉えている → フィルタ調整が有効")
    else:
        print("  ⚠️ 判定: モデルが穴馬を捉えられていない → 特徴量改善が必要")
    
    # ========================================
    # 2. 人気順との相関
    # ========================================
    print("\n" + "=" * 80)
    print("📊 2. 穴馬の人気順分布")
    print("=" * 80)
    
    pop_dist = longshots['popularity_rank'].value_counts().sort_index()
    print("\n人気順 | 頭数 | 割合")
    print("-" * 40)
    for rank, count in pop_dist.head(10).items():
        pct = count / len(longshots) * 100
        bar = "█" * int(pct / 2)
        print(f"{int(rank):2d}番人気 | {count:4d}頭 | {pct:5.1f}% {bar}")
    
    # 人気順1-3位の穴馬
    popular_longshots = len(longshots[longshots['popularity_rank'] <= 3])
    print(f"\n✅ 人気1-3位の穴馬: {popular_longshots}頭 ({popular_longshots/len(longshots)*100:.1f}%)")
    
    # ========================================
    # 3. スキップ理由の分析
    # ========================================
    if df_skipped is not None and 'skip_reason' in df_skipped.columns:
        print("\n" + "=" * 80)
        print("🚫 3. 穴馬のスキップ理由分析")
        print("=" * 80)
        
        # スキップデータも同様にカラム統一
        for old_col, new_col in column_mapping.items():
            if old_col in df_skipped.columns:
                df_skipped[new_col] = df_skipped[old_col]
        
        skipped_longshots = df_skipped[df_skipped['tansho_odds'] >= odds_threshold].copy()
        
        if len(skipped_longshots) > 0:
            skip_reasons = skipped_longshots['skip_reason'].value_counts()
            
            print(f"\nスキップされた穴馬: {len(skipped_longshots)}頭")
            print("\nスキップ理由 | 頭数 | 割合")
            print("-" * 50)
            for reason, count in skip_reasons.items():
                pct = count / len(skipped_longshots) * 100
                reason_jp = {
                    'low_score_diff': '予測スコア差不足',
                    'low_predicted_rank': '予測順位低い',
                    'low_popularity': '人気順位低い',
                    'odds_too_low': 'オッズ低すぎ',
                    'odds_too_high': 'オッズ高すぎ',
                    'multiple_conditions': '複合条件'
                }.get(reason, reason)
                print(f"{reason_jp:15s} | {count:4d}頭 | {pct:5.1f}%")
            
            # 最も多いスキップ理由
            main_reason = skip_reasons.idxmax()
            main_reason_jp = {
                'low_score_diff': '予測スコア差不足',
                'low_predicted_rank': '予測順位が低い',
                'low_popularity': '人気順位が低い',
                'odds_too_low': 'オッズが低すぎる',
                'odds_too_high': 'オッズが高すぎる',
                'multiple_conditions': '複合条件'
            }.get(main_reason, main_reason)
            
            print(f"\n💡 主なスキップ理由: {main_reason_jp}")
            
            if main_reason == 'low_popularity':
                print("   → フィルタの popularity_rank_max を緩めれば穴馬を購入対象に含められる")
            elif main_reason == 'low_predicted_rank':
                print("   → モデルが穴馬を低評価 → 特徴量改善が必要")
            elif main_reason == 'odds_too_high':
                print("   → max_odds を上げれば大穴も対象になる")
    
    # ========================================
    # 4. 的中分析
    # ========================================
    print("\n" + "=" * 80)
    print("🎯 4. 穴馬の的中分析")
    print("=" * 80)
    
    # 確定着順がある場合
    if 'actual_chakujun' in longshots.columns:
        longshots_with_result = longshots.dropna(subset=['actual_chakujun'])
        
        if len(longshots_with_result) > 0:
            # 着順ごとの集計
            win_longshots = len(longshots_with_result[longshots_with_result['actual_chakujun'] == 1])
            place_longshots = len(longshots_with_result[longshots_with_result['actual_chakujun'] <= 3])
            
            print(f"\n的中実績:")
            print(f"  - 1着（単勝的中）: {win_longshots}頭 ({win_longshots/len(longshots_with_result)*100:.1f}%)")
            print(f"  - 3着以内（複勝的中）: {place_longshots}頭 ({place_longshots/len(longshots_with_result)*100:.1f}%)")
            
            # 的中した穴馬の予測順位
            if win_longshots > 0:
                win_longshots_df = longshots_with_result[longshots_with_result['actual_chakujun'] == 1]
                print(f"\n1着穴馬の予測順位:")
                win_rank_dist = win_longshots_df['predicted_rank'].value_counts().sort_index()
                for rank, count in win_rank_dist.items():
                    print(f"  - 予測{int(rank)}位: {count}頭")
                
                avg_win_rank = win_longshots_df['predicted_rank'].mean()
                print(f"  平均予測順位: {avg_win_rank:.1f}位")
                
                if avg_win_rank <= 3:
                    print("  ✅ 的中した穴馬の多くを予測上位で捉えている！")
                else:
                    print("  ⚠️ 的中した穴馬を予測下位に置いている...")
    
    # ========================================
    # 5. 予測順位×人気順のクロス分析
    # ========================================
    print("\n" + "=" * 80)
    print("🔍 5. 予測順位 × 人気順のクロス分析")
    print("=" * 80)
    
    # 予測上位（1-3位）かつ人気薄（4位以下）の穴馬
    predicted_top_unpopular = longshots[
        (longshots['predicted_rank'] <= 3) & 
        (longshots['popularity_rank'] > 3)
    ]
    
    print(f"\n🎯 重要セグメント: 予測上位（1-3位）× 人気薄（4位以下）")
    print(f"  - 該当馬数: {len(predicted_top_unpopular)}頭")
    
    if len(predicted_top_unpopular) > 0:
        print(f"  - 平均オッズ: {predicted_top_unpopular['tansho_odds'].mean():.1f}倍")
        
        if 'actual_chakujun' in predicted_top_unpopular.columns:
            wins = len(predicted_top_unpopular[predicted_top_unpopular['actual_chakujun'] == 1])
            if len(predicted_top_unpopular) > 0:
                win_rate = wins / len(predicted_top_unpopular) * 100
                print(f"  - 単勝的中率: {win_rate:.1f}% ({wins}頭/{len(predicted_top_unpopular)}頭)")
                
                avg_odds = predicted_top_unpopular['tansho_odds'].mean()
                expected_return = win_rate / 100 * avg_odds * 100
                print(f"  - 期待回収率: {expected_return:.1f}%")
                
                if expected_return > 110:
                    print("\n  🔥 このセグメントは高期待値！フィルタ調整で購入対象にすべき！")
        
        print(f"\n  💡 提案: popularity_rank_max を 3 → 6 に変更すれば、これらの馬を購入できる")
    
    # ========================================
    # まとめ
    # ========================================
    print("\n" + "=" * 80)
    print("📝 診断結果まとめ")
    print("=" * 80)
    
    top3_ratio = top3_count / len(longshots)
    
    print(f"\n【モデルの穴馬予測能力】")
    if top3_ratio >= 0.15:
        print(f"  ✅ 予測1-3位に{top3_count}頭（{top3_ratio*100:.1f}%）の穴馬 → モデルは機能している")
        print(f"  💡 推奨: フィルタ調整で購入対象を拡大")
        print(f"     - popularity_rank_max: 3 → 6")
        print(f"     - max_odds: 20 → 30")
    elif top3_ratio >= 0.08:
        print(f"  ⚠️ 予測1-3位に{top3_count}頭（{top3_ratio*100:.1f}%）の穴馬 → やや弱い")
        print(f"  💡 推奨: フィルタ調整 + 特徴量改善の両方")
    else:
        print(f"  🚨 予測1-3位に{top3_count}頭（{top3_ratio*100:.1f}%）の穴馬のみ → モデルが捉えられていない")
        print(f"  💡 推奨: 特徴量改善を優先")
        print(f"     - 人気と実力の乖離を捉える特徴量")
        print(f"     - 展開・ペース予測")
        print(f"     - 前走敗因分析")
    
    print(f"\n【次のアクション】")
    if top3_ratio >= 0.15 and len(predicted_top_unpopular) >= 10:
        print(f"  1. popularity_rank_max を 6 に変更してテスト実行")
        print(f"  2. 期待回収率を確認")
        print(f"  3. 良好なら本運用に採用")
    else:
        print(f"  1. 穴馬特化の特徴量を追加:")
        print(f"     - オッズと予測確率の乖離")
        print(f"     - 前走大敗からの巻き返しパターン")
        print(f"     - 騎手変更効果")
        print(f"  2. モデル再学習")
        print(f"  3. 再度診断を実行")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='穴馬予測分析スクリプト')
    parser.add_argument('--file', type=str, default='results/predicted_results_all.tsv',
                        help='分析対象ファイル（デフォルト: results/predicted_results_all.tsv）')
    parser.add_argument('--skipped', type=str, default='results/predicted_results_skipped.tsv',
                        help='スキップファイル（デフォルト: results/predicted_results_skipped.tsv）')
    parser.add_argument('--odds_threshold', type=float, default=10.0,
                        help='穴馬の基準オッズ（デフォルト: 10.0倍）')
    
    args = parser.parse_args()
    
    analyze_longshot_predictions(
        all_file=args.file,
        skipped_file=args.skipped,
        odds_threshold=args.odds_threshold
    )


if __name__ == '__main__':
    main()
