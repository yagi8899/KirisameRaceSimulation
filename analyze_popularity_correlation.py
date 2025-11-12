#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人気と予測の相関分析
Phase 2-1改の予測が人気に依存しすぎていないか検証
"""

import pandas as pd
import numpy as np

def analyze_popularity_correlation():
    """人気と予測の相関を分析"""
    
    # 最新のバックテスト結果を読み込み
    try:
        # TSVの列名を推測
        df = pd.read_csv('results/predicted_results.tsv', sep='\t', header=None)
        
        # 列名を推測(universal_test.pyの出力形式から)
        # jocode, year, monthday, racenum, baba_jotai, kyori, umaban, bamei, 
        # tansho_odds, tansho_ninkijun_numeric, kakutei_chakujun, predicted_rank, ...
        print(f"[INFO] 総行数: {len(df)}")
        print(f"[INFO] 列数: {len(df.columns)}")
        
        # 必要な列を抽出(列インデックスで)
        # 8: tansho_odds, 9: tansho_ninkijun_numeric, 10: kakutei_chakujun, 11: predicted_rank
        if len(df.columns) >= 12:
            df_analysis = pd.DataFrame({
                'tansho_odds': df[8],
                'popularity': df[9],  # 人気順位
                'actual_rank': df[10],  # 実際の着順
                'predicted_rank': df[11]  # 予測順位
            })
            
            # 数値型に変換
            df_analysis['popularity'] = pd.to_numeric(df_analysis['popularity'], errors='coerce')
            df_analysis['predicted_rank'] = pd.to_numeric(df_analysis['predicted_rank'], errors='coerce')
            df_analysis['actual_rank'] = pd.to_numeric(df_analysis['actual_rank'], errors='coerce')
            
            # 欠損値を削除
            df_analysis = df_analysis.dropna()
            
            print(f"\n[INFO] 分析対象データ数: {len(df_analysis)}")
            
            # ===== 1. 人気順位と予測順位の相関 =====
            correlation = df_analysis['popularity'].corr(df_analysis['predicted_rank'])
            print(f"\n{'='*60}")
            print(f"[分析1] 人気順位 vs 予測順位の相関係数")
            print(f"{'='*60}")
            print(f"相関係数: {correlation:.4f}")
            if correlation > 0.9:
                print("⚠️  超高相関! 予測が人気にほぼ完全依存している可能性が高い")
            elif correlation > 0.7:
                print("⚠️  高相関。予測が人気に強く依存している")
            elif correlation > 0.5:
                print("✓  中程度の相関。人気を参考にしているが独自判断もある")
            else:
                print("✓  低相関。予測が人気から独立している")
            
            # ===== 2. 人気1番が予測1番になった割合 =====
            popular_1_df = df_analysis[df_analysis['popularity'] == 1]
            popular_1_predicted_1 = len(popular_1_df[popular_1_df['predicted_rank'] == 1])
            popular_1_total = len(popular_1_df)
            
            print(f"\n{'='*60}")
            print(f"[分析2] 人気1番馬の予測結果")
            print(f"{'='*60}")
            print(f"人気1番が予測1番になった割合: {popular_1_predicted_1}/{popular_1_total} = {100*popular_1_predicted_1/popular_1_total:.2f}%")
            
            if popular_1_total > 0:
                print(f"\n人気1番の予測順位分布:")
                print(popular_1_df['predicted_rank'].value_counts().sort_index().head(10))
            
            # ===== 3. 予測1番の人気順位分布 =====
            predicted_1_df = df_analysis[df_analysis['predicted_rank'] == 1]
            
            print(f"\n{'='*60}")
            print(f"[分析3] 予測1番に選んだ馬の人気順位分布")
            print(f"{'='*60}")
            print(f"予測1番の総数: {len(predicted_1_df)}")
            print(f"\n人気順位分布:")
            pop_dist = predicted_1_df['popularity'].value_counts().sort_index()
            for rank, count in pop_dist.head(10).items():
                percentage = 100 * count / len(predicted_1_df)
                print(f"  人気{int(rank):2d}番: {count:3d}頭 ({percentage:5.2f}%)")
            
            # 人気1-3番を予測1番に選んだ割合
            top3_popular = len(predicted_1_df[predicted_1_df['popularity'] <= 3])
            top3_percentage = 100 * top3_popular / len(predicted_1_df)
            print(f"\n人気1-3番を予測1番に選んだ割合: {top3_popular}/{len(predicted_1_df)} = {top3_percentage:.2f}%")
            
            if top3_percentage > 80:
                print("⚠️  予測1番の80%以上が人気上位! 人気依存度が高い")
            elif top3_percentage > 60:
                print("⚠️  予測1番の60%以上が人気上位。やや人気依存気味")
            else:
                print("✓  予測1番の選択に多様性がある")
            
            # ===== 4. 人気と予測が完全一致した割合 =====
            match_count = len(df_analysis[df_analysis['popularity'] == df_analysis['predicted_rank']])
            match_rate = 100 * match_count / len(df_analysis)
            
            print(f"\n{'='*60}")
            print(f"[分析4] 人気順位と予測順位の完全一致率")
            print(f"{'='*60}")
            print(f"一致数: {match_count}/{len(df_analysis)} = {match_rate:.2f}%")
            
            if match_rate > 50:
                print("⚠️  50%以上が完全一致! 予測が人気のコピーになっている")
            elif match_rate > 30:
                print("⚠️  30%以上が一致。人気依存度が高い")
            else:
                print("✓  一致率は低く、独自の予測をしている")
            
            # ===== 5. 的中したケースの分析 =====
            hit_df = df_analysis[df_analysis['predicted_rank'] == df_analysis['actual_rank']]
            
            print(f"\n{'='*60}")
            print(f"[分析5] 的中したケースの人気順位分布")
            print(f"{'='*60}")
            print(f"的中数: {len(hit_df)}")
            print(f"的中率: {100*len(hit_df)/len(df_analysis):.2f}%")
            
            if len(hit_df) > 0:
                print(f"\n的中したケースの人気順位分布:")
                hit_pop_dist = hit_df['popularity'].value_counts().sort_index()
                for rank, count in hit_pop_dist.head(10).items():
                    percentage = 100 * count / len(hit_df)
                    print(f"  人気{int(rank):2d}番: {count:2d}回 ({percentage:5.2f}%)")
                
                # 的中ケースの人気1-3番の割合
                hit_top3 = len(hit_df[hit_df['popularity'] <= 3])
                hit_top3_percentage = 100 * hit_top3 / len(hit_df)
                print(f"\n的中の{hit_top3_percentage:.2f}%が人気1-3番")
            
            # ===== 6. 穴馬(人気5番以下)の予測上位率 =====
            underdog_df = df_analysis[df_analysis['popularity'] >= 5]
            underdog_predicted_top3 = len(underdog_df[underdog_df['predicted_rank'] <= 3])
            
            print(f"\n{'='*60}")
            print(f"[分析6] 穴馬(人気5番以下)の予測")
            print(f"{'='*60}")
            print(f"人気5番以下の総数: {len(underdog_df)}")
            print(f"予測1-3番に選んだ数: {underdog_predicted_top3}")
            print(f"予測上位率: {100*underdog_predicted_top3/len(underdog_df):.2f}%")
            
            if underdog_predicted_top3 == 0:
                print("⚠️  穴馬を全く予測上位に選んでいない! 人気完全依存の可能性")
            elif underdog_predicted_top3 < len(underdog_df) * 0.1:
                print("⚠️  穴馬をほぼ予測上位に選んでいない。人気依存度が高い")
            else:
                print("✓  穴馬も予測上位に選択している")
            
            # ===== 7. 結論 =====
            print(f"\n{'='*60}")
            print(f"[総合評価] Phase 2-1改の人気依存度")
            print(f"{'='*60}")
            
            dependency_score = 0
            if correlation > 0.7: dependency_score += 3
            elif correlation > 0.5: dependency_score += 2
            elif correlation > 0.3: dependency_score += 1
            
            if match_rate > 30: dependency_score += 2
            elif match_rate > 20: dependency_score += 1
            
            if top3_percentage > 80: dependency_score += 3
            elif top3_percentage > 60: dependency_score += 2
            elif top3_percentage > 40: dependency_score += 1
            
            if underdog_predicted_top3 == 0: dependency_score += 2
            
            print(f"\n人気依存度スコア: {dependency_score}/10")
            
            if dependency_score >= 8:
                print("⚠️⚠️⚠️  超高依存! 予測は実質的に人気のコピー")
                print("推奨: 人気系特徴量を削除して他の特徴量で再構築")
            elif dependency_score >= 6:
                print("⚠️⚠️  高依存。人気が予測を強く支配している")
                print("推奨: 人気系特徴量の重みを下げるか、他の強い特徴量を追加")
            elif dependency_score >= 4:
                print("⚠️  中程度の依存。人気を参考にしているが独自判断もある")
                print("推奨: このバランスは許容範囲。さらなる改善の余地あり")
            else:
                print("✓  低依存。人気とは独立した予測をしている")
                print("推奨: 現状維持またはさらなる精度向上を目指す")
            
        else:
            print("[ERROR] TSVファイルの列数が不足しています")
            
    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_popularity_correlation()
