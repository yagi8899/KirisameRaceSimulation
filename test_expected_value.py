#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期待値計算の実データテスト

predicted_results.tsvを使って、期待値計算の効果を検証する。
"""

import pandas as pd
import numpy as np
from expected_value_calculator import ExpectedValueCalculator, analyze_expected_value_distribution

def main():
    print("=" * 80)
    print("期待値計算 実データテスト")
    print("=" * 80)
    
    # predicted_results.tsvを読み込み
    print("\n[1] データ読み込み...")
    df = pd.read_csv('results/predicted_results.tsv', sep='\t')
    
    print(f"総レコード数: {len(df):,}")
    print(f"年別件数:")
    print(df['開催年'].value_counts().sort_index())
    
    # カラム名を確認
    print(f"\nカラム名: {df.columns.tolist()}")
    
    # 必要なカラム名を統一
    df_renamed = df.rename(columns={
        '開催年': 'kaisai_year',
        '開催日': 'kaisai_date',
        '競馬場': 'keibajo_code',
        'レース番号': 'race_number',
        '馬番': 'umaban_numeric',
        '予測スコア': 'predicted_score',
        '単勝オッズ': 'tansho_odds',
        '確定着順': 'chakujun_numeric'
    })
    
    # オッズが有効なデータのみ
    df_valid = df_renamed[
        (df_renamed['tansho_odds'].notna()) & 
        (df_renamed['tansho_odds'] > 0)
    ].copy()
    
    print(f"\nオッズ有効データ: {len(df_valid):,}")
    
    # ============================================================
    # [2] 期待値分布の分析
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 期待値分布の分析")
    print("=" * 80)
    
    ev_dist = analyze_expected_value_distribution(df_valid)
    print("\n期待値帯ごとの的中率・回収率:")
    print(ev_dist.to_string(index=False))
    
    # ============================================================
    # [3] 最適閾値の探索
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 最適閾値の探索 (グリッドサーチ)")
    print("=" * 80)
    
    calculator = ExpectedValueCalculator()
    
    threshold_range = [1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
    
    optimization_result = calculator.optimize_threshold(
        df_valid,
        threshold_range=threshold_range
    )
    
    print("\n閾値ごとの性能:")
    results_df = optimization_result['results_df']
    print(results_df.to_string(index=False))
    
    print("\n" + "-" * 80)
    print("最適閾値 (回収率優先):")
    print(f"  閾値: {optimization_result['best_threshold_by_recovery']:.2f}")
    print(f"  回収率: {optimization_result['best_recovery_rate']:.2f}%")
    
    print("\n最適閾値 (利益優先):")
    print(f"  閾値: {optimization_result['best_threshold_by_profit']:.2f}")
    print(f"  利益: {optimization_result['best_profit']:,.0f}円")
    
    print("\n推奨閾値:")
    print(f"  {optimization_result['recommendation']:.2f}")
    
    # ============================================================
    # [4] 年別の性能比較
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 年別性能比較 (推奨閾値適用)")
    print("=" * 80)
    
    best_threshold = optimization_result['recommendation']
    calculator.threshold = best_threshold
    
    for year in sorted(df_valid['kaisai_year'].unique()):
        df_year = df_valid[df_valid['kaisai_year'] == year]
        
        race_groups = df_year.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
        
        total_bets = 0
        total_wins = 0
        total_investment = 0
        total_return = 0
        
        for _, race_df in race_groups:
            race_with_ev = calculator.calculate_race_expected_values(race_df)
            buy_horses = race_with_ev[race_with_ev['should_buy']]
            
            if len(buy_horses) == 0:
                continue
            
            for _, horse in buy_horses.iterrows():
                total_bets += 1
                total_investment += 100
                
                if horse['chakujun_numeric'] == 1:
                    total_wins += 1
                    total_return += horse['tansho_odds'] * 100
        
        hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        recovery_rate = (total_return / total_investment * 100) if total_investment > 0 else 0
        profit = total_return - total_investment
        
        print(f"\n{year}年:")
        print(f"  購入回数: {total_bets:,}")
        print(f"  的中回数: {total_wins}")
        print(f"  的中率: {hit_rate:.2f}%")
        print(f"  投資額: {total_investment:,}円")
        print(f"  払戻額: {total_return:,}円")
        print(f"  回収率: {recovery_rate:.2f}%")
        print(f"  利益: {profit:,}円")
    
    print("\n" + "=" * 80)
    print("テスト完了!")
    print("=" * 80)

if __name__ == '__main__':
    main()
