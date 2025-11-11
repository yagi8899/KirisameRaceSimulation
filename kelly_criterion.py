#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ケリー基準による資金管理モジュール (Kelly Criterion)

競馬投資において、期待値がプラスの賭けに対して最適な資金配分を行う。
ケリー基準は資金増加速度を最大化する理論的に最適な賭け金割合を示す。

主な機能:
- フルケリー基準の計算
- ハーフケリー・クォーターケリー(保守的)の計算
- 最大賭け金制限
- 複数馬の資金配分

数式:
f* = (bp - q) / b
where:
  f* = 最適な賭け金割合
  b = オッズ - 1 (払い戻し倍率)
  p = 勝つ確率
  q = 負ける確率 (1 - p)

Author: KirisameRaceSimulation Team
Date: 2025-11-12
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class KellyCriterion:
    """
    ケリー基準による資金管理クラス
    """
    
    def __init__(
        self, 
        bankroll: float = 1000000,  # 総資金 (デフォルト: 100万円)
        fraction: float = 0.25,  # ケリー基準の分数 (0.25 = クォーターケリー)
        max_bet_percentage: float = 0.05,  # 最大賭け金割合 (5%)
        min_bet_amount: float = 100,  # 最小賭け金額
        max_bet_amount: float = 50000  # 最大賭け金額
    ):
        """
        初期化
        
        Args:
            bankroll (float): 総資金
            fraction (float): ケリー基準の分数
                - 1.0 = フルケリー (最も攻撃的、リスク大)
                - 0.5 = ハーフケリー (バランス型)
                - 0.25 = クォーターケリー (保守的、推奨)
            max_bet_percentage (float): 最大賭け金割合 (総資金に対する%)
            min_bet_amount (float): 最小賭け金額
            max_bet_amount (float): 最大賭け金額
        """
        self.bankroll = bankroll
        self.fraction = fraction
        self.max_bet_percentage = max_bet_percentage
        self.min_bet_amount = min_bet_amount
        self.max_bet_amount = max_bet_amount
        
    def calculate_kelly_fraction(
        self, 
        win_probability: float, 
        odds: float
    ) -> float:
        """
        ケリー基準の賭け金割合を計算
        
        Args:
            win_probability (float): 勝つ確率 (0.0 ~ 1.0)
            odds (float): 単勝オッズ
            
        Returns:
            float: 賭け金割合 (0.0 ~ 1.0)
        """
        # 入力値検証
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        if odds <= 1.0:
            return 0.0
        
        # ケリー基準の計算
        # f* = (bp - q) / b
        # where b = odds - 1, p = win_probability, q = 1 - win_probability
        
        b = odds - 1.0  # 払い戻し倍率 (オッズから1を引く)
        p = win_probability
        q = 1.0 - p
        
        kelly_fraction = (b * p - q) / b
        
        # 負の値は0にクリップ (期待値がマイナスなら賭けない)
        kelly_fraction = max(0.0, kelly_fraction)
        
        # ケリー基準の分数を適用 (保守的にする)
        kelly_fraction *= self.fraction
        
        # 最大賭け金割合でキャップ
        kelly_fraction = min(kelly_fraction, self.max_bet_percentage)
        
        return kelly_fraction
    
    def calculate_bet_amount(
        self, 
        win_probability: float, 
        odds: float,
        current_bankroll: Optional[float] = None
    ) -> float:
        """
        実際の賭け金額を計算
        
        Args:
            win_probability (float): 勝つ確率
            odds (float): 単勝オッズ
            current_bankroll (float, optional): 現在の資金 (Noneなら初期資金を使用)
            
        Returns:
            float: 賭け金額 (円)
        """
        bankroll = current_bankroll if current_bankroll is not None else self.bankroll
        
        # ケリー基準の割合を計算
        kelly_fraction = self.calculate_kelly_fraction(win_probability, odds)
        
        # 賭け金額を計算
        bet_amount = bankroll * kelly_fraction
        
        # 最小・最大賭け金でクリップ
        bet_amount = max(self.min_bet_amount, bet_amount)
        bet_amount = min(self.max_bet_amount, bet_amount)
        
        # 100円単位に丸める
        bet_amount = round(bet_amount / 100) * 100
        
        return bet_amount
    
    def allocate_multiple_bets(
        self,
        race_df: pd.DataFrame,
        win_probability_col: str = 'win_probability',
        odds_col: str = 'tansho_odds',
        current_bankroll: Optional[float] = None
    ) -> pd.DataFrame:
        """
        レース内の複数馬に資金を配分
        
        Args:
            race_df (DataFrame): レースデータ (馬ごとの行)
            win_probability_col (str): 勝率のカラム名
            odds_col (str): オッズのカラム名
            current_bankroll (float, optional): 現在の資金
            
        Returns:
            DataFrame: 賭け金が追加されたデータフレーム
        """
        df = race_df.copy()
        bankroll = current_bankroll if current_bankroll is not None else self.bankroll
        
        # 各馬の賭け金を計算
        df['kelly_fraction'] = df.apply(
            lambda row: self.calculate_kelly_fraction(
                row[win_probability_col],
                row[odds_col]
            ),
            axis=1
        )
        
        # ケリー基準の合計が1.0を超える場合は正規化
        total_kelly = df['kelly_fraction'].sum()
        if total_kelly > self.max_bet_percentage:
            # 最大賭け金割合で正規化
            df['kelly_fraction'] = df['kelly_fraction'] * (self.max_bet_percentage / total_kelly)
        
        # 賭け金額を計算
        df['bet_amount'] = df['kelly_fraction'] * bankroll
        
        # 最小・最大賭け金でクリップ
        df['bet_amount'] = df['bet_amount'].clip(
            lower=self.min_bet_amount,
            upper=self.max_bet_amount
        )
        
        # 100円単位に丸める
        df['bet_amount'] = (df['bet_amount'] / 100).round() * 100
        
        # 期待リターンを計算
        df['expected_profit'] = df['bet_amount'] * (
            df[win_probability_col] * df[odds_col] - 1
        )
        
        return df
    
    def simulate_bankroll_growth(
        self,
        backtest_df: pd.DataFrame,
        initial_bankroll: float = 1000000,
        win_probability_col: str = 'win_probability',
        odds_col: str = 'tansho_odds',
        result_col: str = 'chakujun_numeric'
    ) -> Dict:
        """
        ケリー基準による資金推移をシミュレーション
        
        Args:
            backtest_df (DataFrame): バックテスト用データ
            initial_bankroll (float): 初期資金
            win_probability_col (str): 勝率のカラム名
            odds_col (str): オッズのカラム名
            result_col (str): 着順のカラム名
            
        Returns:
            Dict: シミュレーション結果
        """
        self.bankroll = initial_bankroll
        current_bankroll = initial_bankroll
        
        bankroll_history = [initial_bankroll]
        bet_history = []
        
        # レースIDごとにグループ化
        race_groups = backtest_df.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
        
        for race_id, race_df in race_groups:
            # 購入推奨馬のみ (should_buy == True)
            if 'should_buy' in race_df.columns:
                buy_horses = race_df[race_df['should_buy']].copy()
            else:
                buy_horses = race_df.copy()
            
            if len(buy_horses) == 0:
                continue
            
            # 資金配分を計算
            buy_horses_with_bet = self.allocate_multiple_bets(
                buy_horses,
                win_probability_col=win_probability_col,
                odds_col=odds_col,
                current_bankroll=current_bankroll
            )
            
            # 投資額の合計
            total_bet = buy_horses_with_bet['bet_amount'].sum()
            
            # 払い戻し額の計算
            total_return = 0
            for _, horse in buy_horses_with_bet.iterrows():
                if horse[result_col] == 1:
                    total_return += horse['bet_amount'] * horse[odds_col]
            
            # 資金を更新
            current_bankroll = current_bankroll - total_bet + total_return
            bankroll_history.append(current_bankroll)
            
            # 履歴に記録
            bet_history.append({
                'race_id': race_id,
                'num_bets': len(buy_horses_with_bet),
                'total_bet': total_bet,
                'total_return': total_return,
                'profit': total_return - total_bet,
                'bankroll': current_bankroll
            })
        
        # 最終結果を計算
        final_bankroll = current_bankroll
        total_profit = final_bankroll - initial_bankroll
        roi = (final_bankroll / initial_bankroll - 1) * 100
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': final_bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'bankroll_history': bankroll_history,
            'bet_history': pd.DataFrame(bet_history)
        }
    
    def compare_strategies(
        self,
        backtest_df: pd.DataFrame,
        strategies: List[Dict] = None,
        initial_bankroll: float = 1000000
    ) -> pd.DataFrame:
        """
        複数の戦略を比較
        
        Args:
            backtest_df (DataFrame): バックテスト用データ
            strategies (List[Dict]): 戦略のリスト
                例: [{'name': 'Full Kelly', 'fraction': 1.0},
                     {'name': 'Half Kelly', 'fraction': 0.5}]
            initial_bankroll (float): 初期資金
            
        Returns:
            DataFrame: 戦略比較結果
        """
        if strategies is None:
            strategies = [
                {'name': 'フルケリー', 'fraction': 1.0},
                {'name': 'ハーフケリー', 'fraction': 0.5},
                {'name': 'クォーターケリー', 'fraction': 0.25},
                {'name': '固定100円', 'fraction': 0.0}  # 固定額ベット
            ]
        
        results = []
        
        for strategy in strategies:
            self.fraction = strategy['fraction']
            
            if strategy['fraction'] == 0.0:
                # 固定額ベット (100円)
                sim_result = self._simulate_fixed_bet(
                    backtest_df,
                    initial_bankroll=initial_bankroll,
                    fixed_bet=100
                )
            else:
                sim_result = self.simulate_bankroll_growth(
                    backtest_df,
                    initial_bankroll=initial_bankroll
                )
            
            results.append({
                '戦略': strategy['name'],
                '初期資金': initial_bankroll,
                '最終資金': sim_result['final_bankroll'],
                '利益': sim_result['total_profit'],
                'ROI(%)': sim_result['roi'],
                '資金増加率': sim_result['final_bankroll'] / initial_bankroll
            })
        
        return pd.DataFrame(results)
    
    def _simulate_fixed_bet(
        self,
        backtest_df: pd.DataFrame,
        initial_bankroll: float,
        fixed_bet: float = 100
    ) -> Dict:
        """
        固定額ベットのシミュレーション (比較用)
        """
        current_bankroll = initial_bankroll
        bankroll_history = [initial_bankroll]
        
        race_groups = backtest_df.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
        
        for _, race_df in race_groups:
            if 'should_buy' in race_df.columns:
                buy_horses = race_df[race_df['should_buy']]
            else:
                buy_horses = race_df
            
            if len(buy_horses) == 0:
                continue
            
            total_bet = len(buy_horses) * fixed_bet
            total_return = 0
            
            for _, horse in buy_horses.iterrows():
                if horse['chakujun_numeric'] == 1:
                    total_return += fixed_bet * horse['tansho_odds']
            
            current_bankroll = current_bankroll - total_bet + total_return
            bankroll_history.append(current_bankroll)
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': current_bankroll,
            'total_profit': current_bankroll - initial_bankroll,
            'roi': (current_bankroll / initial_bankroll - 1) * 100,
            'bankroll_history': bankroll_history
        }


# ============================================================================
# メイン実行（テスト用）
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("ケリー基準 テスト実行")
    print("=" * 80)
    
    # テストデータ
    test_case = {
        'win_probability': 0.30,
        'odds': 4.0,
        'bankroll': 1000000
    }
    
    print(f"\n[TEST 1] 単純なケリー基準計算")
    print(f"  勝率: {test_case['win_probability']*100:.1f}%")
    print(f"  オッズ: {test_case['odds']:.1f}倍")
    print(f"  資金: {test_case['bankroll']:,}円")
    print("-" * 80)
    
    # 各戦略で計算
    strategies = [
        ('フルケリー', 1.0),
        ('ハーフケリー', 0.5),
        ('クォーターケリー', 0.25)
    ]
    
    for name, fraction in strategies:
        kelly = KellyCriterion(
            bankroll=test_case['bankroll'],
            fraction=fraction
        )
        
        kelly_frac = kelly.calculate_kelly_fraction(
            test_case['win_probability'],
            test_case['odds']
        )
        bet_amount = kelly.calculate_bet_amount(
            test_case['win_probability'],
            test_case['odds']
        )
        
        print(f"\n{name} (fraction={fraction}):")
        print(f"  賭け金割合: {kelly_frac*100:.2f}%")
        print(f"  賭け金額: {bet_amount:,}円")
    
    print("\n" + "=" * 80)
    print("テスト完了!")
    print("=" * 80)
