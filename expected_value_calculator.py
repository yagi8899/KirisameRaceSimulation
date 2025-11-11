#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期待値計算モジュール (Expected Value Calculator)

競馬予測において、予測勝率とオッズから期待値を計算し、
プラス期待値のレース・馬のみを購入対象とする。

主な機能:
- 単勝期待値の計算 (予測勝率 × オッズ)
- 期待値閾値による購入判断
- バックテストによる最適閾値の探索

Author: KirisameRaceSimulation Team
Date: 2025-11-12
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExpectedValueCalculator:
    """
    期待値計算クラス
    
    予測確率とオッズから期待値を計算し、購入判断を行う。
    """
    
    def __init__(self, threshold: float = 1.2, min_odds: float = 1.1, max_odds: float = 999.9):
        """
        初期化
        
        Args:
            threshold (float): 期待値の閾値 (この値以上なら購入推奨)
            min_odds (float): 最小オッズ (これ未満は除外)
            max_odds (float): 最大オッズ (これ超過は除外)
        """
        self.threshold = threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
        
    def calculate_expected_value(
        self, 
        win_probability: float, 
        odds: float
    ) -> float:
        """
        単勝期待値を計算
        
        期待値 = 予測勝率 × オッズ
        
        例:
        - 予測勝率30%, オッズ4.0倍 → 期待値 = 0.30 × 4.0 = 1.20
        - 期待値 > 1.0 なら期待リターンがプラス
        
        Args:
            win_probability (float): 予測勝率 (0.0 ~ 1.0)
            odds (float): 単勝オッズ
            
        Returns:
            float: 期待値 (1.0が損益分岐点)
        """
        # 入力値検証
        if win_probability <= 0 or win_probability > 1:
            return 0.0
        if odds < self.min_odds or odds > self.max_odds:
            return 0.0
            
        # 期待値計算
        expected_value = win_probability * odds
        
        return expected_value
    
    def should_buy(
        self, 
        win_probability: float, 
        odds: float
    ) -> bool:
        """
        購入すべきかを判断
        
        Args:
            win_probability (float): 予測勝率
            odds (float): 単勝オッズ
            
        Returns:
            bool: True=購入推奨, False=見送り
        """
        ev = self.calculate_expected_value(win_probability, odds)
        return ev >= self.threshold
    
    def calculate_race_expected_values(
        self, 
        race_df: pd.DataFrame,
        prediction_col: str = 'predicted_score',
        odds_col: str = 'tansho_odds'
    ) -> pd.DataFrame:
        """
        レース内の全馬の期待値を計算
        
        Args:
            race_df (DataFrame): レースデータ (馬ごとの行)
            prediction_col (str): 予測スコアのカラム名
            odds_col (str): オッズのカラム名
            
        Returns:
            DataFrame: 期待値が追加されたデータフレーム
        """
        # データフレームをコピー
        df = race_df.copy()
        
        # 予測スコアを勝率に変換 (ソフトマックス)
        # LightGBM LambdaRankの予測スコアは相対的なランキングスコアなので、
        # 確率に変換する必要がある
        scores = df[prediction_col].values
        
        # ソフトマックス関数で確率化
        # exp(score) / sum(exp(all_scores))
        exp_scores = np.exp(scores - np.max(scores))  # オーバーフロー防止
        probabilities = exp_scores / np.sum(exp_scores)
        
        df['win_probability'] = probabilities
        
        # 期待値を計算
        df['expected_value'] = df.apply(
            lambda row: self.calculate_expected_value(
                row['win_probability'], 
                row[odds_col]
            ),
            axis=1
        )
        
        # 購入推奨フラグ
        df['should_buy'] = df['expected_value'] >= self.threshold
        
        # 期待リターン (100円購入時の期待払戻)
        df['expected_return'] = df['expected_value'] * 100
        
        return df
    
    def optimize_threshold(
        self,
        backtest_df: pd.DataFrame,
        threshold_range: List[float] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        prediction_col: str = 'predicted_score',
        odds_col: str = 'tansho_odds',
        result_col: str = 'chakujun_numeric'
    ) -> Dict:
        """
        バックテストで最適な期待値閾値を探索
        
        Args:
            backtest_df (DataFrame): バックテスト用データ
            threshold_range (List[float]): 試行する閾値のリスト
            prediction_col (str): 予測スコアのカラム名
            odds_col (str): オッズのカラム名
            result_col (str): 着順のカラム名
            
        Returns:
            Dict: 最適閾値と各閾値の性能
        """
        results = []
        
        for threshold in threshold_range:
            # 閾値を設定
            self.threshold = threshold
            
            # レースIDごとにグループ化して処理
            race_groups = backtest_df.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
            
            total_bets = 0
            total_wins = 0
            total_investment = 0
            total_return = 0
            
            for _, race_df in race_groups:
                # レース内の期待値を計算
                race_with_ev = self.calculate_race_expected_values(
                    race_df, 
                    prediction_col=prediction_col,
                    odds_col=odds_col
                )
                
                # 購入推奨馬を抽出
                buy_horses = race_with_ev[race_with_ev['should_buy']]
                
                if len(buy_horses) == 0:
                    continue
                
                # 購入処理
                for _, horse in buy_horses.iterrows():
                    total_bets += 1
                    total_investment += 100  # 100円購入と仮定
                    
                    # 1着なら払い戻し
                    if horse[result_col] == 1:
                        total_wins += 1
                        total_return += horse[odds_col] * 100
            
            # 指標を計算
            hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
            recovery_rate = (total_return / total_investment * 100) if total_investment > 0 else 0
            profit = total_return - total_investment
            
            results.append({
                'threshold': threshold,
                'total_bets': total_bets,
                'total_wins': total_wins,
                'hit_rate': hit_rate,
                'total_investment': total_investment,
                'total_return': total_return,
                'recovery_rate': recovery_rate,
                'profit': profit
            })
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)
        
        # 回収率が最も高い閾値を選択
        best_threshold_row = results_df.loc[results_df['recovery_rate'].idxmax()]
        
        # 利益が最も高い閾値も確認
        best_profit_row = results_df.loc[results_df['profit'].idxmax()]
        
        return {
            'results_df': results_df,
            'best_threshold_by_recovery': best_threshold_row['threshold'],
            'best_recovery_rate': best_threshold_row['recovery_rate'],
            'best_threshold_by_profit': best_profit_row['threshold'],
            'best_profit': best_profit_row['profit'],
            'recommendation': best_threshold_row['threshold']  # 回収率優先
        }


def analyze_expected_value_distribution(
    race_df: pd.DataFrame,
    prediction_col: str = 'predicted_score',
    odds_col: str = 'tansho_odds'
) -> pd.DataFrame:
    """
    期待値の分布を分析
    
    Args:
        race_df (DataFrame): レースデータ
        prediction_col (str): 予測スコアのカラム名
        odds_col (str): オッズのカラム名
        
    Returns:
        DataFrame: 期待値帯ごとの的中率・回収率
    """
    calculator = ExpectedValueCalculator()
    
    # レースIDごとにグループ化
    race_groups = race_df.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
    
    all_horses = []
    
    for _, race in race_groups:
        race_with_ev = calculator.calculate_race_expected_values(
            race,
            prediction_col=prediction_col,
            odds_col=odds_col
        )
        all_horses.append(race_with_ev)
    
    # 全馬を統合
    df_all = pd.concat(all_horses, ignore_index=True)
    
    # 期待値を区分
    df_all['ev_range'] = pd.cut(
        df_all['expected_value'],
        bins=[0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 999],
        labels=['0.0-1.0', '1.0-1.2', '1.2-1.4', '1.4-1.6', '1.6-1.8', '1.8-2.0', '2.0+']
    )
    
    # 期待値帯ごとの分析
    summary = df_all.groupby('ev_range').apply(
        lambda g: pd.Series({
            '購入回数': len(g),
            '的中回数': (g['chakujun_numeric'] == 1).sum(),
            '的中率(%)': (g['chakujun_numeric'] == 1).sum() / len(g) * 100 if len(g) > 0 else 0,
            '投資額': len(g) * 100,
            '払戻額': (g[g['chakujun_numeric'] == 1][odds_col] * 100).sum(),
            '回収率(%)': (g[g['chakujun_numeric'] == 1][odds_col] * 100).sum() / (len(g) * 100) * 100 if len(g) > 0 else 0,
            '利益': (g[g['chakujun_numeric'] == 1][odds_col] * 100).sum() - (len(g) * 100)
        })
    ).reset_index()
    
    return summary


# ============================================================================
# メイン実行（テスト用）
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("期待値計算モジュール テスト実行")
    print("=" * 80)
    
    # テストデータ作成
    test_data = pd.DataFrame({
        'kaisai_year': [2023] * 18,
        'kaisai_date': ['2023-01-05'] * 18,
        'keibajo_code': ['06'] * 18,
        'race_number': [1] * 18,
        'umaban_numeric': range(1, 19),
        'predicted_score': np.random.randn(18),
        'tansho_odds': [2.5, 4.0, 6.5, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 
                        25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0],
        'chakujun_numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    })
    
    print("\n[TEST 1] 単純な期待値計算")
    print("-" * 80)
    
    calculator = ExpectedValueCalculator(threshold=1.2)
    
    # レースの期待値を計算
    result = calculator.calculate_race_expected_values(test_data)
    
    print("\n期待値計算結果 (上位5頭):")
    print(result[['umaban_numeric', 'tansho_odds', 'win_probability', 
                  'expected_value', 'should_buy', 'chakujun_numeric']].head())
    
    print(f"\n購入推奨馬数: {result['should_buy'].sum()}")
    print(f"期待値 >= 1.2 の馬: {len(result[result['expected_value'] >= 1.2])}")
    
    print("\n" + "=" * 80)
    print("テスト完了!")
    print("=" * 80)
