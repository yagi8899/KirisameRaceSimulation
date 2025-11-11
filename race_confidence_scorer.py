#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
レース信頼度スコア計算モジュール (Race Confidence Scorer)

レース全体の予測信頼度と、各馬の購入確率を計算する。
的中しやすいレースを選別し、馬ごとに適切な購入確率を設定することで、
無駄な購入を削減し、回収率を向上させる。

主な機能:
- レース全体の信頼度スコア計算 (0-100)
- 各馬の予測信頼度計算 (0-100)
- 信頼度ベースの購入確率設定
- 期待値計算との統合

Author: KirisameRaceSimulation Team
Date: 2025-11-12
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RaceConfidenceScorer:
    """
    レース信頼度スコア計算クラス
    
    レース全体と各馬の信頼度を計算し、購入確率を設定する。
    """
    
    def __init__(
        self,
        race_confidence_threshold: float = 50.0,  # レース信頼度の閾値
        horse_confidence_thresholds: Dict[str, float] = None  # 馬信頼度の閾値
    ):
        """
        初期化
        
        Args:
            race_confidence_threshold (float): レース信頼度の閾値 (この値以上なら購入検討)
            horse_confidence_thresholds (Dict): 馬信頼度の閾値設定
                デフォルト: {
                    'very_high': 70.0,  # 70%以上 → 購入確率100%
                    'high': 50.0,       # 50-70% → 購入確率70%
                    'medium': 30.0,     # 30-50% → 購入確率30%
                    'low': 0.0          # 30%未満 → 購入確率5%
                }
        """
        self.race_confidence_threshold = race_confidence_threshold
        
        if horse_confidence_thresholds is None:
            self.horse_confidence_thresholds = {
                'very_high': 70.0,
                'high': 50.0,
                'medium': 30.0,
                'low': 0.0
            }
        else:
            self.horse_confidence_thresholds = horse_confidence_thresholds
    
    def calculate_race_confidence(
        self,
        race_df: pd.DataFrame,
        prediction_col: str = 'predicted_score'
    ) -> Dict:
        """
        レース全体の信頼度スコアを計算
        
        信頼度を高める要素:
        1. 予測上位3頭のスコア差が大きい (本命が明確)
        2. 出走頭数が少ない (混戦になりにくい)
        3. 予測スコアの分散が大きい (力の差が明確)
        
        Args:
            race_df (DataFrame): レースデータ (馬ごとの行)
            prediction_col (str): 予測スコアのカラム名
            
        Returns:
            Dict: 信頼度スコアと各要素
        """
        df = race_df.copy()
        
        # 予測スコアで降順ソート
        df_sorted = df.sort_values(by=prediction_col, ascending=False).reset_index(drop=True)
        
        # 1. 予測上位3頭のスコア差 (スコア)
        if len(df_sorted) >= 3:
            top1_score = df_sorted.iloc[0][prediction_col]
            top2_score = df_sorted.iloc[1][prediction_col]
            top3_score = df_sorted.iloc[2][prediction_col]
            
            # 1位と2位の差
            diff_1_2 = top1_score - top2_score
            # 1位と3位の差
            diff_1_3 = top1_score - top3_score
            
            # スコア差が大きいほど高得点 (0-40点)
            # 差が1.0以上なら満点、0なら0点
            score_diff_score = min(40, (diff_1_2 + diff_1_3) * 10)
        else:
            score_diff_score = 0
        
        # 2. 出走頭数スコア (0-30点)
        num_horses = len(df)
        # 頭数が少ないほど高得点
        # 8頭以下: 30点, 12頭: 20点, 18頭: 0点
        if num_horses <= 8:
            num_horses_score = 30
        elif num_horses <= 12:
            num_horses_score = 30 - (num_horses - 8) * 2.5
        else:
            num_horses_score = max(0, 20 - (num_horses - 12) * 3.33)
        
        # 3. 予測スコアの分散 (0-30点)
        # 分散が大きい = 力の差が明確
        score_std = df[prediction_col].std()
        # 標準偏差が1.0以上なら満点
        score_variance_score = min(30, score_std * 30)
        
        # 合計スコア (0-100)
        total_confidence = score_diff_score + num_horses_score + score_variance_score
        
        return {
            'race_confidence': min(100, total_confidence),
            'score_diff_score': score_diff_score,
            'num_horses_score': num_horses_score,
            'score_variance_score': score_variance_score,
            'num_horses': num_horses,
            'top3_diff': diff_1_2 + diff_1_3 if len(df_sorted) >= 3 else 0
        }
    
    def calculate_horse_confidence(
        self,
        horse_row: pd.Series,
        race_df: pd.DataFrame,
        prediction_col: str = 'predicted_score'
    ) -> float:
        """
        各馬の予測信頼度を計算
        
        信頼度を高める要素:
        1. 予測順位が高い
        2. 予測スコアが高い
        3. 2位以下との差が大きい
        
        Args:
            horse_row (Series): 馬の行
            race_df (DataFrame): レース全体のデータ
            prediction_col (str): 予測スコアのカラム名
            
        Returns:
            float: 馬の信頼度 (0-100)
        """
        df = race_df.copy()
        
        # 予測順位を計算
        df['pred_rank'] = df[prediction_col].rank(ascending=False, method='min')
        
        horse_score = horse_row[prediction_col]
        horse_rank = df[df.index == horse_row.name]['pred_rank'].values[0]
        
        # 1. 順位スコア (0-50点)
        # 1位: 50点, 2位: 40点, 3位: 30点, 以降減少
        if horse_rank == 1:
            rank_score = 50
        elif horse_rank == 2:
            rank_score = 40
        elif horse_rank == 3:
            rank_score = 30
        elif horse_rank <= 5:
            rank_score = 25
        elif horse_rank <= 8:
            rank_score = 15
        else:
            rank_score = 5
        
        # 2. スコア相対値 (0-30点)
        # レース内での相対的な位置
        min_score = df[prediction_col].min()
        max_score = df[prediction_col].max()
        score_range = max_score - min_score
        
        if score_range > 0:
            relative_score = ((horse_score - min_score) / score_range) * 30
        else:
            relative_score = 15  # 全馬同じなら中間点
        
        # 3. 下位馬との差 (0-20点)
        # 自分より下の馬との平均差が大きいほど高得点
        lower_horses = df[df['pred_rank'] > horse_rank]
        if len(lower_horses) > 0:
            avg_diff = horse_score - lower_horses[prediction_col].mean()
            diff_score = min(20, avg_diff * 10)
        else:
            diff_score = 0
        
        # 合計
        total_confidence = rank_score + relative_score + diff_score
        
        return min(100, total_confidence)
    
    def calculate_purchase_probability(
        self,
        horse_confidence: float
    ) -> float:
        """
        馬の信頼度から購入確率を計算
        
        Args:
            horse_confidence (float): 馬の信頼度 (0-100)
            
        Returns:
            float: 購入確率 (0.0-1.0)
        """
        thresholds = self.horse_confidence_thresholds
        
        if horse_confidence >= thresholds['very_high']:
            # 70%以上 → 購入確率100%
            return 1.0
        elif horse_confidence >= thresholds['high']:
            # 50-70% → 購入確率70%
            return 0.7
        elif horse_confidence >= thresholds['medium']:
            # 30-50% → 購入確率30%
            return 0.3
        else:
            # 30%未満 → 購入確率5%
            return 0.05
    
    def score_race_with_horses(
        self,
        race_df: pd.DataFrame,
        prediction_col: str = 'predicted_score'
    ) -> pd.DataFrame:
        """
        レース全体と各馬の信頼度を計算し、購入確率を設定
        
        Args:
            race_df (DataFrame): レースデータ
            prediction_col (str): 予測スコアのカラム名
            
        Returns:
            DataFrame: 信頼度と購入確率が追加されたデータフレーム
        """
        df = race_df.copy()
        
        # レース全体の信頼度を計算
        race_conf = self.calculate_race_confidence(df, prediction_col=prediction_col)
        
        # 各馬の信頼度を計算
        df['horse_confidence'] = df.apply(
            lambda row: self.calculate_horse_confidence(
                row, 
                df, 
                prediction_col=prediction_col
            ),
            axis=1
        )
        
        # 購入確率を計算
        df['purchase_probability'] = df['horse_confidence'].apply(
            self.calculate_purchase_probability
        )
        
        # レース信頼度を全行に追加
        df['race_confidence'] = race_conf['race_confidence']
        df['race_confidence_details'] = str(race_conf)
        
        # レース信頼度が閾値未満なら全馬購入確率を0に
        if race_conf['race_confidence'] < self.race_confidence_threshold:
            df['purchase_probability'] = 0.0
            df['skip_reason'] = 'low_race_confidence'
        else:
            df['skip_reason'] = None
        
        return df
    
    def integrate_with_expected_value(
        self,
        race_df: pd.DataFrame,
        ev_threshold: float = 1.2,
        prediction_col: str = 'predicted_score',
        odds_col: str = 'tansho_odds'
    ) -> pd.DataFrame:
        """
        期待値計算と統合して最終購入判断
        
        Args:
            race_df (DataFrame): レースデータ
            ev_threshold (float): 期待値の閾値
            prediction_col (str): 予測スコアのカラム名
            odds_col (str): オッズのカラム名
            
        Returns:
            DataFrame: 最終購入判断が追加されたデータフレーム
        """
        from expected_value_calculator import ExpectedValueCalculator
        
        df = race_df.copy()
        
        # 1. 期待値を計算
        ev_calculator = ExpectedValueCalculator(threshold=ev_threshold)
        df = ev_calculator.calculate_race_expected_values(
            df,
            prediction_col=prediction_col,
            odds_col=odds_col
        )
        
        # 2. 信頼度スコアを計算
        df = self.score_race_with_horses(df, prediction_col=prediction_col)
        
        # 3. 最終購入判断
        # 条件: (期待値 >= 閾値) AND (購入確率 > 0)
        df['final_should_buy'] = (df['should_buy']) & (df['purchase_probability'] > 0)
        
        # 4. 実際の購入確率を適用 (確率的購入)
        # 購入確率に基づいてランダムに購入判断
        np.random.seed(42)  # 再現性のため
        random_values = np.random.random(len(df))
        df['probabilistic_buy'] = (df['final_should_buy']) & (random_values < df['purchase_probability'])
        
        return df


def analyze_confidence_distribution(
    backtest_df: pd.DataFrame,
    prediction_col: str = 'predicted_score'
) -> pd.DataFrame:
    """
    信頼度分布と的中率の関係を分析
    
    Args:
        backtest_df (DataFrame): バックテスト用データ
        prediction_col (str): 予測スコアのカラム名
        
    Returns:
        DataFrame: 信頼度帯ごとの的中率・回収率
    """
    scorer = RaceConfidenceScorer()
    
    # レースごとにグループ化
    race_groups = backtest_df.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
    
    all_horses = []
    
    for _, race_df in race_groups:
        scored_race = scorer.score_race_with_horses(race_df, prediction_col=prediction_col)
        all_horses.append(scored_race)
    
    # 全馬を統合
    df_all = pd.concat(all_horses, ignore_index=True)
    
    # 馬信頼度を区分
    df_all['confidence_range'] = pd.cut(
        df_all['horse_confidence'],
        bins=[0, 30, 50, 70, 100],
        labels=['0-30 (低)', '30-50 (中)', '50-70 (高)', '70-100 (最高)']
    )
    
    # 信頼度帯ごとの分析
    summary = df_all.groupby('confidence_range').apply(
        lambda g: pd.Series({
            '馬数': len(g),
            '的中数': (g['chakujun_numeric'] == 1).sum(),
            '的中率(%)': (g['chakujun_numeric'] == 1).sum() / len(g) * 100 if len(g) > 0 else 0,
            '平均オッズ': g['tansho_odds'].mean(),
            '購入確率': g['purchase_probability'].mean()
        })
    ).reset_index()
    
    return summary


# ============================================================================
# メイン実行（テスト用）
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("レース信頼度スコア テスト実行")
    print("=" * 80)
    
    # テストデータ作成
    test_data = pd.DataFrame({
        'kaisai_year': [2023] * 18,
        'kaisai_date': ['2023-01-05'] * 18,
        'keibajo_code': ['06'] * 18,
        'race_number': [1] * 18,
        'umaban_numeric': range(1, 19),
        'predicted_score': [3.5, 2.8, 2.1, 1.5, 1.2, 0.8, 0.5, 0.3, 0.1,
                           -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6, -1.8],
        'tansho_odds': [2.5, 4.0, 6.5, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0,
                       25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0],
        'chakujun_numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    })
    
    print("\n[TEST 1] レース信頼度スコア計算")
    print("-" * 80)
    
    scorer = RaceConfidenceScorer()
    
    # レース全体の信頼度
    race_conf = scorer.calculate_race_confidence(test_data)
    print("\nレース全体の信頼度:")
    for key, value in race_conf.items():
        print(f"  {key}: {value}")
    
    # 各馬の信頼度と購入確率
    scored_race = scorer.score_race_with_horses(test_data)
    
    print("\n各馬の信頼度と購入確率 (上位5頭):")
    print(scored_race[['umaban_numeric', 'predicted_score', 'horse_confidence', 
                       'purchase_probability']].head())
    
    print(f"\n購入確率100%の馬: {len(scored_race[scored_race['purchase_probability'] == 1.0])}")
    print(f"購入確率70%の馬: {len(scored_race[scored_race['purchase_probability'] == 0.7])}")
    print(f"購入確率30%の馬: {len(scored_race[scored_race['purchase_probability'] == 0.3])}")
    print(f"購入確率5%の馬: {len(scored_race[scored_race['purchase_probability'] == 0.05])}")
    
    print("\n" + "=" * 80)
    print("テスト完了!")
    print("=" * 80)
