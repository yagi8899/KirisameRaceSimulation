"""
レース選別閾値の最適化 (Task 2-2)

このモジュールはグリッドサーチを使って以下のパラメータを最適化します:
- ev_threshold: 期待値閾値 (1.0-1.5)
- race_confidence_threshold: レース信頼度閾値 (10-40)
- horse_confidence_thresholds: 馬信頼度閾値 (調整可能)

最適化目標:
- 回収率115-140%
- 的中率24-26%
- 購入回数>0 (実際に購入できること)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools
from tqdm import tqdm


class RaceSelectorOptimizer:
    """レース選別閾値の最適化クラス"""
    
    def __init__(self, data_path: str = 'results/predicted_results.tsv'):
        """
        Args:
            data_path: 予測結果データのパス
        """
        self.df = pd.read_csv(data_path, sep='\t')
        print(f"データ読み込み完了: {len(self.df)}件")
        
    def calculate_performance(
        self,
        ev_threshold: float,
        race_confidence_threshold: float,
        purchase_prob_thresholds: Dict[str, float] = None
    ) -> Dict:
        """
        指定された閾値での購入シミュレーションを実行し、パフォーマンスを計算
        
        Args:
            ev_threshold: 期待値閾値
            race_confidence_threshold: レース信頼度閾値
            purchase_prob_thresholds: 購入確率の閾値辞書
                                    (デフォルト: {very_high: 70, high: 50, medium: 30})
        
        Returns:
            パフォーマンス指標の辞書
        """
        if purchase_prob_thresholds is None:
            purchase_prob_thresholds = {
                'very_high': 70,
                'high': 50,
                'medium': 30
            }
        
        # 期待値とレース信頼度でフィルタリング
        filtered = self.df[
            (self.df['expected_return'] >= ev_threshold) &
            (self.df['レース信頼度'] >= race_confidence_threshold)
        ].copy()
        
        if len(filtered) == 0:
            return {
                'recovery_rate': 0.0,
                'hit_rate': 0.0,
                'purchase_count': 0,
                'total_invested': 0,
                'total_return': 0,
                'profit': 0,
                'avg_odds': 0.0,
                'params': {
                    'ev_threshold': ev_threshold,
                    'race_confidence_threshold': race_confidence_threshold,
                    'purchase_prob_thresholds': purchase_prob_thresholds
                }
            }
        
        # 購入確率を再計算
        def recalc_purchase_prob(confidence):
            if confidence >= purchase_prob_thresholds['very_high']:
                return 1.0  # 100%
            elif confidence >= purchase_prob_thresholds['high']:
                return 0.7  # 70%
            elif confidence >= purchase_prob_thresholds['medium']:
                return 0.3  # 30%
            else:
                return 0.05  # 5%
        
        filtered['購入確率_調整'] = filtered['馬信頼度'].apply(recalc_purchase_prob)
        
        # 確率的購入をシミュレーション (期待値ベース)
        np.random.seed(42)  # 再現性のため
        filtered['購入判定'] = (
            np.random.random(len(filtered)) < filtered['購入確率_調整']
        )
        
        # 購入した馬のみを対象
        purchased = filtered[filtered['購入判定']].copy()
        
        if len(purchased) == 0:
            return {
                'recovery_rate': 0.0,
                'hit_rate': 0.0,
                'purchase_count': 0,
                'total_invested': 0,
                'total_return': 0,
                'profit': 0,
                'avg_odds': 0.0,
                'params': {
                    'ev_threshold': ev_threshold,
                    'race_confidence_threshold': race_confidence_threshold,
                    'purchase_prob_thresholds': purchase_prob_thresholds
                }
            }
        
        # 購入額を計算 (簡易版: 一律1000円)
        purchased['投資額'] = 1000
        
        # 的中判定
        purchased['的中'] = purchased['確定着順'] == 1
        
        # 払戻金計算
        purchased['払戻'] = purchased.apply(
            lambda row: row['投資額'] * row['単勝オッズ'] if row['的中'] else 0,
            axis=1
        )
        
        # パフォーマンス計算
        total_invested = purchased['投資額'].sum()
        total_return = purchased['払戻'].sum()
        profit = total_return - total_invested
        recovery_rate = (total_return / total_invested * 100) if total_invested > 0 else 0
        hit_rate = (purchased['的中'].sum() / len(purchased) * 100) if len(purchased) > 0 else 0
        avg_odds = purchased['単勝オッズ'].mean()
        
        return {
            'recovery_rate': recovery_rate,
            'hit_rate': hit_rate,
            'purchase_count': len(purchased),
            'total_invested': total_invested,
            'total_return': total_return,
            'profit': profit,
            'avg_odds': avg_odds,
            'params': {
                'ev_threshold': ev_threshold,
                'race_confidence_threshold': race_confidence_threshold,
                'purchase_prob_thresholds': purchase_prob_thresholds
            }
        }
    
    def grid_search(
        self,
        ev_thresholds: List[float] = None,
        race_confidence_thresholds: List[float] = None,
        horse_confidence_configs: List[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        グリッドサーチで最適なパラメータを探索
        
        Args:
            ev_thresholds: 期待値閾値のリスト
            race_confidence_thresholds: レース信頼度閾値のリスト
            horse_confidence_configs: 馬信頼度閾値設定のリスト
        
        Returns:
            全パラメータ組み合わせの結果を含むDataFrame
        """
        if ev_thresholds is None:
            ev_thresholds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        if race_confidence_thresholds is None:
            race_confidence_thresholds = [10, 15, 20, 25, 30, 35, 40]
        
        if horse_confidence_configs is None:
            horse_confidence_configs = [
                {'very_high': 70, 'high': 50, 'medium': 30},  # デフォルト
                {'very_high': 60, 'high': 40, 'medium': 20},  # 緩め
                {'very_high': 80, 'high': 60, 'medium': 40},  # 厳しめ
            ]
        
        results = []
        
        # 全組み合わせを試行
        total_combinations = (
            len(ev_thresholds) * 
            len(race_confidence_thresholds) * 
            len(horse_confidence_configs)
        )
        
        print(f"\nグリッドサーチ開始: {total_combinations}通りの組み合わせ")
        
        for ev_th, race_th, horse_conf in tqdm(
            itertools.product(ev_thresholds, race_confidence_thresholds, horse_confidence_configs),
            total=total_combinations,
            desc="最適化中"
        ):
            perf = self.calculate_performance(ev_th, race_th, horse_conf)
            results.append(perf)
        
        # DataFrameに変換
        results_df = pd.DataFrame(results)
        
        # パラメータを展開
        results_df['ev_threshold'] = results_df['params'].apply(lambda x: x['ev_threshold'])
        results_df['race_confidence_threshold'] = results_df['params'].apply(
            lambda x: x['race_confidence_threshold']
        )
        results_df['horse_conf_very_high'] = results_df['params'].apply(
            lambda x: x['purchase_prob_thresholds']['very_high']
        )
        results_df['horse_conf_high'] = results_df['params'].apply(
            lambda x: x['purchase_prob_thresholds']['high']
        )
        results_df['horse_conf_medium'] = results_df['params'].apply(
            lambda x: x['purchase_prob_thresholds']['medium']
        )
        
        # params列は削除
        results_df = results_df.drop('params', axis=1)
        
        return results_df
    
    def find_best_params(
        self,
        results_df: pd.DataFrame,
        min_recovery_rate: float = 115.0,
        min_purchase_count: int = 50
    ) -> pd.DataFrame:
        """
        最適なパラメータを検索
        
        Args:
            results_df: grid_search()の結果
            min_recovery_rate: 最低回収率
            min_purchase_count: 最低購入回数
        
        Returns:
            条件を満たす上位結果のDataFrame
        """
        # 条件でフィルタリング
        filtered = results_df[
            (results_df['recovery_rate'] >= min_recovery_rate) &
            (results_df['purchase_count'] >= min_purchase_count)
        ].copy()
        
        if len(filtered) == 0:
            print(f"\n警告: 回収率{min_recovery_rate}%以上、購入回数{min_purchase_count}件以上の組み合わせが見つかりませんでした。")
            print("条件を緩和して再検索します...")
            
            # 条件を緩和
            filtered = results_df[
                (results_df['recovery_rate'] >= 100.0) &
                (results_df['purchase_count'] >= 10)
            ].copy()
            
            if len(filtered) == 0:
                print("購入回数>0の組み合わせで最良のものを表示します。")
                filtered = results_df[results_df['purchase_count'] > 0].copy()
        
        # 回収率でソート
        filtered = filtered.sort_values('recovery_rate', ascending=False)
        
        return filtered
    
    def print_summary(self, results_df: pd.DataFrame, top_n: int = 10):
        """
        結果サマリーを表示
        
        Args:
            results_df: find_best_params()の結果
            top_n: 表示する上位結果の件数
        """
        print(f"\n{'='*80}")
        print(f"最適化結果サマリー (上位{top_n}件)")
        print(f"{'='*80}\n")
        
        display_cols = [
            'recovery_rate', 'hit_rate', 'purchase_count', 
            'profit', 'avg_odds',
            'ev_threshold', 'race_confidence_threshold',
            'horse_conf_very_high', 'horse_conf_high', 'horse_conf_medium'
        ]
        
        top_results = results_df.head(top_n)[display_cols]
        
        # フォーマットして表示
        for idx, row in top_results.iterrows():
            print(f"ランク {list(top_results.index).index(idx) + 1}")
            print(f"  回収率: {row['recovery_rate']:.2f}%")
            print(f"  的中率: {row['hit_rate']:.2f}%")
            print(f"  購入回数: {row['purchase_count']}回")
            print(f"  損益: {row['profit']:,.0f}円")
            print(f"  平均オッズ: {row['avg_odds']:.2f}倍")
            print(f"  パラメータ:")
            print(f"    - 期待値閾値: {row['ev_threshold']:.2f}")
            print(f"    - レース信頼度閾値: {row['race_confidence_threshold']:.1f}")
            print(f"    - 馬信頼度閾値: very_high={row['horse_conf_very_high']:.0f}, "
                  f"high={row['horse_conf_high']:.0f}, medium={row['horse_conf_medium']:.0f}")
            print()
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        output_path: str = 'results/threshold_optimization_results.csv'
    ):
        """
        結果をCSVファイルに保存
        
        Args:
            results_df: 保存する結果のDataFrame
            output_path: 出力ファイルパス
        """
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n結果を保存しました: {output_path}")


def main():
    """メイン処理"""
    print("="*80)
    print("レース選別閾値最適化 (Task 2-2)")
    print("="*80)
    
    # オプティマイザーの初期化
    optimizer = RaceSelectorOptimizer()
    
    # グリッドサーチ実行
    results_df = optimizer.grid_search(
        ev_thresholds=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        race_confidence_thresholds=[10, 15, 20, 25, 30, 35, 40],
        horse_confidence_configs=[
            {'very_high': 70, 'high': 50, 'medium': 30},  # デフォルト
            {'very_high': 60, 'high': 40, 'medium': 20},  # 緩め
            {'very_high': 80, 'high': 60, 'medium': 40},  # 厳しめ
            {'very_high': 65, 'high': 45, 'medium': 25},  # 中間
        ]
    )
    
    # 結果を保存
    optimizer.save_results(results_df)
    
    # 最適なパラメータを検索
    best_params = optimizer.find_best_params(
        results_df,
        min_recovery_rate=115.0,
        min_purchase_count=50
    )
    
    # サマリー表示
    optimizer.print_summary(best_params, top_n=10)
    
    # トップ結果を詳細表示
    if len(best_params) > 0:
        print("\n" + "="*80)
        print("推奨パラメータ (最高回収率)")
        print("="*80)
        top = best_params.iloc[0]
        print(f"\n期待値閾値: {top['ev_threshold']:.2f}")
        print(f"レース信頼度閾値: {top['race_confidence_threshold']:.1f}")
        print(f"馬信頼度閾値:")
        print(f"  - very_high (100%購入): {top['horse_conf_very_high']:.0f}以上")
        print(f"  - high (70%購入): {top['horse_conf_high']:.0f}以上")
        print(f"  - medium (30%購入): {top['horse_conf_medium']:.0f}以上")
        print(f"\n予測パフォーマンス:")
        print(f"  - 回収率: {top['recovery_rate']:.2f}%")
        print(f"  - 的中率: {top['hit_rate']:.2f}%")
        print(f"  - 購入回数: {top['purchase_count']}回")
        print(f"  - 損益: {top['profit']:,.0f}円")
    else:
        print("\n目標達成可能なパラメータが見つかりませんでした。")
        print("全結果から最良のものを確認してください。")


if __name__ == '__main__':
    main()
