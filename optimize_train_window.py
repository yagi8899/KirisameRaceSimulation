#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最適学習期間探索スクリプト

複数の学習期間でWalk-Forward Validationを実行し、
最も安定した学習期間を特定します。
"""

from walk_forward_validation import run_walk_forward, analyze_results
import pandas as pd


def optimize_train_window(window_candidates=[2, 3, 4, 5, 7, 10], test_years=[2023, 2024, 2025]):
    """
    複数の学習期間でテストして最適値を見つける
    
    Args:
        window_candidates (list): テストする学習期間の候補（年数）
        test_years (list): テスト対象年
    
    Returns:
        dict: 各学習期間の評価結果
    """
    print(f"\n{'='*60}")
    print(f"[*] 最適学習期間探索")
    print(f"   候補: {window_candidates}年")
    print(f"   テスト年: {test_years}")
    print(f"{'='*60}\n")
    
    all_window_results = []
    
    for window in window_candidates:
        print(f"\n{'#'*60}")
        print(f"[#] 学習期間: {window}年でテスト")
        print(f"{'#'*60}\n")
        
        # Walk-Forward Validation実行
        results = run_walk_forward(train_window=window, test_years=test_years)
        
        if not results:
            print(f"[!] 学習期間{window}年のテスト失敗")
            continue
        
        # 統計計算
        df = pd.DataFrame(results)
        tansho_mean = df['単勝的中率'].mean()
        tansho_std = df['単勝的中率'].std()
        tansho_cv = tansho_std / tansho_mean if tansho_mean > 0 else 999
        tansho_min = tansho_mean - 2 * tansho_std
        
        fukusho_mean = df['複勝的中率'].mean()
        sanrenpuku_mean = df['三連複的中率'].mean()
        
        tansho_return = df['単勝回収率'].mean()
        
        all_window_results.append({
            '学習期間': f"{window}年",
            '単勝的中率_平均': tansho_mean,
            '単勝的中率_標準偏差': tansho_std,
            '単勝的中率_変動係数': tansho_cv,
            '単勝的中率_下限95%': tansho_min,
            '複勝的中率_平均': fukusho_mean,
            '三連複的中率_平均': sanrenpuku_mean,
            '単勝回収率_平均': tansho_return,
        })
    
    # 比較表を作成
    comparison_df = pd.DataFrame(all_window_results)
    
    print(f"\n{'='*60}")
    print("[+] 学習期間比較結果")
    print(f"{'='*60}\n")
    
    # 見やすく表示
    for col in ['単勝的中率_平均', '単勝的中率_変動係数', '単勝的中率_下限95%', '単勝回収率_平均']:
        print(f"【{col}】")
        for _, row in comparison_df.iterrows():
            value = row[col]
            if 'CV' in col or '変動係数' in col:
                print(f"  {row['学習期間']:5s}: {value:.3f}")
            else:
                print(f"  {row['学習期間']:5s}: {value:.1%}")
        print()
    
    # 最適期間を判定
    print(f"{'='*60}")
    print("[>] 推奨学習期間")
    print(f"{'='*60}\n")
    
    # 基準1: 変動係数が小さい
    best_by_cv = comparison_df.loc[comparison_df['単勝的中率_変動係数'].idxmin()]
    print(f"最安定（変動係数最小）: {best_by_cv['学習期間']}")
    print(f"  変動係数: {best_by_cv['単勝的中率_変動係数']:.3f}")
    print()
    
    # 基準2: 平均的中率が高い
    best_by_mean = comparison_df.loc[comparison_df['単勝的中率_平均'].idxmax()]
    print(f"最高的中率: {best_by_mean['学習期間']}")
    print(f"  平均的中率: {best_by_mean['単勝的中率_平均']:.1%}")
    print()
    
    # 基準3: 下限が高い（リスク低い）
    best_by_floor = comparison_df.loc[comparison_df['単勝的中率_下限95%'].idxmax()]
    print(f"最高下限（リスク最小）: {best_by_floor['学習期間']}")
    print(f"  下限: {best_by_floor['単勝的中率_下限95%']:.1%}")
    print()
    
    # 総合評価: 変動係数15%未満 & 平均15%超 & 下限10%超
    qualified = comparison_df[
        (comparison_df['単勝的中率_変動係数'] < 0.15) &
        (comparison_df['単勝的中率_平均'] > 0.15) &
        (comparison_df['単勝的中率_下限95%'] > 0.10)
    ]
    
    if len(qualified) > 0:
        # 基準を満たす中で変動係数が最小
        best_overall = qualified.loc[qualified['単勝的中率_変動係数'].idxmin()]
        print(f"[OK] 【総合推奨】: {best_overall['学習期間']}")
        print(f"  変動係数: {best_overall['単勝的中率_変動係数']:.3f}")
        print(f"  平均的中率: {best_overall['単勝的中率_平均']:.1%}")
        print(f"  下限: {best_overall['単勝的中率_下限95%']:.1%}")
        print(f"  -> 本番投入推奨!")
    else:
        print("[NG] 基準を満たす学習期間なし")
        print("   (変動係数<0.15, 平均>15%, 下限>10%)")
        print(f"   -> 次善策: {best_by_cv['学習期間']} (最安定)")
    
    # 結果を保存
    output_file = "results/train_window_optimization.tsv"
    comparison_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8-sig')
    print(f"\n[FILE] 比較結果を {output_file} に保存しました")
    
    return comparison_df


if __name__ == '__main__':
    import sys
    
    # デフォルト設定（実用的な範囲に絞る）
    window_candidates = [5, 7, 10]
    test_years = [2023, 2024, 2025]
    
    # コマンドライン引数で候補を指定可能
    if len(sys.argv) > 1:
        # python optimize_train_window.py 3,5,7
        window_candidates = [int(w) for w in sys.argv[1].split(',')]
    
    if len(sys.argv) > 2:
        # python optimize_train_window.py 3,5,7 2023,2024,2025
        test_years = [int(y) for y in sys.argv[2].split(',')]
    
    # 最適化実行
    results = optimize_train_window(
        window_candidates=window_candidates,
        test_years=test_years
    )
    
    print(f"\n{'='*60}")
    print("[DONE] 最適学習期間探索完了!")
    print(f"{'='*60}")
