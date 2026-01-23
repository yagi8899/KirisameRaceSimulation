#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ranker予測 + Upset分類器の組み合わせ評価

評価する馬券戦略:
1. 三連複: Ranker上位2頭 + Upset候補 のBOX
2. 馬連・ワイド: Ranker上位1頭 + Upset候補 の組み合わせ
3. 単勝・複勝: Rankerで上位かつUpset候補の馬
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


def load_results(file_path: str) -> pd.DataFrame:
    """結果ファイルを読み込む"""
    df = pd.read_csv(file_path, sep='\t')
    return df


def get_race_key(row) -> str:
    """レースを一意に識別するキーを生成"""
    return f"{row['競馬場']}_{row['開催年']}_{row['開催日']}_{row['レース番号']}"


def evaluate_sanrenpuku(df: pd.DataFrame, upset_threshold: float = 0.15,
                        ranker_top_n: int = 2) -> dict:
    """
    三連複評価: Ranker上位N頭 + Upset候補 のBOX
    
    戦略: Ranker上位2頭は軸として固定、Upset候補を相手に流す
    """
    results = {
        'total_races': 0,
        'bet_races': 0,  # 馬券を買ったレース数
        'hits': 0,
        'total_bet': 0,
        'total_return': 0,
        'hit_details': []
    }
    
    # レースごとにグループ化
    race_groups = df.groupby(['競馬場', '開催年', '開催日', 'レース番号'])
    
    for race_key, race_df in race_groups:
        results['total_races'] += 1
        
        # Ranker上位N頭を取得（予測順位が小さい方が上位）
        ranker_top = race_df.nsmallest(ranker_top_n, '予測順位')['馬番'].tolist()
        
        # Upset候補を取得（閾値以上 & 7-12番人気）
        upset_candidates = race_df[
            (race_df['穴馬確率'] >= upset_threshold) & 
            (race_df['人気順'] >= 7) & 
            (race_df['人気順'] <= 12)
        ]['馬番'].tolist()
        
        # Ranker上位がUpset候補に含まれている場合は除外
        upset_candidates = [h for h in upset_candidates if h not in ranker_top]
        
        if len(upset_candidates) == 0:
            continue  # Upset候補がいなければ馬券を買わない
        
        # 馬券の組み合わせ数を計算（Ranker上位2頭 + Upset候補のBOX）
        # 三連複: 上位2頭のうち2頭 + Upset候補のうち1頭
        # = C(2,2) * len(upset) = 1 * len(upset)
        num_bets = len(upset_candidates)
        bet_amount = num_bets * 100  # 1点100円
        
        results['bet_races'] += 1
        results['total_bet'] += bet_amount
        
        # 実際の3着以内を取得
        actual_top3 = race_df[race_df['確定着順'] <= 3]['馬番'].tolist()
        
        # 的中判定: Ranker上位2頭のうち2頭 + Upset候補のうち1頭が3着以内
        ranker_in_top3 = [h for h in ranker_top if h in actual_top3]
        upset_in_top3 = [h for h in upset_candidates if h in actual_top3]
        
        if len(ranker_in_top3) >= 2 and len(upset_in_top3) >= 1:
            # 的中！
            sanrenpuku_odds = race_df['３連複オッズ'].iloc[0]
            if pd.notna(sanrenpuku_odds) and sanrenpuku_odds > 0:
                results['hits'] += 1
                results['total_return'] += sanrenpuku_odds * 100
                results['hit_details'].append({
                    'race_key': f"{race_key[0]}_{race_key[1]}_{race_key[2]}_R{race_key[3]}",
                    'ranker_top': ranker_top,
                    'upset_hit': upset_in_top3,
                    'odds': sanrenpuku_odds,
                    'bet_count': num_bets
                })
    
    # 集計
    results['hit_rate'] = results['hits'] / results['bet_races'] if results['bet_races'] > 0 else 0
    results['roi'] = (results['total_return'] - results['total_bet']) / results['total_bet'] * 100 if results['total_bet'] > 0 else 0
    results['avg_odds'] = np.mean([d['odds'] for d in results['hit_details']]) if results['hit_details'] else 0
    
    return results


def evaluate_umaren_wide(df: pd.DataFrame, upset_threshold: float = 0.15,
                         ranker_top_n: int = 1, bet_type: str = 'umaren') -> dict:
    """
    馬連・ワイド評価: Ranker上位1頭 + Upset候補
    
    bet_type: 'umaren' or 'wide'
    """
    results = {
        'total_races': 0,
        'bet_races': 0,
        'hits': 0,
        'total_bet': 0,
        'total_return': 0,
        'hit_details': []
    }
    
    race_groups = df.groupby(['競馬場', '開催年', '開催日', 'レース番号'])
    
    for race_key, race_df in race_groups:
        results['total_races'] += 1
        
        # Ranker上位N頭
        ranker_top = race_df.nsmallest(ranker_top_n, '予測順位')['馬番'].tolist()
        
        # Upset候補
        upset_candidates = race_df[
            (race_df['穴馬確率'] >= upset_threshold) & 
            (race_df['人気順'] >= 7) & 
            (race_df['人気順'] <= 12)
        ]['馬番'].tolist()
        
        upset_candidates = [h for h in upset_candidates if h not in ranker_top]
        
        if len(upset_candidates) == 0:
            continue
        
        # 馬連: Ranker1頭 × Upset候補
        num_bets = len(ranker_top) * len(upset_candidates)
        bet_amount = num_bets * 100
        
        results['bet_races'] += 1
        results['total_bet'] += bet_amount
        
        # 的中判定
        if bet_type == 'umaren':
            # 馬連: 1-2着の組み合わせ
            actual_top2 = race_df[race_df['確定着順'] <= 2]['馬番'].tolist()
            ranker_in_top2 = [h for h in ranker_top if h in actual_top2]
            upset_in_top2 = [h for h in upset_candidates if h in actual_top2]
            
            if len(ranker_in_top2) >= 1 and len(upset_in_top2) >= 1:
                # 馬連オッズを取得
                umaren_odds = race_df['馬連オッズ'].iloc[0]
                if pd.notna(umaren_odds) and umaren_odds > 0:
                    results['hits'] += 1
                    results['total_return'] += umaren_odds * 100
                    results['hit_details'].append({
                        'race_key': f"{race_key[0]}_{race_key[1]}_{race_key[2]}_R{race_key[3]}",
                        'odds': umaren_odds,
                        'bet_count': num_bets
                    })
        else:
            # ワイド: 1-2, 1-3, 2-3 のいずれか
            actual_top3 = race_df[race_df['確定着順'] <= 3]['馬番'].tolist()
            ranker_in_top3 = [h for h in ranker_top if h in actual_top3]
            upset_in_top3 = [h for h in upset_candidates if h in actual_top3]
            
            if len(ranker_in_top3) >= 1 and len(upset_in_top3) >= 1:
                # ワイドオッズを取得（該当する組み合わせのオッズ）
                # 簡略化: ワイド1-2オッズを使用
                wide_odds = race_df['ワイド1_2オッズ'].iloc[0]
                if pd.notna(wide_odds) and wide_odds > 0:
                    results['hits'] += 1
                    results['total_return'] += wide_odds * 100
                    results['hit_details'].append({
                        'race_key': f"{race_key[0]}_{race_key[1]}_{race_key[2]}_R{race_key[3]}",
                        'odds': wide_odds,
                        'bet_count': num_bets
                    })
    
    results['hit_rate'] = results['hits'] / results['bet_races'] if results['bet_races'] > 0 else 0
    results['roi'] = (results['total_return'] - results['total_bet']) / results['total_bet'] * 100 if results['total_bet'] > 0 else 0
    results['avg_odds'] = np.mean([d['odds'] for d in results['hit_details']]) if results['hit_details'] else 0
    
    return results


def evaluate_fukusho(df: pd.DataFrame, upset_threshold: float = 0.15) -> dict:
    """
    複勝評価: Upset候補の複勝を買う
    
    これは既存の評価と同じだが、Ranker予測順位も考慮した絞り込み
    """
    results = {
        'total_races': 0,
        'bet_races': 0,
        'total_candidates': 0,
        'hits': 0,
        'total_bet': 0,
        'total_return': 0,
        'hit_details': []
    }
    
    race_groups = df.groupby(['競馬場', '開催年', '開催日', 'レース番号'])
    
    for race_key, race_df in race_groups:
        results['total_races'] += 1
        
        # Upset候補
        upset_candidates = race_df[
            (race_df['穴馬確率'] >= upset_threshold) & 
            (race_df['人気順'] >= 7) & 
            (race_df['人気順'] <= 12)
        ]
        
        if len(upset_candidates) == 0:
            continue
        
        results['bet_races'] += 1
        num_bets = len(upset_candidates)
        results['total_candidates'] += num_bets
        results['total_bet'] += num_bets * 100
        
        # 的中判定（3着以内）
        for _, horse in upset_candidates.iterrows():
            if horse['確定着順'] <= 3:
                results['hits'] += 1
                # 複勝オッズを取得（該当馬番のオッズ）
                fukusho_odds = get_fukusho_odds(race_df, horse['馬番'])
                if fukusho_odds > 0:
                    results['total_return'] += fukusho_odds * 100
    
    results['precision'] = results['hits'] / results['total_candidates'] if results['total_candidates'] > 0 else 0
    results['roi'] = (results['total_return'] - results['total_bet']) / results['total_bet'] * 100 if results['total_bet'] > 0 else 0
    
    return results


def get_fukusho_odds(race_df: pd.DataFrame, uma_ban: int) -> float:
    """馬番に対応する複勝オッズを取得"""
    row = race_df.iloc[0]
    
    for i, col in enumerate(['複勝1着馬番', '複勝2着馬番', '複勝3着馬番']):
        if row[col] == uma_ban:
            odds_col = f'複勝{i+1}着オッズ'
            return row[odds_col] if pd.notna(row[odds_col]) else 0
    
    return 0


def evaluate_ranker_upset_combination(df: pd.DataFrame, upset_threshold: float = 0.15,
                                       ranker_filter: int = None) -> dict:
    """
    Ranker順位でフィルタしたUpset候補の評価
    
    ranker_filter: Ranker予測順位がこの値以下のUpset候補のみ選択
                   (例: 6なら予測順位1-6位のUpset候補のみ)
    """
    results = {
        'total_candidates': 0,
        'hits': 0,
        'total_bet': 0,
        'total_return': 0
    }
    
    # Upset候補を抽出
    upset_candidates = df[
        (df['穴馬確率'] >= upset_threshold) & 
        (df['人気順'] >= 7) & 
        (df['人気順'] <= 12)
    ].copy()
    
    # Ranker順位でフィルタ
    if ranker_filter:
        upset_candidates = upset_candidates[upset_candidates['予測順位'] <= ranker_filter]
    
    results['total_candidates'] = len(upset_candidates)
    results['total_bet'] = results['total_candidates'] * 100
    
    # 的中判定
    hits = upset_candidates[upset_candidates['確定着順'] <= 3]
    results['hits'] = len(hits)
    
    # 複勝オッズ計算（簡略化）
    for _, horse in hits.iterrows():
        # 複勝オッズを取得
        race_df = df[(df['競馬場'] == horse['競馬場']) & 
                     (df['開催年'] == horse['開催年']) &
                     (df['開催日'] == horse['開催日']) &
                     (df['レース番号'] == horse['レース番号'])]
        fukusho_odds = get_fukusho_odds(race_df, horse['馬番'])
        results['total_return'] += fukusho_odds * 100
    
    results['precision'] = results['hits'] / results['total_candidates'] if results['total_candidates'] > 0 else 0
    results['roi'] = (results['total_return'] - results['total_bet']) / results['total_bet'] * 100 if results['total_bet'] > 0 else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ranker + Upset組み合わせ評価')
    parser.add_argument('input_file', help='予測結果ファイル（TSV）')
    parser.add_argument('--threshold', type=float, default=0.15, help='Upset閾値')
    parser.add_argument('--by-track', action='store_true', help='競馬場別に評価')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Ranker予測 + Upset分類器 組み合わせ評価")
    print("=" * 70)
    
    df = load_results(args.input_file)
    print(f"\n読み込みデータ: {len(df)} 行")
    print(f"Upset閾値: {args.threshold}")
    
    # 1. 複勝（Upset候補のみ）- 既存評価
    print("\n" + "=" * 70)
    print("【戦略1】複勝: Upset候補を買う")
    print("=" * 70)
    
    fukusho_result = evaluate_fukusho(df, args.threshold)
    print(f"  対象レース: {fukusho_result['bet_races']} / {fukusho_result['total_races']}")
    print(f"  候補数: {fukusho_result['total_candidates']}")
    print(f"  的中数: {fukusho_result['hits']}")
    print(f"  Precision: {fukusho_result['precision']:.2%}")
    print(f"  投資額: {fukusho_result['total_bet']:,}円")
    print(f"  回収額: {fukusho_result['total_return']:,.0f}円")
    print(f"  ROI: {fukusho_result['roi']:+.1f}%")
    
    # 2. 複勝（Ranker予測順位でフィルタ）
    print("\n" + "=" * 70)
    print("【戦略2】複勝: Ranker予測上位のUpset候補に絞る")
    print("=" * 70)
    
    for ranker_filter in [3, 5, 8, None]:
        filter_name = f"Ranker上位{ranker_filter}位以内" if ranker_filter else "フィルタなし"
        result = evaluate_ranker_upset_combination(df, args.threshold, ranker_filter)
        print(f"\n  {filter_name}:")
        print(f"    候補数: {result['total_candidates']}")
        print(f"    的中数: {result['hits']}")
        print(f"    Precision: {result['precision']:.2%}")
        print(f"    ROI: {result['roi']:+.1f}%")
    
    # 3. 馬連（Ranker1位 × Upset候補）
    print("\n" + "=" * 70)
    print("【戦略3】馬連: Ranker予測1位 × Upset候補")
    print("=" * 70)
    
    umaren_result = evaluate_umaren_wide(df, args.threshold, ranker_top_n=1, bet_type='umaren')
    print(f"  対象レース: {umaren_result['bet_races']} / {umaren_result['total_races']}")
    print(f"  的中数: {umaren_result['hits']}")
    print(f"  的中率: {umaren_result['hit_rate']:.2%}")
    print(f"  平均オッズ: {umaren_result['avg_odds']:.1f}倍")
    print(f"  投資額: {umaren_result['total_bet']:,}円")
    print(f"  回収額: {umaren_result['total_return']:,.0f}円")
    print(f"  ROI: {umaren_result['roi']:+.1f}%")
    
    # 4. ワイド（Ranker1位 × Upset候補）
    print("\n" + "=" * 70)
    print("【戦略4】ワイド: Ranker予測1位 × Upset候補")
    print("=" * 70)
    
    wide_result = evaluate_umaren_wide(df, args.threshold, ranker_top_n=1, bet_type='wide')
    print(f"  対象レース: {wide_result['bet_races']} / {wide_result['total_races']}")
    print(f"  的中数: {wide_result['hits']}")
    print(f"  的中率: {wide_result['hit_rate']:.2%}")
    print(f"  平均オッズ: {wide_result['avg_odds']:.1f}倍")
    print(f"  投資額: {wide_result['total_bet']:,}円")
    print(f"  回収額: {wide_result['total_return']:,.0f}円")
    print(f"  ROI: {wide_result['roi']:+.1f}%")
    
    # 5. 三連複（Ranker上位2頭 + Upset候補）
    print("\n" + "=" * 70)
    print("【戦略5】三連複: Ranker予測上位2頭 + Upset候補 BOX")
    print("=" * 70)
    
    sanren_result = evaluate_sanrenpuku(df, args.threshold, ranker_top_n=2)
    print(f"  対象レース: {sanren_result['bet_races']} / {sanren_result['total_races']}")
    print(f"  的中数: {sanren_result['hits']}")
    print(f"  的中率: {sanren_result['hit_rate']:.2%}")
    print(f"  平均オッズ: {sanren_result['avg_odds']:.1f}倍")
    print(f"  投資額: {sanren_result['total_bet']:,}円")
    print(f"  回収額: {sanren_result['total_return']:,.0f}円")
    print(f"  ROI: {sanren_result['roi']:+.1f}%")
    
    if sanren_result['hit_details']:
        print(f"\n  的中レース（上位5件）:")
        for detail in sorted(sanren_result['hit_details'], key=lambda x: x['odds'], reverse=True)[:5]:
            print(f"    {detail['race_key']}: {detail['odds']:.1f}倍 (点数: {detail['bet_count']})")
    
    # サマリー
    print("\n" + "=" * 70)
    print("【サマリー】")
    print("=" * 70)
    print(f"{'戦略':<30} {'ROI':>10} {'的中率/Prec':>12}")
    print("-" * 54)
    print(f"{'複勝（Upset候補）':<28} {fukusho_result['roi']:>+9.1f}% {fukusho_result['precision']:>11.2%}")
    print(f"{'馬連（Ranker1位×Upset）':<26} {umaren_result['roi']:>+9.1f}% {umaren_result['hit_rate']:>11.2%}")
    print(f"{'ワイド（Ranker1位×Upset）':<25} {wide_result['roi']:>+9.1f}% {wide_result['hit_rate']:>11.2%}")
    print(f"{'三連複（Ranker2頭+Upset）':<25} {sanren_result['roi']:>+9.1f}% {sanren_result['hit_rate']:>11.2%}")


if __name__ == '__main__':
    main()
