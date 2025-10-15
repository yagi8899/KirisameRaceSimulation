#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bet_evaluator.py

説明:
  入力ファイル（CSV/TSV）を読み込み、各レースごとに
  単勝・複勝・馬連・馬単・ワイド・三連複の的中数・購入数・的中率・回収率を計算し、
  全レースの集計結果を出力CSVに保存するスクリプト。

使い方:
  python bet_evaluator.py --input data.tsv --output summary.csv

注意:
  - 賭け金は1枚=100円固定。
  - オッズは倍率（払戻 = オッズ * 賭け金）として扱います。
"""

import argparse
import pandas as pd
import os
import sys
from itertools import combinations, permutations

def read_table_auto(path):
    # 自動的に区切り文字を判定して読み込む（優先: タブ, カンマ）
    with open(path, 'r', encoding='utf-8') as f:
        head = f.read(2048)
    if '\t' in head:
        sep = '\t'
    else:
        sep = ','
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    return df

def to_numeric_safe(s):
    try:
        if s is None or s == '':
            return None
        return float(s)
    except:
        return None

def make_race_id(row):
    # レース識別子として複数カラムを連結
    keys = ['競馬場','開催年','開催日','レース番号']
    vals = []
    for k in keys:
        if k in row and str(row[k]) != '':
            vals.append(str(row[k]))
        else:
            vals.append('') 
    return "_".join(vals)

def parse_payouts_for_race(df_race):
    # レース内の代表行（最初の行）から払戻情報を取得する
    row = df_race.iloc[0]
    payouts = {}

    # 単勝オッズは馬ごとにあるので利用しない（馬ごとの列から使う）
    # 複勝: 複勝1着馬番, 複勝1着オッズ, 複勝2着馬番, 複勝2着オッズ, 複勝3着馬番, 複勝3着オッズ
    payouts['fukusho'] = []
    for i in [1,2,3]:
        bcol = f'複勝{i}着馬番'
        ocol = f'複勝{i}着オッズ'
        if bcol in row and ocol in row:
            try:
                b = int(str(row[bcol]).strip())
                o = to_numeric_safe(row[ocol])
                payouts['fukusho'].append((b,o))
            except:
                pass

    # 馬連
    if '馬連馬番1' in row and '馬連馬番2' in row and '馬連オッズ' in row:
        try:
            ml1 = int(str(row['馬連馬番1']).strip())
            ml2 = int(str(row['馬連馬番2']).strip())
            mlo = to_numeric_safe(row['馬連オッズ'])
            payouts['馬連'] = { tuple(sorted((ml1,ml2))): mlo }
        except:
            payouts['馬連'] = {}
    else:
        payouts['馬連'] = {}

    # 馬単
    if '馬単馬番1' in row and '馬単馬番2' in row and '馬単オッズ' in row:
        try:
            m1 = int(str(row['馬単馬番1']).strip())
            m2 = int(str(row['馬単馬番2']).strip())
            mo = to_numeric_safe(row['馬単オッズ'])
            payouts['馬単'] = { (m1,m2): mo }
        except:
            payouts['馬単'] = {}
    else:
        payouts['馬単'] = {}

    # ワイド: ワイド1_2馬番1, ワイド1_2馬番2, ワイド1_2オッズ, ワイド2_3..., ワイド1_3...
    payouts['ワイド'] = {}
    for tag in ['1_2','2_3','1_3']:
        b1 = f'ワイド{tag}馬番1'
        b2 = f'ワイド{tag}馬番2'
        ocol = f'ワイド{tag}オッズ'
        if b1 in row and b2 in row and ocol in row:
            try:
                w1 = int(str(row[b1]).strip())
                w2 = int(str(row[b2]).strip())
                wo = to_numeric_safe(row[ocol])
                payouts['ワイド'][tuple(sorted((w1,w2)))] = wo
            except:
                pass

    # ３連複オッズ
    if '３連複オッズ' in row:
        payouts['三連複'] = to_numeric_safe(row['３連複オッズ'])
        # もし３連複の組番が別列にある場合は今は対応していない（多くのデータはオッズだけ）
    else:
        payouts['三連複'] = None

    return payouts

def evaluate(df):
    # 必須の数値カラムを整形
    # horse number column name candidates: '馬番'
    if '馬番' not in df.columns:
        raise RuntimeError('入力データに「馬番」列が見つかりません。')

    # ensure numeric columns
    df['馬番_int'] = df['馬番'].apply(lambda x: int(str(x).strip()) if str(x).strip()!='' else None)
    # 確定着順
    if '確定着順' not in df.columns:
        raise RuntimeError('入力データに「確定着順」列が見つかりません。')
    df['着順_int'] = df['確定着順'].apply(lambda x: int(str(x).strip()) if str(x).strip()!='' else None)
    # 予測順位
    if '予測順位' not in df.columns:
        raise RuntimeError('入力データに「予測順位」列が見つかりません。')
    df['予測順位_int'] = df['予測順位'].apply(lambda x: int(float(str(x).strip())) if str(x).strip()!='' else None)

    # 単勝オッズ（馬ごと）
    if '単勝オッズ' in df.columns:
        df['単勝_odds'] = df['単勝オッズ'].apply(lambda x: to_numeric_safe(x))
    else:
        df['単勝_odds'] = None

    # group by race
    df['race_id'] = df.apply(make_race_id, axis=1)
    races = df['race_id'].unique()

    # initialize accumulators
    types = ['単勝','複勝','馬連','馬単','ワイド','三連複']
    stats = {t: {'tickets':0, 'hits':0, 'spent':0.0, 'return':0.0} for t in types}

    # per-race detailed output
    per_race = []

    for race in races:
        df_r = df[df['race_id']==race].copy()
        # map horse number to finish and to predicted rank and to single win odds
        horse_finish = {}
        horse_pred = {}
        horse_win_odds = {}
        for _, r in df_r.iterrows():
            hn = r['馬番_int']
            horse_finish[hn] = r['着順_int']
            horse_pred[hn] = r['予測順位_int']
            horse_win_odds[hn] = r['単勝_odds']

        payouts = parse_payouts_for_race(df_r)

        # Predicted set for ranks 1..3 (include duplicates -> set of horses whose predicted rank <=3)
        predicted_top3 = [hn for hn,pr in horse_pred.items() if pr is not None and pr<=3]
        predicted_top1 = [hn for hn,pr in horse_pred.items() if pr is not None and pr==1]

        # ACTUAL top positions
        # find numbers of 1st,2nd,3rd horses
        finish_to_horse = {}
        for hn, pos in horse_finish.items():
            if pos is not None:
                finish_to_horse.setdefault(pos, hn)
        actual1 = finish_to_horse.get(1, None)
        actual2 = finish_to_horse.get(2, None)
        actual3 = finish_to_horse.get(3, None)
        actual_top2_set = set([h for h in (actual1,actual2) if h is not None])
        actual_top3_set = set([h for h in (actual1,actual2,actual3) if h is not None])

        # ----- 単勝 -----
        unit = 100.0
        for hn in predicted_top1:
            stats['単勝']['tickets'] += 1
            stats['単勝']['spent'] += unit
            # win?
            if actual1 is not None and hn == actual1:
                stats['単勝']['hits'] += 1
                # payout: use horse's 単勝オッズ if available, else 0
                o = horse_win_odds.get(hn)
                if o is None:
                    payout = 0.0
                else:
                    payout = (o) * unit
                stats['単勝']['return'] += payout

        # ----- 複勝 -----
        # For each predicted horse (pred <=3), buy one複勝 ticket
        # Determine payout: find the finishing position of that horse and use corresponding 複勝N着オッズ mapping if available.
        # If mapping not present, we skip payout (assume 0)
        # Build dict place->(horse,odds) from payouts['fukusho']
        fukusho_map = {}
        for t in payouts.get('fukusho', []):
            try:
                b,o = t
                fukusho_map[b] = o
            except:
                pass

        for hn in predicted_top3:
            stats['複勝']['tickets'] += 1
            stats['複勝']['spent'] += unit
            pos = horse_finish.get(hn)
            if pos is not None and pos <= 3:
                stats['複勝']['hits'] += 1
                # payout: if fukusho_map contains the horse -> use that odds; else fallback to None -> assume 0
                o = fukusho_map.get(hn)
                if o is None:
                    payout = 0.0
                else:
                    payout = o * unit
                stats['複勝']['return'] += payout

        # ----- 馬連 -----
        # buy all unordered pairs among predicted_top3 (combinations 2)
        pairs = list(combinations(predicted_top3, 2))
        for pair in pairs:
            pair_sorted = tuple(sorted(pair))
            stats['馬連']['tickets'] += 1
            stats['馬連']['spent'] += unit
            # win if actual top2 set == pair_sorted set (unordered)
            if actual_top2_set == set(pair_sorted):
                stats['馬連']['hits'] += 1
                # if payout listed for this pair, use it; else 0
                o = payouts.get('馬連', {}).get(pair_sorted)
                if o is None:
                    payout = 0.0
                else:
                    payout = o * unit
                stats['馬連']['return'] += payout

        # ----- 馬単 -----
        # buy all ordered pairs among predicted_top3 (permutations of 2)
        ordered_pairs = list(permutations(predicted_top3, 2))
        for op in ordered_pairs:
            stats['馬単']['tickets'] += 1
            stats['馬単']['spent'] += unit
            # win if actual1 == op[0] and actual2 == op[1]
            if actual1 is not None and actual2 is not None and actual1 == op[0] and actual2 == op[1]:
                stats['馬単']['hits'] += 1
                o = payouts.get('馬単', {}).get((op[0],op[1]))
                if o is None:
                    payout = 0.0
                else:
                    payout = o * unit
                stats['馬単']['return'] += payout

        # ----- ワイド -----
        # buy all unordered pairs among predicted_top3 (same as 馬連 pairs)
        for pair in pairs:
            pair_sorted = tuple(sorted(pair))
            stats['ワイド']['tickets'] += 1
            stats['ワイド']['spent'] += unit
            # win if both horses finish in top3 (wide pays if both in top3 regardless of order) BUT
            # typical ワイド is any two horses finishing in top3 (i.e., both among the three placings).
            # We'll treat win condition as: both horses in actual_top3_set.
            if pair_sorted[0] in actual_top3_set and pair_sorted[1] in actual_top3_set:
                stats['ワイド']['hits'] += 1
                o = payouts.get('ワイド', {}).get(pair_sorted)
                if o is None:
                    payout = 0.0
                else:
                    payout = o * unit
                stats['ワイド']['return'] += payout

        # ----- 三連複 -----
        # buy all unordered 3-combinations among predicted_top3 (if predicted_top3 has <3 horses, no tickets)
        triplets = list(combinations(predicted_top3, 3))
        for trip in triplets:
            stats['三連複']['tickets'] += 1
            stats['三連複']['spent'] += unit
            # win if set(trip) == actual_top3_set
            if set(trip) == actual_top3_set:
                stats['三連複']['hits'] += 1
                o = payouts.get('三連複')
                if o is None:
                    payout = 0.0
                else:
                    payout = o * unit
                stats['三連複']['return'] += payout

        # collect per-race stats if needed
        per_race.append({
            'race_id': race,
            'predicted_top1_count': len(predicted_top1),
            'predicted_top3_count': len(predicted_top3),
            'actual_top3': list(actual_top3_set),
        })

    # finalize metrics
    results = []
    for t in types:
        s = stats[t]
        tickets = s['tickets']
        hits = s['hits']
        spent = s['spent']
        ret = s['return']
        hit_rate = (hits / tickets) if tickets>0 else None
        roi = (ret / spent * 100.0) if spent>0 else None
        results.append({
            'bet_type': t,
            'tickets': tickets,
            'hits': hits,
            'hit_rate': hit_rate,
            'spent_yen': spent,
            'return_yen': ret,
            'return_rate_percent': roi
        })
    return results, stats

def save_results(results, out_path):
    df = pd.DataFrame(results)
    # format percentages
    df['hit_rate'] = df['hit_rate'].apply(lambda x: '{:.2%}'.format(x) if x is not None else 'N/A')
    df['return_rate_percent'] = df['return_rate_percent'].apply(lambda x: '{:.2f}%'.format(x) if x is not None else 'N/A')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to: {out_path}")

def main():
    p = argparse.ArgumentParser(description='Evaluate betting results from predicted ranks and actual finishes.')
    p.add_argument('--input', '-i', required=True, help='入力データファイルパス (CSV or TSV)')
    p.add_argument('--output', '-o', required=True, help='出力サマリーCSVファイルパス')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print('入力ファイルが見つかりません:', args.input, file=sys.stderr)
        sys.exit(1)

    df = read_table_auto(args.input)
    results, stats = evaluate(df)
    save_results(results, args.output)

    # also print a human-readable summary
    print("Summary:")
    for r in results:
        print(f"- {r['bet_type']}: tickets={r['tickets']}, hits={r['hits']}, hit_rate={r['hit_rate']}, spent={r['spent_yen']:.0f}円, return={r['return_yen']:.0f}円, return_rate={r['return_rate_percent']}")

if __name__ == '__main__':
    main()
