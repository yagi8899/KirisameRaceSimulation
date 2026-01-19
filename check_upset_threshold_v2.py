#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
閾値調査スクリプト（修正版）
実際の穴馬（7-12番人気かつ3着以内）の確率分布を調べる
"""

import pandas as pd
import numpy as np

def analyze_upset_threshold():
    """穴馬の確率分布を分析"""
    
    # 阪神短距離の結果ファイルを読み込み（一番候補が多かったやつ）
    try:
        file_path = "results/predicted_results_hanshin_turf_3ageup_short_trainunknown_test2023_all.tsv"
        df = pd.read_csv(file_path, sep='\t')
        print(f"[INFO] データ読み込み完了: {len(df)}件")
        
        # 穴馬の定義：7-12番人気かつ3着以内
        upset_horses = df[
            (df['人気順'] >= 7) & (df['人気順'] <= 12) & 
            (df['確定着順'] <= 3)
        ].copy()
        
        print(f"\n[UPSET] 実際の穴馬: {len(upset_horses)}頭")
        
        if len(upset_horses) > 0:
            print(f"[UPSET] 人気範囲: {upset_horses['人気順'].min()}〜{upset_horses['人気順'].max()}番人気")
            print(f"[UPSET] 穴馬確率範囲: {upset_horses['穴馬確率'].min():.4f}〜{upset_horses['穴馬確率'].max():.4f}")
            
            # 穴馬の詳細
            print(f"\n[UPSET] 穴馬詳細（上位10頭）:")
            for idx, row in upset_horses.head(10).iterrows():
                print(f"  {row['馬名']:8} {row['人気順']:2d}番人気 → {row['確定着順']}着 確率{row['穴馬確率']:.4f}")
                
        else:
            print("[UPSET] 穴馬が0頭！これが問題の原因")
            
        # 全体の人気と着順分布
        print(f"\n[STATS] 人気別着順分布:")
        for ninki in range(7, 13):
            horses_in_ninki = df[df['人気順'] == ninki]
            if len(horses_in_ninki) > 0:
                in_top3 = len(horses_in_ninki[horses_in_ninki['確定着順'] <= 3])
                print(f"  {ninki:2d}番人気: {len(horses_in_ninki):3d}頭中 {in_top3:2d}頭が3着以内 ({in_top3/len(horses_in_ninki)*100:.1f}%)")
                
        # 穴馬候補の分析
        upset_candidates = df[df['穴馬候補'] == True]
        print(f"\n[CANDIDATE] 穴馬候補: {len(upset_candidates)}頭")
        if len(upset_candidates) > 0:
            print(f"[CANDIDATE] 人気範囲: {upset_candidates['人気順'].min()}〜{upset_candidates['人気順'].max()}番人気")
            print(f"[CANDIDATE] 確率範囲: {upset_candidates['穴馬確率'].min():.4f}〜{upset_candidates['穴馬確率'].max():.4f}")
            candidate_hits = len(upset_candidates[upset_candidates['確定着順'] <= 3])
            print(f"[CANDIDATE] 3着以内: {candidate_hits}頭 ({candidate_hits/len(upset_candidates)*100:.1f}%)")
            
        # 閾値分析
        print(f"\n[THRESHOLD] 現在の閾値: 0.4")
        print(f"[THRESHOLD] 0.4以上の馬数: {len(df[df['穴馬確率'] >= 0.4])}頭")
        print(f"[THRESHOLD] 0.3以上の馬数: {len(df[df['穴馬確率'] >= 0.3])}頭")
        print(f"[THRESHOLD] 0.2以上の馬数: {len(df[df['穴馬確率'] >= 0.2])}頭")
        print(f"[THRESHOLD] 0.1以上の馬数: {len(df[df['穴馬確率'] >= 0.1])}頭")
        
    except Exception as e:
        print(f"[ERROR] ファイル読み込みエラー: {e}")
        print(f"[DEBUG] 試行ファイル: {file_path}")

if __name__ == "__main__":
    analyze_upset_threshold()