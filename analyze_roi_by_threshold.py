"""
閾値別ROI（収支）分析スクリプト
複勝配当を使って各閾値での期待収支を計算する
"""
import pandas as pd
import numpy as np
import sys

def get_fukusho_payout(row):
    """複勝配当を取得"""
    umaban = row['馬番']
    chakujun = row['確定着順']
    
    # 3着以内でなければ0
    if chakujun > 3:
        return 0
    
    # 複勝1-3着の馬番とオッズをチェック
    for i in [1, 2, 3]:
        umaban_col = f'複勝{i}着馬番'
        odds_col = f'複勝{i}着オッズ'
        if umaban_col in row.index and odds_col in row.index:
            if row[umaban_col] == umaban:
                return row[odds_col] if pd.notna(row[odds_col]) else 0
    return 0

def analyze_roi(file_path, by_track=False):
    """ROI分析"""
    # データ読み込み
    df = pd.read_csv(file_path, sep='\t')
    print(f'総レコード: {len(df)}')
    
    # 7-12番人気のみ
    df_upset = df[(df['人気順'] >= 7) & (df['人気順'] <= 12)].copy()
    print(f'7-12番人気: {len(df_upset)}')
    
    # 実際の穴馬（3着以内）
    df_upset['is_actual_upset'] = (df_upset['確定着順'] <= 3).astype(int)
    print(f'実際の穴馬: {df_upset["is_actual_upset"].sum()}')
    
    # 複勝配当を取得
    df_upset['fukusho_payout'] = df_upset.apply(get_fukusho_payout, axis=1)
    print(f'\n複勝配当取得成功: {(df_upset["fukusho_payout"] > 0).sum()}件')
    print(f'平均複勝配当（的中時）: {df_upset[df_upset["fukusho_payout"] > 0]["fukusho_payout"].mean():.2f}倍')
    
    def calc_roi_table(data, label="全体"):
        """ROIテーブル計算"""
        print(f'\n{"="*90}')
        print(f'[{label}] 各閾値でのROI（収支）分析')
        print(f'{"="*90}')
        print(f'{"閾値":>6} {"候補数":>8} {"的中":>6} {"投資額":>10} {"払戻額":>10} {"収支":>10} {"ROI":>8} {"Precision":>10}')
        print('-'*90)
        
        best_roi = -100
        best_threshold = 0.10
        best_profit = -999999
        
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            candidates = data[data['穴馬確率'] >= threshold]
            n_candidates = len(candidates)
            
            if n_candidates == 0:
                continue
            
            n_hit = candidates['is_actual_upset'].sum()
            investment = n_candidates * 100
            payout = candidates['fukusho_payout'].sum() * 100
            profit = payout - investment
            roi = (payout / investment - 1) * 100 if investment > 0 else 0
            precision = n_hit / n_candidates * 100 if n_candidates > 0 else 0
            
            marker = ''
            if roi > best_roi:
                best_roi = roi
                best_threshold = threshold
                best_profit = profit
                marker = ' **BEST**'
            
            print(f'{threshold:>6.2f} {n_candidates:>8} {n_hit:>6} {investment:>10,} {payout:>10,.0f} {profit:>+10,.0f} {roi:>+7.1f}% {precision:>9.2f}%{marker}')
        
        print('-'*90)
        print(f'Best ROI: threshold={best_threshold}, ROI={best_roi:+.1f}%, profit={best_profit:+,.0f}yen')
        return best_threshold, best_roi, best_profit
    
    # 全体分析
    calc_roi_table(df_upset, "全体")
    
    # 競馬場別分析
    if by_track:
        for track in df_upset['競馬場'].unique():
            track_data = df_upset[df_upset['競馬場'] == track]
            if len(track_data) > 0:
                calc_roi_table(track_data, track)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "check_results/predicted_results_all.tsv"
    by_track = "--by-track" in sys.argv
    analyze_roi(file_path, by_track)
