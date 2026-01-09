"""
データ品質チェックスクリプト

目的:
- 欠損値の確認
- 外れ値の検出
- データ分布の確認
- 特徴量の統計サマリー

実行:
    python data_quality_check.py
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# データベース接続情報
DB_PARAMS = {
    'dbname': 'keiba',
    'user': 'postgres',
    'password': 'ahtaht88',
    'host': 'localhost',
    'port': '5432'
}

def check_missing_values(conn):
    """欠損値チェック"""
    print("=" * 80)
    print("【1. 欠損値チェック】")
    print("=" * 80)
    
    # 主要なテーブルの欠損値確認
    query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(se.bamei) as bamei_count,
        COUNT(se.kishu_code) as kishu_code_count,
        COUNT(se.chokyoshi_code) as chokyoshi_code_count,
        COUNT(se.futan_juryo) as futan_count,
        COUNT(se.tansho_odds) as tansho_odds_count,
        COUNT(se.tansho_ninkijun) as ninki_jun_count,
        COUNT(se.kakutei_chakujun) as kakutei_chakujun_count
    FROM jvd_se se
    INNER JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen 
        AND se.kaisai_tsukihi = ra.kaisai_tsukihi 
        AND se.keibajo_code = ra.keibajo_code 
        AND se.race_bango = ra.race_bango
    WHERE CAST(ra.kaisai_nen AS INTEGER) >= 2020 
        AND CAST(ra.kaisai_nen AS INTEGER) <= 2023
    """
    
    df = pd.read_sql(query, conn)
    
    print(f"\n総レコード数: {df['total_records'].iloc[0]:,}件\n")
    
    columns = {
        'bamei_count': '馬名',
        'kishu_code_count': '騎手コード',
        'chokyoshi_code_count': '調教師コード',
        'futan_count': '斤量',
        'tansho_odds_count': '単勝オッズ',
        'ninki_jun_count': '人気順',
        'kakutei_chakujun_count': '確定着順'
    }
    
    total = df['total_records'].iloc[0]
    
    for col, name in columns.items():
        count = df[col].iloc[0]
        missing = total - count
        missing_rate = (missing / total * 100) if total > 0 else 0
        status = "✅" if missing_rate < 1 else "⚠️" if missing_rate < 5 else "❌"
        print(f"{status} {name:15s}: {count:,}件 / {total:,}件 (欠損率: {missing_rate:.2f}%)")


def check_outliers(conn):
    """外れ値チェック"""
    print("\n" + "=" * 80)
    print("【2. 外れ値チェック】")
    print("=" * 80)
    
    # 数値データの統計量取得
    query = """
    SELECT 
        CAST(se.futan_juryo AS FLOAT) as futan,
        NULLIF(CAST(se.tansho_odds AS FLOAT), 0) / 10 as tansho_odds,
        CAST(se.tansho_ninkijun AS INTEGER) as ninki_jun,
        CAST(se.kakutei_chakujun AS INTEGER) as kakutei_chakujun
    FROM jvd_se se
    INNER JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen 
        AND se.kaisai_tsukihi = ra.kaisai_tsukihi 
        AND se.keibajo_code = ra.keibajo_code 
        AND se.race_bango = ra.race_bango
    WHERE CAST(ra.kaisai_nen AS INTEGER) >= 2020 
        AND CAST(ra.kaisai_nen AS INTEGER) <= 2023
        AND se.kakutei_chakujun NOT IN ('00', '99', '')
        AND se.kakutei_chakujun ~ '^[0-9]+$'
    """
    
    df = pd.read_sql(query, conn)
    
    # 確定着順を数値化
    df['kakutei_chakujun'] = pd.to_numeric(df['kakutei_chakujun'], errors='coerce')
    
    print(f"\n分析対象: {len(df):,}件\n")
    
    # 斤量チェック
    print("【斤量（futan）】")
    print(f"  平均: {df['futan'].mean():.1f}kg")
    print(f"  中央値: {df['futan'].median():.1f}kg")
    print(f"  標準偏差: {df['futan'].std():.1f}kg")
    print(f"  最小値: {df['futan'].min():.1f}kg")
    print(f"  最大値: {df['futan'].max():.1f}kg")
    
    # 異常値検出（3σルール）
    mean = df['futan'].mean()
    std = df['futan'].std()
    outliers = df[(df['futan'] < mean - 3*std) | (df['futan'] > mean + 3*std)]
    print(f"  外れ値（3σ超）: {len(outliers)}件 ({len(outliers)/len(df)*100:.2f}%)")
    
    # 単勝オッズチェック
    print("\n【単勝オッズ（tansho_odds）】")
    print(f"  平均: {df['tansho_odds'].mean():.1f}倍")
    print(f"  中央値: {df['tansho_odds'].median():.1f}倍")
    print(f"  標準偏差: {df['tansho_odds'].std():.1f}倍")
    print(f"  最小値: {df['tansho_odds'].min():.1f}倍")
    print(f"  最大値: {df['tansho_odds'].max():.1f}倍")
    
    # オッズの分布
    print(f"  1~3倍: {len(df[df['tansho_odds'] <= 3])}件 ({len(df[df['tansho_odds'] <= 3])/len(df)*100:.1f}%)")
    print(f"  3~10倍: {len(df[(df['tansho_odds'] > 3) & (df['tansho_odds'] <= 10)])}件 ({len(df[(df['tansho_odds'] > 3) & (df['tansho_odds'] <= 10)])/len(df)*100:.1f}%)")
    print(f"  10~50倍: {len(df[(df['tansho_odds'] > 10) & (df['tansho_odds'] <= 50)])}件 ({len(df[(df['tansho_odds'] > 10) & (df['tansho_odds'] <= 50)])/len(df)*100:.1f}%)")
    print(f"  50倍超: {len(df[df['tansho_odds'] > 50])}件 ({len(df[df['tansho_odds'] > 50])/len(df)*100:.1f}%)")
    
    # 人気順チェック
    print("\n【人気順（ninki_jun）】")
    print(f"  平均: {df['ninki_jun'].mean():.1f}番人気")
    print(f"  中央値: {df['ninki_jun'].median():.1f}番人気")
    print(f"  最小値: {df['ninki_jun'].min():.0f}番人気")
    print(f"  最大値: {df['ninki_jun'].max():.0f}番人気")
    
    # 着順チェック
    print("\n【確定着順（kakutei_chakujun）】")
    print(f"  平均: {df['kakutei_chakujun'].mean():.1f}着")
    print(f"  中央値: {df['kakutei_chakujun'].median():.1f}着")
    print(f"  最小値: {df['kakutei_chakujun'].min():.0f}着")
    print(f"  最大値: {df['kakutei_chakujun'].max():.0f}着")


def check_race_conditions(conn):
    """レース条件の分布チェック"""
    print("\n" + "=" * 80)
    print("【3. レース条件の分布】")
    print("=" * 80)
    
    # 馬場状態の分布（芝）
    query = """
    SELECT 
        babajotai_code_shiba as baba_jotai,
        COUNT(*) as count
    FROM jvd_ra
    WHERE CAST(kaisai_nen AS INTEGER) >= 2020 
        AND CAST(kaisai_nen AS INTEGER) <= 2023
        AND CAST(track_code AS INTEGER) BETWEEN 10 AND 22
        AND babajotai_code_shiba IS NOT NULL
    GROUP BY babajotai_code_shiba
    ORDER BY count DESC
    """
    
    df = pd.read_sql(query, conn)
    
    print("\n【芝馬場状態】")
    total = df['count'].sum()
    for _, row in df.iterrows():
        baba = row['baba_jotai']
        count = row['count']
        ratio = count / total * 100
        baba_name = {'1': '良', '2': '稍重', '3': '重', '4': '不良'}.get(baba, baba)
        print(f"  {baba_name:5s}: {count:,}件 ({ratio:.1f}%)")
    
    # ダート馬場状態の分布
    query = """
    SELECT 
        babajotai_code_dirt as baba_jotai,
        COUNT(*) as count
    FROM jvd_ra
    WHERE CAST(kaisai_nen AS INTEGER) >= 2020 
        AND CAST(kaisai_nen AS INTEGER) <= 2023
        AND CAST(track_code AS INTEGER) BETWEEN 23 AND 24
        AND babajotai_code_dirt IS NOT NULL
    GROUP BY babajotai_code_dirt
    ORDER BY count DESC
    """
    
    df = pd.read_sql(query, conn)
    
    print("\n【ダート馬場状態】")
    total = df['count'].sum()
    for _, row in df.iterrows():
        baba = row['baba_jotai']
        count = row['count']
        ratio = count / total * 100
        baba_name = {'1': '良', '2': '稍重', '3': '重', '4': '不良'}.get(baba, baba)
        print(f"  {baba_name:5s}: {count:,}件 ({ratio:.1f}%)")
    
    # 距離の分布
    query = """
    SELECT 
        CAST(kyori AS INTEGER) as kyori,
        COUNT(*) as count
    FROM jvd_ra
    WHERE CAST(kaisai_nen AS INTEGER) >= 2020 
        AND CAST(kaisai_nen AS INTEGER) <= 2023
        AND kyori IS NOT NULL
    GROUP BY CAST(kyori AS INTEGER)
    ORDER BY CAST(kyori AS INTEGER)
    """
    
    df = pd.read_sql(query, conn)
    
    print("\n【距離分布】")
    print(f"  最短距離: {df['kyori'].min()}m")
    print(f"  最長距離: {df['kyori'].max()}m")
    print(f"  平均距離: {(df['kyori'] * df['count']).sum() / df['count'].sum():.0f}m")
    
    # 距離帯別
    df['distance_category'] = pd.cut(df['kyori'], 
                                      bins=[0, 1400, 1800, 2200, 9999],
                                      labels=['短距離(~1400m)', '中短距離(1401-1800m)', 
                                              '中長距離(1801-2200m)', '長距離(2201m~)'])
    
    category_counts = df.groupby('distance_category')['count'].sum()
    total = category_counts.sum()
    
    for category, count in category_counts.items():
        ratio = count / total * 100
        print(f"  {category}: {count:,}件 ({ratio:.1f}%)")


def check_past_performance(conn):
    """過去成績データの確認"""
    print("\n" + "=" * 80)
    print("【4. 過去成績データの品質】")
    print("=" * 80)
    
    # 馬名別のレース数を確認
    query = """
    SELECT 
        COUNT(DISTINCT bamei) as unique_horses,
        AVG(race_count) as avg_races_per_horse,
        MIN(race_count) as min_races,
        MAX(race_count) as max_races
    FROM (
        SELECT 
            bamei,
            COUNT(*) as race_count
        FROM jvd_se
        WHERE CAST(kaisai_nen AS INTEGER) >= 2020 
            AND CAST(kaisai_nen AS INTEGER) <= 2023
            AND kakutei_chakujun ~ '^[0-9]+$'
        GROUP BY bamei
    ) horse_stats
    """
    
    df = pd.read_sql(query, conn)
    
    print("\n【馬の出走経験】")
    print(f"ユニークな馬: {df['unique_horses'].iloc[0]:,}頭")
    print(f"馬当たり平均レース数: {df['avg_races_per_horse'].iloc[0]:.1f}レース")
    print(f"最少レース数: {df['min_races'].iloc[0]:.0f}レース")
    print(f"最多レース数: {df['max_races'].iloc[0]:.0f}レース")
    
    print("\n⚠️  注意: 過去3走データの不完全な馬が含まれる可能性があります")
    print("       (デビュー戦・2戦目・3戦目の馬は特徴量が不完全)")


def check_data_integrity(conn):
    """データ整合性チェック"""
    print("\n" + "=" * 80)
    print("【5. データ整合性チェック】")
    print("=" * 80)
    
    # 取消・除外レースの数
    query = """
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN kakutei_chakujun IN ('00', '99', '') THEN 1 ELSE 0 END) as invalid_finish,
        SUM(CASE WHEN tansho_odds IS NULL OR tansho_odds = '0' OR tansho_odds = '' THEN 1 ELSE 0 END) as no_odds,
        SUM(CASE WHEN futan_juryo IS NULL OR futan_juryo = '' THEN 1 ELSE 0 END) as no_weight
    FROM jvd_se se
    INNER JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen 
        AND se.kaisai_tsukihi = ra.kaisai_tsukihi 
        AND se.keibajo_code = ra.keibajo_code 
        AND se.race_bango = ra.race_bango
    WHERE CAST(ra.kaisai_nen AS INTEGER) >= 2020 
        AND CAST(ra.kaisai_nen AS INTEGER) <= 2023
    """
    
    df = pd.read_sql(query, conn)
    
    total = df['total'].iloc[0]
    invalid = df['invalid_finish'].iloc[0]
    no_odds = df['no_odds'].iloc[0]
    no_weight = df['no_weight'].iloc[0]
    
    print(f"\n総レコード数: {total:,}件\n")
    print(f"❌ 取消・除外（着順00/99/空白）: {invalid:,}件 ({invalid/total*100:.2f}%)")
    print(f"❌ オッズ情報なし: {no_odds:,}件 ({no_odds/total*100:.2f}%)")
    print(f"❌ 斤量情報なし: {no_weight:,}件 ({no_weight/total*100:.2f}%)")
    
    valid = total - invalid - no_odds - no_weight
    print(f"\n✅ 学習可能なデータ: {valid:,}件 ({valid/total*100:.1f}%)")


def generate_summary_report(conn):
    """サマリーレポート生成"""
    print("\n" + "=" * 80)
    print("【6. データ品質サマリー】")
    print("=" * 80)
    
    # 年度別データ件数
    query = """
    SELECT 
        ra.kaisai_nen as nendo,
        COUNT(*) as total_records,
        COUNT(DISTINCT ra.keibajo_code) as unique_tracks,
        COUNT(DISTINCT se.bamei) as unique_horses
    FROM jvd_se se
    INNER JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen 
        AND se.kaisai_tsukihi = ra.kaisai_tsukihi 
        AND se.keibajo_code = ra.keibajo_code 
        AND se.race_bango = ra.race_bango
    WHERE CAST(ra.kaisai_nen AS INTEGER) >= 2020 
        AND CAST(ra.kaisai_nen AS INTEGER) <= 2023
    GROUP BY ra.kaisai_nen
    ORDER BY ra.kaisai_nen
    """
    
    df = pd.read_sql(query, conn)
    
    print("\n【年度別データ概要】")
    for _, row in df.iterrows():
        print(f"  {row['nendo']}年: {row['total_records']:,}件 "
              f"(競馬場{row['unique_tracks']}箇所, 馬{row['unique_horses']:,}頭)")


def main():
    """メイン処理"""
    print("=" * 80)
    print("データ品質チェック開始")
    print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # DB接続
        conn = psycopg2.connect(**DB_PARAMS)
        
        # 各種チェック実行
        check_missing_values(conn)
        check_outliers(conn)
        check_race_conditions(conn)
        check_past_performance(conn)
        check_data_integrity(conn)
        generate_summary_report(conn)
        
        # 結論
        print("\n" + "=" * 80)
        print("【結論と推奨アクション】")
        print("=" * 80)
        print("""
✅ 実施すべき対策:

1. 取消・除外データの除外
   - kakutei_chakujun IN ('00', '99', '') を学習データから除外

2. デビュー戦・2戦目の馬の扱い
   - 過去3走データがない馬は別途処理が必要
   - または除外するか、デフォルト値を設定

3. 外れ値の処理
   - 単勝オッズ999.9倍以上は外れ値として除外検討
   - 斤量の異常値（40kg未満、65kg超）をチェック

4. 欠損値の補完
   - 騎手コード、調教師コードの欠損を確認
   - 必要に応じてデフォルト値を設定

次のステップ:
→ 馬場状態別・距離別の過去成績特徴量の実装
→ データクリーニング処理をmodel_creator.pyに追加
""")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("[OK] データ品質チェック完了!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
