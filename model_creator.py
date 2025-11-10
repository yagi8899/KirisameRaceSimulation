import psycopg2
import os
from pathlib2 import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.metrics import ndcg_score
from keiba_constants import get_track_name, format_model_description
from datetime import datetime


def create_universal_model(track_code, kyoso_shubetsu_code, surface_type, 
                          min_distance, max_distance, model_filename, output_dir='models',
                          year_start=2013, year_end=2022):
    """
    汎用的な競馬予測モデル作成関数
    
    Args:
        track_code (str): 競馬場コード ('01'=札幌, '02'=函館, ..., '09'=阪神, '10'=小倉)
        kyoso_shubetsu_code (str): 競争種別コード ('12'=3歳, '13'=3歳以上)
        surface_type (str): 'turf' or 'dirt' (芝またはダート)
        min_distance (int): 最小距離 (例: 1000)
        max_distance (int): 最大距離 (例: 3600, 上限なしの場合は9999)
        model_filename (str): 保存するモデルファイル名 (例: 'hanshin_turf_3ageup.sav')
        output_dir (str): モデル保存先ディレクトリ (デフォルト: 'models')
        year_start (int): 学習データ開始年 (デフォルト: 2013)
        year_end (int): 学習データ終了年 (デフォルト: 2022)
    
    Returns:
        None: モデルファイルを保存
    """
    
    # カレントディレクトリをスクリプト配置箇所に変更
    os.chdir(Path(__file__).parent)
    print(f"作業ディレクトリ:{os.getcwd()}")
    
    # モデル保存用ディレクトリを作成（存在しない場合）
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 モデル保存先: {output_path.absolute()}")

    # PostgreSQL コネクションの作成
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )

    # トラック条件を動的に設定
    if surface_type.lower() == 'turf':
        # 芝の場合
        # TODO 芝は10～22と広く範囲指定する。芝とダートで精度に違いが出るようであれば対象トラックコードを減らすなどの工夫が必要かも。
        track_condition = "cast(rase.track_code as integer) between 10 and 22"
        baba_condition = "ra.babajotai_code_shiba"
    else:
        # ダートの場合
        track_condition = "cast(rase.track_code as integer) between 23 and 24"
        baba_condition = "ra.babajotai_code_dirt"

    # 距離条件を設定
    if max_distance == 9999:
        distance_condition = f"cast(rase.kyori as integer) >= {min_distance}"
    else:
        distance_condition = f"cast(rase.kyori as integer) between {min_distance} and {max_distance}"
    
    # 競争種別を設定
    if kyoso_shubetsu_code == '12':
        # 3歳戦
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) = 12"
    elif kyoso_shubetsu_code == '13':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) >= 13"

    # SQLクエリを動的に生成
    sql = f"""
    select * from (
        select
        ra.kaisai_nen,
        ra.kaisai_tsukihi,
        ra.keibajo_code,
        CASE 
            WHEN ra.keibajo_code = '01' THEN '札幌' 
            WHEN ra.keibajo_code = '02' THEN '函館' 
            WHEN ra.keibajo_code = '03' THEN '福島' 
            WHEN ra.keibajo_code = '04' THEN '新潟' 
            WHEN ra.keibajo_code = '05' THEN '東京' 
            WHEN ra.keibajo_code = '06' THEN '中山' 
            WHEN ra.keibajo_code = '07' THEN '中京' 
            WHEN ra.keibajo_code = '08' THEN '京都' 
            WHEN ra.keibajo_code = '09' THEN '阪神' 
            WHEN ra.keibajo_code = '10' THEN '小倉' 
            ELSE '' 
        END keibajo_name,
        ra.race_bango,
        ra.kyori,
        ra.tenko_code,
        {baba_condition} as babajotai_code,
        ra.grade_code,
        ra.kyoso_joken_code,
        ra.kyoso_shubetsu_code,
        ra.track_code,
        ra.shusso_tosu,
        seum.ketto_toroku_bango,
        trim(seum.bamei),
        seum.wakuban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.kishu_name,
        seum.chokyoshi_name,
        seum.futan_juryo,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        seum.seibetsu_code,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        18 - cast(seum.kakutei_chakujun as integer) + 1 as kakutei_chakujun_numeric, 
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,  --上位着順ほど1に近くなる
        AVG(
            1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float))
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS past_avg_sotai_chakujun,
        AVG(
            cast(ra.kyori as integer) /
            NULLIF(
                FLOOR(cast(seum.soha_time as integer) / 1000) * 60 +
                FLOOR((cast(seum.soha_time as integer) % 1000) / 10) +
                (cast(seum.soha_time as integer) % 10) * 0.1,
                0
            )
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS time_index,
        SUM(
            CASE 
                WHEN seum.kakutei_chakujun = '01' THEN 100
                WHEN seum.kakutei_chakujun = '02' THEN 80
                WHEN seum.kakutei_chakujun = '03' THEN 60
                WHEN seum.kakutei_chakujun = '04' THEN 40
                WHEN seum.kakutei_chakujun = '05' THEN 30
                WHEN seum.kakutei_chakujun = '06' THEN 20
                WHEN seum.kakutei_chakujun = '07' THEN 10
                ELSE 5 
            END
            * CASE 
                WHEN ra.grade_code = 'A' THEN 3.00                                                                                          --G1 (1.00→3.00に強化)
                WHEN ra.grade_code = 'B' THEN 2.00                                                                                          --G2 (0.80→2.00に強化)
                WHEN ra.grade_code = 'C' THEN 1.50                                                                                          --G3 (0.60→1.50に強化)
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '999' THEN 1.00       --OP (0.50→1.00に調整)
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '016' THEN 0.80       --3勝クラス (0.40→0.80に調整)
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '010' THEN 0.60       --2勝クラス (0.30→0.60に調整)
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '005' THEN 0.40       --1勝クラス (0.20→0.40に調整)
                ELSE 0.20                                                                                                                   --未勝利 (0.10→0.20に調整)
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING  
        ) AS past_score,  --グレード別スコア
        CASE 
            WHEN AVG(
                CASE 
                    WHEN cast(seum.kohan_3f as integer) > 0 AND cast(seum.kohan_3f as integer) < 999 THEN
                    CAST(seum.kohan_3f AS FLOAT) / 10
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) IS NOT NULL THEN
            AVG(
                CASE 
                    WHEN cast(seum.kohan_3f as integer) > 0 AND cast(seum.kohan_3f as integer) < 999 THEN
                    CAST(seum.kohan_3f AS FLOAT) / 10
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) - 
            CASE
                WHEN cast(ra.kyori as integer) <= 1600 THEN 33.5
                WHEN cast(ra.kyori as integer) <= 2000 THEN 35.0
                WHEN cast(ra.kyori as integer) <= 2400 THEN 36.0
                ELSE 37.0
            END
            ELSE 0
        END AS kohan_3f_index
    from
        jvd_ra ra 
        inner join ( 
            select
                se.kaisai_nen
                , se.kaisai_tsukihi
                , se.keibajo_code
                , se.race_bango
                , se.kakutei_chakujun
                , se.ketto_toroku_bango
                , se.bamei
                , se.wakuban
                , se.umaban
                , se.barei
                , se.seibetsu_code
                , se.kishu_code
                , se.chokyoshi_code
                , trim(se.kishumei_ryakusho) as kishu_name
                , trim(se.chokyoshimei_ryakusho) as chokyoshi_name
                , se.futan_juryo
                , se.tansho_odds
                , se.tansho_ninkijun
                , se.kohan_3f
                , se.soha_time
            from
                jvd_se se
            where 
                se.kohan_3f <> '000' 
                and se.kohan_3f <> '999'
        ) seum 
            on ra.kaisai_nen = seum.kaisai_nen 
            and ra.kaisai_tsukihi = seum.kaisai_tsukihi 
            and ra.keibajo_code = seum.keibajo_code 
            and ra.race_bango = seum.race_bango 
    where
        cast(ra.kaisai_nen as integer) between {year_start} and {year_end}    --学習データ年範囲
    ) rase 
    where 
    rase.keibajo_code = '{track_code}'                                        --競馬場指定
    and {kyoso_shubetsu_condition}                                            --競争種別
    and {track_condition}                                                     --芝/ダート
    and {distance_condition}                                                  --距離条件
    """

    # モデル説明を生成
    model_desc = format_model_description(track_code, kyoso_shubetsu_code, surface_type, min_distance, max_distance)
    print(f"🏇 モデル作成開始: {model_desc}")
    
    # SQLをログファイルに出力（常に上書き）
    log_filepath = Path('sql_log.txt')
    with open(log_filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== モデル作成SQL ===\n")
        f.write(f"モデル: {model_desc}\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{sql}\n")
    print(f"📝 SQLをログファイルに出力: {log_filepath}")
    
    # SELECT結果をDataFrame
    df = pd.read_sql_query(sql=sql, con=conn)
    
    if len(df) == 0:
        print("❌ 指定した条件に合致するデータが見つかりませんでした。条件を確認してください。")
        return

    print(f"📊 データ件数: {len(df)}件")

    # 着順スコアが0のデータは無効扱いにして除外
    df = df[df['chakujun_score'] > 0]

    # 🔥修正: データ前処理を適切に実施
    # 騎手コード・調教師コード・馬名などの文字列列を保持したまま、数値列のみを処理
    print("🔍 データ型確認...")
    print(f"  kishu_code型（修正前）: {df['kishu_code'].dtype}")
    print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
    print(f"  kishu_codeユニーク数: {df['kishu_code'].nunique()}")
    
    # 数値化する列を明示的に指定（文字列列は除外）
    numeric_columns = [
        'wakuban', 'umaban_numeric', 'barei', 'futan_juryo', 'tansho_odds',
        'kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'kyori', 'shusso_tosu',
        'tenko_code', 'babajotai_code', 'grade_code', 'kyoso_joken_code',
        'kyoso_shubetsu_code', 'track_code', 'seibetsu_code',
        'kakutei_chakujun_numeric', 'chakujun_score', 'past_avg_sotai_chakujun',
        'time_index', 'past_score', 'kohan_3f_index'
    ]
    
    # 数値化する列のみ処理（文字列列は保持）
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値を0で埋める（数値列のみ）
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # 文字列型の列はそのまま保持（kishu_code, chokyoshi_code, bamei など）
    print(f"  kishu_code型（修正後）: {df['kishu_code'].dtype}")
    print(f"  kishu_codeサンプル: {df['kishu_code'].head(5).tolist()}")
    print("✅ データ前処理完了（文字列列を保持）")

    # past_avg_sotai_chakujunはSQLで計算済みの単純移動平均を使用
    # (EWM実験の結果、単純平均の方が複勝・三連複で安定した性能を示した)

    X = df.loc[:, [
        # "futan_juryo",
        "past_score",
        "kohan_3f_index",
        "past_avg_sotai_chakujun",
        "time_index",
    ]].astype(float)
    
    # 高性能な派生特徴量を追加！
    # 枠番と頭数の比率（内枠有利度）
    max_wakuban = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['wakuban'].transform('max')
    df['wakuban_ratio'] = df['wakuban'] / max_wakuban
    X['wakuban_ratio'] = df['wakuban_ratio']
    
    # 斤量と馬齢の比率（若馬の負担能力）
    df['futan_per_barei'] = df['futan_juryo'] / df['barei'].replace(0, 1)
    X['futan_per_barei'] = df['futan_per_barei']

    # 🚀 高精度二次特徴量を追加（予測スコア重複回避 + 精度向上）
    # シンプルな特徴量から始めて過学習を防ぐ
    
    # 馬番×距離の相互作用（内外枠の距離適性）
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000  # スケール調整
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    # 🔥改善された特徴量🔥
    # 2. futan_per_bareiの非線形変換
    df['futan_per_barei_log'] = np.log(df['futan_per_barei'].clip(lower=0.1))
    X['futan_per_barei_log'] = df['futan_per_barei_log']
    
    # 期待斤量からの差分（年齢別期待斤量との差）
    expected_weight_by_age = {2: 48, 3: 52, 4: 55, 5: 57, 6: 57, 7: 56, 8: 55}
    df['futan_deviation'] = df.apply(
        lambda row: row['futan_juryo'] - expected_weight_by_age.get(row['barei'], 55), 
        axis=1
    )
    X['futan_deviation'] = df['futan_deviation']
    
    # # 4. 複数のピーク年齢パターン
    # df['barei_peak_distance'] = abs(df['barei'] - 4)  # 4歳をピークと仮定（既存）
    # X['barei_peak_distance'] = df['barei_peak_distance']
    
    # # 3歳短距離ピーク（早熟型）
    # df['barei_peak_short'] = abs(df['barei'] - 3)
    # X['barei_peak_short'] = df['barei_peak_short']
    
    # # 5歳長距離ピーク（晩成型）
    # df['barei_peak_long'] = abs(df['barei'] - 5)
    # X['barei_peak_long'] = df['barei_peak_long']

    # 5. 枠番バイアススコア（枠番の歴史的優位性を数値化）
    # 枠番別の歴史的着順分布を計算
    wakuban_stats = df.groupby('wakuban').agg({
        'kakutei_chakujun_numeric': ['mean', 'std', 'count']
    }).round(4)
    wakuban_stats.columns = ['waku_avg_rank', 'waku_std_rank', 'waku_count']
    wakuban_stats = wakuban_stats.reset_index()
    
    # 全体平均からの偏差でバイアススコアを計算
    overall_avg_rank = df['kakutei_chakujun_numeric'].mean()
    wakuban_stats['wakuban_bias_score'] = (overall_avg_rank - wakuban_stats['waku_avg_rank']) / wakuban_stats['waku_std_rank']
    wakuban_stats['wakuban_bias_score'] = wakuban_stats['wakuban_bias_score'].fillna(0)  # NaNを0で埋める
    
    # DataFrameにマージ
    df = df.merge(wakuban_stats[['wakuban', 'wakuban_bias_score']], on='wakuban', how='left')
    # X['wakuban_bias_score'] = df['wakuban_bias_score']

    # レース内での馬番相対位置（頭数による正規化）
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # 研究用特徴量 追加
    # 斤量偏差値（レース内で標準化）
    # レース内の平均と標準偏差を計算して、各馬の斤量がどれくらい重い/軽いかを表現
    race_group = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['futan_juryo']
    df['futan_mean'] = race_group.transform('mean')
    df['futan_std'] = race_group.transform('std')
    
    # 標準偏差が0の場合（全頭同じ斤量）は0にする
    df['futan_zscore'] = np.where(
        df['futan_std'] > 0,
        (df['futan_juryo'] - df['futan_mean']) / df['futan_std'],
        0
    )
    X['futan_zscore'] = df['futan_zscore']
    
    # レース内での斤量順位（パーセンタイル）
    # 0.0=最軽量、1.0=最重量
    df['futan_percentile'] = race_group.transform(lambda x: x.rank(pct=True))
    X['futan_percentile'] = df['futan_percentile']

    # 🔥新機能: 距離適性スコアを追加（3種類）🔥
    print("🏇 距離適性スコアを計算中...")
    
    # 距離カテゴリ分類関数
    def categorize_distance(kyori):
        """距離を4カテゴリに分類"""
        if kyori <= 1400:
            return 'short'  # 短距離
        elif kyori <= 1800:
            return 'mile'   # マイル
        elif kyori <= 2400:
            return 'middle' # 中距離
        else:
            return 'long'   # 長距離
    
    # 今回のレースの距離カテゴリを追加
    df['distance_category'] = df['kyori'].apply(categorize_distance)
    
    # 🔥重要: 馬場情報も先に追加（df_sortedで使うため）🔥
    # 芝/ダート分類関数
    def categorize_surface(track_code):
        """トラックコードから芝/ダートを判定"""
        track_code_int = int(track_code)
        if 10 <= track_code_int <= 22:
            return 'turf'  # 芝
        elif 23 <= track_code_int <= 24:
            return 'dirt'  # ダート
        else:
            return 'unknown'
    
    # 馬場状態分類関数
    def categorize_baba_condition(baba_code):
        """馬場状態コードを分類"""
        if baba_code == 1:
            return 'good'      # 良
        elif baba_code == 2:
            return 'slightly'  # 稍重
        elif baba_code == 3:
            return 'heavy'     # 重
        elif baba_code == 4:
            return 'bad'       # 不良
        else:
            return 'unknown'
    
    # 今回のレースの馬場情報を追加
    df['surface_type'] = df['track_code'].apply(categorize_surface)
    df['baba_condition'] = df['babajotai_code'].apply(categorize_baba_condition)
    
    # 時系列順にソート（馬ごとに過去データを参照するため）
    df_sorted = df.sort_values(['ketto_toroku_bango', 'kaisai_nen', 'kaisai_tsukihi']).copy()
    
    # 距離カテゴリ別適性スコア（同じカテゴリでの過去5走の平均相対着順）
    def calc_distance_category_score(group):
        """各馬の距離カテゴリ別適性を計算"""
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            # 現在のレースの距離カテゴリ
            current_category = group.iloc[idx]['distance_category']
            
            # 過去のレース（同じカテゴリ）から直近5走を取得
            past_same_category = group.iloc[:idx][
                group.iloc[:idx]['distance_category'] == current_category
            ].tail(5)
            
            if len(past_same_category) > 0:
                # 相対着順の平均（1 - 着順/18）
                avg_score = (1 - (past_same_category['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_category_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_category_score
    ).values
    
    # 近似距離での成績（±200m以内、直近10走）
    def calc_similar_distance_score(group):
        """近似距離での適性を計算"""
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_kyori = group.iloc[idx]['kyori']
            
            # 過去のレース（±200m以内）から直近10走を取得
            past_similar = group.iloc[:idx][
                abs(group.iloc[:idx]['kyori'] - current_kyori) <= 200
            ].tail(10)
            
            if len(past_similar) > 0:
                avg_score = (1 - (past_similar['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['similar_distance_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_similar_distance_score
    ).values
    
    # 距離変化対応力（前走からの距離変化±100m以上時の成績、直近5走）
    def calc_distance_change_adaptability(group):
        """距離変化時の対応力を計算"""
        scores = []
        for idx in range(len(group)):
            if idx < 2:  # 最低2走分のデータが必要
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            # ✅ 修正: 過去6走分を取得（前走との差分を見るため）
            past_races = group.iloc[max(0, idx-6):idx].copy()
            
            if len(past_races) >= 3:  # ✅ 修正: 最低3走必要（差分2個）
                # 距離の変化量を計算
                past_races['kyori_diff'] = past_races['kyori'].diff().abs()
                
                # ✅ 修正: 最新5走のみを評価（最初の1行はNaNなので除外）
                past_races_eval = past_races.tail(5)
                
                # 距離変化が100m以上のレースを抽出
                changed_races = past_races_eval[past_races_eval['kyori_diff'] >= 100]
                
                if len(changed_races) > 0:
                    avg_score = (1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # ✅ 修正: 変化なしは中立
            else:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['distance_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_distance_change_adaptability
    ).values
    
    # 元のインデックス順に戻す
    df = df.copy()
    df['distance_category_score'] = df_sorted.sort_index()['distance_category_score']
    df['similar_distance_score'] = df_sorted.sort_index()['similar_distance_score']
    df['distance_change_adaptability'] = df_sorted.sort_index()['distance_change_adaptability']
    
    # 特徴量に追加
    X['distance_category_score'] = df['distance_category_score']
    X['similar_distance_score'] = df['similar_distance_score']
    # X['distance_change_adaptability'] = df['distance_change_adaptability']
    
    print(f"✅ 距離適性スコアを3種類追加しました！")
    print(f"  - distance_category_score: 距離カテゴリ別適性（直近5走）")
    print(f"  - similar_distance_score: 近似距離での成績（±200m、直近10走）")
    print(f"  - distance_change_adaptability: 距離変化対応力（±100m以上、直近5走）")
    
    # 🔥新機能: 馬場適性スコアを追加（3種類）🔥
    print("🌿 馬場適性スコアを計算中...")
    
    # 馬場情報は既にdf_sortedに含まれているので、そのまま使用
    # 1️⃣ 芝/ダート別適性スコア（直近10走）
    def calc_surface_score(group):
        """各馬の芝/ダート別適性を計算"""
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_surface = group.iloc[idx]['surface_type']
            
            # 同じ馬場タイプでの過去成績（直近10走）
            past_same_surface = group.iloc[:idx][
                group.iloc[:idx]['surface_type'] == current_surface
            ].tail(10)
            
            if len(past_same_surface) > 0:
                avg_score = (1 - (past_same_surface['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['surface_aptitude_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_surface_score
    ).values
    
    # 2️⃣ 馬場状態別適性スコア（良/稍重/重/不良、直近10走）
    def calc_baba_condition_score(group):
        """各馬の馬場状態別適性を計算"""
        scores = []
        for idx in range(len(group)):
            if idx == 0:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            current_condition = group.iloc[idx]['baba_condition']
            
            # 同じ馬場状態での過去成績（直近10走）
            past_same_condition = group.iloc[:idx][
                group.iloc[:idx]['baba_condition'] == current_condition
            ].tail(10)
            
            if len(past_same_condition) > 0:
                avg_score = (1 - (past_same_condition['kakutei_chakujun_numeric'] / 18.0)).mean()
                scores.append(avg_score)
            else:
                scores.append(0.5)  # ✅ 修正: データなしは中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_condition_score'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_condition_score
    ).values
    
    # 3️⃣ 馬場変化対応力（前走と異なる馬場状態での成績、直近5走）
    def calc_baba_change_adaptability(group):
        """馬場状態変化時の対応力を計算"""
        scores = []
        for idx in range(len(group)):
            if idx < 2:  # 最低2走分のデータが必要
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
                continue
            
            # ✅ 修正: 過去6走分を取得（前走との変化を見るため）
            past_races = group.iloc[max(0, idx-6):idx].copy()
            
            if len(past_races) >= 3:  # ✅ 修正: 最低3走必要
                # 馬場状態の変化を検出（前走と異なる馬場状態）
                past_races['baba_changed'] = past_races['baba_condition'].shift(1) != past_races['baba_condition']
                
                # ✅ 修正: 最新5走のみを評価
                past_races_eval = past_races.tail(5)
                
                # 馬場状態が変化したレースのみ抽出
                changed_races = past_races_eval[past_races_eval['baba_changed'] == True]
                
                if len(changed_races) > 0:
                    avg_score = (1 - (changed_races['kakutei_chakujun_numeric'] / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # ✅ 修正: 変化なしは中立
            else:
                scores.append(0.5)  # ✅ 修正: データ不足は中立値
        
        return pd.Series(scores, index=group.index)
    
    df_sorted['baba_change_adaptability'] = df_sorted.groupby('ketto_toroku_bango', group_keys=False).apply(
        calc_baba_change_adaptability
    ).values
    
    # 元のインデックス順に戻す
    df['surface_aptitude_score'] = df_sorted.sort_index()['surface_aptitude_score']
    df['baba_condition_score'] = df_sorted.sort_index()['baba_condition_score']
    df['baba_change_adaptability'] = df_sorted.sort_index()['baba_change_adaptability']
    
    # 特徴量に追加
    X['surface_aptitude_score'] = df['surface_aptitude_score']
    # X['baba_condition_score'] = df['baba_condition_score']
    X['baba_change_adaptability'] = df['baba_change_adaptability']
    
    print(f"✅ 馬場適性スコアを3種類追加しました！")
    print(f"  - surface_aptitude_score: 芝/ダート別適性（直近10走）")
    print(f"  - baba_condition_score: 馬場状態別適性（良/稍重/重/不良、直近10走）")
    print(f"  - baba_change_adaptability: 馬場変化対応力（直近5走）")

    # 🔥新機能: 騎手・調教師の動的能力スコアを追加（4種類）🔥
    print("🏇 騎手・調教師の能力スコアを計算中...")
    
    # ✅ 修正: race_bangoを追加して時系列リークを防止
    df_sorted_kishu = df.sort_values(['kishu_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    # 1️⃣ 騎手の実力補正スコア（期待着順との差分、直近3ヶ月）
    def calc_kishu_skill_adjusted_score(group):
        """騎手の純粋な技術を評価（馬の実力を補正）"""
        scores = []
        
        for idx in range(len(group)):
            # 騎手コードがない場合はスキップ
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            
            # 3ヶ月前の日付
            three_months_ago = current_date - pd.DateOffset(months=3)
            
            # 過去3ヶ月のレースを抽出（未来のデータは見ない！）
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                recent_races = past_races[past_races['kaisai_date'] >= three_months_ago]
                
                if len(recent_races) >= 3:  # 最低3レース必要
                    # ✅ 修正: 騎手の純粋な成績を評価（馬の実力補正ではなく、騎手の平均成績）
                    # 着順をスコア化（1着=1.0, 18着=0.0）
                    recent_races['rank_score'] = 1.0 - ((18 - recent_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                    
                    # 騎手の平均スコアを計算
                    avg_score = recent_races['rank_score'].mean()
                    
                    # 0-1の範囲にクリップ（既に範囲内だが念のため）
                    normalized_score = max(0.0, min(1.0, avg_score))
                    
                    scores.append(normalized_score)
                else:
                    scores.append(0.5)  # データ不足は中立
            else:
                scores.append(0.5)  # 初回は中立
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_skill_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_skill_adjusted_score
    ).values
    
    # 2️⃣ 騎手の人気差スコア（オッズ補正、直近3ヶ月）
    def calc_kishu_popularity_adjusted_score(group):
        """騎手の人気補正スコア（人気より上位に来れるか）"""
        scores = []
        
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            
            three_months_ago = current_date - pd.DateOffset(months=3)
            
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                recent_races = past_races[past_races['kaisai_date'] >= three_months_ago]
                
                if len(recent_races) >= 3:
                    # オッズが0や異常値の場合を除外
                    valid_races = recent_races[recent_races['tansho_odds'] > 0]
                    
                    if len(valid_races) >= 3:
                        # ✅ 修正: オッズベースの期待成績と実際の成績を比較
                        # オッズが低い = 期待値が高い（1に近い）
                        # オッズが高い = 期待値が低い（0に近い）
                        max_odds = valid_races['tansho_odds'].max()
                        valid_races['odds_expectation'] = 1.0 - (valid_races['tansho_odds'] / (max_odds + 1.0))
                        
                        # 実際の成績スコア
                        valid_races['actual_score'] = 1.0 - ((18 - valid_races['kakutei_chakujun_numeric'] + 1) / 18.0)
                        
                        # 期待を上回った度合い（プラスなら期待以上）
                        valid_races['performance_diff'] = valid_races['actual_score'] - valid_races['odds_expectation']
                        
                        # 平均差分をスコア化（0.5が中立）
                        avg_diff = valid_races['performance_diff'].mean()
                        normalized_score = 0.5 + (avg_diff * 0.5)  # ±0.5の範囲に収める
                        normalized_score = max(0.0, min(1.0, normalized_score))
                        
                        scores.append(normalized_score)
                    else:
                        scores.append(0.5)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_popularity_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_popularity_adjusted_score
    ).values
    
    # 3️⃣ 騎手の芝/ダート別スコア（馬場適性考慮、直近6ヶ月）
    def calc_kishu_surface_score(group):
        """騎手の馬場タイプ別直近6ヶ月成績"""
        scores = []
        
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['kishu_code']) or group.iloc[idx]['kishu_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            current_surface = group.iloc[idx]['surface_type']
            
            six_months_ago = current_date - pd.DateOffset(months=6)
            
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                # 同じ馬場タイプでの直近6ヶ月
                recent_same_surface = past_races[
                    (past_races['kaisai_date'] >= six_months_ago) &
                    (past_races['surface_type'] == current_surface)
                ]
                
                if len(recent_same_surface) >= 5:  # 最低5レース必要
                    avg_score = (1 - ((18 - recent_same_surface['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_kishu['kishu_surface_score'] = df_sorted_kishu.groupby('kishu_code', group_keys=False).apply(
        calc_kishu_surface_score
    ).values
    
    # ✅ 修正: race_bangoを追加して時系列リークを防止
    df_sorted_chokyoshi = df.sort_values(['chokyoshi_code', 'kaisai_nen', 'kaisai_tsukihi', 'race_bango']).copy()
    
    # 4️⃣ 調教師の直近3ヶ月成績スコア
    def calc_chokyoshi_recent_score(group):
        """調教師の直近3ヶ月成績"""
        scores = []
        
        for idx in range(len(group)):
            if pd.isna(group.iloc[idx]['chokyoshi_code']) or group.iloc[idx]['chokyoshi_code'] == '':
                scores.append(0.5)
                continue
                
            current_date = pd.to_datetime(
                str(int(group.iloc[idx]['kaisai_nen'])) + str(int(group.iloc[idx]['kaisai_tsukihi'])).zfill(4),
                format='%Y%m%d'
            )
            
            three_months_ago = current_date - pd.DateOffset(months=3)
            
            past_races = group.iloc[:idx]
            
            if len(past_races) > 0:
                past_races = past_races.copy()
                past_races['kaisai_date'] = pd.to_datetime(
                    past_races['kaisai_nen'].astype(str) + past_races['kaisai_tsukihi'].astype(str).str.zfill(4),
                    format='%Y%m%d'
                )
                recent_races = past_races[past_races['kaisai_date'] >= three_months_ago]
                
                if len(recent_races) >= 5:  # ✅ 修正: 5レースに変更（10レースでは大部分が中立値になる）
                    avg_score = (1 - ((18 - recent_races['kakutei_chakujun_numeric'] + 1) / 18.0)).mean()
                    scores.append(avg_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return pd.Series(scores, index=group.index)
    
    df_sorted_chokyoshi['chokyoshi_recent_score'] = df_sorted_chokyoshi.groupby('chokyoshi_code', group_keys=False).apply(
        calc_chokyoshi_recent_score
    ).values
    
    # 元のインデックス順に戻す
    df['kishu_skill_score'] = df_sorted_kishu.sort_index()['kishu_skill_score']
    df['kishu_popularity_score'] = df_sorted_kishu.sort_index()['kishu_popularity_score']
    df['kishu_surface_score'] = df_sorted_kishu.sort_index()['kishu_surface_score']
    df['chokyoshi_recent_score'] = df_sorted_chokyoshi.sort_index()['chokyoshi_recent_score']
    
    # 特徴量に追加
    X['kishu_skill_score'] = df['kishu_skill_score']
    X['kishu_popularity_score'] = df['kishu_popularity_score']
    X['kishu_surface_score'] = df['kishu_surface_score']
    X['chokyoshi_recent_score'] = df['chokyoshi_recent_score']
    
    print(f"✅ 騎手・調教師スコアを4種類追加しました！")
    print(f"  - kishu_skill_score: 騎手の実力補正スコア（馬の実力を考慮、直近3ヶ月）")
    print(f"  - kishu_popularity_score: 騎手の人気差スコア（オッズ補正、直近3ヶ月）")
    print(f"  - kishu_surface_score: 騎手の芝/ダート別適性（直近6ヶ月）")
    print(f"  - chokyoshi_recent_score: 調教師の直近成績（直近3ヶ月）")

    # 過去レースで「人気薄なのに好走した回数」
    # df['upset_count'] = df.groupby('ketto_toroku_bango').apply(
    #     lambda g: ((g['tansho_ninkijun_numeric'] >= 5) & (g['kakutei_chakujun_numeric'] <= 3)).sum()
    # )
    # X['upset_count'] = df['upset_count']

    # # 研究用特徴量 追加
    
    # カテゴリ変数を作成
    # X['kyori'] = X['kyori'].astype('category')
    # X['tenko_code'] = X['tenko_code'].astype('category')
    # X['babajotai_code'] = X['babajotai_code'].astype('category')
    # X['seibetsu_code'] = X['seibetsu_code'].astype('category')
    categorical_features = []

    #目的変数を設定
    y = df['kakutei_chakujun_numeric'].astype(int)

    # レースごとのグループを正しく計算
    df['group_id'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango']).ngroup()
    groups = df['group_id'].values
    print(f"グループ数: {len(set(groups))}")  # ユニークなグループの数
    print(f"データ数: {len(groups)}")  # 全データ数

    # データとグループ数が一致していることを確認
    if len(groups) != len(X):
        raise ValueError(f"データ件数({len(X)})とグループの数({len(groups)})が一致しません！")

    # 🔥改善1: 時系列分割を年単位で明確化🔥
    # 年単位で訓練/テストを分割（ばらつき削減のため）
    # 例: 2013-2020年を訓練、2021-2022年をテスト
    
    # 学習データの年範囲から訓練/テスト年を計算
    all_years = sorted(df['kaisai_nen'].unique())
    total_years = len(all_years)
    
    # 75%を訓練データに（年単位で）
    train_year_count = int(total_years * 0.75)
    train_years = all_years[:train_year_count]
    test_years = all_years[train_year_count:]
    
    print(f"📅 訓練データ年: {train_years} ({len(train_years)}年間)")
    print(f"📅 テストデータ年: {test_years} ({len(test_years)}年間)")
    
    # 年単位でインデックスを分割
    train_indices = df[df['kaisai_nen'].isin(train_years)].index
    test_indices = df[df['kaisai_nen'].isin(test_years)].index
    
    # データ分割
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]
    
    print(f"✅ 訓練データ件数: {len(X_train)}件")
    print(f"✅ テストデータ件数: {len(X_test)}件")

    # Optunaのobjective関数
    def objective(trial):
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',                                                              # 上位の並び順を重視
            'ndcg_eval_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 5)
        }

        # 🔥修正: グループサイズを正しい順序で計算（データの順序を保持）
        # sort=Falseで元のデータ順を維持しながらグループサイズを抽出
        train_df_with_group = pd.DataFrame({'group': groups_train}).reset_index(drop=True)
        train_group_sizes = train_df_with_group.groupby('group', sort=False).size().values
        
        test_df_with_group = pd.DataFrame({'group': groups_test}).reset_index(drop=True)
        test_group_sizes = test_df_with_group.groupby('group', sort=False).size().values
        
        dtrain = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_test, label=y_test, group=test_group_sizes, categorical_feature=categorical_features)

        tmp_model = lgb.train(
            param,
            dtrain,
            valid_sets=[dvalid],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 🔥修正: レースごとにNDCGを計算して平均を返す（正しい評価方法）
        preds = tmp_model.predict(X_test, num_iteration=tmp_model.best_iteration)
        
        # テストデータをレースごとに分割してNDCGを計算
        ndcg_scores = []
        start_idx = 0
        for group_size in test_group_sizes:
            end_idx = start_idx + group_size
            y_true_group = y_test.iloc[start_idx:end_idx].values
            y_pred_group = preds[start_idx:end_idx]
            
            # レース内に2頭以上いる場合のみNDCG計算（1頭だとエラーになる）
            if len(y_true_group) > 1:
                # 2次元配列として渡す
                ndcg = ndcg_score([y_true_group], [y_pred_group], k=5)
                ndcg_scores.append(ndcg)
            
            start_idx = end_idx
        
        # 全レースのNDCG平均を返す
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    # 💡推奨: 本番運用前に上位パラメータを複数シード（random_state=42,43,44...）で
    #        再学習し、NDCG/ROI/的中率の安定性を確認することを強く推奨
    # TODO 将来の改善: lgb.Dataset の weight パラメータにオッズベースの重みを導入し、
    #      穴馬（高オッズ馬）の予測精度を向上させることでROI最適化を図る
    
    # 🔥改善2: Optunaのシード固定（再現性向上のため）🔥
    print("🔍 ハイパーパラメータ最適化を開始...")
    sampler = optuna.samplers.TPESampler(seed=42)  # シードを固定
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50)

    print('Best trial:')
    print(study.best_trial.params)

    # 最適パラメータを使って再学習
    best_params = study.best_trial.params

    best_params.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'verbosity': 0,  # 学習の進捗を表示
        'random_state': 42,
    })

    # 🔥修正: グループデータを正しく準備（データの順序を保持）
    # レースごとの出走頭数を計算（sort=Falseで元の順序を維持）
    train_df_with_group = pd.DataFrame({'group': groups_train}).reset_index(drop=True)
    train_group_sizes = train_df_with_group.groupby('group', sort=False).size().values
    
    test_df_with_group = pd.DataFrame({'group': groups_test}).reset_index(drop=True)
    test_group_sizes = test_df_with_group.groupby('group', sort=False).size().values
    
    print(f"訓練データのレース数: {len(train_group_sizes)}")
    print(f"テストデータのレース数: {len(test_group_sizes)}")
    
    # LightGBM用のデータセットを作成
    dtrain = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, categorical_feature=categorical_features)
    dvalid = lgb.Dataset(X_test, label=y_test, group=test_group_sizes, categorical_feature=categorical_features)

    # 最適化されたパラメータでモデルを学習
    print("🔥 最適化されたパラメータでモデルを学習するよ！")
    model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dvalid],
        valid_names=['テストデータ'],
        num_boost_round=1000,  # 最大反復回数
        callbacks=[
            lgb.early_stopping(50),  # 50回改善がなければ早期終了（学習の安定化）
            lgb.log_evaluation(10)   # 10回ごとに結果表示
        ]
    )

    # 特徴量の重要度を確認する（モデルがどの情報を重視しているか）
    importance = model.feature_importance()
    feature_names = X.columns
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("特徴量の重要度:")
    print(feature_importance)

    # モデルを保存する
    model_filepath = output_path / model_filename
    pickle.dump(model, open(model_filepath, 'wb'))
    print(f"✅ モデルを {model_filepath} に保存しました")

    # コネクションをクローズ
    conn.close()


# 旧来の関数を維持（互換性のため）
def make_model():
    """
    旧バージョン互換性のための関数
    阪神競馬場の３歳以上芝中長距離モデルを作成
    
    注意: 互換性のため、ルートディレクトリに保存されます
    """
    create_universal_model(
        track_code='09',  # 阪神
        kyoso_shubetsu_code='13',  # 3歳以上
        surface_type='turf',  # 芝
        min_distance=1700,  # 中長距離
        max_distance=9999,  # 上限なし
        model_filename='hanshin_shiba_3ageup_model.sav',
        output_dir='.'  # ルートディレクトリに保存（既存の動作を維持）
    )


if __name__ == '__main__':
    # テスト実行：旧バージョンと同じモデルを作成
    make_model()