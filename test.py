import psycopg2
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np

def predict_and_save_results():
    # PostgreSQL コネクションの作成
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )

    # SQLクエリ
    sql = """
    select * from (
        select
        ra.kaisai_nen,
        ra.kaisai_tsukihi,
        ra.race_bango,
        seum.umaban,
        seum.bamei,
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
        ra.kyori,
        ra.tenko_code,
        ra.babajotai_code_shiba,
        ra.grade_code,
        ra.kyoso_joken_code,
        ra.kyoso_shubetsu_code,
        ra.track_code,
        seum.ketto_toroku_bango,
        seum.wakuban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.futan_juryo,
        seum.seibetsu_code,
        CASE WHEN seum.seibetsu_code = '1' THEN '1' ELSE '0' END AS mare_horse,
        CASE WHEN seum.seibetsu_code = '2' THEN '1' ELSE '0' END AS femare_horse,
        CASE WHEN seum.seibetsu_code = '3' THEN '1' ELSE '0' END AS sen_horse,
        CASE WHEN ra.babajotai_code_shiba = '1' THEN '1' ELSE '0' END AS baba_good,
        CASE WHEN ra.babajotai_code_shiba = '2' THEN '1' ELSE '0' END AS baba_slightly_heavy,
        CASE WHEN ra.babajotai_code_shiba = '3' THEN '1' ELSE '0' END AS baba_heavy,
        CASE WHEN ra.babajotai_code_shiba = '4' THEN '1' ELSE '0' END AS baba_defective,
        CASE WHEN ra.tenko_code = '1' THEN '1' ELSE '0' END AS tenko_fine,
        CASE WHEN ra.tenko_code = '2' THEN '1' ELSE '0' END AS tenko_cloudy,
        CASE WHEN ra.tenko_code = '3' THEN '1' ELSE '0' END AS tenko_rainy,
        CASE WHEN ra.tenko_code = '4' THEN '1' ELSE '0' END AS tenko_drizzle,
        CASE WHEN ra.tenko_code = '5' THEN '1' ELSE '0' END AS tenko_snow,
        CASE WHEN ra.tenko_code = '6' THEN '1' ELSE '0' END AS tenko_light_snow,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        nullif(cast(seum.kakutei_chakujun as integer), 0) as kakutei_chakujun_numeric,
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,
        -1 as sotai_chakujun_numeric,
        -1 AS time_index,
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
                WHEN ra.grade_code = 'A' THEN 1.00
                WHEN ra.grade_code = 'B' THEN 0.80
                WHEN ra.grade_code = 'C' THEN 0.60
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '999' THEN 0.50
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '016' THEN 0.40
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '010' THEN 0.30
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '005' THEN 0.20
                ELSE 0.10
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING  
        ) AS past_score
        ,0 as kohan_3f_sec
        ,0 AS kohan_3f_index
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1a), '') as integer), 0) as 複勝1着馬番
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1b), '') as float), 0) / 100 as 複勝1着オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1c), '') as integer), 0) as 複勝1着人気
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_2a), '') as integer), 0) as 複勝2着馬番
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_2b), '') as float), 0) / 100 as 複勝2着オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_2c), '') as integer), 0) as 複勝2着人気
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_3a), '') as integer), 0) as 複勝3着馬番
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_3b), '') as float), 0) / 100 as 複勝3着オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_3c), '') as integer), 0) as 複勝3着人気
        ,cast(substring(trim(hr.haraimodoshi_umaren_1a), 1, 2) as integer) as 馬連馬番1
        ,cast(substring(trim(hr.haraimodoshi_umaren_1a), 3, 2) as integer) as 馬連馬番2
        ,nullif(cast(nullif(trim(hr.haraimodoshi_umaren_1b), '') as float), 0) / 100 as 馬連オッズ
        ,cast(substring(trim(hr.haraimodoshi_wide_1a), 1, 2) as integer) as ワイド1_2馬番1
        ,cast(substring(trim(hr.haraimodoshi_wide_1a), 3, 2) as integer) as ワイド1_2馬番2
        ,cast(substring(trim(hr.haraimodoshi_wide_2a), 1, 2) as integer) as ワイド2_3着馬番1
        ,cast(substring(trim(hr.haraimodoshi_wide_2a), 3, 2) as integer) as ワイド2_3着馬番2
        ,cast(substring(trim(hr.haraimodoshi_wide_3a), 1, 2) as integer) as ワイド1_3着馬番1
        ,cast(substring(trim(hr.haraimodoshi_wide_3a), 3, 2) as integer) as ワイド1_3着馬番2
        ,nullif(cast(nullif(trim(hr.haraimodoshi_wide_1b), '') as float), 0) / 100 as ワイド1_2オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_wide_2b), '') as float), 0) / 100 as ワイド2_3オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_wide_3b), '') as float), 0) / 100 as ワイド1_3オッズ
        ,cast(substring(trim(hr.haraimodoshi_umatan_1a), 1, 2) as integer) as 馬単馬番1
        ,cast(substring(trim(hr.haraimodoshi_umatan_1a), 3, 2) as integer) as 馬単馬番2
        ,nullif(cast(nullif(trim(hr.haraimodoshi_umatan_1b), '') as float), 0) / 100 as 馬単オッズ
        ,nullif(cast(nullif(trim(hr.haraimodoshi_sanrenpuku_1b), '') as float), 0) / 100 as ３連複オッズ
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
                , se.futan_juryo
                , se.kishu_code
                , se.chokyoshi_code
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
        inner join jvd_hr hr
            on ra.kaisai_nen = hr.kaisai_nen 
            and ra.kaisai_tsukihi = hr.kaisai_tsukihi 
            and ra.keibajo_code = hr.keibajo_code 
            and ra.race_bango = hr.race_bango
    where
        cast(ra.kaisai_nen as integer) = 2023 
    ) rase 
    where 
    rase.keibajo_code = '09'
    and cast(rase.kyoso_shubetsu_code as integer) >= cast('13' as integer)
    and cast(rase.track_code as integer) between 10 and 22
    and cast(rase.kyori as integer) >= 1700
    """

    # データを取得
    df = pd.read_sql_query(sql=sql, con=conn)

    # 馬名だけは保存しておく
    horse_names = df['bamei'].copy()
    
    # 数値データだけを前処理
    numeric_columns = df.columns.drop(['bamei', 'keibajo_name'])  # 馬名以外の列を取得
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[numeric_columns] = df[numeric_columns].replace('0', np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # 保存しておいた馬名を戻す
    df['bamei'] = horse_names

    # 特徴量を選択
    X = df.loc[:, [
        "kyori",
        "tenko_code",  
        "babajotai_code_shiba",
        "seibetsu_code",
        # "umaban_numeric", 
        # "barei",
        "futan_juryo",
        "past_score",
        "kohan_3f_index",
        "sotai_chakujun_numeric",
        "time_index",
        # "mare_horse",
        # "femare_horse",
        # "sen_horse",
        # "baba_good",
        # "baba_slightly_heavy",
        # "baba_heavy",
        # "baba_defective",
        # "tenko_fine",
        # "tenko_cloudy",
        # "tenko_rainy",
        # "tenko_drizzle",
        # "tenko_snow",
        # "tenko_light_snow",
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
    
    # 馬齢の非線形変換（競走馬のピーク年齢効果）
    # df['barei_squared'] = df['barei'] ** 2
    # X['barei_squared'] = df['barei_squared']
    df['barei_peak_distance'] = abs(df['barei'] - 4)  # 4歳をピークと仮定
    X['barei_peak_distance'] = df['barei_peak_distance']

    # レース内での馬番相対位置（頭数による正規化）
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # # 微小な個体識別子を追加（重複完全回避のため）
    # # 馬番ベースの極小調整値
    # df['micro_adjustment'] = df['umaban_numeric'] / 1000000  # 0.000001〜0.000018程度
    # X['micro_adjustment'] = df['micro_adjustment']

    # カテゴリ変数を作成
    X['kyori'] = X['kyori'].astype('category')
    X['tenko_code'] = X['tenko_code'].astype('category')
    X['babajotai_code_shiba'] = X['babajotai_code_shiba'].astype('category')
    X['seibetsu_code'] = X['seibetsu_code'].astype('category')

    # お試し特徴量だよ
        
    # kohan_3f_indexを距離に応じた値にする！
    distance_bins = [0, 1600, 2000, 2400, 10000]
    default_values = {
        0: 33.5,   # 短距離（〜1600m）
        1: 35.0,   # マイル（〜2000m）
        2: 36.0,   # 中距離（〜2400m）
        3: 37.0    # 長距離（2400m〜）
    }
    
    # 距離のビンに応じて基準タイムを割り当て
    df['distance_bin'] = pd.cut(df['kyori'], bins=distance_bins, labels=False)
    df['kohan_3f_base'] = df['distance_bin'].map(default_values)
    
    # 基準タイムから少しだけランダムにずらす（実際のレースっぽく）
    np.random.seed(42)  # 再現性のためシード固定
    df['kohan_3f_sec'] = df['kohan_3f_base'] + np.random.normal(0, 0, len(df))
    
    # kohan_3f_indexを計算（main.pyと同じ計算方法）
    df['kohan_3f_index'] = df['kohan_3f_sec'] - df['kohan_3f_base']
    X['kohan_3f_index'] = df['kohan_3f_index']

    # モデルをロード
    with open('hanshin_shiba_3ageup_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)

    # シグモイド関数を定義
    def sigmoid(x):
        """値を0-1の範囲に収めるよ～"""
        import numpy as np
        return 1 / (1 + np.exp(-x))

    # 予測を実行して、57y5t rvfnnシグモイド関数で変換
    raw_scores = model.predict(X)
    df['predicted_chakujun_score'] = sigmoid(raw_scores)

    # データをソート
    df = df.sort_values(by=['kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'umaban'], ascending=True)

    # グループ内でのスコア順位を計算
    df['score_rank'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['predicted_chakujun_score'].rank(method='min', ascending=False)

    # kakutei_chakujun_numeric と score_rank を整数に変換
    df['kakutei_chakujun_numeric'] = df['kakutei_chakujun_numeric'].fillna(0).astype(int)
    df['tansho_ninkijun_numeric'] = df['tansho_ninkijun_numeric'].fillna(0).astype(int)
    df['score_rank'] = df['score_rank'].fillna(0).astype(int)

    # 必要な列を選択
    output_columns = ['keibajo_name',
                      'kaisai_nen', 
                      'kaisai_tsukihi', 
                      'race_bango', 
                      'umaban', 
                      'bamei', 
                      'tansho_odds', 
                      'tansho_ninkijun_numeric', 
                      'kakutei_chakujun_numeric', 
                      'score_rank', 
                      'predicted_chakujun_score',
                      '複勝1着馬番',
                      '複勝1着オッズ',
                      '複勝1着人気',
                      '複勝2着馬番',
                      '複勝2着オッズ',
                      '複勝2着人気',
                      '複勝3着馬番',
                      '複勝3着オッズ',
                      '複勝3着人気',
                      '馬連馬番1',
                      '馬連馬番2',
                      '馬連オッズ',
                      'ワイド1_2馬番1',
                      'ワイド1_2馬番2',
                      'ワイド2_3着馬番1',
                      'ワイド2_3着馬番2',
                      'ワイド1_3着馬番1',
                      'ワイド1_3着馬番2',
                      'ワイド1_2オッズ',
                      'ワイド2_3オッズ',
                      'ワイド1_3オッズ',
                      '馬単馬番1',
                      '馬単馬番2',
                      '馬単オッズ',
                      '３連複オッズ',]
    output_df = df[output_columns]

    # # 単勝的中列を追加（的中すると〇を付ける）
    # output_df['単勝的中'] = np.where(output_df['kakutei_chakujun_numeric'] == 1, '〇', '×')
    # # 複勝的中列を追加（的中すると〇を付ける）
    # output_df['複勝的中'] = np.where(output_df['kakutei_chakujun_numeric'].isin([1, 2, 3]), '〇', '×')

    # 列名を変更
    output_df = output_df.rename(columns={
        'keibajo_name': '競馬場',
        'kaisai_nen': '開催年',
        'kaisai_tsukihi': '開催日',
        'race_bango': 'レース番号',
        'umaban': '馬番',
        'bamei': '馬名',
        'tansho_odds': '単勝オッズ',
        'tansho_ninkijun_numeric': '人気順',
        'kakutei_chakujun_numeric': '確定着順',
        'score_rank': '予測順位',
        'predicted_chakujun_score': '予測スコア'
    })

    # 正しいレース数の計算方法はこれ～！
    race_count = len(output_df.groupby(['開催年', '開催日', 'レース番号']))

    # 単勝の的中率と回収率
    tansho_hit = (output_df['確定着順'] == 1) & (output_df['予測順位'] == 1)
    race_count = len(output_df.groupby(['開催年', '開催日', 'レース番号']))
    tansho_hitrate = 100 * tansho_hit.sum() / race_count
    tansho_recoveryrate = 100 * (tansho_hit * output_df['単勝オッズ']).sum() / race_count

    # 複勝の的中率と回収率も同じやり方で修正
    fukusho_hit = (output_df['確定着順'].isin([1, 2, 3])) & (output_df['予測順位'].isin([1, 2, 3]))
    fukusho_hitrate = fukusho_hit.sum() / (race_count * 3) * 100  # 3着以内の的中率
    
    # 的中馬だけ取り出す
    hit_rows = output_df[fukusho_hit].copy()

    def extract_odds(row):
        if row['確定着順'] == 1:
            return row['複勝1着オッズ']
        elif row['確定着順'] == 2:
            return row['複勝2着オッズ']
        elif row['確定着順'] == 3:
            return row['複勝3着オッズ']
        else:
            return 0

    # 的中馬に対応する払戻を計算（100円賭けたとして）
    hit_rows['的中オッズ'] = hit_rows.apply(extract_odds, axis=1)
    total_payout = (hit_rows['的中オッズ'] * 100).sum()

    # 総購入額（毎レースで3頭に100円ずつ）
    total_bet = race_count * 3 * 100
    fukusho_recoveryrate = total_payout / total_bet * 100

    # 馬連の的中率と回収率 (上位2頭が着順1-2に来たかどうか)
    umaren_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: set([1, 2]).issubset(set(x.sort_values('予測スコア', ascending=False).head(2)['確定着順'].values))
    )
    umaren_hitrate = 100 * umaren_hit.sum() / race_count
    umaren_recoveryrate = 100 * (umaren_hit * output_df.groupby(['開催年', '開催日', 'レース番号'])['馬連オッズ'].first()).sum() / race_count

    # ワイドは予測上位3頭から2頭選ぶ組み合わせがどれか1つでも的中すればOK！
    wide_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: any([
            # 予測1位と2位の馬が3着以内に来たかチェック
            len(set(x.sort_values('予測スコア', ascending=False).iloc[[0, 1]]['確定着順'].values) & {1, 2, 3}) == 2,
            # 予測1位と3位の馬が3着以内に来たかチェック
            len(set(x.sort_values('予測スコア', ascending=False).iloc[[0, 2]]['確定着順'].values) & {1, 2, 3}) == 2,
            # 予測2位と3位の馬が3着以内に来たかチェック
            len(set(x.sort_values('予測スコア', ascending=False).iloc[[1, 2]]['確定着順'].values) & {1, 2, 3}) == 2
        ])
    )
    wide_hitrate = wide_hit.sum() / (race_count * 3) * 100

    wide_odds_sum = 0
    for name, race_group in output_df.groupby(['開催年', '開催日', 'レース番号']):
        top_horses = race_group.sort_values('予測スコア', ascending=False).head(3)
        
        # 上位3頭から2頭選ぶ組み合わせのどれかが的中したらOK
        if len(set(top_horses.iloc[[0, 1]]['確定着順'].values) & {1, 2, 3}) == 2:
            wide_odds_sum += race_group['ワイド1_2オッズ'].values[0]
        elif len(set(top_horses.iloc[[0, 2]]['確定着順'].values) & {1, 2, 3}) == 2:
            wide_odds_sum += race_group['ワイド1_3オッズ'].values[0]
        elif len(set(top_horses.iloc[[1, 2]]['確定着順'].values) & {1, 2, 3}) == 2:
            wide_odds_sum += race_group['ワイド2_3オッズ'].values[0]

    wide_total_payout = (wide_odds_sum * 100)

    # ワイドの総購入額（毎レースで3頭に100円ずつ）
    wide_recoveryrate = wide_total_payout / total_bet * 100

    # 馬単の的中率と回収率 (上位2頭が順番通りに着順1-2に来たかどうか)
    umatan_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: list(x.sort_values('予測スコア', ascending=False).head(2)['確定着順'].values) == [1, 2]
    )
    umatan_hitrate = 100 * umatan_hit.sum() / race_count
    
    # レースごとの馬単オッズで集計するように修正！！
    umatan_odds_sum = 0
    for name, race_group in output_df.groupby(['開催年', '開催日', 'レース番号']):
        top_horses = race_group.sort_values('予測スコア', ascending=False).head(2)
        # 上位2頭が順番通りに1-2に来たかチェック
        if list(top_horses['確定着順'].values) == [1, 2]:
            # 的中したらそのレースの馬単オッズを加算
            umatan_odds_sum += race_group['馬単オッズ'].iloc[0]

    # 正しい回収率計算（レース数×100円賭けた場合の回収率）
    umatan_recoveryrate = 100 * umatan_odds_sum / race_count

    # 三連複の的中率と回収率 (上位3頭が着順1-2-3に来たかどうか)
    sanrenpuku_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: set([1, 2, 3]).issubset(set(x.sort_values('予測スコア', ascending=False).head(3)['確定着順'].values))
    )
    sanrenpuku_hitrate = 100 * sanrenpuku_hit.sum() / len(sanrenpuku_hit)
    sanrenpuku_recoveryrate = 100 * (sanrenpuku_hit * output_df.groupby(['開催年', '開催日', 'レース番号'])['３連複オッズ'].first()).sum() / len(sanrenpuku_hit)

    # 結果をデータフレームにまとめる
    summary_df = pd.DataFrame({
        '的中数': [tansho_hit.sum(), fukusho_hit.sum(), umaren_hit.sum(), wide_hit.sum(), umatan_hit.sum(), sanrenpuku_hit.sum()],
        '的中率(%)': [tansho_hitrate, fukusho_hitrate, umaren_hitrate, wide_hitrate, umatan_hitrate, sanrenpuku_hitrate],
        '回収率(%)': [tansho_recoveryrate, fukusho_recoveryrate, umaren_recoveryrate, wide_recoveryrate, umatan_recoveryrate, sanrenpuku_recoveryrate]
    }, index=['単勝', '複勝', '馬連', 'ワイド', '馬単', '３連複'])

    # # 2つの結果を結合する（これがマジ大事！）
    # output_df = pd.concat([output_df, pd.DataFrame([[""] * len(output_df.columns)], columns=output_df.columns)])
    # output_df = pd.concat([output_df, pd.DataFrame([[""] * len(output_df.columns)], columns=output_df.columns)])
    # output_df = pd.concat([output_df, pd.DataFrame([[""] * len(output_df.columns)], columns=output_df.columns)])

    # # ここのコードちょっと変えるだけ！
    # # 空行挿入したいなら、これの方が確実だよ！
    # rows_to_add = len(summary_df)  # 6行必要（単勝、複勝、馬連、ワイド、馬単、３連複）
    # empty_df = pd.DataFrame({col: [""] * rows_to_add for col in output_df.columns})
    # output_df = pd.concat([output_df, empty_df], ignore_index=True)

    # # 結果の表を追加（ループ処理はそのまま使えるよ）
    # start_row = len(output_df) - rows_to_add
    # for i, (bet_type, row) in enumerate(summary_df.iterrows()):
    #     output_df.iloc[start_row + i, 0] = bet_type
    #     output_df.iloc[start_row + i, 1] = f"的中数: {row['的中数']}"
    #     output_df.iloc[start_row + i, 2] = f"的中率: {row['的中率(%)']:.2f}%"
    #     output_df.iloc[start_row + i, 3] = f"回収率: {row['回収率(%)']:.2f}%"

    # 結果をTSVに保存
    output_file = 'predicted_results.tsv'
    output_df.to_csv(output_file, index=False, sep='\t', encoding='utf-8-sig')
    print(f"予測結果を {output_file} に保存しました！")

    # 的中率と回収率を別ファイルに保存（一緒に的中数も！）
    summary_file = 'betting_summary.tsv'
    summary_df.to_csv(summary_file, index=True, sep='\t', encoding='utf-8-sig')
    print(f"的中率・回収率・的中数を {summary_file} に保存しました！")

if __name__ == '__main__':
    predict_and_save_results()