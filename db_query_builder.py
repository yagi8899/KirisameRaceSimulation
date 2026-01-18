"""
競馬データ取得用SQLクエリビルダー

このモジュールは、model_creator.py、universal_test.py、model_explainer.py等で
重複していたSQL生成ロジックを共通化するためのものです。

model_creator.pyのSQL構造をベースに、払い戻し情報（jvd_hr）の結合を
オプションで追加できるようになっています。
"""

from typing import Optional, Tuple


def build_race_data_query(
    track_code: str,
    year_start: int,
    year_end: int,
    surface_type: str = 'turf',
    distance_min: int = 1000,
    distance_max: int = 4000,
    kyoso_shubetsu_code: Optional[str] = None,
    include_payout: bool = False
) -> str:
    """
    競馬データ取得用SQLクエリを動的生成
    
    Args:
        track_code: 競馬場コード（'01'=札幌, '05'=東京, '09'=阪神など）
        year_start: 開始年（例: 2020）
        year_end: 終了年（例: 2023）
        surface_type: 馬場タイプ（'turf'=芝, 'dirt'=ダート）
        distance_min: 最小距離（例: 1800）
        distance_max: 最大距離（例: 2400）、9999を指定すると「以上」条件になる
        kyoso_shubetsu_code: 競走種別コード（'12'=3歳戦, '13'=3歳以上戦など）
        include_payout: 払い戻し情報（jvd_hr）を含むか（universal_test.py用）
    
    Returns:
        str: 実行可能なSQLクエリ
    """
    # 芝/ダート条件
    if surface_type == 'turf':
        track_condition = "cast(rase.track_code as integer) between 10 and 22"
        baba_condition = "ra.babajotai_code_shiba"
    else:
        track_condition = "cast(rase.track_code as integer) between 23 and 29"
        baba_condition = "ra.babajotai_code_dirt"
    
    # 距離条件
    if distance_max == 9999:
        distance_condition = f"cast(rase.kyori as integer) >= {distance_min}"
    else:
        distance_condition = f"cast(rase.kyori as integer) between {distance_min} and {distance_max}"
    
    # 競争種別条件
    if kyoso_shubetsu_code == '12':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) = 12"
    elif kyoso_shubetsu_code == '13':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) >= 13"
    else:
        kyoso_shubetsu_condition = "1=1"  # 条件なし
    
    # 払い戻し情報の結合（universal_test.py用）
    if include_payout:
        payout_join = """inner join jvd_hr hr
            on ra.kaisai_nen = hr.kaisai_nen 
            and ra.kaisai_tsukihi = hr.kaisai_tsukihi 
            and ra.keibajo_code = hr.keibajo_code 
            and ra.race_bango = hr.race_bango"""
        
        payout_columns = """,nullif(cast(nullif(trim(hr.haraimodoshi_fukusho_1a), '') as integer), 0) as 複勝1着馬番
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
        ,nullif(cast(nullif(trim(hr.haraimodoshi_sanrenpuku_1b), '') as float), 0) / 100 as ３連複オッズ"""
    else:
        payout_join = ""
        payout_columns = ""
    
    # SQLクエリ組み立て（model_creator.pyベース）
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
        trim(seum.bamei) as bamei,
        seum.wakuban,
        seum.umaban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.kishu_name,
        seum.chokyoshi_name,
        seum.futan_juryo,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        seum.seibetsu_code,
        seum.corner_1,
        seum.corner_2,
        seum.corner_3,
        seum.corner_4,
        seum.kyakushitsu_hantei,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        seum.kakutei_chakujun,
        18 - cast(seum.kakutei_chakujun as integer) + 1 as kakutei_chakujun_numeric, 
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,  --上位着順ほど1に近くなる
        AVG(
            (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0)))
            * CASE
                WHEN seum.time_sa LIKE '-%' THEN 1.00  -- 1着(マイナス値) → 係数1.00(満点)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85   -- 0.5秒差以内 → 0.85倍(15%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70  -- 1.0秒差以内 → 0.70倍(30%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50  -- 2.0秒差以内 → 0.50倍(50%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30  -- 3.0秒差以内 → 0.30倍(70%減)
                ELSE 0.20  -- 3.0秒超 → 0.20倍(大敗はほぼ無視)
            END
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
        END AS kohan_3f_index,
        -- 騎手スコア: 過去30走の平均着順スコア（1着=高スコア）
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            ) >= 10 THEN
            AVG(1.0 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0))) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            )
            ELSE 0.5
        END AS kishu_skill_score,
        -- 騎手の路面別スコア: 過去50走の同一路面平均成績（1着=高スコア）
        CASE 
            WHEN COUNT(
                CASE 
                    WHEN CASE 
                        WHEN cast(ra.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(ra.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END = 
                    CASE 
                        WHEN cast(ra.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(ra.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END
                    THEN 1 ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
            ) >= 5 THEN
            AVG(
                CASE 
                    WHEN CASE 
                        WHEN cast(ra.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(ra.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END = 
                    CASE 
                        WHEN cast(ra.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(ra.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END
                    THEN 1.0 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0))
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
            )
            ELSE 0.5
        END AS kishu_surface_score,
        -- 調教師スコア: 過去20走の平均着順スコア（1着=高スコア）
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY seum.chokyoshi_code
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) >= 5 THEN
            AVG(1.0 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0))) OVER (
                PARTITION BY seum.chokyoshi_code
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            )
            ELSE 0.5
        END AS chokyoshi_recent_score,
        -- 馬番パーセンタイル: レース内での馬番の相対位置（0～1）
        PERCENT_RANK() OVER (
            PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
            ORDER BY cast(seum.umaban as integer)
        ) AS umaban_percentile,
        -- 斤量Z-score: レース内での斤量の標準化スコア
        CASE 
            WHEN STDDEV(cast(seum.futan_juryo as float)) OVER (
                PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
            ) > 0 THEN
            (cast(seum.futan_juryo as float) - AVG(cast(seum.futan_juryo as float)) OVER (
                PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
            )) / STDDEV(cast(seum.futan_juryo as float)) OVER (
                PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
            )
            ELSE 0
        END AS futan_zscore,
        -- 斤量パーセンタイル: レース内での斤量の相対位置（0～1）
        PERCENT_RANK() OVER (
            PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
            ORDER BY cast(seum.futan_juryo as float)
        ) AS futan_percentile,
        -- 短距離スコア: 1000-1400mでの過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(ra.kyori as integer) BETWEEN 1000 AND 1400
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_short,
        -- マイルスコア: 1401-1800mでの過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(ra.kyori as integer) BETWEEN 1401 AND 1800
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_mile,
        -- 中距離スコア: 1801-2400mでの過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(ra.kyori as integer) BETWEEN 1801 AND 2400
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_middle,
        -- 長距離スコア: 2401m以上での過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(ra.kyori as integer) >= 2401
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_long,
        -- 前走距離差: 前走との距離差（m）
        cast(ra.kyori as integer) - LAG(cast(ra.kyori as integer)) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
        ) AS zenso_kyori_sa,
        -- 長距離経験回数: 過去の2200m以上レース経験回数
        COUNT(
            CASE WHEN cast(ra.kyori as integer) >= 2200 THEN 1 ELSE NULL END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS long_distance_experience_count,
        -- スタート指数: 過去の1コーナー平均位置と馬番平均の差（完全版）
        CASE 
            WHEN COUNT(
                CASE WHEN seum.corner_1 IS NOT NULL AND seum.corner_1 <> '' THEN 1 ELSE NULL END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) >= 1 THEN
            AVG(
                CASE 
                    WHEN seum.corner_1 IS NOT NULL AND seum.corner_1 <> '' 
                    THEN cast(seum.corner_1 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) - 
            AVG(cast(seum.umaban as float)) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
            ELSE 0
        END AS start_index,
        -- コーナー位置スコア: 過去3走の全コーナー平均位置
        (
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_1 IS NOT NULL AND seum.corner_1 <> '' 
                    THEN cast(seum.corner_1 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0) +
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_2 IS NOT NULL AND seum.corner_2 <> '' 
                    THEN cast(seum.corner_2 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0) +
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_3 IS NOT NULL AND seum.corner_3 <> '' 
                    THEN cast(seum.corner_3 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0) +
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_4 IS NOT NULL AND seum.corner_4 <> '' 
                    THEN cast(seum.corner_4 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0)
        ) / 4.0 AS corner_position_score,
        -- 路面適性スコア: 同一路面での過去3走平均着順スコア
        AVG(
            CASE 
                WHEN CASE 
                    WHEN cast(ra.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                    WHEN cast(ra.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                    ELSE 'unknown'
                END = 
                CASE 
                    WHEN cast(ra.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                    WHEN cast(ra.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                    ELSE 'unknown'
                END
                THEN 1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0)
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS surface_aptitude_score{payout_columns}
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
                , se.time_sa
                , se.corner_1
                , se.corner_2
                , se.corner_3
                , se.corner_4
                , se.kyakushitsu_hantei
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
        {payout_join}
    where
        cast(ra.kaisai_nen as integer) between {year_start} and {year_end}    --学習データ年範囲
    ) rase 
    where 
    rase.keibajo_code = '{track_code}'                                        --競馬場指定
    and {kyoso_shubetsu_condition}                                            --競争種別
    and {track_condition}                                                     --芝/ダート
    and {distance_condition}                                                  --距離条件
    """
    
    return sql


def build_sokuho_race_data_query(
    track_code: str,
    surface_type: str = 'turf',
    distance_min: int = 1000,
    distance_max: int = 4000,
    kyoso_shubetsu_code: Optional[str] = None
) -> str:
    """
    速報データ予測用SQLクエリを動的生成
    
    apd_sokuho_jvd_seから速報データを取得し、jvd_seの過去データと結合して
    ウィンドウ関数で特徴量を計算する。最終的に速報データのみを返す。
    
    Args:
        track_code: 競馬場コード（'01'=札幌, '05'=東京, '09'=阪神など）
        surface_type: 馬場タイプ（'turf'=芝, 'dirt'=ダート）
        distance_min: 最小距離（例: 1800）
        distance_max: 最大距離（例: 2400）、9999を指定すると「以上」条件になる
        kyoso_shubetsu_code: 競走種別コード（'12'=3歳戦, '13'=3歳以上戦など）
    
    Returns:
        str: 実行可能なSQLクエリ
    """
    # 年数範囲: 直近5年分のデータのみ取得（パフォーマンス最適化）
    from datetime import datetime
    current_year = datetime.now().year
    start_year = current_year - 5
    
    # 芝/ダート条件
    if surface_type == 'turf':
        track_condition = "cast(rase.track_code as integer) between 10 and 22"
        baba_condition = "babajotai_code_shiba"
    else:
        track_condition = "cast(rase.track_code as integer) between 23 and 29"
        baba_condition = "babajotai_code_dirt"
    
    # 距離条件
    if distance_max == 9999:
        distance_condition = f"cast(rase.kyori as integer) >= {distance_min}"
    else:
        distance_condition = f"cast(rase.kyori as integer) between {distance_min} and {distance_max}"
    
    # 競争種別条件
    if kyoso_shubetsu_code == '12':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) = 12"
    elif kyoso_shubetsu_code == '13':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) >= 13"
    else:
        kyoso_shubetsu_condition = "1=1"  # 条件なし
    
    # SQLクエリ組み立て: 過去データと速報データをUNION ALLで結合し、ウィンドウ関数で特徴量を計算
    sql = f"""
    select * from (
        select
        seum.kaisai_nen,
        seum.kaisai_tsukihi,
        seum.keibajo_code,
        CASE 
            WHEN seum.keibajo_code = '01' THEN '札幌' 
            WHEN seum.keibajo_code = '02' THEN '函館' 
            WHEN seum.keibajo_code = '03' THEN '福島' 
            WHEN seum.keibajo_code = '04' THEN '新潟' 
            WHEN seum.keibajo_code = '05' THEN '東京' 
            WHEN seum.keibajo_code = '06' THEN '中山' 
            WHEN seum.keibajo_code = '07' THEN '中京' 
            WHEN seum.keibajo_code = '08' THEN '京都' 
            WHEN seum.keibajo_code = '09' THEN '阪神' 
            WHEN seum.keibajo_code = '10' THEN '小倉' 
            ELSE '' 
        END keibajo_name,
        seum.race_bango,
        seum.kyori,
        seum.tenko_code,
        seum.babajotai_code,
        seum.grade_code,
        seum.kyoso_joken_code,
        seum.kyoso_shubetsu_code,
        seum.track_code,
        seum.shusso_tosu,
        seum.ketto_toroku_bango,
        trim(seum.bamei) as bamei,
        seum.wakuban,
        seum.umaban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.kishu_name,
        seum.chokyoshi_name,
        seum.futan_juryo,
        nullif(cast(seum.tansho_odds as float), 0) / 10 as tansho_odds,
        seum.seibetsu_code,
        seum.corner_1,
        seum.corner_2,
        seum.corner_3,
        seum.corner_4,
        seum.kyakushitsu_hantei,
        nullif(cast(seum.tansho_ninkijun as integer), 0) as tansho_ninkijun_numeric,
        seum.is_sokuho,
        seum.kakutei_chakujun_numeric,
        seum.chakujun_score,
        AVG(
            (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0)))
            * CASE
                WHEN seum.time_sa LIKE '-%' THEN 1.00  -- 1着(マイナス値) → 係数1.00(満点)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85   -- 0.5秒差以内 → 0.85倍(15%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70  -- 1.0秒差以内 → 0.70倍(30%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50  -- 2.0秒差以内 → 0.50倍(50%減)
                WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30  -- 3.0秒差以内 → 0.30倍(70%減)
                ELSE 0.20  -- 3.0秒超 → 0.20倍(大敗はほぼ無視)
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS past_avg_sotai_chakujun,
        AVG(
            cast(seum.kyori as integer) /
            NULLIF(
                FLOOR(cast(seum.soha_time as integer) / 1000) * 60 +
                FLOOR((cast(seum.soha_time as integer) % 1000) / 10) +
                (cast(seum.soha_time as integer) % 10) * 0.1,
                0
            )
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
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
                WHEN seum.grade_code = 'A' THEN 3.00                                                                                          --G1 (1.00→3.00に強化)
                WHEN seum.grade_code = 'B' THEN 2.00                                                                                          --G2 (0.80→2.00に強化)
                WHEN seum.grade_code = 'C' THEN 1.50                                                                                          --G3 (0.60→1.50に強化)
                WHEN seum.grade_code <> 'A' AND seum.grade_code <> 'B' AND seum.grade_code <> 'C' AND seum.kyoso_joken_code = '999' THEN 1.00       --OP (0.50→1.00に調整)
                WHEN seum.grade_code <> 'A' AND seum.grade_code <> 'B' AND seum.grade_code <> 'C' AND seum.kyoso_joken_code = '016' THEN 0.80       --3勝クラス (0.40→0.80に調整)
                WHEN seum.grade_code <> 'A' AND seum.grade_code <> 'B' AND seum.grade_code <> 'C' AND seum.kyoso_joken_code = '010' THEN 0.60       --2勝クラス (0.30→0.60に調整)
                WHEN seum.grade_code <> 'A' AND seum.grade_code <> 'B' AND seum.grade_code <> 'C' AND seum.kyoso_joken_code = '005' THEN 0.40       --1勝クラス (0.20→0.40に調整)
                ELSE 0.20                                                                                                                   --未勝利 (0.10→0.20に調整)
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
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
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
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
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) - 
            CASE
                WHEN cast(seum.kyori as integer) <= 1600 THEN 33.5
                WHEN cast(seum.kyori as integer) <= 2000 THEN 35.0
                WHEN cast(seum.kyori as integer) <= 2400 THEN 36.0
                ELSE 37.0
            END
            ELSE 0
        END AS kohan_3f_index,
        -- 騎手スコア: 過去30走の平均着順スコア（1着=高スコア）
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            ) >= 10 THEN
            AVG(1.0 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0))) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            )
            ELSE 0.5
        END AS kishu_skill_score,
        -- 騎手の路面別スコア: 過去50走の同一路面平均成績（1着=高スコア）
        CASE 
            WHEN COUNT(
                CASE 
                    WHEN CASE 
                        WHEN cast(seum.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(seum.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END = 
                    CASE 
                        WHEN cast(seum.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(seum.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END
                    THEN 1 ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
            ) >= 5 THEN
            AVG(
                CASE 
                    WHEN CASE 
                        WHEN cast(seum.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(seum.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END = 
                    CASE 
                        WHEN cast(seum.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                        WHEN cast(seum.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                        ELSE 'unknown'
                    END
                    THEN 1.0 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0))
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.kishu_code
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
            )
            ELSE 0.5
        END AS kishu_surface_score,
        -- 調教師スコア: 過去20走の平均着順スコア（1着=高スコア）
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY seum.chokyoshi_code
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) >= 5 THEN
            AVG(1.0 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0))) OVER (
                PARTITION BY seum.chokyoshi_code
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            )
            ELSE 0.5
        END AS chokyoshi_recent_score,
        -- 馬番パーセンタイル: レース内での馬番の相対位置（0～1）
        PERCENT_RANK() OVER (
            PARTITION BY seum.kaisai_nen, seum.kaisai_tsukihi, seum.keibajo_code, seum.race_bango
            ORDER BY cast(seum.umaban as integer)
        ) AS umaban_percentile,
        -- 斤量Z-score: レース内での斤量の標準化スコア
        CASE 
            WHEN STDDEV(cast(seum.futan_juryo as float)) OVER (
                PARTITION BY seum.kaisai_nen, seum.kaisai_tsukihi, seum.keibajo_code, seum.race_bango
            ) > 0 THEN
            (cast(seum.futan_juryo as float) - AVG(cast(seum.futan_juryo as float)) OVER (
                PARTITION BY seum.kaisai_nen, seum.kaisai_tsukihi, seum.keibajo_code, seum.race_bango
            )) / STDDEV(cast(seum.futan_juryo as float)) OVER (
                PARTITION BY seum.kaisai_nen, seum.kaisai_tsukihi, seum.keibajo_code, seum.race_bango
            )
            ELSE 0
        END AS futan_zscore,
        -- 斤量パーセンタイル: レース内での斤量の相対位置（0～1）
        PERCENT_RANK() OVER (
            PARTITION BY seum.kaisai_nen, seum.kaisai_tsukihi, seum.keibajo_code, seum.race_bango
            ORDER BY cast(seum.futan_juryo as float)
        ) AS futan_percentile,
        -- 短距離スコア: 1000-1400mでの過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(seum.kyori as integer) BETWEEN 1000 AND 1400
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_short,
        -- マイルスコア: 1401-1800mでの過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(seum.kyori as integer) BETWEEN 1401 AND 1800
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_mile,
        -- 中距離スコア: 1801-2400mでの過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(seum.kyori as integer) BETWEEN 1801 AND 2400
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_middle,
        -- 長距離スコア: 2401m以上での過去5走の平均成績スコア
        AVG(
            CASE 
                WHEN cast(seum.kyori as integer) >= 2401
                THEN (1 - (cast(seum.kakutei_chakujun as float) / NULLIF(cast(seum.shusso_tosu as float), 0)))
                    * CASE
                        WHEN seum.time_sa LIKE '-%' THEN 1.00
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
                        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 30 THEN 0.30
                        ELSE 0.20
                    END
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS past_score_long,
        -- 前走距離差: 前走との距離差（m）
        cast(seum.kyori as integer) - LAG(cast(seum.kyori as integer)) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
        ) AS zenso_kyori_sa,
        -- 長距離経験回数: 過去の2200m以上レース経験回数
        COUNT(
            CASE WHEN cast(seum.kyori as integer) >= 2200 THEN 1 ELSE NULL END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS long_distance_experience_count,
        -- スタート指数: 過去の1コーナー平均位置と馬番平均の差（完全版）
        CASE 
            WHEN COUNT(
                CASE WHEN seum.corner_1 IS NOT NULL AND seum.corner_1 <> '' THEN 1 ELSE NULL END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) >= 1 THEN
            AVG(
                CASE 
                    WHEN seum.corner_1 IS NOT NULL AND seum.corner_1 <> '' 
                    THEN cast(seum.corner_1 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) - 
            AVG(cast(seum.umaban as float)) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
            ELSE 0
        END AS start_index,
        -- コーナー位置スコア: 過去3走の全コーナー平均位置
        (
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_1 IS NOT NULL AND seum.corner_1 <> '' 
                    THEN cast(seum.corner_1 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0) +
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_2 IS NOT NULL AND seum.corner_2 <> '' 
                    THEN cast(seum.corner_2 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0) +
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_3 IS NOT NULL AND seum.corner_3 <> '' 
                    THEN cast(seum.corner_3 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0) +
            COALESCE(AVG(
                CASE 
                    WHEN seum.corner_4 IS NOT NULL AND seum.corner_4 <> '' 
                    THEN cast(seum.corner_4 as float)
                    ELSE NULL
                END
            ) OVER (
                PARTITION BY seum.ketto_toroku_bango
                ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ), 0)
        ) / 4.0 AS corner_position_score,
        -- 路面適性スコア: 同一路面での過去3走平均着順スコア
        AVG(
            CASE 
                WHEN CASE 
                    WHEN cast(seum.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                    WHEN cast(seum.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                    ELSE 'unknown'
                END = 
                CASE 
                    WHEN cast(seum.track_code as integer) BETWEEN 10 AND 22 THEN 'turf'
                    WHEN cast(seum.track_code as integer) BETWEEN 23 AND 29 THEN 'dirt'
                    ELSE 'unknown'
                END
                THEN 1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0)
                ELSE NULL
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(seum.kaisai_nen as integer), cast(seum.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS surface_aptitude_score
    from (
        -- 過去データ（jvd_se）
        select
            se.kaisai_nen,
            se.kaisai_tsukihi,
            se.keibajo_code,
            se.race_bango,
            cast(se.race_bango as integer) as race_bango_int,
            se.kakutei_chakujun,
            18 - cast(se.kakutei_chakujun as integer) + 1 as kakutei_chakujun_numeric,
            1.0 / nullif(cast(se.kakutei_chakujun as integer), 0) as chakujun_score,
            se.ketto_toroku_bango,
            se.bamei,
            se.wakuban,
            se.umaban,
            se.barei,
            se.seibetsu_code,
            se.kishu_code,
            se.chokyoshi_code,
            trim(se.kishumei_ryakusho) as kishu_name,
            trim(se.chokyoshimei_ryakusho) as chokyoshi_name,
            se.futan_juryo,
            se.tansho_odds,
            se.tansho_ninkijun,
            se.kohan_3f,
            se.soha_time,
            se.time_sa,
            se.corner_1,
            se.corner_2,
            se.corner_3,
            se.corner_4,
            se.kyakushitsu_hantei,
            past_ra.kyori,
            past_ra.tenko_code,
            past_ra.{baba_condition} as babajotai_code,
            past_ra.shusso_tosu,
            past_ra.grade_code,
            past_ra.kyoso_joken_code,
            past_ra.kyoso_shubetsu_code,
            past_ra.track_code,
            0 as is_sokuho
        from jvd_se se
        inner join jvd_ra past_ra
            on se.kaisai_nen = past_ra.kaisai_nen
            and se.kaisai_tsukihi = past_ra.kaisai_tsukihi
            and se.keibajo_code = past_ra.keibajo_code
            and se.race_bango = past_ra.race_bango
        where se.kohan_3f <> '000' and se.kohan_3f <> '999'
            and cast(se.kaisai_nen as integer) >= {start_year}
        
        UNION ALL
        
        -- 速報データ（今回のレース）
        select
            sokuho_se.kaisai_nen,
            sokuho_se.kaisai_tsukihi,
            sokuho_se.keibajo_code,
            sokuho_se.race_bango,
            cast(sokuho_se.race_bango as integer) as race_bango_int,
            null as kakutei_chakujun,
            null as kakutei_chakujun_numeric,
            null as chakujun_score,
            sokuho_se.ketto_toroku_bango,
            sokuho_se.bamei,
            sokuho_se.wakuban,
            sokuho_se.umaban,
            sokuho_se.barei,
            sokuho_se.seibetsu_code,
            sokuho_se.kishu_code,
            sokuho_se.chokyoshi_code,
            trim(sokuho_se.kishumei_ryakusho) as kishu_name,
            trim(sokuho_se.chokyoshimei_ryakusho) as chokyoshi_name,
            sokuho_se.futan_juryo,
            sokuho_se.tansho_odds,
            sokuho_se.tansho_ninkijun,
            null as kohan_3f,
            null as soha_time,
            null as time_sa,
            null as corner_1,
            null as corner_2,
            null as corner_3,
            null as corner_4,
            null as kyakushitsu_hantei,
            sokuho_ra.kyori,
            sokuho_ra.tenko_code,
            sokuho_ra.{baba_condition} as babajotai_code,
            sokuho_ra.shusso_tosu,
            sokuho_ra.grade_code,
            sokuho_ra.kyoso_joken_code,
            sokuho_ra.kyoso_shubetsu_code,
            sokuho_ra.track_code,
            1 as is_sokuho
        from apd_sokuho_jvd_se sokuho_se
        inner join apd_sokuho_jvd_ra sokuho_ra
            on sokuho_se.kaisai_nen = sokuho_ra.kaisai_nen
            and sokuho_se.kaisai_tsukihi = sokuho_ra.kaisai_tsukihi
            and sokuho_se.keibajo_code = sokuho_ra.keibajo_code
            and sokuho_se.race_bango = sokuho_ra.race_bango
    ) seum
    ) rase
    where 
    rase.is_sokuho = 1                                                        -- 速報データのみ抽出
    and rase.keibajo_code = '{track_code}'                                    -- 競馬場指定
    and {kyoso_shubetsu_condition}                                            -- 競争種別
    and {track_condition}                                                     -- 芝/ダート
    and {distance_condition}                                                  -- 距離条件
    """
    
    return sql
