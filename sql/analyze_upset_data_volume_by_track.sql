-- ================================================================================
-- 全競馬場における穴馬データ量の調査
-- 
-- 目的: 条件別穴馬分類器を作成する際のデータ量を検証
-- 定義: 7-12番人気で3着以内に入る馬を「穴馬」とする
-- ================================================================================

-- ================================================================================
-- 1. 全体統計（全競馬場・全条件）
-- ================================================================================
SELECT
    '全体統計' AS category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) AS upset_count,
    ROUND(SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS upset_rate_pct
FROM (
    SELECT
        CASE
            WHEN CAST(CAST(n.tansho_ninkijun AS INTEGER) AS INTEGER) BETWEEN 7 AND 12
                AND CAST(n.kakutei_chakujun AS INTEGER) BETWEEN 1 AND 3
            THEN 1
            ELSE 0
        END AS is_upset
    FROM jvd_se n
    INNER JOIN jvd_ra r ON n.kaisai_nen = r.kaisai_nen
        AND n.kaisai_tsukihi = r.kaisai_tsukihi
        AND n.keibajo_code = r.keibajo_code
        AND n.race_bango = r.race_bango
    WHERE CAST(CAST(n.tansho_ninkijun AS INTEGER) AS INTEGER) BETWEEN 7 AND 12  -- 7-12番人気のみ
        AND CAST(n.kakutei_chakujun AS INTEGER) != 0           -- レース完走馬のみ
        AND CAST(r.kyori AS INTEGER) >= 1000                   -- 1000m以上
) AS base_data;

-- ================================================================================
-- 2. 競馬場別の穴馬データ量
-- ================================================================================
SELECT
    track_data.keibajo_code,
    CASE track_data.keibajo_code
        WHEN '01' THEN '札幌'
        WHEN '02' THEN '函館'
        WHEN '03' THEN '福島'
        WHEN '04' THEN '新潟'
        WHEN '05' THEN '東京'
        WHEN '06' THEN '中山'
        WHEN '07' THEN '中京'
        WHEN '08' THEN '京都'
        WHEN '09' THEN '阪神'
        WHEN '10' THEN '小倉'
        ELSE '不明'
    END AS track_name,
    COUNT(*) AS total_records,
    SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) AS upset_count,
    ROUND(SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS upset_rate_pct,
    CASE
        WHEN COUNT(*) >= 1000 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 100 THEN '✅ 十分'
        WHEN COUNT(*) >= 500 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 50 THEN '⚠️ やや不足'
        ELSE '不足'
    END AS data_sufficiency
FROM (
    SELECT
        r.keibajo_code,
        CASE
            WHEN CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
                AND CAST(n.kakutei_chakujun AS INTEGER) BETWEEN 1 AND 3
            THEN 1
            ELSE 0
        END AS is_upset
    FROM jvd_se n
    INNER JOIN jvd_ra r ON n.kaisai_nen = r.kaisai_nen
        AND n.kaisai_tsukihi = r.kaisai_tsukihi
        AND n.keibajo_code = r.keibajo_code
        AND n.race_bango = r.race_bango
    WHERE CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
        AND CAST(n.kakutei_chakujun AS INTEGER) != 0
        AND CAST(r.kyori AS INTEGER) >= 1000
) AS track_data
GROUP BY track_data.keibajo_code
ORDER BY total_records DESC;

-- ================================================================================
-- 3. 競馬場 × 芝ダ区分別の穴馬データ量
-- ================================================================================
SELECT
    surface_data.keibajo_code,
    CASE surface_data.keibajo_code
        WHEN '01' THEN '札幌'
        WHEN '02' THEN '函館'
        WHEN '03' THEN '福島'
        WHEN '04' THEN '新潟'
        WHEN '05' THEN '東京'
        WHEN '06' THEN '中山'
        WHEN '07' THEN '中京'
        WHEN '08' THEN '京都'
        WHEN '09' THEN '阪神'
        WHEN '10' THEN '小倉'
        ELSE '不明'
    END AS track_name,
    surface_data.surface_type,
    COUNT(*) AS total_records,
    SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) AS upset_count,
    ROUND(SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS upset_rate_pct,
    CASE
        WHEN COUNT(*) >= 1000 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 100 THEN '十分'
        WHEN COUNT(*) >= 500 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 50 THEN 'やや不足'
        ELSE '不足'
    END AS data_sufficiency
FROM (
    SELECT
        r.keibajo_code,
        r.track_code,
        CASE r.track_code
            WHEN '10' THEN '芝'
            WHEN '11' THEN '芝'
            WHEN '12' THEN '芝'
            WHEN '13' THEN '芝'
            WHEN '14' THEN '芝'
            WHEN '15' THEN '芝'
            WHEN '16' THEN '芝'
            WHEN '17' THEN '芝'
            WHEN '18' THEN '芝'
            WHEN '19' THEN '芝'
            WHEN '20' THEN '芝'
            WHEN '21' THEN '芝'
            WHEN '22' THEN '芝'
            WHEN '23' THEN 'ダート'
            WHEN '24' THEN 'ダート'
            WHEN '25' THEN 'ダート'
            WHEN '26' THEN 'ダート'
            WHEN '27' THEN 'ダート'
            WHEN '28' THEN 'ダート'
            WHEN '29' THEN 'ダート'
            ELSE '不明'
        END AS surface_type,
        CASE
            WHEN CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
                AND CAST(n.kakutei_chakujun AS INTEGER) BETWEEN 1 AND 3
            THEN 1
            ELSE 0
        END AS is_upset
    FROM jvd_se n
    INNER JOIN jvd_ra r ON n.kaisai_nen = r.kaisai_nen
        AND n.kaisai_tsukihi = r.kaisai_tsukihi
        AND n.keibajo_code = r.keibajo_code
        AND n.race_bango = r.race_bango
    WHERE CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
        AND CAST(n.kakutei_chakujun AS INTEGER) != 0
        AND CAST(r.kyori AS INTEGER) >= 1000
) AS surface_data
GROUP BY surface_data.keibajo_code, surface_data.surface_type
ORDER BY track_name, surface_type;

-- ================================================================================
-- 4. 競馬場 × 距離帯別の穴馬データ量
-- ================================================================================
SELECT
    distance_data.keibajo_code,
    CASE distance_data.keibajo_code
        WHEN '01' THEN '札幌'
        WHEN '02' THEN '函館'
        WHEN '03' THEN '福島'
        WHEN '04' THEN '新潟'
        WHEN '05' THEN '東京'
        WHEN '06' THEN '中山'
        WHEN '07' THEN '中京'
        WHEN '08' THEN '京都'
        WHEN '09' THEN '阪神'
        WHEN '10' THEN '小倉'
        ELSE '不明'
    END AS track_name,
    distance_data.distance_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) AS upset_count,
    ROUND(SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS upset_rate_pct,
    CASE
        WHEN COUNT(*) >= 1000 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 100 THEN '十分'
        WHEN COUNT(*) >= 500 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 50 THEN 'やや不足'
        ELSE '不足'
    END AS data_sufficiency
FROM (
    SELECT
        r.keibajo_code,
        r.kyori,
        CASE
            WHEN CAST(r.kyori AS INTEGER) BETWEEN 1000 AND 1400 THEN '短距離(1000-1400m)'
            WHEN CAST(r.kyori AS INTEGER) BETWEEN 1401 AND 1800 THEN 'マイル(1401-1800m)'
            WHEN CAST(r.kyori AS INTEGER) BETWEEN 1801 AND 2200 THEN '中距離(1801-2200m)'
            WHEN CAST(r.kyori AS INTEGER) >= 2201 THEN '長距離(2201m~)'
            ELSE '不明'
        END AS distance_category,
        CASE
            WHEN CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
                AND CAST(n.kakutei_chakujun AS INTEGER) BETWEEN 1 AND 3
            THEN 1
            ELSE 0
        END AS is_upset
    FROM jvd_se n
    INNER JOIN jvd_ra r ON n.kaisai_nen = r.kaisai_nen
        AND n.kaisai_tsukihi = r.kaisai_tsukihi
        AND n.keibajo_code = r.keibajo_code
        AND n.race_bango = r.race_bango
    WHERE CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
        AND CAST(n.kakutei_chakujun AS INTEGER) != 0
        AND CAST(r.kyori AS INTEGER) >= 1000
) AS distance_data
GROUP BY distance_data.keibajo_code, distance_data.distance_category
ORDER BY track_name, distance_category;

-- ================================================================================
-- 5. 競馬場 × 芝ダ × 距離帯別の穴馬データ量（詳細版）
-- ================================================================================
SELECT
    detailed_data.keibajo_code,
    CASE detailed_data.keibajo_code
        WHEN '01' THEN '札幌'
        WHEN '02' THEN '函館'
        WHEN '03' THEN '福島'
        WHEN '04' THEN '新潟'
        WHEN '05' THEN '東京'
        WHEN '06' THEN '中山'
        WHEN '07' THEN '中京'
        WHEN '08' THEN '京都'
        WHEN '09' THEN '阪神'
        WHEN '10' THEN '小倉'
        ELSE '不明'
    END AS track_name,
    detailed_data.surface_type,
    detailed_data.distance_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) AS upset_count,
    ROUND(SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS upset_rate_pct,
    CASE
        WHEN COUNT(*) >= 1000 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 100 THEN '✅ 十分'
        WHEN COUNT(*) >= 500 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 50 THEN '⚠️ やや不足'
        ELSE '❌ 不足'
    END AS data_sufficiency
FROM (
    SELECT
        r.keibajo_code,
        r.track_code,
        r.kyori,
        CASE r.track_code
            WHEN '10' THEN '芝'
            WHEN '11' THEN '芝'
            WHEN '12' THEN '芝'
            WHEN '13' THEN '芝'
            WHEN '14' THEN '芝'
            WHEN '15' THEN '芝'
            WHEN '16' THEN '芝'
            WHEN '17' THEN '芝'
            WHEN '18' THEN '芝'
            WHEN '19' THEN '芝'
            WHEN '20' THEN '芝'
            WHEN '21' THEN '芝'
            WHEN '22' THEN '芝'
            WHEN '23' THEN 'ダート'
            WHEN '24' THEN 'ダート'
            WHEN '25' THEN 'ダート'
            WHEN '26' THEN 'ダート'
            WHEN '27' THEN 'ダート'
            WHEN '28' THEN 'ダート'
            WHEN '29' THEN 'ダート'
            ELSE '不明'
        END AS surface_type,
        CASE
            WHEN CAST(r.kyori AS INTEGER) BETWEEN 1000 AND 1400 THEN '短距離(1000-1400m)'
            WHEN CAST(r.kyori AS INTEGER) BETWEEN 1401 AND 1800 THEN 'マイル(1401-1800m)'
            WHEN CAST(r.kyori AS INTEGER) BETWEEN 1801 AND 2200 THEN '中距離(1801-2200m)'
            WHEN CAST(r.kyori AS INTEGER) >= 2201 THEN '長距離(2201m~)'
            ELSE '不明'
        END AS distance_category,
        CASE
            WHEN CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
                AND CAST(n.kakutei_chakujun AS INTEGER) BETWEEN 1 AND 3
            THEN 1
            ELSE 0
        END AS is_upset
    FROM jvd_se n
    INNER JOIN jvd_ra r ON n.kaisai_nen = r.kaisai_nen
        AND n.kaisai_tsukihi = r.kaisai_tsukihi
        AND n.keibajo_code = r.keibajo_code
        AND n.race_bango = r.race_bango
    WHERE CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
        AND CAST(n.kakutei_chakujun AS INTEGER) != 0
        AND CAST(r.kyori AS INTEGER) >= 1000
) AS detailed_data
GROUP BY detailed_data.keibajo_code, detailed_data.surface_type, detailed_data.distance_category
ORDER BY track_name, surface_type, distance_category;

-- ================================================================================
-- 6. 緩和案：地域別の穴馬データ量
-- ================================================================================
SELECT
    region_data.region,
    COUNT(*) AS total_records,
    SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) AS upset_count,
    ROUND(SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS upset_rate_pct,
    CASE
        WHEN COUNT(*) >= 1000 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 100 THEN '✅ 十分'
        WHEN COUNT(*) >= 500 AND SUM(CASE WHEN is_upset = 1 THEN 1 ELSE 0 END) >= 50 THEN '⚠️ やや不足'
        ELSE '❌ 不足'
    END AS data_sufficiency
FROM (
    SELECT
        r.keibajo_code,
        CASE
            WHEN r.keibajo_code IN ('01', '02') THEN '北海道(札幌+函館)'
            WHEN r.keibajo_code IN ('03', '04') THEN '東北・北関東(福島+新潟)'
            WHEN r.keibajo_code IN ('05', '06') THEN '関東(東京+中山)'
            WHEN r.keibajo_code IN ('07', '08', '09') THEN '関西(中京+京都+阪神)'
            WHEN r.keibajo_code = '10' THEN '九州(小倉)'
            ELSE '不明'
        END AS region,
        CASE
            WHEN CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
                AND CAST(n.kakutei_chakujun AS INTEGER) BETWEEN 1 AND 3
            THEN 1
            ELSE 0
        END AS is_upset
    FROM jvd_se n
    INNER JOIN jvd_ra r ON n.kaisai_nen = r.kaisai_nen
        AND n.kaisai_tsukihi = r.kaisai_tsukihi
        AND n.keibajo_code = r.keibajo_code
        AND n.race_bango = r.race_bango
    WHERE CAST(n.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
        AND CAST(n.kakutei_chakujun AS INTEGER) != 0
        AND CAST(r.kyori AS INTEGER) >= 1000
) AS region_data
GROUP BY region_data.region
ORDER BY region;
