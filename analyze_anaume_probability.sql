-- 穴馬の3着以内確率分析（過去10年）
-- 2026年1月19日作成

-- =====================================================
-- 1. 人気順位別の3着以内率（詳細版）
-- =====================================================
SELECT 
    CAST(se.tansho_ninkijun AS INTEGER) AS 人気順位,
    COUNT(*) AS 出走回数,
    SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS 三着以内回数,
    ROUND(
        100.0 * SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 三着以内率_パーセント,
    SUM(CASE WHEN se.kakutei_chakujun = '01' THEN 1 ELSE 0 END) AS 一着回数,
    ROUND(
        100.0 * SUM(CASE WHEN se.kakutei_chakujun = '01' THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 一着率_パーセント
FROM jvd_se se
INNER JOIN jvd_ra ra ON 
    se.kaisai_nen = ra.kaisai_nen 
    AND se.kaisai_tsukihi = ra.kaisai_tsukihi
    AND se.keibajo_code = ra.keibajo_code
    AND se.race_bango = ra.race_bango
WHERE 
    CAST(ra.kaisai_nen AS INTEGER) >= 2016  -- 過去10年程度
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND se.tansho_ninkijun <> '00'
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 1 AND 18  -- 異常値除外
GROUP BY CAST(se.tansho_ninkijun AS INTEGER)
ORDER BY CAST(se.tansho_ninkijun AS INTEGER);

-- =====================================================
-- 2. 穴馬カテゴリ別の3着以内率（サマリー版）
-- =====================================================
SELECT 
    CASE 
        WHEN CAST(se.tansho_ninkijun AS INTEGER) = 1 THEN '1番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 2 AND 3 THEN '2-3番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 4 AND 5 THEN '4-5番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 6 AND 9 THEN '6-9番人気（中穴）'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) >= 10 THEN '10番人気以下（大穴）'
    END AS 人気区分,
    COUNT(*) AS 出走回数,
    SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS 三着以内回数,
    ROUND(
        100.0 * SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 三着以内率_パーセント,
    SUM(CASE WHEN se.kakutei_chakujun = '01' THEN 1 ELSE 0 END) AS 一着回数,
    ROUND(
        100.0 * SUM(CASE WHEN se.kakutei_chakujun = '01' THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 一着率_パーセント
FROM jvd_se se
INNER JOIN jvd_ra ra ON 
    se.kaisai_nen = ra.kaisai_nen 
    AND se.kaisai_tsukihi = ra.kaisai_tsukihi
    AND se.keibajo_code = ra.keibajo_code
    AND se.race_bango = ra.race_bango
WHERE 
    CAST(ra.kaisai_nen AS INTEGER) >= 2016
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND se.tansho_ninkijun <> '00'
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 1 AND 18
GROUP BY 
    CASE 
        WHEN CAST(se.tansho_ninkijun AS INTEGER) = 1 THEN '1番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 2 AND 3 THEN '2-3番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 4 AND 5 THEN '4-5番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 6 AND 9 THEN '6-9番人気（中穴）'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) >= 10 THEN '10番人気以下（大穴）'
    END
ORDER BY 
    MIN(CAST(se.tansho_ninkijun AS INTEGER));

-- =====================================================
-- 3. 年度別 穴馬（10番人気以下）の3着以内率推移
-- =====================================================
SELECT 
    ra.kaisai_nen AS 年度,
    COUNT(*) AS 十番人気以下出走回数,
    SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS 三着以内回数,
    ROUND(
        100.0 * SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 三着以内率_パーセント
FROM jvd_se se
INNER JOIN jvd_ra ra ON 
    se.kaisai_nen = ra.kaisai_nen 
    AND se.kaisai_tsukihi = ra.kaisai_tsukihi
    AND se.keibajo_code = ra.keibajo_code
    AND se.race_bango = ra.race_bango
WHERE 
    CAST(ra.kaisai_nen AS INTEGER) >= 2016
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) >= 10
GROUP BY ra.kaisai_nen
ORDER BY ra.kaisai_nen;

-- =====================================================
-- 4. 競馬場別 穴馬（10番人気以下）の3着以内率
-- =====================================================
SELECT 
    ra.keibajo_code AS 競馬場コード,
    CASE ra.keibajo_code
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
    END AS 競馬場名,
    COUNT(*) AS 十番人気以下出走回数,
    SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS 三着以内回数,
    ROUND(
        100.0 * SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 三着以内率_パーセント
FROM jvd_se se
INNER JOIN jvd_ra ra ON 
    se.kaisai_nen = ra.kaisai_nen 
    AND se.kaisai_tsukihi = ra.kaisai_tsukihi
    AND se.keibajo_code = ra.keibajo_code
    AND se.race_bango = ra.race_bango
WHERE 
    CAST(ra.kaisai_nen AS INTEGER) >= 2016
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) >= 10
GROUP BY ra.keibajo_code
ORDER BY 三着以内率_パーセント DESC;

-- =====================================================
-- 5. 路面×距離別 穴馬（10番人気以下）の3着以内率
-- =====================================================
SELECT 
    CASE ra.track_code
        WHEN '10' THEN '芝直線'
        WHEN '11' THEN '芝左'
        WHEN '12' THEN '芝左'
        WHEN '13' THEN '芝左'
        WHEN '14' THEN '芝左'
        WHEN '15' THEN '芝左'
        WHEN '16' THEN '芝左'
        WHEN '17' THEN '芝右'
        WHEN '18' THEN '芝右'
        WHEN '19' THEN '芝右'
        WHEN '20' THEN '芝右'
        WHEN '21' THEN '芝右'
        WHEN '22' THEN '芝右'
        WHEN '23' THEN 'ダート左'
        WHEN '24' THEN 'ダート右'
        WHEN '25' THEN 'ダート左'
        WHEN '26' THEN 'ダート左'
        ELSE '不明'
    END AS 路面種別,
    CASE 
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1000 AND 1400 THEN '短距離(1000-1400m)'
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1401 AND 1800 THEN 'マイル(1401-1800m)'
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1801 AND 2400 THEN '中距離(1801-2400m)'
        WHEN CAST(ra.kyori AS INTEGER) >= 2401 THEN '長距離(2401m~)'
        ELSE '不明'
    END AS 距離区分,
    COUNT(*) AS 十番人気以下出走回数,
    SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS 三着以内回数,
    ROUND(
        100.0 * SUM(CASE WHEN CAST(se.kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 三着以内率_パーセント
FROM jvd_se se
INNER JOIN jvd_ra ra ON 
    se.kaisai_nen = ra.kaisai_nen 
    AND se.kaisai_tsukihi = ra.kaisai_tsukihi
    AND se.keibajo_code = ra.keibajo_code
    AND se.race_bango = ra.race_bango
WHERE 
    CAST(ra.kaisai_nen AS INTEGER) >= 2016
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) >= 10
    AND ra.kyori IS NOT NULL
    AND ra.track_code IS NOT NULL
GROUP BY 
    CASE ra.track_code
        WHEN '10' THEN '芝直線'
        WHEN '11' THEN '芝左'
        WHEN '12' THEN '芝左'
        WHEN '13' THEN '芝左'
        WHEN '14' THEN '芝左'
        WHEN '15' THEN '芝左'
        WHEN '16' THEN '芝左'
        WHEN '17' THEN '芝右'
        WHEN '18' THEN '芝右'
        WHEN '19' THEN '芝右'
        WHEN '20' THEN '芝右'
        WHEN '21' THEN '芝右'
        WHEN '22' THEN '芝右'
        WHEN '23' THEN 'ダート左'
        WHEN '24' THEN 'ダート右'
        WHEN '25' THEN 'ダート左'
        WHEN '26' THEN 'ダート左'
        ELSE '不明'
    END,
    CASE 
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1000 AND 1400 THEN '短距離(1000-1400m)'
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1401 AND 1800 THEN 'マイル(1401-1800m)'
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1801 AND 2400 THEN '中距離(1801-2400m)'
        WHEN CAST(ra.kyori AS INTEGER) >= 2401 THEN '長距離(2401m~)'
        ELSE '不明'
    END
ORDER BY 路面種別, 距離区分;

-- =====================================================
-- 使い方メモ
-- =====================================================
-- 1. 人気順位別詳細: 1番人気～18番人気の個別成績
-- 2. カテゴリ別サマリー: 5区分に分けた見やすい集計
-- 3. 年度別推移: 穴馬の3着以内率が年々変わっているか確認
-- 4. 競馬場別: どの競馬場が穴が出やすいか
-- 5. 路面×距離別: 芝短距離、ダート長距離などでの穴馬傾向
