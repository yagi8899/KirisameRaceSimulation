-- 穴馬（7-12番人気）の3着以内確率分析（過去10年）
-- 2026年1月19日作成
-- 2026年1月20日更新：7-12番人気に特化、騎手・調教師・前走・馬場状態・クラス別分析追加

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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014  -- 過去10年程度
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
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 4 AND 6 THEN '4-6番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12 THEN '7-12番人気（穴馬対象）'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) >= 13 THEN '13番人気以下（大穴）'
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND se.tansho_ninkijun <> '00'
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 1 AND 18
GROUP BY 
    CASE 
        WHEN CAST(se.tansho_ninkijun AS INTEGER) = 1 THEN '1番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 2 AND 3 THEN '2-3番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 4 AND 6 THEN '4-6番人気'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12 THEN '7-12番人気（穴馬対象）'
        WHEN CAST(se.tansho_ninkijun AS INTEGER) >= 13 THEN '13番人気以下（大穴）'
    END
ORDER BY 
    MIN(CAST(se.tansho_ninkijun AS INTEGER));

-- =====================================================
-- 3. 年度別 穴馬（7-12番人気）の3着以内率推移
-- =====================================================
SELECT 
    ra.kaisai_nen AS 年度,
    COUNT(*) AS 穴馬出走回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
GROUP BY ra.kaisai_nen
ORDER BY ra.kaisai_nen;

-- =====================================================
-- 4. 競馬場別 穴馬（7-12番人気）の3着以内率
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
    COUNT(*) AS 穴馬出走回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
GROUP BY ra.keibajo_code
ORDER BY 三着以内率_パーセント DESC;

-- =====================================================
-- 5. 路面×距離別 穴馬（7-12番人気）の3着以内率
-- =====================================================
SELECT 
    CASE 
        WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22') THEN '芝'
        WHEN ra.track_code IN ('23','24','25','26','29') THEN 'ダート'
        ELSE '不明'
    END AS 路面種別,
    CASE 
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1000 AND 1400 THEN '短距離(1000-1400m)'
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1401 AND 1800 THEN 'マイル(1401-1800m)'
        WHEN CAST(ra.kyori AS INTEGER) BETWEEN 1801 AND 2400 THEN '中距離(1801-2400m)'
        WHEN CAST(ra.kyori AS INTEGER) >= 2401 THEN '長距離(2401m~)'
        ELSE '不明'
    END AS 距離区分,
    COUNT(*) AS 穴馬出走回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
    AND ra.kyori IS NOT NULL
    AND ra.track_code IS NOT NULL
GROUP BY 
    CASE 
        WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22') THEN '芝'
        WHEN ra.track_code IN ('23','24','25','26','29') THEN 'ダート'
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
-- 6. 馬場状態別 穴馬（7-12番人気）の3着以内率
-- =====================================================
SELECT 
    CASE 
        WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22') THEN '芝'
        WHEN ra.track_code IN ('23','24','25','26','29') THEN 'ダート'
        ELSE '不明'
    END AS 路面種別,
    CASE 
        WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22') THEN
            CASE ra.babajotai_code_shiba
                WHEN '1' THEN '良'
                WHEN '2' THEN '稍重'
                WHEN '3' THEN '重'
                WHEN '4' THEN '不良'
                ELSE '不明'
            END
        ELSE
            CASE ra.babajotai_code_dirt
                WHEN '1' THEN '良'
                WHEN '2' THEN '稍重'
                WHEN '3' THEN '重'
                WHEN '4' THEN '不良'
                ELSE '不明'
            END
    END AS 馬場状態,
    COUNT(*) AS 穴馬出走回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
    AND ra.track_code IS NOT NULL
GROUP BY 
    CASE 
        WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22') THEN '芝'
        WHEN ra.track_code IN ('23','24','25','26','29') THEN 'ダート'
        ELSE '不明'
    END,
    CASE 
        WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22') THEN
            CASE ra.babajotai_code_shiba
                WHEN '1' THEN '良'
                WHEN '2' THEN '稍重'
                WHEN '3' THEN '重'
                WHEN '4' THEN '不良'
                ELSE '不明'
            END
        ELSE
            CASE ra.babajotai_code_dirt
                WHEN '1' THEN '良'
                WHEN '2' THEN '稍重'
                WHEN '3' THEN '重'
                WHEN '4' THEN '不良'
                ELSE '不明'
            END
    END
ORDER BY 路面種別, 馬場状態;

-- =====================================================
-- 7. クラス別 穴馬（7-12番人気）の3着以内率
-- =====================================================
SELECT 
    CASE 
        WHEN ra.grade_code = 'A' THEN 'G1'
        WHEN ra.grade_code = 'B' THEN 'G2'
        WHEN ra.grade_code = 'C' THEN 'G3'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '999' THEN 'OP（オープン）'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '016' THEN '3勝クラス'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '010' THEN '2勝クラス'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '005' THEN '1勝クラス'
        ELSE '未勝利・新馬'
    END AS クラス,
    COUNT(*) AS 穴馬出走回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
GROUP BY 
    CASE 
        WHEN ra.grade_code = 'A' THEN 'G1'
        WHEN ra.grade_code = 'B' THEN 'G2'
        WHEN ra.grade_code = 'C' THEN 'G3'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '999' THEN 'OP（オープン）'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '016' THEN '3勝クラス'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '010' THEN '2勝クラス'
        WHEN ra.grade_code NOT IN ('A','B','C') AND ra.kyoso_joken_code = '005' THEN '1勝クラス'
        ELSE '未勝利・新馬'
    END
ORDER BY 三着以内率_パーセント DESC;

-- =====================================================
-- 8. 馬齢・性別 穴馬（7-12番人気）の3着以内率
-- =====================================================
SELECT 
    se.barei AS 馬齢,
    CASE se.sei_code
        WHEN '1' THEN '牡'
        WHEN '2' THEN '牝'
        WHEN '3' THEN 'セン'
        ELSE '不明'
    END AS 性別,
    COUNT(*) AS 穴馬出走回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
    AND se.barei IS NOT NULL
    AND se.sei_code IS NOT NULL
GROUP BY se.barei, se.sei_code
ORDER BY se.barei, se.sei_code;

-- =====================================================
-- 9. 騎手別 穴馬（7-12番人気）の3着以内率 TOP30
-- =====================================================
SELECT 
    se.kishu_code AS 騎手コード,
    se.kishu_mei AS 騎手名,
    COUNT(*) AS 穴馬騎乗回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
    AND se.kishu_code IS NOT NULL
GROUP BY se.kishu_code, se.kishu_mei
HAVING COUNT(*) >= 50  -- 50回以上騎乗した騎手のみ
ORDER BY 三着以内率_パーセント DESC
LIMIT 30;

-- =====================================================
-- 10. 調教師別 穴馬（7-12番人気）の3着以内率 TOP30
-- =====================================================
SELECT 
    se.chokyoshi_code AS 調教師コード,
    se.chokyoshi_mei AS 調教師名,
    COUNT(*) AS 穴馬管理回数,
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
    CAST(ra.kaisai_nen AS INTEGER) >= 2014
    AND se.kakutei_chakujun IS NOT NULL
    AND se.kakutei_chakujun <> '00'
    AND se.tansho_ninkijun IS NOT NULL
    AND CAST(se.tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
    AND se.chokyoshi_code IS NOT NULL
GROUP BY se.chokyoshi_code, se.chokyoshi_mei
HAVING COUNT(*) >= 30  -- 30回以上管理した調教師のみ
ORDER BY 三着以内率_パーセント DESC
LIMIT 30;

-- =====================================================
-- 11. 前走着順別 穴馬（7-12番人気）の巻き返し率
-- =====================================================
WITH zenso_data AS (
    SELECT 
        se.kaisai_nen,
        se.kaisai_tsukihi,
        se.keibajo_code,
        se.race_bango,
        se.umaban,
        se.kakutei_chakujun,
        se.tansho_ninkijun,
        -- 前走着順を取得（直近の過去レース）
        LAG(se.kakutei_chakujun) OVER (
            PARTITION BY se.ketto_toroku_bango 
            ORDER BY se.kaisai_nen, se.kaisai_tsukihi, se.keibajo_code, se.race_bango
        ) AS zenso_chakujun
    FROM jvd_se se
    INNER JOIN jvd_ra ra ON 
        se.kaisai_nen = ra.kaisai_nen 
        AND se.kaisai_tsukihi = ra.kaisai_tsukihi
        AND se.keibajo_code = ra.keibajo_code
        AND se.race_bango = ra.race_bango
    WHERE 
        CAST(ra.kaisai_nen AS INTEGER) >= 2014
        AND se.kakutei_chakujun IS NOT NULL
        AND se.kakutei_chakujun <> '00'
)
SELECT 
    CASE 
        WHEN CAST(zenso_chakujun AS INTEGER) <= 3 THEN '前走3着以内'
        WHEN CAST(zenso_chakujun AS INTEGER) BETWEEN 4 AND 6 THEN '前走4-6着'
        WHEN CAST(zenso_chakujun AS INTEGER) BETWEEN 7 AND 9 THEN '前走7-9着'
        WHEN CAST(zenso_chakujun AS INTEGER) >= 10 THEN '前走10着以下'
        ELSE '前走なし'
    END AS 前走着順区分,
    COUNT(*) AS 穴馬出走回数,
    SUM(CASE WHEN CAST(kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS 三着以内回数,
    ROUND(
        100.0 * SUM(CASE WHEN CAST(kakutei_chakujun AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 
        2
    ) AS 三着以内率_パーセント
FROM zenso_data
WHERE 
    tansho_ninkijun IS NOT NULL
    AND CAST(tansho_ninkijun AS INTEGER) BETWEEN 7 AND 12
GROUP BY 
    CASE 
        WHEN CAST(zenso_chakujun AS INTEGER) <= 3 THEN '前走3着以内'
        WHEN CAST(zenso_chakujun AS INTEGER) BETWEEN 4 AND 6 THEN '前走4-6着'
        WHEN CAST(zenso_chakujun AS INTEGER) BETWEEN 7 AND 9 THEN '前走7-9着'
        WHEN CAST(zenso_chakujun AS INTEGER) >= 10 THEN '前走10着以下'
        ELSE '前走なし'
    END
ORDER BY 三着以内率_パーセント DESC;

-- =====================================================
-- 使い方メモ
-- =====================================================
-- 1. 人気順位別詳細: 1番人気～18番人気の個別成績
-- 2. カテゴリ別サマリー: 7-12番人気を明確に区分
-- 3. 年度別推移: 穴馬の3着以内率が年々変わっているか確認
-- 4. 競馬場別: どの競馬場が穴が出やすいか
-- 5. 路面×距離別: 芝短距離、ダート長距離などでの穴馬傾向
-- 6. 馬場状態別: 良・稍重・重・不良での穴馬率
-- 7. クラス別: 新馬・未勝利・1勝～3勝・オープンでの穴馬率
-- 8. 馬齢・性別: 若馬・古馬、牡・牝・センでの穴馬率
-- 9. 騎手別TOP30: 穴馬を好走させる騎手ランキング
-- 10. 調教師別TOP30: 穴馬を仕上げる調教師ランキング
-- 11. 前走着順別: 前走大敗から巻き返す確率