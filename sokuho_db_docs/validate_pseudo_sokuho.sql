-- ========================================
-- 疑似速報データ検証SQL
-- ========================================
-- create_pseudo_sokuho_data.sqlで生成したデータの整合性を確認
--
-- Usage:
--   psql -h localhost -U postgres -d keiba -v target_year_start=2020 -v target_year_end=2023 -f validate_pseudo_sokuho.sql
--
-- Parameters:
--   target_year_start: 検証対象の開始年（例: 2020）
--   target_year_end:   検証対象の終了年（例: 2023）
-- ========================================

-- 文字コード設定（Windows環境対応）
SET client_encoding TO 'UTF8';

\echo '========================================'
\echo 'Pseudo Sokuho Data Validation'
\echo '========================================'
\echo 'Target Year: ' :target_year_start ' - ' :target_year_end
\echo ''

-- ========================================
-- 1. レース件数の一致確認
-- ========================================
\echo '[1/6] Race count check...'

WITH original AS (
    SELECT COUNT(*) as cnt
    FROM jvd_ra
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
),
sokuho AS (
    SELECT COUNT(*) as cnt
    FROM apd_sokuho_jvd_ra
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
)
SELECT 
    'jvd_ra' as テーブル,
    original.cnt as 元データ件数,
    sokuho.cnt as 速報データ件数,
    CASE 
        WHEN original.cnt = sokuho.cnt THEN '✓ OK'
        ELSE '✗ NG: 件数不一致'
    END as 判定
FROM original, sokuho;

\echo '';

-- ========================================
-- 2. 馬毎レース件数の一致確認
-- ========================================
\echo '[2/6] Horse race entry count check...'

WITH original AS (
    SELECT COUNT(*) as cnt
    FROM jvd_se
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
),
sokuho AS (
    SELECT COUNT(*) as cnt
    FROM apd_sokuho_jvd_se
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
)
SELECT 
    'jvd_se' as テーブル,
    original.cnt as 元データ件数,
    sokuho.cnt as 速報データ件数,
    CASE 
        WHEN original.cnt = sokuho.cnt THEN '✓ OK'
        ELSE '✗ NG: 件数不一致'
    END as 判定
FROM original, sokuho;

\echo '';

-- ========================================
-- 3. 結果情報マスク確認（apd_sokuho_jvd_ra）
-- ========================================
\echo '[3/6] Race result masking check...'

WITH mask_check AS (
    SELECT 
        COUNT(*) as total_records,
        COUNT(*) FILTER (WHERE nyusen_tosu IS NULL) as nyusen_tosu_null,
        COUNT(*) FILTER (WHERE lap_time IS NULL) as lap_time_null,
        COUNT(*) FILTER (WHERE kohan_3f IS NULL) as kohan_3f_null,
        COUNT(*) FILTER (WHERE corner_tsuka_juni_1 IS NULL) as corner_null
    FROM apd_sokuho_jvd_ra
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
)
SELECT 
    'apd_sokuho_jvd_ra' as テーブル,
    total_records as 総レコード数,
    CASE 
        WHEN nyusen_tosu_null = total_records THEN '✓' 
        ELSE '✗ (' || (total_records - nyusen_tosu_null) || '件残存)'
    END as nyusen_tosu,
    CASE 
        WHEN lap_time_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - lap_time_null) || '件残存)'
    END as lap_time,
    CASE 
        WHEN kohan_3f_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - kohan_3f_null) || '件残存)'
    END as kohan_3f,
    CASE 
        WHEN corner_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - corner_null) || '件残存)'
    END as corner_tsuka_juni
FROM mask_check;

\echo '';

-- ========================================
-- 4. 結果情報マスク確認（apd_sokuho_jvd_se）
-- ========================================
\echo '[4/6] Horse result masking check...'

WITH mask_check AS (
    SELECT 
        COUNT(*) as total_records,
        COUNT(*) FILTER (WHERE kakutei_chakujun IS NULL) as kakutei_chakujun_null,
        COUNT(*) FILTER (WHERE soha_time IS NULL) as soha_time_null,
        COUNT(*) FILTER (WHERE time_sa IS NULL) as time_sa_null,
        COUNT(*) FILTER (WHERE corner_1 IS NULL) as corner_1_null,
        COUNT(*) FILTER (WHERE kohan_3f IS NULL) as kohan_3f_null,
        COUNT(*) FILTER (WHERE kakutoku_honshokin IS NULL) as kakutoku_honshokin_null
    FROM apd_sokuho_jvd_se
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
)
SELECT 
    'apd_sokuho_jvd_se' as テーブル,
    total_records as 総レコード数,
    CASE 
        WHEN kakutei_chakujun_null = total_records THEN '✓' 
        ELSE '✗ (' || (total_records - kakutei_chakujun_null) || '件残存)'
    END as kakutei_chakujun,
    CASE 
        WHEN soha_time_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - soha_time_null) || '件残存)'
    END as soha_time,
    CASE 
        WHEN time_sa_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - time_sa_null) || '件残存)'
    END as time_sa,
    CASE 
        WHEN corner_1_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - corner_1_null) || '件残存)'
    END as corner_1_4,
    CASE 
        WHEN kohan_3f_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - kohan_3f_null) || '件残存)'
    END as kohan_3f,
    CASE 
        WHEN kakutoku_honshokin_null = total_records THEN '✓'
        ELSE '✗ (' || (total_records - kakutoku_honshokin_null) || '件残存)'
    END as kakutoku_honshokin
FROM mask_check;

\echo '';

-- ========================================
-- 5. 主キー重複チェック
-- ========================================
\echo '[5/6] Primary key duplicate check...'

-- apd_sokuho_jvd_ra
WITH dup_check AS (
    SELECT 
        kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango,
        COUNT(*) as cnt
    FROM apd_sokuho_jvd_ra
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
    GROUP BY kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango
    HAVING COUNT(*) > 1
)
SELECT 
    'apd_sokuho_jvd_ra' as テーブル,
    CASE 
        WHEN COUNT(*) = 0 THEN '✓ OK: 重複なし'
        ELSE '✗ NG: ' || COUNT(*) || '件の重複'
    END as 判定
FROM dup_check;

-- apd_sokuho_jvd_se
WITH dup_check AS (
    SELECT 
        kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango, umaban, ketto_toroku_bango,
        COUNT(*) as cnt
    FROM apd_sokuho_jvd_se
    WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
    GROUP BY kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango, umaban, ketto_toroku_bango
    HAVING COUNT(*) > 1
)
SELECT 
    'apd_sokuho_jvd_se' as テーブル,
    CASE 
        WHEN COUNT(*) = 0 THEN '✓ OK: 重複なし'
        ELSE '✗ NG: ' || COUNT(*) || '件の重複'
    END as 判定
FROM dup_check;

\echo '';

-- ========================================
-- 6. サンプルデータ出力（目視確認用）
-- ========================================
\echo '[6/6] Sample data output (latest race)...'
\echo '';
\echo '--- apd_sokuho_jvd_ra サンプル ---';

SELECT 
    kaisai_nen || kaisai_tsukihi as 開催年月日,
    keibajo_code as 競馬場,
    race_bango as R,
    kyosomei_ryakusho_10 as レース名,
    kyori as 距離,
    shusso_tosu as 出走頭数,
    tenko_code as 天候,
    babajotai_code_shiba as 馬場芝,
    CASE 
        WHEN nyusen_tosu IS NULL THEN '✓' 
        ELSE '✗ データ残存'
    END as 結果マスク
FROM apd_sokuho_jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
ORDER BY kaisai_nen DESC, kaisai_tsukihi DESC, keibajo_code, race_bango
LIMIT 1;

\echo '';
\echo '--- apd_sokuho_jvd_se サンプル ---';

SELECT 
    s.kaisai_nen || s.kaisai_tsukihi as 開催年月日,
    s.keibajo_code as 競馬場,
    s.race_bango as R,
    s.umaban as 馬番,
    s.bamei as 馬名,
    s.kishumei_ryakusho as 騎手,
    s.futan_juryo as 斤量,
    s.tansho_odds as オッズ,
    s.tansho_ninkijun as 人気,
    CASE 
        WHEN s.kakutei_chakujun IS NULL AND s.soha_time IS NULL AND s.corner_1 IS NULL THEN '✓' 
        ELSE '✗ データ残存'
    END as 結果マスク
FROM apd_sokuho_jvd_se s
WHERE cast(s.kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
ORDER BY s.kaisai_nen DESC, s.kaisai_tsukihi DESC, s.keibajo_code, s.race_bango, cast(s.umaban as integer)
LIMIT 5;

\echo '';
\echo '========================================';
\echo 'Pseudo Sokuho Data Validation Complete!'
\echo '========================================';
\echo '';
