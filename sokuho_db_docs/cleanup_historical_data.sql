-- ============================================
-- 過去成績データ削除スクリプト
-- 対象テーブル: jvd_ra（レース結果）, jvd_se（馬成績）
-- ============================================
-- 
-- 【警告】
-- このスクリプトは過去成績データを削除します。
-- 削除後は疑似速報データ（sokuho_ra, sokuho_se）のみが残ります。
-- 本番環境では絶対に実行しないでください！
--
-- 使い方:
--   psql -h localhost -U postgres -d keiba \
--     -v target_year_start=2023 \
--     -v target_year_end=2023 \
--     -f cleanup_historical_data.sql
-- ============================================

-- 文字コード設定（Windows環境対応）
SET client_encoding TO 'UTF8';

\echo ''
\echo '========================================'
\echo 'Historical Data Cleanup'
\echo 'Target Year: ' :target_year_start ' - ' :target_year_end
\echo '========================================'
\echo ''

-- 削除前の件数確認
\echo '[Before] Historical data count'
\echo ''

\echo 'jvd_ra (Race Results):'
SELECT 
    cast(kaisai_nen as integer) AS year,
    COUNT(*) AS count
FROM jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
GROUP BY kaisai_nen
ORDER BY year;

\echo ''
\echo 'jvd_se (Horse Results):'
SELECT 
    cast(kaisai_nen as integer) AS year,
    COUNT(*) AS count
FROM jvd_se
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
GROUP BY kaisai_nen
ORDER BY year;

\echo ''
\echo '----------------------------------------'
\echo 'Executing deletion...'
\echo '----------------------------------------'
\echo ''

-- トランザクション開始
BEGIN;

-- jvd_se（馬成績）の削除
\echo 'Deleting jvd_se...'
DELETE FROM jvd_se
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

-- jvd_ra（レース結果）の削除
\echo 'Deleting jvd_ra...'
DELETE FROM jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

COMMIT;

\echo ''
\echo '----------------------------------------'
\echo 'Deletion complete'
\echo '----------------------------------------'
\echo ''

-- 削除後の件数確認
\echo '[After] Historical data count (should be 0)'
\echo ''

\echo 'jvd_ra (Race Results):'
SELECT 
    cast(kaisai_nen as integer) AS year,
    COUNT(*) AS count
FROM jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
GROUP BY kaisai_nen
ORDER BY year;

\echo ''
\echo 'jvd_se (Horse Results):'
SELECT 
    cast(kaisai_nen as integer) AS year,
    COUNT(*) AS count
FROM jvd_se
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
GROUP BY kaisai_nen
ORDER BY year;

\echo ''
\echo '========================================'
\echo 'Historical Data Cleanup Complete!'
\echo ''
\echo '* Pseudo sokuho data (sokuho_ra, sokuho_se) remains'
\echo '========================================'
\echo ''
