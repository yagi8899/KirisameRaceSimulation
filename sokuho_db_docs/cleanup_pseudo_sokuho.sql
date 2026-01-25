-- ========================================
-- 疑似速報データクリーンアップSQL
-- ========================================
-- apd_sokuho_jvd_ra, apd_sokuho_jvd_se テーブルの疑似データを削除
--
-- Usage:
--   psql -h localhost -U postgres -d keiba -v target_year_start=2020 -v target_year_end=2023 -f cleanup_pseudo_sokuho.sql
--
-- Parameters:
--   target_year_start: 削除対象の開始年（例: 2020）
--   target_year_end:   削除対象の終了年（例: 2023）
-- ========================================

-- 文字コード設定（Windows環境対応）
SET client_encoding TO 'UTF8';

\echo '========================================'
\echo 'Pseudo Sokuho Data Cleanup'
\echo '========================================'
\echo 'Target Year: ' :target_year_start ' - ' :target_year_end
\echo ''
\echo 'WARNING: This operation cannot be undone!'
\echo ''

BEGIN;

-- 削除前の件数確認
\echo 'Record count before deletion:'

SELECT 
    'apd_sokuho_jvd_ra' as テーブル,
    COUNT(*) as 件数
FROM apd_sokuho_jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
UNION ALL
SELECT 
    'apd_sokuho_jvd_se' as テーブル,
    COUNT(*) as 件数
FROM apd_sokuho_jvd_se
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo ''
\echo 'Deleting data...'

-- 馬毎レース情報を削除
DELETE FROM apd_sokuho_jvd_se 
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  Done: apd_sokuho_jvd_se deleted'

-- レース情報を削除
DELETE FROM apd_sokuho_jvd_ra 
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  Done: apd_sokuho_jvd_ra deleted'
\echo ''

-- 削除後の件数確認
\echo 'Record count after deletion:'

SELECT 
    'apd_sokuho_jvd_ra' as テーブル,
    COUNT(*) as 残存件数
FROM apd_sokuho_jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end
UNION ALL
SELECT 
    'apd_sokuho_jvd_se' as テーブル,
    COUNT(*) as 残存件数
FROM apd_sokuho_jvd_se
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

COMMIT;

\echo ''
\echo '========================================'
\echo 'Cleanup Complete!'
\echo '========================================'
\echo ''
