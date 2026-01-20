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

\echo '========================================';
\echo '疑似速報データクリーンアップ';
\echo '========================================';
\echo '対象年: ' :target_year_start ' - ' :target_year_end;
\echo '';
\echo '⚠️  この操作は元に戻せません！';
\echo '';

BEGIN;

-- 削除前の件数確認
\echo '削除前のデータ件数:';

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

\echo '';
\echo 'データを削除中...';

-- 馬毎レース情報を削除
DELETE FROM apd_sokuho_jvd_se 
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  ✓ apd_sokuho_jvd_se 削除完了: ' :'ROW_COUNT' '件';

-- レース情報を削除
DELETE FROM apd_sokuho_jvd_ra 
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  ✓ apd_sokuho_jvd_ra 削除完了: ' :'ROW_COUNT' '件';
\echo '';

-- 削除後の件数確認
\echo '削除後のデータ件数:';

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

\echo '';
\echo '========================================';
\echo 'クリーンアップ完了！';
\echo '========================================';
\echo '';
