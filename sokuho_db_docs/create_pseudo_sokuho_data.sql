-- ========================================
-- 疑似速報データ生成SQL
-- ========================================
-- 過去の確定データ（jvd_ra, jvd_se）から速報テーブル形式（apd_sokuho_jvd_ra, apd_sokuho_jvd_se）へ変換
-- レース結果情報をマスクして、実運用と同等の速報予測環境を再現する
--
-- Usage:
--   psql -h localhost -U postgres -d keiba -v target_year_start=2020 -v target_year_end=2023 -f create_pseudo_sokuho_data.sql
--
-- Parameters:
--   target_year_start: 変換対象の開始年（例: 2020）
--   target_year_end:   変換対象の終了年（例: 2023）
-- ========================================

\echo '========================================';
\echo '疑似速報データ生成開始';
\echo '========================================';
\echo '対象年: ' :target_year_start ' - ' :target_year_end;
\echo '';

-- ========================================
-- トランザクション開始
-- ========================================
BEGIN;

\echo '既存データをクリア中...';

-- 既存の疑似速報データを削除（指定年度のみ）
DELETE FROM apd_sokuho_jvd_ra 
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

DELETE FROM apd_sokuho_jvd_se 
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  ✓ 既存データクリア完了';
\echo '';

-- ========================================
-- 1. レース情報（apd_sokuho_jvd_ra）の生成
-- ========================================
\echo 'レース情報（apd_sokuho_jvd_ra）を生成中...';

INSERT INTO apd_sokuho_jvd_ra (
    record_id,
    data_kubun,
    data_sakusei_nengappi,
    kaisai_nen,
    kaisai_tsukihi,
    keibajo_code,
    kaisai_kai,
    kaisai_nichime,
    race_bango,
    yobi_code,
    tokubetsu_kyoso_bango,
    kyosomei_hondai,
    kyosomei_fukudai,
    kyosomei_kakkonai,
    kyosomei_hondai_eur,
    kyosomei_fukudai_eur,
    kyosomei_kakkonai_eur,
    kyosomei_ryakusho_10,
    kyosomei_ryakusho_6,
    kyosomei_ryakusho_3,
    kyosomei_kubun,
    jusho_kaiji,
    grade_code,
    grade_code_henkomae,
    kyoso_shubetsu_code,
    kyoso_kigo_code,
    juryo_shubetsu_code,
    kyoso_joken_code_2sai,
    kyoso_joken_code_3sai,
    kyoso_joken_code_4sai,
    kyoso_joken_code_5sai_ijo,
    kyoso_joken_code,
    kyoso_joken_meisho,
    kyori,
    kyori_henkomae,
    track_code,
    track_code_henkomae,
    course_kubun,
    course_kubun_henkomae,
    honshokin,
    honshokin_henkomae,
    fukashokin,
    fukashokin_henkomae,
    hasso_jikoku,
    hasso_jikoku_henkomae,
    toroku_tosu,
    shusso_tosu,
    nyusen_tosu,
    tenko_code,
    babajotai_code_shiba,
    babajotai_code_dirt,
    lap_time,
    shogai_mile_time,
    zenhan_3f,
    zenhan_4f,
    kohan_3f,
    kohan_4f,
    corner_tsuka_juni_1,
    corner_tsuka_juni_2,
    corner_tsuka_juni_3,
    corner_tsuka_juni_4,
    record_koshin_kubun
)
SELECT 
    record_id,
    data_kubun,
    to_char(current_date, 'YYYYMMDD') as data_sakusei_nengappi, -- 現在日付を設定
    kaisai_nen,
    kaisai_tsukihi,
    keibajo_code,
    kaisai_kai,
    kaisai_nichime,
    race_bango,
    yobi_code,
    tokubetsu_kyoso_bango,
    kyosomei_hondai,
    kyosomei_fukudai,
    kyosomei_kakkonai,
    kyosomei_hondai_eur,
    kyosomei_fukudai_eur,
    kyosomei_kakkonai_eur,
    kyosomei_ryakusho_10,
    kyosomei_ryakusho_6,
    kyosomei_ryakusho_3,
    kyosomei_kubun,
    jusho_kaiji,
    grade_code,
    grade_code_henkomae,
    kyoso_shubetsu_code,
    kyoso_kigo_code,
    juryo_shubetsu_code,
    kyoso_joken_code_2sai,
    kyoso_joken_code_3sai,
    kyoso_joken_code_4sai,
    kyoso_joken_code_5sai_ijo,
    kyoso_joken_code,
    kyoso_joken_meisho,
    kyori,
    kyori_henkomae,
    track_code,
    track_code_henkomae,
    course_kubun,
    course_kubun_henkomae,
    honshokin,
    honshokin_henkomae,
    fukashokin,
    fukashokin_henkomae,
    hasso_jikoku,
    hasso_jikoku_henkomae,
    toroku_tosu,
    shusso_tosu,
    NULL as nyusen_tosu,              -- 🚫 結果情報：マスク
    tenko_code,
    babajotai_code_shiba,
    babajotai_code_dirt,
    NULL as lap_time,                 -- 🚫 結果情報：マスク
    NULL as shogai_mile_time,         -- 🚫 結果情報：マスク
    NULL as zenhan_3f,                -- 🚫 結果情報：マスク
    NULL as zenhan_4f,                -- 🚫 結果情報：マスク
    NULL as kohan_3f,                 -- 🚫 結果情報：マスク
    NULL as kohan_4f,                 -- 🚫 結果情報：マスク
    NULL as corner_tsuka_juni_1,      -- 🚫 結果情報：マスク
    NULL as corner_tsuka_juni_2,      -- 🚫 結果情報：マスク
    NULL as corner_tsuka_juni_3,      -- 🚫 結果情報：マスク
    NULL as corner_tsuka_juni_4,      -- 🚫 結果情報：マスク
    NULL as record_koshin_kubun       -- 🚫 結果情報：マスク
FROM jvd_ra
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  ✓ レース情報生成完了: ' :'ROW_COUNT' '件';
\echo '';

-- ========================================
-- 2. 馬毎レース情報（apd_sokuho_jvd_se）の生成
-- ========================================
\echo '馬毎レース情報（apd_sokuho_jvd_se）を生成中...';

INSERT INTO apd_sokuho_jvd_se (
    record_id,
    data_kubun,
    data_sakusei_nengappi,
    kaisai_nen,
    kaisai_tsukihi,
    keibajo_code,
    kaisai_kai,
    kaisai_nichime,
    race_bango,
    wakuban,
    umaban,
    ketto_toroku_bango,
    bamei,
    umakigo_code,
    seibetsu_code,
    hinshu_code,
    moshoku_code,
    barei,
    tozai_shozoku_code,
    chokyoshi_code,
    chokyoshimei_ryakusho,
    banushi_code,
    banushimei,
    fukushoku_hyoji,
    yobi_1,
    futan_juryo,
    futan_juryo_henkomae,
    blinker_shiyo_kubun,
    yobi_2,
    kishu_code,
    kishu_code_henkomae,
    kishumei_ryakusho,
    kishumei_ryakusho_henkomae,
    kishu_minarai_code,
    kishu_minarai_code_henkomae,
    bataiju,
    zogen_fugo,
    zogen_sa,
    ijo_kubun_code,
    nyusen_juni,
    kakutei_chakujun,
    dochaku_kubun,
    dochaku_tosu,
    soha_time,
    chakusa_code_1,
    chakusa_code_2,
    chakusa_code_3,
    corner_1,
    corner_2,
    corner_3,
    corner_4,
    tansho_odds,
    tansho_ninkijun,
    kakutoku_honshokin,
    kakutoku_fukashokin,
    yobi_3,
    yobi_4,
    kohan_4f,
    kohan_3f,
    aiteuma_joho_1,
    aiteuma_joho_2,
    aiteuma_joho_3,
    time_sa,
    record_koshin_kubun,
    mining_kubun,
    yoso_soha_time,
    yoso_gosa_plus,
    yoso_gosa_minus,
    yoso_juni,
    kyakushitsu_hantei
)
SELECT 
    record_id,
    data_kubun,
    to_char(current_date, 'YYYYMMDD') as data_sakusei_nengappi, -- 現在日付を設定
    kaisai_nen,
    kaisai_tsukihi,
    keibajo_code,
    kaisai_kai,
    kaisai_nichime,
    race_bango,
    wakuban,
    umaban,
    ketto_toroku_bango,
    bamei,
    umakigo_code,
    seibetsu_code,
    hinshu_code,
    moshoku_code,
    barei,
    tozai_shozoku_code,
    chokyoshi_code,
    chokyoshimei_ryakusho,
    banushi_code,
    banushimei,
    fukushoku_hyoji,
    yobi_1,
    futan_juryo,
    futan_juryo_henkomae,
    blinker_shiyo_kubun,
    yobi_2,
    kishu_code,
    kishu_code_henkomae,
    kishumei_ryakusho,
    kishumei_ryakusho_henkomae,
    kishu_minarai_code,
    kishu_minarai_code_henkomae,
    bataiju,
    zogen_fugo,
    zogen_sa,
    ijo_kubun_code,
    NULL as nyusen_juni,              -- 🚫 結果情報：マスク
    NULL as kakutei_chakujun,         -- 🚫 結果情報：マスク
    NULL as dochaku_kubun,            -- 🚫 結果情報：マスク
    NULL as dochaku_tosu,             -- 🚫 結果情報：マスク
    NULL as soha_time,                -- 🚫 結果情報：マスク
    NULL as chakusa_code_1,           -- 🚫 結果情報：マスク
    NULL as chakusa_code_2,           -- 🚫 結果情報：マスク
    NULL as chakusa_code_3,           -- 🚫 結果情報：マスク
    NULL as corner_1,                 -- 🚫 結果情報：マスク
    NULL as corner_2,                 -- 🚫 結果情報：マスク
    NULL as corner_3,                 -- 🚫 結果情報：マスク
    NULL as corner_4,                 -- 🚫 結果情報：マスク
    tansho_odds,
    tansho_ninkijun,
    NULL as kakutoku_honshokin,       -- 🚫 結果情報：マスク
    NULL as kakutoku_fukashokin,      -- 🚫 結果情報：マスク
    yobi_3,
    yobi_4,
    NULL as kohan_4f,                 -- 🚫 結果情報：マスク
    NULL as kohan_3f,                 -- 🚫 結果情報：マスク
    NULL as aiteuma_joho_1,           -- 🚫 結果情報：マスク
    NULL as aiteuma_joho_2,           -- 🚫 結果情報：マスク
    NULL as aiteuma_joho_3,           -- 🚫 結果情報：マスク
    NULL as time_sa,                  -- 🚫 結果情報：マスク
    NULL as record_koshin_kubun,      -- 🚫 結果情報：マスク
    mining_kubun,
    yoso_soha_time,
    yoso_gosa_plus,
    yoso_gosa_minus,
    yoso_juni,
    kyakushitsu_hantei
FROM jvd_se
WHERE cast(kaisai_nen as integer) BETWEEN :target_year_start AND :target_year_end;

\echo '  ✓ 馬毎レース情報生成完了: ' :'ROW_COUNT' '件';
\echo '';

-- ========================================
-- コミット
-- ========================================
COMMIT;

\echo '========================================';
\echo '疑似速報データ生成完了！';
\echo '========================================';
\echo '';
\echo '次のステップ:';
\echo '  1. validate_pseudo_sokuho.sql で検証を実行';
\echo '  2. build_sokuho_race_data_query() を使って予測実行';
\echo '';
