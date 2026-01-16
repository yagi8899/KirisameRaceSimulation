# 速報用馬毎レース情報（jvd_seベース） (apd_sokuho_jvd_se)

## テーブル情報

| 項目                           | 値                                                                                                   |
|:-------------------------------|:-----------------------------------------------------------------------------------------------------|
| システム名                     | keiba                                                                                                |
| サブシステム名                 |                                                                                                      |
| スキーマ名                     | public                                                                                               |
| 物理テーブル名                 | apd_sokuho_jvd_se                                                                                    |
| 論理テーブル名                 | 速報用馬毎レース情報（jvd_seベース）                                                                 |
| 作成者                         | swamp                                                                                                |
| 作成日                         | 2026/01/16                                                                                           |
| RDBMS                          | PostgreSQL 17.0 on x86_64-windows, compiled by msvc-19.41.34120, 64-bit 17.0                         |



## カラム情報

| No. | 論理名                         | 物理名                         | データ型                       | Not Null | デフォルト           | 備考                           |
|----:|:-------------------------------|:-------------------------------|:-------------------------------|:---------|:---------------------|:-------------------------------|
|   1 | レコード種別ID                 | record_id                      | character varying(2)           |          |                      |                                |
|   2 | データ区分                     | data_kubun                     | character varying(1)           |          |                      |                                |
|   3 | データ作成年月日               | data_sakusei_nengappi          | character varying(8)           |          |                      |                                |
|   4 | 開催年                         | kaisai_nen                     | character varying(4)           | Yes (PK) |                      |                                |
|   5 | 開催月日                       | kaisai_tsukihi                 | character varying(4)           | Yes (PK) |                      |                                |
|   6 | 競馬場コード                   | keibajo_code                   | character varying(2)           | Yes (PK) |                      |                                |
|   7 | 開催回[第N回]                  | kaisai_kai                     | character varying(2)           |          |                      |                                |
|   8 | 開催日目[N日目]                | kaisai_nichime                 | character varying(2)           |          |                      |                                |
|   9 | レース番号                     | race_bango                     | character varying(2)           | Yes (PK) |                      |                                |
|  10 | 枠番                           | wakuban                        | character varying(1)           |          |                      |                                |
|  11 | 馬番                           | umaban                         | character varying(2)           | Yes (PK) |                      |                                |
|  12 | 血統登録番号                   | ketto_toroku_bango             | character varying(10)          | Yes (PK) |                      |                                |
|  13 | 馬名                           | bamei                          | character varying(36)          |          |                      |                                |
|  14 | 馬記号コード                   | umakigo_code                   | character varying(2)           |          |                      |                                |
|  15 | 性別コード                     | seibetsu_code                  | character varying(1)           |          |                      |                                |
|  16 | 品種コード                     | hinshu_code                    | character varying(1)           |          |                      |                                |
|  17 | 毛色コード                     | moshoku_code                   | character varying(2)           |          |                      |                                |
|  18 | 馬齢                           | barei                          | character varying(2)           |          |                      |                                |
|  19 | 東西所属コード                 | tozai_shozoku_code             | character varying(1)           |          |                      |                                |
|  20 | 調教師コード                   | chokyoshi_code                 | character varying(5)           |          |                      |                                |
|  21 | 調教師名略称                   | chokyoshimei_ryakusho          | character varying(8)           |          |                      |                                |
|  22 | 馬主コード                     | banushi_code                   | character varying(6)           |          |                      |                                |
|  23 | 馬主名(法人格無)               | banushimei                     | character varying(64)          |          |                      |                                |
|  24 | 服色標示                       | fukushoku_hyoji                | character varying(60)          |          |                      |                                |
|  25 | 予備                           | yobi_1                         | character varying(60)          |          |                      |                                |
|  26 | 負担重量                       | futan_juryo                    | character varying(3)           |          |                      |                                |
|  27 | 変更前負担重量                 | futan_juryo_henkomae           | character varying(3)           |          |                      |                                |
|  28 | ブリンカー使用区分             | blinker_shiyo_kubun            | character varying(1)           |          |                      |                                |
|  29 | 予備                           | yobi_2                         | character varying(1)           |          |                      |                                |
|  30 | 騎手コード                     | kishu_code                     | character varying(5)           |          |                      |                                |
|  31 | 変更前騎手コード               | kishu_code_henkomae            | character varying(5)           |          |                      |                                |
|  32 | 騎手名略称                     | kishumei_ryakusho              | character varying(8)           |          |                      |                                |
|  33 | 変更前騎手名略称               | kishumei_ryakusho_henkomae     | character varying(8)           |          |                      |                                |
|  34 | 騎手見習コード                 | kishu_minarai_code             | character varying(1)           |          |                      |                                |
|  35 | 変更前騎手見習コード           | kishu_minarai_code_henkomae    | character varying(1)           |          |                      |                                |
|  36 | 馬体重                         | bataiju                        | character varying(3)           |          |                      |                                |
|  37 | 増減符号                       | zogen_fugo                     | character varying(1)           |          |                      |                                |
|  38 | 増減差                         | zogen_sa                       | character varying(3)           |          |                      |                                |
|  39 | 異常区分コード                 | ijo_kubun_code                 | character varying(1)           |          |                      |                                |
|  40 | 入線順位                       | nyusen_juni                    | character varying(2)           |          |                      |                                |
|  41 | 確定着順                       | kakutei_chakujun               | character varying(2)           |          |                      |                                |
|  42 | 同着区分                       | dochaku_kubun                  | character varying(1)           |          |                      |                                |
|  43 | 同着頭数                       | dochaku_tosu                   | character varying(1)           |          |                      |                                |
|  44 | 走破タイム                     | soha_time                      | character varying(4)           |          |                      |                                |
|  45 | 着差コード                     | chakusa_code_1                 | character varying(3)           |          |                      |                                |
|  46 | ＋着差コード                   | chakusa_code_2                 | character varying(3)           |          |                      |                                |
|  47 | ＋＋着差コード                 | chakusa_code_3                 | character varying(3)           |          |                      |                                |
|  48 | 1コーナーでの順位              | corner_1                       | character varying(2)           |          |                      |                                |
|  49 | 2コーナーでの順位              | corner_2                       | character varying(2)           |          |                      |                                |
|  50 | 3コーナーでの順位              | corner_3                       | character varying(2)           |          |                      |                                |
|  51 | 4コーナーでの順位              | corner_4                       | character varying(2)           |          |                      |                                |
|  52 | 単勝オッズ                     | tansho_odds                    | character varying(4)           |          |                      |                                |
|  53 | 単勝人気順                     | tansho_ninkijun                | character varying(2)           |          |                      |                                |
|  54 | 獲得本賞金                     | kakutoku_honshokin             | character varying(8)           |          |                      |                                |
|  55 | 獲得付加賞金                   | kakutoku_fukashokin            | character varying(8)           |          |                      |                                |
|  56 | 予備                           | yobi_3                         | character varying(3)           |          |                      |                                |
|  57 | 予備                           | yobi_4                         | character varying(3)           |          |                      |                                |
|  58 | 後4ハロンタイム                | kohan_4f                       | character varying(3)           |          |                      |                                |
|  59 | 後3ハロンタイム                | kohan_3f                       | character varying(3)           |          |                      |                                |
|  60 | 1着馬(相手馬)情報1             | aiteuma_joho_1                 | character varying(46)          |          |                      |                                |
|  61 | 1着馬(相手馬)情報2             | aiteuma_joho_2                 | character varying(46)          |          |                      |                                |
|  62 | 1着馬(相手馬)情報3             | aiteuma_joho_3                 | character varying(46)          |          |                      |                                |
|  63 | タイム差                       | time_sa                        | character varying(4)           |          |                      |                                |
|  64 | レコード更新区分               | record_koshin_kubun            | character varying(1)           |          |                      |                                |
|  65 | マイニング区分                 | mining_kubun                   | character varying(1)           |          |                      |                                |
|  66 | マイニング予想走破タイム       | yoso_soha_time                 | character varying(5)           |          |                      |                                |
|  67 | マイニング予想誤差(信頼度)＋   | yoso_gosa_plus                 | character varying(4)           |          |                      |                                |
|  68 | マイニング予想誤差(信頼度)－   | yoso_gosa_minus                | character varying(4)           |          |                      |                                |
|  69 | マイニング予想順位             | yoso_juni                      | character varying(2)           |          |                      |                                |
|  70 | 今回レース脚質判定             | kyakushitsu_hantei             | character varying(1)           |          |                      |                                |



## インデックス情報

| No. | インデックス名                 | カラムリスト                             | ユニーク   | オプション                     | 
|----:|:-------------------------------|:-----------------------------------------|:-----------|:-------------------------------|
|   1 | apd_sokuho_jvd_se_pk           | kaisai_nen,kaisai_tsukihi,keibajo_code,race_bango,umaban,ketto_toroku_bango | Yes        |                                |
|   2 | apd_sokuho_jvd_se_idx1         | expr                                     |            |                                |
|   3 | apd_sokuho_jvd_se_idx2         | ketto_toroku_bango                       |            |                                |



## 制約情報

| No. | 制約名                         | 種類                           | 制約定義                       |
|----:|:-------------------------------|:-------------------------------|:-------------------------------|
|   1 | 2200_587469_11_not_null        | CHECK                          | umaban IS NOT NULL             |
|   2 | 2200_587469_12_not_null        | CHECK                          | ketto_toroku_bango IS NOT NULL |
|   3 | 2200_587469_4_not_null         | CHECK                          | kaisai_nen IS NOT NULL         |
|   4 | 2200_587469_5_not_null         | CHECK                          | kaisai_tsukihi IS NOT NULL     |
|   5 | 2200_587469_6_not_null         | CHECK                          | keibajo_code IS NOT NULL       |
|   6 | 2200_587469_9_not_null         | CHECK                          | race_bango IS NOT NULL         |
|   7 | apd_sokuho_jvd_se_pk           | PRIMARY KEY                    | kaisai_nen,kaisai_tsukihi,keibajo_code,race_bango,umaban,ketto_toroku_bango |



## 外部キー情報

| No. | 外部キー名                     | カラムリスト                             | 参照先                         | 参照先カラムリスト                       | ON DELETE    | ON UPDATE    |
|----:|:-------------------------------|:-----------------------------------------|:-------------------------------|:-----------------------------------------|:-------------|:-------------|



## 外部キー情報(PK側)

| No. | 外部キー名                     | カラムリスト                             | 参照元                         | 参照元カラムリスト                       | ON DELETE    | ON UPDATE    |
|----:|:-------------------------------|:-----------------------------------------|:-------------------------------|:-----------------------------------------|:-------------|:-------------|



## トリガー情報

| No. | トリガー名                     | イベント                                 | タイミング           | 条件                           |
|----:|:-------------------------------|:-----------------------------------------|:---------------------|:-------------------------------|



## RDBMS固有の情報

| No. | プロパティ名                   | プロパティ値                                                                                         |
|----:|:-------------------------------|:-----------------------------------------------------------------------------------------------------|
|   1 | schemaname                     | public                                                                                               |
|   2 | tablename                      | apd_sokuho_jvd_se                                                                                    |
|   3 | tableowner                     | postgres                                                                                             |
|   4 | tablespace                     |                                                                                                      |
|   5 | hasindexes                     | True                                                                                                 |
|   6 | hasrules                       | False                                                                                                |
|   7 | hastriggers                    | False                                                                                                |
|   8 | rowsecurity                    | False                                                                                                |


