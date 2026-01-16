# 速報用レース詳細（jvd_raベース） (apd_sokuho_jvd_ra)

## テーブル情報

| 項目                           | 値                                                                                                   |
|:-------------------------------|:-----------------------------------------------------------------------------------------------------|
| システム名                     | keiba                                                                                                |
| サブシステム名                 |                                                                                                      |
| スキーマ名                     | public                                                                                               |
| 物理テーブル名                 | apd_sokuho_jvd_ra                                                                                    |
| 論理テーブル名                 | 速報用レース詳細（jvd_raベース）                                                                     |
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
|  10 | 曜日コード                     | yobi_code                      | character varying(1)           |          |                      |                                |
|  11 | 特別競走番号                   | tokubetsu_kyoso_bango          | character varying(4)           |          |                      |                                |
|  12 | 競走名本題                     | kyosomei_hondai                | character varying(60)          |          |                      |                                |
|  13 | 競走名副題                     | kyosomei_fukudai               | character varying(60)          |          |                      |                                |
|  14 | 競走名カッコ内                 | kyosomei_kakkonai              | character varying(60)          |          |                      |                                |
|  15 | 競走名本題欧字                 | kyosomei_hondai_eur            | character varying(120)         |          |                      |                                |
|  16 | 競走名副題欧字                 | kyosomei_fukudai_eur           | character varying(120)         |          |                      |                                |
|  17 | 競走名カッコ内欧字             | kyosomei_kakkonai_eur          | character varying(120)         |          |                      |                                |
|  18 | 競走名略称10文字               | kyosomei_ryakusho_10           | character varying(20)          |          |                      |                                |
|  19 | 競走名略称6文字                | kyosomei_ryakusho_6            | character varying(12)          |          |                      |                                |
|  20 | 競走名略称3文字                | kyosomei_ryakusho_3            | character varying(6)           |          |                      |                                |
|  21 | 競走名区分                     | kyosomei_kubun                 | character varying(1)           |          |                      |                                |
|  22 | 重賞回次[第N回]                | jusho_kaiji                    | character varying(3)           |          |                      |                                |
|  23 | グレードコード                 | grade_code                     | character varying(1)           |          |                      |                                |
|  24 | 変更前グレードコード           | grade_code_henkomae            | character varying(1)           |          |                      |                                |
|  25 | 競走種別コード                 | kyoso_shubetsu_code            | character varying(2)           |          |                      |                                |
|  26 | 競走記号コード                 | kyoso_kigo_code                | character varying(3)           |          |                      |                                |
|  27 | 重量種別コード                 | juryo_shubetsu_code            | character varying(1)           |          |                      |                                |
|  28 | 競走条件コード 2歳条件         | kyoso_joken_code_2sai          | character varying(3)           |          |                      |                                |
|  29 | 競走条件コード 3歳条件         | kyoso_joken_code_3sai          | character varying(3)           |          |                      |                                |
|  30 | 競走条件コード 4歳条件         | kyoso_joken_code_4sai          | character varying(3)           |          |                      |                                |
|  31 | 競走条件コード 5歳以上条件     | kyoso_joken_code_5sai_ijo      | character varying(3)           |          |                      |                                |
|  32 | 競走条件コード 最若年条件      | kyoso_joken_code               | character varying(3)           |          |                      |                                |
|  33 | 競走条件名称                   | kyoso_joken_meisho             | character varying(60)          |          |                      |                                |
|  34 | 距離                           | kyori                          | character varying(4)           |          |                      |                                |
|  35 | 変更前距離                     | kyori_henkomae                 | character varying(4)           |          |                      |                                |
|  36 | トラックコード                 | track_code                     | character varying(2)           |          |                      |                                |
|  37 | 変更前トラックコード           | track_code_henkomae            | character varying(2)           |          |                      |                                |
|  38 | コース区分                     | course_kubun                   | character varying(2)           |          |                      |                                |
|  39 | 変更前コース区分               | course_kubun_henkomae          | character varying(2)           |          |                      |                                |
|  40 | 本賞金                         | honshokin                      | character varying(56)          |          |                      |                                |
|  41 | 変更前本賞金                   | honshokin_henkomae             | character varying(40)          |          |                      |                                |
|  42 | 付加賞金                       | fukashokin                     | character varying(40)          |          |                      |                                |
|  43 | 変更前付加賞金                 | fukashokin_henkomae            | character varying(24)          |          |                      |                                |
|  44 | 発走時刻                       | hasso_jikoku                   | character varying(4)           |          |                      |                                |
|  45 | 変更前発走時刻                 | hasso_jikoku_henkomae          | character varying(4)           |          |                      |                                |
|  46 | 登録頭数                       | toroku_tosu                    | character varying(2)           |          |                      |                                |
|  47 | 出走頭数                       | shusso_tosu                    | character varying(2)           |          |                      |                                |
|  48 | 入線頭数                       | nyusen_tosu                    | character varying(2)           |          |                      |                                |
|  49 | 天候コード                     | tenko_code                     | character varying(1)           |          |                      |                                |
|  50 | 芝馬場状態コード               | babajotai_code_shiba           | character varying(1)           |          |                      |                                |
|  51 | ダート馬場状態コード           | babajotai_code_dirt            | character varying(1)           |          |                      |                                |
|  52 | ラップタイム                   | lap_time                       | character varying(75)          |          |                      |                                |
|  53 | 障害マイルタイム               | shogai_mile_time               | character varying(4)           |          |                      |                                |
|  54 | 前3ハロン                      | zenhan_3f                      | character varying(3)           |          |                      |                                |
|  55 | 前4ハロン                      | zenhan_4f                      | character varying(3)           |          |                      |                                |
|  56 | 後3ハロン                      | kohan_3f                       | character varying(3)           |          |                      |                                |
|  57 | 後4ハロン                      | kohan_4f                       | character varying(3)           |          |                      |                                |
|  58 | コーナー通過順位1              | corner_tsuka_juni_1            | character varying(72)          |          |                      |                                |
|  59 | コーナー通過順位2              | corner_tsuka_juni_2            | character varying(72)          |          |                      |                                |
|  60 | コーナー通過順位3              | corner_tsuka_juni_3            | character varying(72)          |          |                      |                                |
|  61 | コーナー通過順位4              | corner_tsuka_juni_4            | character varying(72)          |          |                      |                                |
|  62 | レコード更新区分               | record_koshin_kubun            | character varying(1)           |          |                      |                                |



## インデックス情報

| No. | インデックス名                 | カラムリスト                             | ユニーク   | オプション                     | 
|----:|:-------------------------------|:-----------------------------------------|:-----------|:-------------------------------|
|   1 | apd_sokuho_jvd_ra_pk           | kaisai_nen,kaisai_tsukihi,keibajo_code,race_bango | Yes        |                                |
|   2 | apd_sokuho_jvd_ra_idx1         | expr                                     |            |                                |
|   3 | apd_sokuho_jvd_ra_idx2         | tokubetsu_kyoso_bango                    |            |                                |



## 制約情報

| No. | 制約名                         | 種類                           | 制約定義                       |
|----:|:-------------------------------|:-------------------------------|:-------------------------------|
|   1 | 2200_587433_4_not_null         | CHECK                          | kaisai_nen IS NOT NULL         |
|   2 | 2200_587433_5_not_null         | CHECK                          | kaisai_tsukihi IS NOT NULL     |
|   3 | 2200_587433_6_not_null         | CHECK                          | keibajo_code IS NOT NULL       |
|   4 | 2200_587433_9_not_null         | CHECK                          | race_bango IS NOT NULL         |
|   5 | apd_sokuho_jvd_ra_pk           | PRIMARY KEY                    | kaisai_nen,kaisai_tsukihi,keibajo_code,race_bango |



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
|   2 | tablename                      | apd_sokuho_jvd_ra                                                                                    |
|   3 | tableowner                     | postgres                                                                                             |
|   4 | tablespace                     |                                                                                                      |
|   5 | hasindexes                     | True                                                                                                 |
|   6 | hasrules                       | False                                                                                                |
|   7 | hastriggers                    | False                                                                                                |
|   8 | rowsecurity                    | False                                                                                                |


