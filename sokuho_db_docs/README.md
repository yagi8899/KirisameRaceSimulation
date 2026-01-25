# 疑似速報データ作成ガイド

## 📋 概要

このフォルダには、過去の確定データから疑似速報データを生成するためのSQLスクリプトが含まれています。
速報予測の検証環境を構築する際に使用します。

## 🗂️ フォルダ構成

```
sokuho_db_docs/
├── 2023/                                    # 2023年用バッチファイル
│   ├── create_pseudo_sokuho_2023.bat        # 疑似速報データ生成
│   ├── validate_pseudo_sokuho_2023.bat      # 疑似速報データ検証
│   ├── cleanup_pseudo_sokuho_2023.bat       # 疑似速報データ削除
│   └── cleanup_historical_data_2023.bat     # 過去成績データ削除
├── 2024/                                    # 2024年用バッチファイル
│   ├── create_pseudo_sokuho_2024.bat
│   ├── validate_pseudo_sokuho_2024.bat
│   ├── cleanup_pseudo_sokuho_2024.bat
│   └── cleanup_historical_data_2024.bat
├── 2025/                                    # 2025年用バッチファイル
│   ├── create_pseudo_sokuho_2025.bat
│   ├── validate_pseudo_sokuho_2025.bat
│   ├── cleanup_pseudo_sokuho_2025.bat
│   └── cleanup_historical_data_2025.bat
├── create_pseudo_sokuho_data.sql            # 疑似速報データ生成SQL
├── validate_pseudo_sokuho.sql               # 疑似速報データ検証SQL
├── cleanup_pseudo_sokuho.sql                # 疑似速報データ削除SQL
├── cleanup_historical_data.sql              # 過去成績データ削除SQL
├── public.apd_sokuho_jvd_ra.md              # レース情報テーブル仕様書
├── public.apd_sokuho_jvd_se.md              # 馬毎レース情報テーブル仕様書
└── README.md                                # このファイル
```

## 🗂️ ファイル一覧

### SQLスクリプト（共通）

| ファイル | 説明 |
|---------|------|
| `create_pseudo_sokuho_data.sql` | 疑似速報データを生成するメインスクリプト |
| `validate_pseudo_sokuho.sql` | 生成データの整合性を検証するスクリプト |
| `cleanup_pseudo_sokuho.sql` | 疑似速報データを削除するスクリプト |
| `cleanup_historical_data.sql` | 過去成績データ（jvd_ra, jvd_se）を削除するスクリプト |

### 仕様書

| ファイル | 説明 |
|---------|------|
| `public.apd_sokuho_jvd_ra.md` | レース情報テーブルの仕様書 |
| `public.apd_sokuho_jvd_se.md` | 馬毎レース情報テーブルの仕様書 |

### バッチファイル（年度別）

各年度フォルダ内に以下の4種類のバッチファイルがあります：

| バッチファイル | 説明 |
|---------------|------|
| `create_pseudo_sokuho_YYYY.bat` | 疑似速報データを生成 |
| `validate_pseudo_sokuho_YYYY.bat` | 疑似速報データを検証 |
| `cleanup_pseudo_sokuho_YYYY.bat` | 疑似速報データを削除 |
| `cleanup_historical_data_YYYY.bat` | 過去成績データ（jvd_ra, jvd_se）を削除 |

---

## 🚀 使い方

### 前提条件

- PostgreSQLがインストールされていること
- `keiba`データベースに接続できること
- `jvd_ra`, `jvd_se`テーブルに確定データが存在すること
- `apd_sokuho_jvd_ra`, `apd_sokuho_jvd_se`テーブルが作成済みであること

### 🎯 クイックスタート（バッチファイル使用）

各年度フォルダのバッチファイルをダブルクリックするだけで実行できます：

```
1. 2023/create_pseudo_sokuho_2023.bat    → 疑似速報データ生成
2. 2023/validate_pseudo_sokuho_2023.bat  → データ検証
3. 2023/cleanup_historical_data_2023.bat → 過去成績削除（本番環境再現時）
4. （テスト実行）
5. 2023/cleanup_pseudo_sokuho_2023.bat   → クリーンアップ（必要時）
```

### Step 1: 疑似速報データの生成

```bash
# 2020年〜2023年のデータを生成する場合
psql -h localhost -U postgres -d keiba \
  -v target_year_start=2020 \
  -v target_year_end=2023 \
  -f create_pseudo_sokuho_data.sql
```

**パラメータ**:
| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `target_year_start` | 変換対象の開始年 | 2020 |
| `target_year_end` | 変換対象の終了年 | 2023 |

**処理内容**:
1. 指定年度の既存疑似データを削除
2. `jvd_ra` → `apd_sokuho_jvd_ra` へ変換（結果情報をマスク）
3. `jvd_se` → `apd_sokuho_jvd_se` へ変換（結果情報をマスク）

### Step 2: データの検証

```bash
# 生成データの整合性を確認
psql -h localhost -U postgres -d keiba \
  -v target_year_start=2020 \
  -v target_year_end=2023 \
  -f validate_pseudo_sokuho.sql
```

**検証項目**:
1. レース件数の一致確認
2. 馬毎レース件数の一致確認
3. 結果情報がマスクされているか確認（`apd_sokuho_jvd_ra`）
4. 結果情報がマスクされているか確認（`apd_sokuho_jvd_se`）
5. オッズ情報がコピーされているか確認
6. 年度別データ件数の確認

### Step 3: データの削除（必要な場合）

```bash
# 疑似速報データを削除
psql -h localhost -U postgres -d keiba \
  -v target_year_start=2020 \
  -v target_year_end=2023 \
  -f cleanup_pseudo_sokuho.sql
```

⚠️ **注意**: この操作は元に戻せません！

### Step 4: 過去成績データの削除（本番環境再現時）

検証環境で本番に近い状態をテストしたい場合、過去成績データ（jvd_ra, jvd_se）を削除します。

```bash
# 過去成績データを削除（疑似速報データのみ残す）
psql -h localhost -U postgres -d keiba \
  -v target_year_start=2020 \
  -v target_year_end=2023 \
  -f cleanup_historical_data.sql
```

⚠️ **警告**: 
- この操作は元に戻せません！
- jvd_ra, jvd_seの該当年度データが完全に削除されます
- 本番環境では絶対に実行しないでください！
- バッチファイル使用時は2段階の確認があります

---

## 🧪 検証環境構築フロー

### フルフロー（本番環境再現）

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 疑似速報データ生成                                        │
│    create_pseudo_sokuho_YYYY.bat                            │
│    → apd_sokuho_jvd_ra, apd_sokuho_jvd_se にデータ作成      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 疑似速報データ検証                                        │
│    validate_pseudo_sokuho_YYYY.bat                          │
│    → 件数・マスク状態を確認                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 過去成績データ削除（本番再現時のみ）                       │
│    cleanup_historical_data_YYYY.bat                         │
│    → jvd_ra, jvd_se の該当年度データを削除                   │
│    ⚠️ 確定結果が参照できなくなります                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. テスト実行                                                │
│    sokuho_prediction.py 等でテスト                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. クリーンアップ（必要時）                                   │
│    cleanup_pseudo_sokuho_YYYY.bat                           │
│    → 疑似速報データを削除                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 マスクされる情報

### レース情報（apd_sokuho_jvd_ra）

以下のカラムは`NULL`でマスクされます：

| カラム名 | 説明 |
|---------|------|
| `nyusen_tosu` | 入線頭数 |
| `lap_time` | ラップタイム |
| `shogai_mile_time` | 障害マイルタイム |
| `zenhan_3f` / `zenhan_4f` | 前半3F/4F |
| `kohan_3f` / `kohan_4f` | 後半3F/4F |
| `corner_tsuka_juni_1〜4` | コーナー通過順位 |
| `record_koshin_kubun` | レコード更新区分 |

### 馬毎レース情報（apd_sokuho_jvd_se）

以下のカラムは`NULL`またはデフォルト値でマスクされます：

| カラム名 | 説明 | マスク値 |
|---------|------|---------|
| `kakutei_chakujun` | 確定着順 | NULL |
| `nyusen_chakujun` | 入線着順 | NULL |
| `soha_time` | 走破タイム | NULL |
| `chakusa_code_*` | 着差コード | NULL |
| `time_sa` | タイム差 | NULL |
| `kohan_3f` | 後半3F | NULL |
| `corner_juni_*` | コーナー順位 | NULL |
| `weight_zogen_sa` | 体重増減差 | NULL |
| `ijyo_kubun_code` | 異常区分コード | '0' |

### 保持される情報

以下の情報は速報時点で利用可能なため、そのままコピーされます：

- 馬名、血統、騎手、調教師
- 斤量、馬体重（当日発表分）
- 単勝・複勝オッズ（直前）
- 枠番、馬番
- 前走成績（過去5走分）

---

## 🔧 実行例

### 全年度のデータを生成

```bash
# 2015年〜2024年の10年分を生成
psql -h localhost -U postgres -d keiba \
  -v target_year_start=2015 \
  -v target_year_end=2024 \
  -f create_pseudo_sokuho_data.sql
```

### 検証結果の例

```
[1/6] レース件数の一致確認...
 テーブル  | 元データ件数 | 速報データ件数 | 判定
---------+-------------+---------------+------
 jvd_ra  |       45000 |         45000 | ✓ OK

[3/6] レース結果情報がマスクされているか確認...
 テーブル           | 総レコード数 | nyusen_tosu | lap_time | kohan_3f
-------------------+-------------+-------------+----------+---------
 apd_sokuho_jvd_ra |       45000 |           ✓ |        ✓ |        ✓
```

---

## ⚠️ 注意事項

1. **大量データの場合**: 10年分のデータは数百万レコードになるため、処理に時間がかかります
2. **ディスク容量**: 元データとほぼ同量のディスク容量が必要です
3. **トランザクション**: 各スクリプトはトランザクション内で実行されるため、エラー時は自動ロールバックされます
4. **既存データ**: 同一年度のデータは上書きされます（DELETE → INSERT）

---

## 📁 テーブル仕様

詳細なテーブル仕様は以下を参照：

- [apd_sokuho_jvd_ra（レース情報）](./public.apd_sokuho_jvd_ra.md)
- [apd_sokuho_jvd_se（馬毎レース情報）](./public.apd_sokuho_jvd_se.md)

---

## 🔗 関連ファイル

- `sokuho_prediction.py` - 速報予測メインスクリプト
- `SOKUHO_PREDICTION_GUIDE.md` - 速報予測ガイド

---

**作成日**: 2026年1月24日
**更新日**: 2026年1月24日（年度別フォルダ構成、過去成績削除スクリプト追加）
