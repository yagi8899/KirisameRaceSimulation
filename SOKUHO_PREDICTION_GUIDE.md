# 速報データ予測ガイド

このドキュメントは、速報データ（今週予定されているレース情報）を使用した実際の予測を行うための完全ガイドです。

## 📌 概要

`sokuho_prediction.py`は、`apd_sokuho_jvd_ra`と`apd_sokuho_jvd_se`テーブルから速報データを取得し、学習済みモデルで予測を行い、購入推奨結果を出力します。

## 🎯 主な機能

- ✅ 速報データ（今週のレース）の自動取得
- ✅ 過去データと結合して特徴量を計算
- ✅ モデルによる順位予測
- ✅ 購入推奨判定（予測順位・人気順・オッズ・スコア差でフィルタリング）
- ✅ 結果をTSVファイルで出力

## 🗄️ テーブル構造

### 速報用レース情報テーブル（`apd_sokuho_jvd_ra`）

速報データのレース情報を格納するテーブルです。

**主要カラム**：
- `kaisai_nen` - 開催年
- `kaisai_tsukihi` - 開催月日（MMDD形式）
- `keibajo_code` - 競馬場コード
- `race_bango` - レース番号
- `kyori` - 距離
- `track_code` - トラックコード（芝/ダート判定）
- `grade_code` - グレードコード
- `kyoso_joken_code` - 競走条件コード
- `kyoso_shubetsu_code` - 競走種別コード
- `shusso_tosu` - 出走頭数

詳細は [sokuho_db_docs/public.apd_sokuho_jvd_ra.md](sokuho_db_docs/public.apd_sokuho_jvd_ra.md) を参照してください。

### 速報用馬毎レース情報テーブル（`apd_sokuho_jvd_se`）

速報データの出走馬情報を格納するテーブルです。

**主要カラム**：
- `ketto_toroku_bango` - 血統登録番号（馬の識別子）
- `bamei` - 馬名
- `umaban` - 馬番
- `wakuban` - 枠番
- `barei` - 馬齢
- `kishu_code` - 騎手コード
- `chokyoshi_code` - 調教師コード
- `futan_juryo` - 負担重量
- `tansho_odds` - 単勝オッズ
- `tansho_ninkijun` - 単勝人気順

**注意**: 速報データには以下のカラムは含まれません（レース後に確定）：
- `kakutei_chakujun` - 確定着順
- `soha_time` - 走破タイム
- `corner_1-4` - コーナー通過順位
- `kohan_3f` - 後3ハロンタイム

詳細は [sokuho_db_docs/public.apd_sokuho_jvd_se.md](sokuho_db_docs/public.apd_sokuho_jvd_se.md) を参照してください。

### 過去データテーブル（`jvd_se`）

速報データの特徴量計算には、過去データ（`jvd_se`）と結合して使用します。
`ketto_toroku_bango`（血統登録番号）で紐付け、その馬の過去成績から特徴量を計算します。

## 🔄 データフロー

```
1. 速報データ取得
   └─ apd_sokuho_jvd_ra（レース情報）
   └─ apd_sokuho_jvd_se（出走馬情報）

2. 過去データと結合（SQLで実行）
   └─ jvd_se（過去の出走馬データ）とUNION ALL
   └─ ketto_toroku_bango で結合

3. 特徴量計算（SQLのウィンドウ関数）
   └─ past_avg_sotai_chakujun（過去3走の相対着順平均）
   └─ time_index（過去3走の平均速度）
   └─ past_score（グレード別過去実績）
   └─ kohan_3f_index（過去3走の後半3F指数）

4. Pythonで高度な特徴量を追加
   └─ similar_distance_score（近似距離での過去10走成績）
   └─ start_index（過去10走のスタート能力）
   └─ kishu_skill_score（騎手の直近3ヶ月成績）
   └─ など24個の特徴量

5. モデルで予測
   └─ LightGBM LambdaRankモデル

6. 購入推奨判定
   └─ 予測順位 ≤ 3
   └─ 人気順 ≤ 3
   └─ オッズ 1.5倍〜20倍
   └─ 予測1位と2位のスコア差 ≥ 0.05

7. 結果出力
   └─ sokuho_results/sokuho_prediction_{model_name}_{timestamp}.tsv
```

## 📝 使用方法

### 標準モデル（40個全て）で予測

```bash
python sokuho_prediction.py --model standard
```

このコマンドは、`model_configs.json`に定義されている全ての標準モデル（40個）で予測を実行します。

### カスタムモデルで予測

```bash
python sokuho_prediction.py --model custom --model-name tokyo_turf_3ageup_long
```

特定のカスタムモデルのみで予測を実行します。

## 📊 出力ファイル

### ファイル名形式

```
sokuho_results/sokuho_prediction_{モデル名}_{タイムスタンプ}.tsv
```

例: `sokuho_prediction_tokyo_turf_3ageup_long_20260116_143025.tsv`

### 出力カラム

| カラム名 | 説明 |
|---------|------|
| 競馬場 | 競馬場名 |
| 開催年 | 開催年（YYYY） |
| 開催日 | 開催日（MMDD） |
| レース番号 | レース番号 |
| 芝ダ区分 | 芝/ダートの区分 |
| 距離 | 距離（m） |
| 馬番 | 馬番 |
| 馬名 | 馬名 |
| 単勝オッズ | 単勝オッズ |
| 人気順 | 人気順 |
| 予測順位 | モデルが予測した順位 |
| 予測スコア | 予測スコア（0-1の範囲） |
| 購入推奨 | 購入推奨フラグ（True/False） |
| スコア差 | レース内での予測1位と2位のスコア差 |
| スキップ理由 | 購入推奨でない場合の理由 |

### スキップ理由の種類

| スキップ理由 | 説明 |
|-------------|------|
| `low_score_diff` | 予測スコア差が小さい（本命が不明確） |
| `low_predicted_rank` | 予測順位が4位以下 |
| `low_popularity` | 人気順が4位以下 |
| `odds_too_low` | オッズが1.5倍未満 |
| `odds_too_high` | オッズが20倍超 |
| `multiple_conditions` | 複数の条件に該当 |

## ⚙️ 設定値

### 購入推奨フィルタのデフォルト値

- **予測順位上限**: 3位以内
- **人気順上限**: 3位以内
- **オッズ範囲**: 1.5倍〜20倍
- **最小スコア差**: 0.05（予測1位と2位の差）

これらの値は`sokuho_prediction.py`の`add_sokuho_purchase_logic()`関数で変更可能です。

## 🚨 注意事項

### 速報データの鮮度

- 速報データは常に最新のものが`apd_sokuho_jvd_*`テーブルに格納されている前提です
- 古い速報データは自動的に削除されているため、時系列チェックは不要です

### モデルファイルが見つからない場合

モデルファイル（`.sav`）が`models/`ディレクトリに存在しない場合、そのモデルはスキップされます（警告ログが出力されます）。

### 対象データが0件の場合

- モデルの条件（競馬場・距離・馬場・競走種別）に合致する速報データが存在しない場合、そのモデルはスキップされます
- 全モデルで0件の場合、「速報データが登録されていない可能性があります」という警告が表示されます

## 📂 関連ファイル

- `sokuho_prediction.py` - メインスクリプト
- `db_query_builder.py` - SQLクエリ生成（`build_sokuho_race_data_query()`関数）
- `model_configs.json` - モデル設定ファイル
- `data_preprocessing.py` - データ前処理
- `feature_engineering.py` - 基本特徴量生成
- `keiba_constants.py` - 定数定義

## 🔧 トラブルシューティング

### データベース接続エラー

```
[ERROR] データ取得エラー: connection refused
```

→ PostgreSQLが起動していることを確認してください。

### モデルファイルが見つからない

```
[SKIP] モデルファイルが見つかりません: models/tokyo_turf_3ageup_long.sav
```

→ `model_creator.py`または`batch_model_creator.py`でモデルを作成してください。

### 速報データが0件

```
[SKIP] 対象データが0件です
```

→ `apd_sokuho_jvd_*`テーブルに速報データが登録されているか確認してください。

## 📈 今後の拡張予定

- [ ] オッズ更新対応（定期的に再予測）
- [ ] 複数日の速報データ対応（日付フィルタオプション）
- [ ] 信頼度スコアの追加
- [ ] 期待値計算の追加

## 📞 問い合わせ

バグ報告や機能要望は、プロジェクトの管理者に連絡してください。
