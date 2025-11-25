# 競馬予測モデル 汎用化プロジェクト 🏇

このプロジェクトは機械学習を使った競馬の順位予想システムです。
従来の阪神競馬場限定から、複数の競馬場・条件に対応できるように汎用化されました！

## 📁 ファイル構成

### 🆕 新しいファイル（汎用化対応）
- `model_creator.py` - 汎用的なモデル作成関数
- `batch_model_creator.py` - 複数モデル一括作成スクリプト
- `universal_test.py` - 汎用テストスクリプト
- `keiba_constants.py` - 共通定数・ユーティリティ関数
- `model_configs.json` - モデル設定（JSON形式）
- `model_config_loader.py` - 設定ファイル読み込みユーティリティ

### 🔬 分析・デバッグツール
- `model_explainer.py` - SHAP分析によるモデル解釈ツール
- `analyze_shap_results.py` - SHAP値の統計分析スクリプト
- `test_ewm.py` - EWM(指数加重移動平均)テストスクリプト
- `debug_ewm_detail.py` - EWM vs SQL平均の詳細比較ツール
- `analyze_ewm_issue.py` - EWM性能問題の根本原因分析
- `compare_models.py` - モデル間性能比較ツール
- `shap_analysis/` - SHAP分析結果の保存ディレクトリ
- `shap_analysis_report.md` - SHAP分析レポート

### 📚 既存ファイル（互換性維持）
- `create_model_hanshin_shiba_3ageup.py` - 阪神芝中長距離モデル作成（旧版）
- `test.py` - 阪神芝中長距離モデルテスト（旧版）
- `hanshin_shiba_3ageup_model.sav` - 既存の阪神モデルファイル

## 🚀 使い方（基本編）

### 📋 事前準備

1. **PostgreSQLの準備**
   - JRA公式データがインポートされたデータベース `keiba` を準備
   - データベース接続情報を各Pythonファイルで設定

2. **設定ファイルの確認**
   - `model_configs.json` で作成したいモデルを確認・編集

### 🎯 コマンド一覧（よく使うやつ）

#### 🔥 モデル作成コマンド

```bash
# 【メイン】24個の標準モデルを一括作成（推奨・2013~2022年）
python batch_model_creator.py

# 年範囲を指定してモデル作成（例: 2020~2023年）
python batch_model_creator.py 2020-2023

# 単一年でモデル作成（例: 2023年のみ）
python batch_model_creator.py 2023

# カスタムモデルのみ作成（デフォルト2013~2022年）
python batch_model_creator.py custom

# カスタムモデルを年範囲指定で作成
python batch_model_creator.py custom 2020-2023
```

#### 🎯 テスト実行コマンド

```bash
# 【メイン】全モデル一括テスト（推奨・デフォルト2023年）
python universal_test.py multi

# 年範囲を指定して全モデルテスト（例: 2020~2023年）
python universal_test.py multi 2020-2023

# 単一年で全モデルテスト（例: 2024年のみ）
python universal_test.py multi 2024

# 単一モデルテスト（阪神芝中長距離のみ・デフォルト2023年）
python universal_test.py

# 単一モデルを年範囲指定でテスト
python universal_test.py 2020-2023

# 特定モデルのみテスト（注意: 年範囲指定は未対応）
python universal_test.py single tokyo_turf_3ageup_long.sav
```

### 📊 実行手順（初回セットアップ）

```bash
# 1. 仮想環境を作成・アクティベート
python -m venv venv
source venv/Scripts/activate  # Windows bash

# 2. 必要なパッケージをインストール
pip install -r requirements.txt

# 3. データベース接続設定を確認
# （universal_test.py, model_creator.py内のDB接続情報を編集）

# 4. 標準モデルを24個一括作成（時間かかります！）
python batch_model_creator.py

# 5. 作成されたモデルで2023年データをテスト
python universal_test.py multi
```

### 🎪 日常的な使い方

```bash
# モデルが既に作成済みの場合
python universal_test.py multi

# 新しいモデル設定を追加した場合
python batch_model_creator.py custom

# 特定のモデルだけテストしたい場合
python universal_test.py single hanshin_turf_3ageup_long.sav
```

### 5. 単一モデルの個別作成

```python
from model_creator import create_universal_model

create_universal_model(
    track_code='05',           # 東京競馬場
    kyoso_shubetsu_code='13',  # 3歳以上
    surface_type='turf',       # 芝
    min_distance=1000,         # 1000m以上
    max_distance=1600,         # 1600m以下
    model_filename='tokyo_turf_short.sav'
)
```

## 🏟️ 競馬場コード

| コード | 競馬場 | コード | 競馬場 |
|--------|---------|--------|---------|
| 01     | 札幌    | 06     | 中山    |
| 02     | 函館    | 07     | 中京    |
| 03     | 福島    | 08     | 京都    |
| 04     | 新潟    | 09     | 阪神    |
| 05     | 東京    | 10     | 小倉    |

## 🏃 競争種別コード

| コード | 種別 |
|--------|------|
| 10     | 2歳  |
| 13     | 3歳以上 |

## 🌱 路面タイプ

| 指定値 | 路面 |
|--------|------|
| turf   | 芝   |
| dirt   | ダート |

## 📊 生成されるファイル詳細

### 📁 フォルダ構成
```
KirisameRaceSimulation2/
├── models/                    # 学習済みモデル
├── results/                   # テスト結果
├── venv/                     # 仮想環境（.gitignore済み）
├── model_configs.json        # モデル設定
└── *.py                      # Pythonスクリプト
```

### 🎯 モデル作成時に生成されるファイル

| ファイル名 | 場所 | 説明 |
|-----------|------|------|
| `{モデル名}.sav` | `models/` | 学習済みモデルファイル（pickle形式） |

**例：**
```
models/
├── tokyo_turf_3ageup_long.sav    # 東京芝中長距離
├── tokyo_turf_3ageup_short.sav   # 東京芝短距離
├── kyoto_dirt_3age_long.sav      # 京都ダート中長距離
└── hanshin_turf_3ageup_long.sav  # 阪神芝中長距離
```

### 📈 テスト実行時に生成されるファイル

| ファイル名 | 説明 | 主な内容 |
|-----------|------|----------|
| `predicted_results.tsv` | **通常レース統合結果（38列）** | スコア差が基準値以上のレースの全馬（分析用列なし） |
| `predicted_results_skipped.tsv` | **スキップレース統合結果（43列）** | スコア差が基準値未満のレースの全馬（分析用列あり） |
| `predicted_results_{モデル名}.tsv` | 個別モデル詳細結果 | 各モデル単体の予測結果 |
| `betting_summary_{モデル名}.tsv` | 的中率・回収率まとめ | 単勝・複勝・馬連などの成績 |
| `model_comparison.tsv` | モデル性能比較表 | 全モデルの成績を横並び比較 |

### 📋 予測結果ファイルの列構成

#### `predicted_results.tsv` (通常レース統合結果 - 38列)
| 列名 | 説明 | 例 |
|------|------|-----|
| 競馬場 | 競馬場名 | 阪神 |
| 開催年 | 開催年 | 2023 |
| 開催日 | 開催月日 | 1223 |
| レース番号 | レース番号 | 11 |
| **芝ダ区分** | 芝・ダート区分 | 芝 / ダート |
| 馬番 | 馬番 | 7 |
| 馬名 | 馬名 | キタサンブラック |
| 単勝オッズ | 単勝オッズ | 2.8 |
| 人気順 | 人気順 | 1 |
| 確定着順 | 実際の着順 | 1 |
| 予測順位 | 予測順位 | 1 |
| 予測スコア | 予測スコア(0-1) | 0.85 |

#### `predicted_results_skipped.tsv` (スキップレース統合結果 - 43列)
**通常レースの38列に加えて以下の5列:**

| 列名 | 説明 | 例 |
|------|------|-----|
| スコア差 | 1位と2位の予測スコア差 | 0.006 |
| スキップ理由 | スキップされた理由 | low_score_diff |
| 購入推奨 | 購入推奨フラグ | False |
| 購入額 | 購入額 | 0 |
| 現在資金 | 現在の資金残高 | 1000000 |

#### `betting_summary_{モデル名}.tsv` (的中率・回収率)
| 券種 | 的中数 | 購入数 | 的中率(%) | 投資額 | 払戻額 | 回収率(%) |
|------|--------|--------|-----------|--------|--------|-----------|
| 単勝 | 15 | 65 | 23.08 | 6500 | 9850 | 151.54 |
| 複勝 | 42 | 65 | 64.62 | 6500 | 6200 | 95.38 |
| 馬連 | 8 | 65 | 12.31 | 6500 | 5430 | 83.54 |

## 💡 使用例

### 例1: 地方競馬場の短距離戦モデル
```python
# 小倉競馬場の芝短距離モデル
create_universal_model(
    track_code='10',
    kyoso_shubetsu_code='13',
    surface_type='turf',
    min_distance=1000,
    max_distance=1400,
    model_filename='kokura_turf_sprint.sav'
)
```

### 例2: ダート長距離専門モデル
```python
# 東京競馬場のダート長距離モデル
create_universal_model(
    track_code='05',
    kyoso_shubetsu_code='13',
    surface_type='dirt',
    min_distance=2000,
    max_distance=9999,
    model_filename='tokyo_dirt_long.sav'
)
```

### 例3: 2歳戦専門モデル
```python
# 阪神競馬場の2歳戦芝モデル
create_universal_model(
    track_code='09',
    kyoso_shubetsu_code='10',
    surface_type='turf',
    min_distance=1000,
    max_distance=9999,
    model_filename='hanshin_turf_2age.sav'
)
```

## 🔧 カスタマイズ

### 新しいモデル条件を追加したい場合

**方法1: JSONファイルを直接編集（推奨）**

`model_configs.json`の`custom_models`セクションに新しい設定を追加：

```json
{
  "custom_models": [
    {
      "track_code": "04",
      "kyoso_shubetsu_code": "13",
      "surface_type": "turf",
      "min_distance": 1000,
      "max_distance": 1600,
      "model_filename": "niigata_turf_3ageup_short.sav",
      "description": "新潟芝短距離3歳以上"
    }
  ]
}
```

**方法2: プログラムで動的に追加**

```python
from model_config_loader import add_custom_model

add_custom_model(
    track_code='04',
    kyoso_shubetsu_code='13',
    surface_type='turf',
    min_distance=1000,
    max_distance=1600,
    model_filename='niigata_turf_3ageup_short.sav',
    description='新潟芝短距離3歳以上'
)
```

### 競馬場やコードを追加したい場合

`keiba_constants.py`で競馬場コードや各種定数を管理しています

### 特徴量を変更したい場合

`model_creator.py`の特徴量選択部分を編集してください：

```python
X = df.loc[:, [
    "kyori",
    "tenko_code",  
    "babajotai_code",
    # ここに新しい特徴量を追加
]].astype(float)
```

## ⚡ パフォーマンス

- ハイパーパラメータ最適化: Optuna使用（50試行）
- 機械学習アルゴリズム: LightGBM（LambdaRank）
- 時系列分割: 75%訓練用、25%テスト用

## 🔍 結果の見方

### 的中率・回収率
- **的中率**: 予想が当たった割合（%）
- **回収率**: 投資に対する払戻の割合（%）
- **回収率100%超**: 利益が出ている状態 🎉

### 予測結果ファイル
- TSV形式で保存（Excelで開けます）
- 各馬の予測順位と実際の着順を比較可能
- オッズ情報も含まれているので収支計算ができます

## ⚠️ トラブルシューティング

### 🔍 よくある問題と解決方法

#### 1. データベース接続エラー
```
psycopg2.OperationalError: could not connect to server
```
**解決方法：**
- PostgreSQLサービスが起動しているか確認
- `universal_test.py` と `model_creator.py` のDB接続情報を確認
- データベース `keiba` が存在するか確認

#### 2. モデルファイルが見つからない
```
❌ モデルファイル tokyo_turf_3ageup_long.sav が見つかりません
```
**解決方法：**
- まず `python batch_model_creator.py` でモデルを作成
- `models/` フォルダが存在するか確認

#### 3. パッケージ不足エラー
```
ModuleNotFoundError: No module named 'lightgbm'
```
**解決方法：**
- 仮想環境がアクティベートされているか確認
- `pip install -r requirements.txt` を実行

#### 4. メモリ不足・処理が重い
**解決方法：**
- 他のアプリケーションを終了してメモリを確保
- バッチ処理の場合は時間をおいて再実行
- より高性能なマシンでの実行を検討

#### 5. ファイル書き込みエラー
```
PermissionError: [Errno 13] Permission denied
```
**解決方法：**
- 結果ファイル（.tsv）がExcelなどで開かれていないか確認
- `results/` フォルダの書き込み権限を確認

### 🧪 デバッグ用コマンド

```bash
# 詳細ログを出力してテスト実行
python universal_test.py multi 2>&1 | tee debug_log.txt

# 特定モデルのみで問題を切り分け
python universal_test.py single hanshin_turf_3ageup_long.sav

# Pythonとパッケージバージョン確認
python --version
pip list | grep -E "(lightgbm|pandas|numpy|scikit-learn)"
```

## 💾 システム要件

### 推奨環境
- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.8以上（3.10推奨）
- **RAM**: 8GB以上（16GB推奨）
- **ストレージ**: 空き容量10GB以上
- **PostgreSQL**: 12以上

### 処理時間の目安
- **モデル作成**: 1個あたり5-15分（ハイパラ最適化含む）
- **24個一括作成**: 2-6時間
- **テスト実行**: 1個あたり1-3分
- **全モデルテスト**: 15-30分

## 🚨 注意事項

1. **データベース接続**: PostgreSQLに競馬データが必要です
2. **計算時間**: 最適化処理で数分〜数十分かかる場合があります
3. **メモリ使用量**: 大量データを扱うため、8GB以上のRAM推奨
4. **投資は自己責任**: 実際の馬券購入は慎重に！

## 📋 詳細コマンドリファレンス

### 🔧 batch_model_creator.py

**概要**: JSON設定に基づいてモデルを一括作成

```bash
# 標準モデル24個を一括作成（デフォルト: 2013~2022年）
python batch_model_creator.py

# 年範囲を指定して標準モデル作成（例: 2020~2023年）
python batch_model_creator.py 2020-2023

# 単一年で標準モデル作成（例: 2023年のみ）
python batch_model_creator.py 2023

# カスタムモデルのみ作成（デフォルト: 2013~2022年）
python batch_model_creator.py custom

# カスタムモデルを年範囲指定で作成
python batch_model_creator.py custom 2020-2023

# カスタムモデルを単一年で作成
python batch_model_creator.py custom 2023
```

**処理内容:**
- `model_configs.json` から設定を読み込み
- Optuna使用でハイパーパラメータ最適化（50試行）
- `models/` フォルダに `.sav` ファイル保存
- **🆕 年範囲指定**: コマンドライン引数で学習データの年範囲を柔軟に変更可能

### 🎯 universal_test.py

**概要**: 作成済みモデルでテスト実行

```bash
# 全モデル一括テスト（デフォルト: 2023年）
python universal_test.py multi

# 年範囲を指定して全モデルテスト（例: 2020~2023年）
python universal_test.py multi 2020-2023

# 単一年で全モデルテスト（例: 2024年のみ）
python universal_test.py multi 2024

# 単一モデルテスト（デフォルト: 2023年）
python universal_test.py

# 単一モデルを年範囲指定でテスト
python universal_test.py 2020-2023

# 単一モデルを単一年でテスト
python universal_test.py 2024

# 特定モデルのみテスト（注意: 年範囲指定は未対応）
python universal_test.py single hanshin_turf_3ageup_long.sav
```

**処理内容:**
- 2023年データ（デフォルト）でテスト実行
- **🆕 年範囲指定**: コマンドライン引数でテストデータの年範囲を柔軟に変更可能
- 的中率・回収率計算
- `results/` フォルダに結果保存

### 🛠️ model_creator.py

**概要**: 単一モデル作成用関数（プログラムから呼び出し）

```python
from model_creator import create_universal_model

# 基本的な使い方
create_universal_model(
    track_code='05',           # 競馬場コード
    kyoso_shubetsu_code='13',  # 競争種別コード
    surface_type='turf',       # 路面タイプ
    min_distance=1700,         # 最小距離
    max_distance=9999,         # 最大距離
    model_filename='my_model.sav',  # 保存ファイル名
    year_start=2013,           # 学習データ開始年（デフォルト: 2013）
    year_end=2022              # 学習データ終了年（デフォルト: 2022）
)

# 年範囲を指定した例
create_universal_model(
    track_code='05',
    kyoso_shubetsu_code='13',
    surface_type='turf',
    min_distance=1700,
    max_distance=9999,
    model_filename='tokyo_turf_2020_2023.sav',
    year_start=2020,           # 2020年から
    year_end=2023              # 2023年まで
)
```

### 📝 model_config_loader.py

**概要**: JSON設定管理ユーティリティ

```python
from model_config_loader import load_model_configs, add_custom_model

# 設定読み込み
configs = load_model_configs()

# カスタムモデル追加
add_custom_model(
    track_code='04',
    kyoso_shubetsu_code='13',
    surface_type='turf',
    min_distance=1000,
    max_distance=1600,
    model_filename='niigata_turf_short.sav',
    description='新潟芝短距離3歳以上'
)
```

### 🗃️ keiba_constants.py

**概要**: 共通定数・ユーティリティ関数

```python
from keiba_constants import get_track_name, get_surface_name, get_kyoso_shubetsu_name

# 競馬場名取得
track_name = get_track_name('05')  # → '東京'

# 路面名取得  
surface_name = get_surface_name(1)  # → '芝'

# 競争種別名取得
kyoso_name = get_kyoso_shubetsu_name('13')  # → '3歳以上'
```

## 🎯 実践的な使用パターン

### パターン1: 初回セットアップ
```bash
# 1. 環境準備
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# 2. 標準モデル一括作成（数時間かかります・デフォルト2013~2022年）
python batch_model_creator.py

# 3. 全モデルテスト実行
python universal_test.py multi
```

### パターン2: 新しい競馬場を追加
```bash
# 1. model_configs.json の custom_models に設定追加
# 2. カスタムモデル作成
python batch_model_creator.py custom

# 3. 新モデルも含めてテスト
python universal_test.py multi
```

### パターン3: 特定モデルの性能確認
```bash
# 特定モデルのみテスト
python universal_test.py single tokyo_turf_3ageup_long.sav

# 結果確認
cat results/betting_summary_tokyo_turf_3ageup_long.tsv
```

### パターン4: 🆕 年範囲を指定してモデル作成&テスト
```bash
# 2020年~2023年のデータでモデル作成
python batch_model_creator.py 2020-2023

# 同じ期間でテスト実行
python universal_test.py multi 2020-2023

# 2023年のみでモデル作成
python batch_model_creator.py 2023

# 2023年のみでテスト実行
python universal_test.py multi 2023

# カスタムモデルも2020~2023年で作成&テスト
python batch_model_creator.py custom 2020-2023
python universal_test.py multi 2020-2023
```

### パターン5: 複数の期間でモデルを比較
```bash
# 2013~2022年のモデル作成&テスト
python batch_model_creator.py 2013-2022
python universal_test.py multi 2023
# → 2023年データで検証

# 2020~2023年のモデル作成&テスト
python batch_model_creator.py 2020-2023
python universal_test.py multi 2024
# → 2024年データで検証（最新データでの性能確認）

# 結果比較
# → results/model_comparison.tsv で期間ごとのモデル性能を比較可能
```

### パターン6: 過去データでのバックテスト
```bash
# 2013~2019年でモデル作成
python batch_model_creator.py 2013-2019

# 2020~2022年でテスト（未知データでの性能評価）
python universal_test.py multi 2020-2022
```

## 🔬 SHAP分析機能

### SHAP分析とは？
SHAP (SHapley Additive exPlanations) は、機械学習モデルの予測理由を数値化・可視化する技術です。
各特徴量がどれだけ予測に貢献したかを定量的に評価できます。

### 使い方

```bash
# SHAP分析実行（モデルの予測理由を分析）
python model_explainer.py

# SHAP値の統計分析（特徴量重要度ランキング）
python analyze_shap_results.py
```

### 分析結果の見方

**生成されるファイル:**
- `shap_analysis/{モデル名}_importance.csv` - 特徴量重要度ランキング
- `shap_analysis_report.md` - 分析レポート

**重要度の解釈:**
- **SHAP値が高い特徴量**: モデルの予測に大きく貢献
- **SHAP値が低い特徴量**: モデルにほとんど使われていない（削除候補）

### SHAP分析結果サマリー

**最重要特徴量（top5）:**
1. `past_avg_sotai_chakujun` (0.211) - 過去平均相対着順
2. `time_index` (0.165) - タイムインデックス
3. `past_score` (0.132) - 過去スコア
4. `umaban_kyori_interaction` (0.127) - 馬番×距離の相互作用
5. `kohan_3f_index` (0.113) - 後半3Fインデックス

**削除候補特徴量（SHAP < 0.005）:**
- `barei_peak_short`, `barei_peak_distance`, `baba_condition_score`
- `futan_juryo`, `distance_change_adaptability`, `wakuban_bias_score`

## 📊 EWM実験レポート

### 実験概要
`past_avg_sotai_chakujun`（過去平均相対着順）にEWM（指数加重移動平均）を適用し、
単純移動平均と比較検証しました。

### 実験結果サマリー

| 手法 | span | 単勝的中率 | 複勝的中率 | 三連複的中率 | 評価 |
|------|------|------------|------------|--------------|------|
| **SQL平均** | - | **23.08%** | **49.23%** | **7.69%** | ✅ 安定 |
| EWM | 3 | 18.46% | - | - | ❌ 最悪 |
| EWM | 5 | 20.00% | - | - | ❌ 改善不足 |
| EWM | 7 | 26.15% | 47.18% | 6.15% | ⚠️ 単勝↑複勝↓ |

### 結論
**EWMは競馬予測には不向き**と判明：
- ✅ 単勝予測（span=7）: 26.15%と改善
- ❌ 複勝・三連複: 悪化（47.18% → 49.23%が理想）
- 🔍 原因: EWMの「慣性」問題 - 馬の調子急降下を検出しづらい

**最終判断:** SQL単純移動平均を継続使用

詳細: `shap_analysis_report.md` と各分析スクリプト参照

## 🎯 今後の拡張予定

- [x] SHAP分析による特徴量重要度評価 ✅
- [x] EWM(指数加重移動平均)の検証 ✅
- [ ] 低重要度特徴量の削除による精度向上
- [ ] 騎手・調教師情報の追加
- [ ] 血統情報の組み込み
- [ ] リアルタイム予測機能
- [ ] Web UI の作成
- [ ] クラウド対応
- [ ] 他年度データでの自動検証
- [ ] モデル性能の可視化機能

## 📞 サポート・お問い合わせ

**問題が発生した場合：**
1. エラーメッセージの全文をコピー
2. 実行したコマンドを記録
3. 環境情報（OS、Pythonバージョン）を確認
4. GitHubのIssuesで報告またはメール連絡

**連絡先**: GitHub Issues または [メールアドレス]

---

🔥 **楽しい競馬予想ライフを！** 🔥

何か質問があったら気軽に聞いてね〜✨