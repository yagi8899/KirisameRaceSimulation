# Walk-Forward Validation システム設計書

## 📋 概要

本システムは、競馬予測モデルのWalk-Forward Validation（WFV）を自動化し、最適な学習期間の決定とモデルの汎化性能評価を実現する。

### 目的
1. 年ごとにモデルをローリング作成し、未来データで評価
2. 複数の学習期間を比較し、最適期間を決定
3. モデルの安定性・汎化性能を評価
4. 本運用前のバックテスト基盤を提供

---

## 🏗️ システム構成

### ファイル構成
```
KirisameRaceSimulation2/
├── walk_forward_validation.py       # メインスクリプト
├── walk_forward_config.json         # WFV設定ファイル
├── model_configs.json               # モデル定義（既存、変更なし）
├── walk_forward_results/            # 実行結果保存先
│   ├── period_5/                    # 5年期間の結果
│   │   ├── models/
│   │   │   ├── 2023/
│   │   │   │   ├── tokyo_turf_3ageup_long_2018-2022.sav
│   │   │   │   └── ...
│   │   │   ├── 2024/
│   │   │   └── 2025/
│   │   ├── test_results/
│   │   │   ├── 2023/
│   │   │   ├── 2024/
│   │   │   └── 2025/
│   │   └── summary_period_5.tsv
│   ├── period_7/
│   ├── period_10/
│   ├── period_comparison.tsv        # 期間比較レポート
│   ├── progress.json                # 進捗管理ファイル
│   └── execution.log                # 実行ログ
└── README.md
```

---

## ⚙️ 設定ファイル仕様

### walk_forward_config.json

```json
{
  "walk_forward_validation": {
    "test_years": [2023, 2024, 2025],
    "output_dir": "walk_forward_results",
    
    "execution_mode": "single_period",
    
    "single_period_settings": {
      "training_period": 10,
      "rolling_type": "fixed",
      "models": "all"
    },
    
    "compare_periods_settings": {
      "training_periods": [5, 7, 10],
      "models": ["tokyo_turf_3ageup_long", "tokyo_turf_3ageup_short"]
    },
    
    "execution": {
      "on_model_creation_error": "skip",
      "on_test_error": "skip",
      "retry_count": 3,
      "continue_on_error": true
    },
    
    "logging": {
      "level": "INFO",
      "file": "walk_forward_results/execution.log",
      "console": true
    }
  }
}
```

#### 設定項目詳細

##### test_years (必須)
- 型: `list[int]`
- 説明: テスト対象年のリスト
- 例: `[2023, 2024, 2025]`

##### output_dir (必須)
- 型: `string`
- 説明: 結果保存先ディレクトリ
- デフォルト: `"walk_forward_results"`

##### execution_mode (必須)
- 型: `"single_period"` | `"compare_periods"`
- 説明: 実行モード
  - `single_period`: 単一学習期間でWFV実行
  - `compare_periods`: 複数期間を比較実験

##### single_period_settings
- **training_period** (必須):
  - 型: `int`
  - 説明: 学習期間（年数）
  - 例: `10` → 10年分のデータで学習
  
- **rolling_type** (必須):
  - 型: `"fixed"`
  - 説明: ローリング方式（固定窓のみサポート）
  
- **models** (必須):
  - 型: `"all"` | `"standard"` | `"custom"` | `list[string]`
  - 説明: 対象モデル
    - `"all"`: 全モデル（standard + custom）
    - `"standard"`: standard_modelsのみ
    - `"custom"`: custom_modelsのみ
    - `["model1", "model2"]`: 個別指定

##### compare_periods_settings
- **training_periods** (必須):
  - 型: `list[int]`
  - 説明: 比較する学習期間のリスト
  - 例: `[5, 7, 10]`
  
- **models** (必須):
  - 型: 同上

##### execution
- **on_model_creation_error**:
  - 型: `"skip"` | `"stop"` | `"retry"`
  - デフォルト: `"skip"`
  - 説明: モデル作成失敗時の挙動
  
- **on_test_error**:
  - 型: `"skip"` | `"stop"` | `"retry"`
  - デフォルト: `"skip"`
  - 説明: テスト実行失敗時の挙動
  
- **retry_count**:
  - 型: `int`
  - デフォルト: `3`
  - 説明: リトライ回数
  
- **continue_on_error**:
  - 型: `boolean`
  - デフォルト: `true`
  - 説明: エラー発生時も処理を継続

##### logging
- **level**:
  - 型: `"DEBUG"` | `"INFO"` | `"WARNING"` | `"ERROR"`
  - デフォルト: `"INFO"`
  
- **file**:
  - 型: `string`
  - 説明: ログファイルパス
  
- **console**:
  - 型: `boolean`
  - デフォルト: `true`
  - 説明: コンソール出力の有無

---

## 📝 ファイル命名規則

### モデルファイル
```
{base_name}_{train_start}-{train_end}.sav

例:
tokyo_turf_3ageup_long_2018-2022.sav
tokyo_turf_3ageup_long_2019-2023.sav
```

**重要:** モデル作成とテスト実行で命名規則が一致していること
- モデル作成時: `{base_name}_{train_start}-{train_end}.sav`
- テスト実行時: 同じ命名規則で検索

### テスト結果ファイル
```
predicted_results_{model_name}_{train_period}_test{test_year}.tsv

例:
predicted_results_tokyo_turf_3ageup_long_2018-2022_test2023.tsv
```

### サマリーファイル
```
summary_period_{period}.tsv

例:
summary_period_5.tsv
summary_period_10.tsv
```

---

## 🔄 実行フロー

### 1. 単一期間モード (execution_mode: "single_period")

```
1. 設定ファイル読み込み
2. model_configs.jsonから対象モデル取得
3. 各テスト年でループ:
   a. 学習期間を計算
      train_start = test_year - training_period
      train_end = test_year - 1
   
   b. 各モデルでループ:
      - モデル作成（create_universal_model）
      - 進捗保存（progress.json）
   
   c. 各モデルでループ:
      - テスト実行（universal_test機能）
      - 結果保存
   
4. サマリー生成
5. レポート出力
```

### 2. 期間比較モード (execution_mode: "compare_periods")

```
1. 設定ファイル読み込み
2. model_configs.jsonから対象モデル取得
3. 各学習期間でループ:
   a. period_X/ ディレクトリ作成
   
   b. 各テスト年でループ:
      - 学習期間を計算
      - 各モデルでモデル作成
      - 各モデルでテスト実行
   
   c. この期間のサマリー生成
   
4. 全期間の比較レポート生成
5. 最適期間の推定
```

---

## 📊 出力レポート仕様

### 1. summary_period_{period}.tsv

各学習期間の詳細成績。

**列構成:**
```
モデル名
学習期間（例: 2018-2022）
テスト年
レース数
購入推奨馬数
単勝的中数
単勝的中率
単勝回収率
複勝的中数
複勝的中率
複勝回収率
馬連的中数
馬連的中率
馬連回収率
ワイド的中数
ワイド的中率
ワイド回収率
馬単的中数
馬単的中率
馬単回収率
三連複的中数
三連複的中率
三連複回収率
期待回収率
```

**例:**
```tsv
モデル名	学習期間	テスト年	レース数	購入推奨馬数	単勝的中数	単勝的中率	単勝回収率	...
tokyo_turf_long	2018-2022	2023	65	116	30	25.9%	148.3%	...
tokyo_turf_long	2019-2023	2024	40	70	14	20.0%	95.2%	...
tokyo_turf_long	2020-2024	2025	57	95	19	20.0%	110.5%	...
平均	-	-	54	93.7	21	22.0%	118.0%	...
標準偏差	-	-	13	23.1	8.5	3.5%	27.0%	...
```

### 2. period_comparison.tsv

複数学習期間の比較サマリー。

**列構成:**
```
学習期間
平均回収率（単勝）
標準偏差（単勝）
変動係数（単勝）
平均回収率（複勝）
標準偏差（複勝）
変動係数（複勝）
平均的中率（単勝）
平均的中率（複勝）
推奨度
```

**例:**
```tsv
学習期間	平均回収率	標準偏差	変動係数	...	推奨度
5年	105.2%	18.3%	0.17	...	✅ 最安定
7年	102.8%	22.5%	0.22	...	⭐ やや安定
10年	95.6%	28.4%	0.30	...	⚠️ やや不安定
```

**推奨度判定基準:**
- ✅ 最安定: 変動係数 < 0.20 かつ 平均回収率 > 100%
- ⭐ やや安定: 変動係数 < 0.25
- ⚠️ やや不安定: 変動係数 < 0.35
- 🚨 不安定: 変動係数 >= 0.35

### 3. model_stability_report.tsv

各モデルの安定性評価。

**列構成:**
```
モデル名
平均回収率
標準偏差
変動係数
最良年
最悪年
判定
```

---

## 💾 進捗管理 (progress.json)

実行途中での中断・再開を可能にする進捗ファイル。

**構造:**
```json
{
  "execution_mode": "compare_periods",
  "training_periods": [5, 7, 10],
  "test_years": [2023, 2024, 2025],
  "progress": {
    "period_5": {
      "2023": {
        "tokyo_turf_3ageup_long": {
          "model_created": true,
          "model_tested": true,
          "model_path": "walk_forward_results/period_5/models/2023/tokyo_turf_3ageup_long_2018-2022.sav"
        },
        "tokyo_turf_3ageup_short": {
          "model_created": true,
          "model_tested": false,
          "model_path": "walk_forward_results/period_5/models/2023/tokyo_turf_3ageup_short_2018-2022.sav"
        }
      },
      "2024": {},
      "2025": {}
    },
    "period_7": {},
    "period_10": {}
  },
  "last_updated": "2026-01-13T15:30:00",
  "started_at": "2026-01-13T10:00:00"
}
```

### 再開ロジック
1. `progress.json`が存在しない → 新規実行
2. 存在する → 未完了タスクを検出
3. モデル単位でスキップ判定:
   - `model_created=true` → モデル作成をスキップ
   - `model_tested=true` → テスト実行をスキップ
   - 両方falseまたは未記録 → 実行

---

## ⌨️ コマンドライン引数

### 基本実行
```bash
python walk_forward_validation.py
```
- 設定ファイル必須: `walk_forward_config.json`
- 設定ファイルに完全準拠

### オプション引数

#### --config
```bash
python walk_forward_validation.py --config my_config.json
```
- 設定ファイルのパスを指定
- デフォルト: `walk_forward_config.json`

#### --resume
```bash
python walk_forward_validation.py --resume
```
- 前回の実行を途中から再開
- `progress.json`を読み込み、未完了タスクのみ実行

#### --dry-run
```bash
python walk_forward_validation.py --dry-run
```
- 実行計画のみ表示（実際には実行しない）
- 確認用

#### --clean
```bash
python walk_forward_validation.py --clean
```
- 前回の実行結果を削除してクリーンスタート
- `progress.json`も削除

#### --verbose
```bash
python walk_forward_validation.py --verbose
```
- ログレベルをDEBUGに変更（詳細表示）

---

## 🚨 エラーハンドリング

### モデル作成失敗時
```
設定: on_model_creation_error = "skip"

動作:
1. エラーログを記録
2. 該当モデルをスキップ
3. 次のモデルに進む
4. progress.jsonに失敗を記録
```

### テスト実行失敗時
```
設定: on_test_error = "skip"

動作:
1. エラーログを記録
2. 該当テストをスキップ
3. 次のテストに進む
4. progress.jsonに失敗を記録
```

### リトライロジック
```
設定: retry_count = 3

動作:
1. 失敗時、最大3回リトライ
2. 3回とも失敗 → スキップ
3. リトライ間隔: 10秒
```

---

## 📐 設計上の重要事項

### 1. モデル作成とテストの齟齬防止

**問題:** ファイル名の不一致でテスト時にモデルが見つからない

**対策:**
```python
def get_model_filename(base_name, train_start, train_end):
    """モデルファイル名を統一的に生成"""
    return f"{base_name}_{train_start}-{train_end}.sav"

# モデル作成時
filename = get_model_filename("tokyo_turf_3ageup_long", 2018, 2022)
create_model(..., model_filename=filename)

# テスト実行時
filename = get_model_filename("tokyo_turf_3ageup_long", 2018, 2022)
test_model(model_filename=filename)
```

### 2. ディレクトリ構造の一貫性

**期間ごとに完全分離:**
```
period_5/
  models/2023/  # 2018-2022学習
  models/2024/  # 2019-2023学習
  test_results/2023/
  test_results/2024/

period_7/
  models/2023/  # 2016-2022学習
  ...
```

**混在を防ぐ:**
- 各期間は独立したディレクトリ
- モデルファイル名に学習期間を含める

### 3. model_configs.jsonとの連携

```python
from model_config_loader import get_all_models

# model_configs.jsonから設定を読み込み
all_models = get_all_models()

# フィルタリング
target_models = filter_models(all_models, config['models'])

# 各モデルの設定を完全に引き継ぐ
for model_config in target_models:
    create_universal_model(
        track_code=model_config['track_code'],
        surface_type=model_config['surface_type'],
        min_distance=model_config['min_distance'],
        max_distance=model_config['max_distance'],
        kyoso_shubetsu_code=model_config['kyoso_shubetsu_code'],
        train_year_start=train_start,
        train_year_end=train_end,
        model_filename=get_model_filename(...)
    )
```

---

## 📈 実行時間の見積もり

### 前提条件
- モデル1個の作成時間: 約15分
- モデル1個のテスト時間: 約3分

### 単一期間モード（10年、全24モデル、3年テスト）
```
モデル作成: 24モデル × 3年 × 15分 = 1,080分（18時間）
テスト実行: 24モデル × 3年 × 3分 = 216分（3.6時間）
合計: 約21.6時間
```

### 期間比較モード（3期間、2モデル、3年テスト）
```
モデル作成: 2モデル × 3期間 × 3年 × 15分 = 270分（4.5時間）
テスト実行: 2モデル × 3期間 × 3年 × 3分 = 54分（0.9時間）
合計: 約5.4時間
```

---

## 🎯 実装フェーズ

### Phase 1: 最小機能版（目標: 2日）
- [x] 設計完了
- [ ] walk_forward_validation.py作成
- [ ] 単一期間モード実装
- [ ] progress.json管理
- [ ] 基本サマリー生成
- [ ] 動作確認（小規模テスト）

### Phase 2: 期間比較機能（目標: 1日）
- [ ] 期間比較モード実装
- [ ] period_comparison.tsv生成
- [ ] 最適期間推定ロジック

### Phase 3: エラーハンドリング強化（目標: 0.5日）
- [ ] リトライロジック
- [ ] 詳細エラーログ
- [ ] 異常終了時の復旧

### Phase 4: 拡張機能（将来）
- [ ] 並列実行
- [ ] 通知機能（Slack等）
- [ ] 可視化（グラフ生成）
- [ ] Expanding window対応

---

## 📚 関連ドキュメント

- `README.md`: プロジェクト全体の説明
- `model_configs.json`: モデル定義
- `walk_forward_config.json`: WFV設定（本システム）

---

## 🔗 依存関係

### 既存モジュール
- `model_creator.py`: `create_universal_model()`
- `model_config_loader.py`: `get_all_models()`
- `universal_test.py`: テスト実行機能
- `db_query_builder.py`: SQL生成
- `data_preprocessing.py`: データ前処理
- `feature_engineering.py`: 特徴量作成

### 新規パッケージ
- なし（既存環境で動作）

---

## 🛠️ 開発ガイドライン

### コーディング規約
- Python 3.8以上
- PEP 8準拠
- 型ヒント使用推奨
- Docstring必須

### テスト方針
1. 小規模テスト（1モデル、1年）
2. 中規模テスト（2モデル、2年、2期間）
3. 本番規模テスト（全モデル、3年）

### コミット規約
- `feat:` 新機能
- `fix:` バグ修正
- `docs:` ドキュメント
- `refactor:` リファクタリング

---

## 📝 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|----------|
| 2026-01-13 | 1.0 | 初版作成 |

---

**END OF DOCUMENT**
