# 穴馬候補判定閾値 設定ガイド

## 概要

`upset_threshold_config.json` は、穴馬候補を判定する際の確率閾値を管理する設定ファイルです。
競馬場・芝ダ区分・距離区分ごとに異なる閾値を設定できます。

## ファイル場所

```
KirisameRaceSimulation/upset_threshold_config.json
```

## 基本構造

```json
{
  "default_threshold": 0.20,
  
  "thresholds_by_condition": {
    "by_track": { "競馬場コード": 閾値 },
    "by_surface": { "turf/dirt": 閾値 },
    "by_distance": { "short/long": 閾値 },
    "by_track_surface": { "競馬場_芝ダ": 閾値 },
    "by_track_surface_distance": { "競馬場_芝ダ_距離": 閾値 }
  }
}
```

## 競馬場コード一覧

| コード | 競馬場 | コード | 競馬場 |
|--------|--------|--------|--------|
| 01 | 札幌 | 06 | 中山 |
| 02 | 函館 | 07 | 中京 |
| 03 | 福島 | 08 | 京都 |
| 04 | 新潟 | 09 | 阪神 |
| 05 | 東京 | 10 | 小倉 |

## 芝ダ区分・距離区分

- **芝ダ区分**: `turf`（芝）, `dirt`（ダート）
- **距離区分**: `short`（1800m以下）, `long`（1801m以上）

## 閾値の優先順位

複数の設定がある場合、**より具体的な設定が優先**されます：

1. `by_track_surface_distance` （例: `02_turf_short`）← 最優先
2. `by_track_surface` （例: `02_turf`）
3. `by_track` （例: `02`）
4. `by_surface` （例: `turf`）
5. `by_distance` （例: `short`）
6. `default_threshold` ← フォールバック

## 使い方

### Step 1: 最適閾値を検証する

```bash
# 検証スクリプトを実行
python analyze_upset_threshold.py
```

出力例:
```
[SIMULATE] 各閾値でのPrecision/Recall/F1
      閾値      候補数     TP     FP     FN  Precision     Recall       F1
------------------------------------------------------------------------------------------
    0.10      268     33    235     26     12.31%     55.93%    20.18 ✅ 目標達成
    0.15      227     32    195     27     14.10%     54.24%    22.38 ✅ 目標達成
    0.20      197     31    166     28     15.74%     52.54%    24.22 ✅ 目標達成  ← F1最大
    0.25      174     23    151     36     13.22%     38.98%    19.74 ✅ 目標達成
```

### Step 2: 設定ファイルを編集

検証結果に基づいて `upset_threshold_config.json` を編集：

```json
{
  "default_threshold": 0.20,
  
  "thresholds_by_condition": {
    "by_track_surface_distance": {
      "02_turf_short": 0.20,
      "02_turf_long": 0.15,
      "02_dirt_short": 0.25,
      "02_dirt_long": 0.20,
      "05_turf_short": 0.18,
      "09_turf_long": 0.22
    }
  }
}
```

### Step 3: 予測を実行

設定は自動的に読み込まれます：

```bash
python universal_test.py --model hakodate_turf_3ageup_short ...
```

ログ出力:
```
[UPSET-THRESHOLD] 02_turf_short の閾値を使用: 0.20
```

## 設定例

### 例1: 全競馬場で同じ閾値を使う（デフォルト）

```json
{
  "default_threshold": 0.20
}
```

### 例2: 芝とダートで閾値を分ける

```json
{
  "default_threshold": 0.20,
  "thresholds_by_condition": {
    "by_surface": {
      "turf": 0.20,
      "dirt": 0.25
    }
  }
}
```

### 例3: 特定の競馬場だけ調整

```json
{
  "default_threshold": 0.20,
  "thresholds_by_condition": {
    "by_track": {
      "02": 0.18,
      "05": 0.22
    }
  }
}
```

### 例4: 細かく条件別に設定

```json
{
  "default_threshold": 0.20,
  "thresholds_by_condition": {
    "by_track_surface_distance": {
      "02_turf_short": 0.20,
      "02_turf_long": 0.15,
      "05_turf_short": 0.18,
      "05_turf_long": 0.22,
      "09_dirt_short": 0.25
    }
  }
}
```

## 閾値チューニングのコツ

| 目的 | 閾値の方向 | トレードオフ |
|------|------------|--------------|
| Precision重視（的中率UP） | 閾値を上げる（0.25-0.30） | 候補数が減り、見逃しが増える |
| Recall重視（検出率UP） | 閾値を下げる（0.10-0.15） | 候補数が増え、空振りが増える |
| バランス重視 | F1スコア最大の閾値 | 一般的にはこれがおすすめ |

## 注意事項

1. **設定ファイルが存在しない場合**: デフォルト閾値 `0.20` が使用されます
2. **JSONの構文エラーがある場合**: エラーメッセージが表示され、デフォルト閾値が使用されます
3. **新しい競馬場条件を追加する場合**: 必ず検証してから閾値を決定してください

## 関連ファイル

- `universal_test.py`: 閾値を読み込んで穴馬候補を判定
- `analyze_upset_threshold.py`: 閾値の最適化分析
- `calculate_precision_recall.py`: Precision/Recall計算

## 更新履歴

| 日付 | 変更内容 |
|------|----------|
| 2026-01-21 | 初版作成。全条件でデフォルト閾値 0.20（函館2025年データでPrecision 15.74%達成） |
