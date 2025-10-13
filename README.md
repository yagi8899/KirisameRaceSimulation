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

### 📚 既存ファイル（互換性維持）
- `create_model_hanshin_shiba_3ageup.py` - 阪神芝中長距離モデル作成（旧版）
- `test.py` - 阪神芝中長距離モデルテスト（旧版）
- `hanshin_shiba_3ageup_model.sav` - 既存の阪神モデルファイル

## 🚀 使い方

### 1. 設定ファイルの確認・編集

まず `model_configs.json` で作成したいモデルを設定：

```json
{
  "standard_models": [
    {
      "track_code": "05",
      "kyoso_shubetsu_code": "13",
      "surface_type": "turf",
      "min_distance": 1700,
      "max_distance": 9999,
      "model_filename": "tokyo_turf_3ageup_long.sav",
      "description": "東京芝中長距離3歳以上"
    }
  ]
}
```

### 2. 標準モデルの一括作成

```bash
# JSON設定に基づいて標準モデルを一括作成
python batch_model_creator.py
```

### 3. カスタムモデルの作成

```bash
# JSON設定に基づいてカスタムモデルを作成
python batch_model_creator.py custom
```

### 4. モデルのテスト

```bash
# 単一モデルテスト（従来互換）
python universal_test.py

# 全モデル比較テスト（JSON設定に基づく）
python universal_test.py multi
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

## 📊 生成されるファイル

### モデル作成時
- `models/{モデル名}.sav` - 学習済みモデルファイル

### テスト実行時（すべて`results/`フォルダに保存）
- `results/predicted_results_{モデル名}.tsv` - 予測結果詳細（追記保存）
- `results/betting_summary_{モデル名}.tsv` - 的中率・回収率サマリー
- `results/model_comparison.tsv` - 複数モデル比較結果（複数テスト時）
- `results/predicted_results.tsv` - 単一モデルテスト結果（追記保存）

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

## 🚨 注意事項

1. **データベース接続**: PostgreSQLに競馬データが必要です
2. **計算時間**: 最適化処理で数分〜数十分かかる場合があります
3. **メモリ使用量**: 大量データを扱うため、8GB以上のRAM推奨
4. **投資は自己責任**: 実際の馬券購入は慎重に！

## 🎯 今後の拡張予定

- [ ] 騎手・調教師情報の追加
- [ ] 血統情報の組み込み
- [ ] リアルタイム予測機能
- [ ] Web UI の作成
- [ ] クラウド対応

---

🔥 **楽しい競馬予想ライフを！** 🔥

何か質問があったら気軽に聞いてね〜✨