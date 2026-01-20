# Phase 3.5 特徴量追加・削除プラン 📋

## 📊 概要

**作成日**: 2026年1月20日  
**目的**: Precision 1.78% → 8%達成のための特徴量再構成  
**背景**: Phase 3実装後の特徴量重要度分析により、kishu_changed (1,923) と class_downgrade (112) の効果が極めて低いことが判明

---

## 🎯 目標

- **現状**: Precision 1.78% max (最適閾値0.35)
- **目標**: Precision 8%以上
- **手段**: 効果的な特徴量5個を追加、ノイズの多い特徴量7個を削除

---

## 📈 特徴量数の変化

| 状態 | Phase 3 | Phase 3削除 | Phase 3.5 | 合計 |
|------|---------|------------|-----------|------|
| 追加 | 6個 | - | 5個 | 11個 |
| 削除 | - | 2個 | 5個 | 7個 |
| **合計特徴量数** | **28個** | **26個** | **26個** | **26個** |
| **UPSET関連** | 6個 | 4個 | 9個 | 9個 |

---

## ➕ 追加予定の特徴量 (5個)

### 1. zenso_ninki_gap (前走人気着順ギャップ)
- **計算方法**: `LAG(popularity_rank - kakutei_chakujun)`
- **期待効果**: +10% (過小評価馬の検出)
- **SQL実装**: 容易 (単純なLAG処理)
- **リーク懸念**: なし (前走確定データのみ)

### 2. zenso_nigeba (前走逃げ成功フラグ)
- **計算方法**: `LAG(corner_1 == 1)`
- **期待効果**: +5% (展開依存性の検出)
- **SQL実装**: 容易 (単純なLAG処理)

### 3. zenso_taihai (前走大敗フラグ)
- **計算方法**: `LAG(kakutei_chakujun > 10)`
- **期待効果**: +5% (巻き返しパターンの検出)
- **SQL実装**: 容易 (単純なLAG処理)

### 4. zenso_agari_rank (前走上がり順位)
- **計算方法**: `RANK() OVER (ORDER BY kohan_3f) → LAG`
- **期待効果**: +10% (隠れた実力の検出)
- **SQL実装**: やや複雑 (2段階処理)
- **備考**: フェーズ2からフェーズ3.5に昇格

### 5. saikin_kaikakuritsu (直近3走改善率)
- **計算方法**: `COUNT(今回 < 前回) / 3走`
- **期待効果**: +8% (調子の上向き検出)
- **SQL実装**: やや複雑 (WINDOW関数)
- **備考**: class_downgradeの代替

**合計期待効果**: +38% (Precision向上)

---

## ➖ 削除予定の特徴量 (7個)

### Phase 3から削除済み (2個)

#### 1. kishu_changed (騎手変更フラグ) ❌
- **削除理由**: 特徴量重要度1,923 (極めて低い)
- **削除日**: 2026-01-19
- **影響**: ほぼなし (他特徴量で代替可能)

#### 2. class_downgrade (クラス降級フラグ) ❌
- **削除理由**: 特徴量重要度112 (極めて低い)
- **削除日**: 2026-01-19
- **影響**: saikin_kaikakuritsuで代替

### Phase 3.5で削除予定 (5個)

#### 3. wakuban_inner (内枠フラグ) ❌
- **削除理由**: 短距離専用特徴量、汎用UPSETモデルには不要

#### 4. wakuban_outer (外枠フラグ) ❌
- **削除理由**: 短距離専用特徴量、汎用UPSETモデルには不要

#### 5. estimated_running_style (推定脚質) ❌
- **削除理由**: 推定値でノイズが多い、corner系特徴量で代替可能

#### 6. tenko_code (天候コード) ❌
- **削除理由**: 効果が不明瞭、馬場状態で代替可能

#### 7. distance_change (距離変化) ❌
- **削除理由**: 距離適性スコア (similar_distance_score) で吸収可能

---

## 📝 実装手順

### Step 1: SQL実装 (db_query_builder.py)

両方のクエリに同じ特徴量を追加:
- `build_race_data_query()` - 訓練用クエリ
- `build_sokuho_race_data_query()` - 速報予測用クエリ

#### 1.1 zenso_ninki_gap
```sql
LAG(
    cast(seum.popularity as float) - cast(seum.kakutei_chakujun as float)
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
) AS zenso_ninki_gap
```

#### 1.2 zenso_nigeba
```sql
LAG(
    CASE WHEN seum.corner_1 = '01' THEN 1 ELSE 0 END
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
) AS zenso_nigeba
```

#### 1.3 zenso_taihai
```sql
LAG(
    CASE WHEN cast(seum.kakutei_chakujun as integer) > 10 THEN 1 ELSE 0 END
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
) AS zenso_taihai
```

#### 1.4 zenso_agari_rank (2段階処理)
```sql
-- サブクエリでレース内ランキングを計算
WITH agari_ranks AS (
    SELECT 
        seum.*,
        RANK() OVER (
            PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
            ORDER BY CASE 
                WHEN seum.kohan_3f = '000' OR seum.kohan_3f = '999' THEN 9999 
                ELSE cast(seum.kohan_3f as integer) 
            END
        ) AS agari_rank
    FROM ...
)
-- メインクエリでLAG
SELECT 
    ...,
    LAG(agari_rank) OVER (
        PARTITION BY ketto_toroku_bango
        ORDER BY kaisai_nen, kaisai_tsukihi
    ) AS zenso_agari_rank
FROM agari_ranks
```

#### 1.5 saikin_kaikakuritsu (WINDOW関数)
```sql
-- 過去3走で着順が改善した回数 / 3
(
    CASE WHEN cast(seum.kakutei_chakujun as integer) < LAG(cast(seum.kakutei_chakujun as integer), 1) THEN 1 ELSE 0 END +
    CASE WHEN LAG(cast(seum.kakutei_chakujun as integer), 1) < LAG(cast(seum.kakutei_chakujun as integer), 2) THEN 1 ELSE 0 END +
    CASE WHEN LAG(cast(seum.kakutei_chakujun as integer), 2) < LAG(cast(seum.kakutei_chakujun as integer), 3) THEN 1 ELSE 0 END
) / 3.0 AS saikin_kaikakuritsu
```

### Step 2: Python実装 (feature_engineering.py)

`add_upset_specific_features()` 関数を更新:

```python
def add_upset_specific_features(df):
    """穴馬予測用の特徴量を追加"""
    
    # 既存の4特徴量はそのまま維持
    # past_score_std, past_chakujun_variance, zenso_oikomi_power, zenso_kakoi_komon
    
    # 新規5特徴量を追加
    # zenso_ninki_gap: SQLで計算済み (fillna(-1))
    if 'zenso_ninki_gap' in df.columns:
        df['zenso_ninki_gap'] = df['zenso_ninki_gap'].fillna(-1)
    
    # zenso_nigeba: SQLで計算済み (fillna(0))
    if 'zenso_nigeba' in df.columns:
        df['zenso_nigeba'] = df['zenso_nigeba'].fillna(0)
    
    # zenso_taihai: SQLで計算済み (fillna(0))
    if 'zenso_taihai' in df.columns:
        df['zenso_taihai'] = df['zenso_taihai'].fillna(0)
    
    # zenso_agari_rank: SQLで計算済み (fillna(-1))
    if 'zenso_agari_rank' in df.columns:
        df['zenso_agari_rank'] = df['zenso_agari_rank'].fillna(-1)
    
    # saikin_kaikakuritsu: SQLで計算済み (fillna(0.5 = 中立))
    if 'saikin_kaikakuritsu' in df.columns:
        df['saikin_kaikakuritsu'] = df['saikin_kaikakuritsu'].fillna(0.5)
    
    # 削除予定の特徴量を除外
    drop_cols = ['wakuban_inner', 'wakuban_outer', 'estimated_running_style', 
                 'tenko_code', 'distance_change']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return df
```

### Step 3: analyze_upset_patterns.py更新

特徴量リストを更新:

```python
# 削除: kishu_changed, class_downgrade, wakuban_inner, wakuban_outer, 
#       estimated_running_style, tenko_code, distance_change
# 追加: zenso_ninki_gap, zenso_nigeba, zenso_taihai, zenso_agari_rank, saikin_kaikakuritsu

feature_cols = [
    # 既存の22個の特徴量...
    
    # Phase 3 (4個残存)
    'past_score_std', 'past_chakujun_variance', 
    'zenso_oikomi_power', 'zenso_kakoi_komon',
    
    # Phase 3.5 (5個追加)
    'zenso_ninki_gap', 'zenso_nigeba', 'zenso_taihai',
    'zenso_agari_rank', 'saikin_kaikakuritsu'
]
```

### Step 4: モデル再訓練

```bash
# 全期間のモデルを再訓練 (walk_forward_validation)
python walk_forward_validation.py --with-upset

# または個別モデル再訓練
python train_upset_classifier.py --years 2015-2024
```

### Step 5: 効果検証

```bash
# Precision評価
python analyze_upset_threshold.py "models/upset_classifier_2015-2024.sav"

# 特徴量重要度チェック
python analyze_upset_model_features.py "models/upset_classifier_2015-2024.sav"

# テスト実行
python universal_test.py 2025
```

---

## 📊 期待される結果

### Before (Phase 3実装後)
- **特徴量数**: 28個
- **UPSET特徴量**: 6個 (うち2個が低効果)
- **Precision**: 1.78% max (閾値0.35)
- **問題点**: kishu_changed, class_downgradeが機能していない

### After (Phase 3.5実装後)
- **特徴量数**: 26個 (-2個)
- **UPSET特徴量**: 9個 (+3個実質)
- **Precision**: 8%以上 (目標)
- **改善点**: 
  - 過小評価馬の検出 (zenso_ninki_gap)
  - 隠れた実力の発見 (zenso_agari_rank)
  - 展開依存性の把握 (zenso_nigeba, zenso_taihai)
  - 調子の波の検出 (saikin_kaikakuritsu)

---

## ⚠️ 注意事項

### データリークのチェック
- ✅ 全ての新特徴量は過去確定データのみ使用
- ✅ popularity_rankは前走の値を使用 (今回レースのオッズは不使用)
- ✅ LAG関数で前走データを取得 (未来データは参照しない)

### NULL処理の方針
- **数値特徴量**: -1 (経験不足を明示)
- **フラグ特徴量**: 0 (該当しない)
- **割合特徴量**: 0.5 (中立値)

### 実装の順序
1. SQLクエリ実装 → テスト
2. Python特徴量追加 → テスト
3. モデル再訓練
4. Precision評価
5. 目標未達の場合 → Phase 2の4特徴量を追加検討

---

## 📅 スケジュール

- **2026-01-20**: SQL実装 (5特徴量)
- **2026-01-21**: Python実装 + テスト
- **2026-01-22**: モデル再訓練 (2015-2024)
- **2026-01-23**: Precision評価 + 効果検証
- **2026-01-24**: 目標達成確認 or Phase 2追加検討

---

## 🔗 関連ドキュメント

- [UPSET_PREDICTION_FEATURES.md](UPSET_PREDICTION_FEATURES.md) - 特徴量設計ドキュメント
- [FEATURE_LIST.md](FEATURE_LIST.md) - 全特徴量一覧
- [analyze_upset_model_features.py](analyze_upset_model_features.py) - 重要度分析スクリプト
- [analyze_upset_threshold.py](analyze_upset_threshold.py) - 閾値最適化スクリプト

---

**最終更新**: 2026年1月20日  
**作成者**: GitHub Copilot  
**ステータス**: 🔄 実装準備中
