# SHAP分析レポート - kyoto_dirt_3ageup_long

## 📊 実行日: 2026年01月18日

---

## 🎯 重要な発見

### 1️⃣ **過去成績系の特徴量が圧倒的に重要**

**Top 3の特徴量:**
1. **relative_ability** (0.006) - relative_ability
   - SHAP値: 0.006
   - LightGBM Gain: 2084.4

2. **surface_aptitude_score** (0.002) - surface_aptitude_score
   - SHAP値: 0.002
   - LightGBM Gain: 280.7

3. **past_avg_sotai_chakujun** (0.002) - 過去の相対着順
   - SHAP値: 0.002 (ぶっちぎり1位)
   - LightGBM Gain: 95.8
   - 意味: 直近3走の相対着順(1-(着順/出走頭数))の平均
   - **結論**: 馬の直近パフォーマンスが最も重要!

**Top3だけで全体影響の62.1%を占める!**
- relative_ability: 0.006 / 0.016 = 36.3%
- surface_aptitude_score: 0.002 / 0.016 = 15.6%
- past_avg_sotai_chakujun: 0.002 / 0.016 = 10.3%

---

### 2️⃣ **カテゴリ別特徴量の重要度**

**特徴量カテゴリ別寄与率:**
- **馬場適性系** (17.3%) - 2個の特徴量
- **過去成績系** (12.3%) - 3個の特徴量
- **騎手系** (9.9%) - 2個の特徴量
- **斤量系** (6.5%) - 4個の特徴量
- **調教師系** (2.7%) - 1個の特徴量

**分析:**
- 馬場適性系が17.3%でトップ
- モデルは馬の基本能力を最も重視している

---

### 3️⃣ **削除推奨特徴量の分析**

**削除候補(SHAP < 0.005): 23個**

- `surface_aptitude_score` (SHAP=0.002477) ❌
- `past_avg_sotai_chakujun` (SHAP=0.001629) ❌
- `right_direction_score` (SHAP=0.001540) ❌
- `kishu_skill_score` (SHAP=0.000988) ❌
- `futan_zscore` (SHAP=0.000881) ❌
- `kishu_surface_score` (SHAP=0.000584) ❌
- `kyuyo_kikan` (SHAP=0.000538) ❌
- `chokyoshi_recent_score` (SHAP=0.000428) ❌
- `kohan_3f_index` (SHAP=0.000264) ❌
- `time_index` (SHAP=0.000258) ❌
- `left_direction_score` (SHAP=0.000163) ❌
- `futan_per_barei` (SHAP=0.000081) ❌
- `past_score_mean` (SHAP=0.000080) ❌
- `current_class_score` (SHAP=0.000073) ❌
- `past_score` (SHAP=0.000065) ❌
- `futan_deviation` (SHAP=0.000062) ❌
- `wakuban_ratio` (SHAP=0.000000) ❌
- `umaban_kyori_interaction` (SHAP=0.000000) ❌
- `umaban_percentile` (SHAP=0.000000) ❌
- `class_score_change` (SHAP=0.000000) ❌
- `futan_percentile` (SHAP=0.000000) ❌
- `similar_distance_score` (SHAP=0.000000) ❌
- `current_direction_match` (SHAP=0.000000) ❌

**削除による影響:**
- 特徴量数: 24個 → 1個
- 削減率: 95.8%
- 失われる情報量: 63.75%

**期待効果:**
- 過学習リスク減少
- 学習速度向上
- モデルの解釈性向上

---

### 4️⃣ **累積寄与率分析**

- **累積寄与率 50%**: Top2個の特徴量
- **累積寄与率 70%**: Top4個の特徴量
- **累積寄与率 80%**: Top6個の特徴量
- **累積寄与率 90%**: Top8個の特徴量

**パレートの法則:**
- 上位2個（全体の8.3%）で全体の50%を説明
- 理想的な重要度分布を実現！

---

## 🔥 改善提案

### ✅ すぐできる改善

#### 1. **不要な特徴量を削除(次元削減)**
削除候補(SHAP < 0.005):
- `surface_aptitude_score` (0.002477) ❌
- `past_avg_sotai_chakujun` (0.001629) ❌
- `right_direction_score` (0.001540) ❌
- `kishu_skill_score` (0.000988) ❌
- `futan_zscore` (0.000881) ❌
- `kishu_surface_score` (0.000584) ❌
- `kyuyo_kikan` (0.000538) ❌
- `chokyoshi_recent_score` (0.000428) ❌
- `kohan_3f_index` (0.000264) ❌
- `time_index` (0.000258) ❌
- `left_direction_score` (0.000163) ❌
- `futan_per_barei` (0.000081) ❌
- `past_score_mean` (0.000080) ❌
- `current_class_score` (0.000073) ❌
- `past_score` (0.000065) ❌
- `futan_deviation` (0.000062) ❌
- `wakuban_ratio` (0.000000) ❌
- `umaban_kyori_interaction` (0.000000) ❌
- `umaban_percentile` (0.000000) ❌
- `class_score_change` (0.000000) ❌
- `futan_percentile` (0.000000) ❌
- `similar_distance_score` (0.000000) ❌
- `current_direction_match` (0.000000) ❌

#### 2. **Top3特徴量の強化**

**past_avg_sotai_chakujun強化案:**
- 現在: 直近3走の平均
- 改善: **指数加重平均**(最新レースを重視)
  - 3走前: 重み0.2
  - 2走前: 重み0.3
  - 1走前: 重み0.5

---

## 📈 統計サマリー

- **全特徴量数**: 24個
- **SHAP値合計**: 0.0159
- **SHAP値平均**: 0.0007
- **SHAP値中央値**: 0.0001
- **SHAP値標準偏差**: 0.0013
- **LightGBM Gain相関**: 0.9274

---

## 🎲 次のアクション

### 優先度高(すぐやる)
1. ✅ **23個の不要特徴量を削除**
2. ✅ **Top3特徴量を強化**
3. ✅ **モデル再学習**

### 優先度中(検証後に実施)
4. ⏳ **中位特徴量の改善**（非線形変換、相互作用追加）
5. ⏳ **過去成績参照期間の調整**（3走→5走など）
6. ⏳ **騎手特徴量の精緻化**（競馬場別に分割）

### 優先度低(余裕があれば)
7. 🔮 **騎手×馬の相性特徴量を追加**
8. 🔮 **賞金額ベースの特徴量を追加**

---

## 💡 結論

**SHAP分析から得られた最大の知見:**

> **「relative_abilityが全体の36.3%を占め、他のすべてを圧倒している」**

現在のモデルは:
- ✅ 馬の過去成績を正しく評価できている
- ✅ 騎手の能力も適切に考慮している
- ✅ 斤量の影響も捉えている
- ❌ ノイズ特徴量が多すぎる(24個中23個は不要)
- ❌ Top特徴量の作り方に改善余地あり

**次のステップ:**
1. 不要特徴量を削除して1個に減らす
2. Top3特徴量を強化（指数加重平均など）
3. モデルを再学習して的中率を確認
