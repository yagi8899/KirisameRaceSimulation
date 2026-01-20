# 穴馬予測特徴量 設計ドキュメント 🎯

## 📋 概要

このドキュメントは、回収率向上のために穴馬（高配当馬）を的中させるための特徴量設計をまとめたものです。
**オッズ情報を意図的に使わず**、純粋な実力・状態・適性から穴馬を見抜くことを目指します。

## 🔧 実装方針（2026年1月20日更新）

### SQL実装を基本とする理由

本プロジェクトでは**特徴量をSQL側で計算することを基本方針**とします。

**理由**:
1. **速報予測対応**: オッズ確定前の速報予測でも、過去レースデータから計算した特徴量は利用可能
2. **データ一貫性**: 訓練時と推論時で同じSQLクエリを使用するため、データ不整合のリスクが低い
3. **パフォーマンス**: SQLiteのWINDOW関数・集計を活用し、大量データ処理が効率的
4. **保守性**: `db_query_builder.py`で一元管理され、特徴量追加時の影響範囲が明確

### 実装場所

- **メインSQL**: `db_query_builder.py`の`build_race_data_query()`関数内
- **速報用SQL**: `build_sokuho_race_data_query()`関数（同じ特徴量を使用）
- **Python補完**: SQLで実装困難な特徴量のみ`feature_engineering.py`で計算

### 優先順位の再編成方針

1. **SQL実装が容易**: WINDOW関数・LAGで完結する特徴量を最優先
2. **効果が高い**: UPSET_PREDICTION_FEATURES.mdで「超重要」「重要」に分類された特徴量
3. **前提条件クリア**: 既存カラム（kyuyo_kikan、class_score_changeなど）が利用可能な特徴量

---

## � ランキング学習における穴馬特徴量の役割

### LambdaRankと穴馬予測の関係

本プロジェクトではLightGBMの`lambdarank`を使用しており、これは**リストワイズ学習**の一種。
従来の回帰・分類とは以下の点で根本的に異なる：

| 観点 | 回帰・分類 | ランキング学習（LambdaRank） |
|------|-----------|------------------------------|
| 学習単位 | 個々の馬（サンプル単位） | **レース単位（クエリ単位）** |
| 最適化対象 | 予測値の精度 | **NDCG（順位の並び替え精度）** |
| 比較方法 | 絶対的な評価 | **同一レース内での相対比較** |
| 重要なこと | 各馬の予測スコア | **馬同士の順位付けの正しさ** |

### ランキング学習視点での穴馬特徴量設計

#### 1. レース内相対化の重要性
穴馬特徴量は**同一レース内で差別化できる**ことが重要。
例えば、`past_score_std`（成績のムラ）が高い馬が1頭だけいれば、
その馬は他馬との相対比較で「穴馬候補」として浮上する。

```
NG: 全馬が同じような値を取る特徴量 → 順位付けに貢献しない
OK: レース内でバラつきがある特徴量 → 差別化に貢献
```

#### 2. NDCGへの貢献
NDCGは「上位に正解（好走馬）がいるほど高評価」する指標。
穴馬特徴量は以下を狙う：
- **人気馬と穴馬の分離**: 似たような予測スコアにならないよう差別化
- **穴馬の正確な引き上げ**: 本来3着以内に来る穴馬を上位に持ってくる

#### 3. ペアワイズ比較への影響
LambdaRankは内部で「馬A vs 馬B」のペア比較を行う。
穴馬特徴量は「人気馬 vs 穴馬」の比較で、穴馬の勝ちパターンを学習させる：
- 穴馬の`zenso_agari_rank`が1位 → 人気馬より上位に行く可能性
- 穴馬の`chokyoshi_upset_rate`が高い → 人気馬より好走しやすい

### 特徴量スケールの注意点

ランキング学習でも**特徴量のスケール**は重要：
- 極端に大きい値を取る特徴量があると、他の特徴量の影響が薄れる
- ただし、決定木ベース（LightGBM）なので正規化は必須ではない
- **同一レース内での分散**を意識した設計が効果的

---

## �🎯 穴馬予測の基本方針

### 穴馬の定義
- 人気が低い（相対的に評価されていない）
- しかし実力・適性・状態が良い
- 条件が合えば上位に食い込む馬

### 予測アプローチ
1. **「人気にならない理由」を特徴量化**
   - 成績の不安定性、前走惨敗、展開不利など
2. **「実力が隠れている要素」を発見**
   - 上がり最速、追い込み力、クラス降級など
3. **「条件適性の極端さ」を捉える**
   - この条件だけ強い馬、脚質がハマる展開など

---

## 🔴 超重要（即実装推奨）

### 1. 成績の不安定性（標準偏差・分散）

**目的:** ムラっ気のある馬 = 当たればデカい穴馬候補

**現状の問題:**
- 平均成績しか見ておらず、安定性を評価できていない
- 「平均的に弱い馬」と「たまに激走する馬」が区別できない

**ランキング学習での効果:**
- 🎯 **レース内分散**: 同一レース内で「安定型」と「ムラ型」を分離できる
- 🎯 **ペア比較**: 人気馬（低std）vs 穴馬（高std）で、穴馬が好走するパターンを学習
- 🎯 **NDCG貢献**: 「たまに激走する穴馬」を正しく上位に配置

**追加特徴量:**

#### `past_score_std` (成績スコア標準偏差)
- **計算方法:** 過去5走の成績スコアの標準偏差
- **範囲:** 0.0～0.5程度
- **解釈:**
  - 高値(0.3+): ムラっ気あり → 穴馬候補
  - 低値(0.1-): 安定型 → 本命候補
- **SQL実装:**
```sql
STDDEV(
    (1.0 - cast(seum.kakutei_chakujun as float) / NULLIF(cast(ra.shusso_tosu as float), 0))
    * CASE
        WHEN seum.time_sa LIKE '-%' THEN 1.00
        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 5 THEN 0.85
        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 10 THEN 0.70
        WHEN CAST(REPLACE(seum.time_sa, '+', '') AS INTEGER) <= 20 THEN 0.50
        ELSE 0.20
    END
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
) AS past_score_std
```

#### `past_chakujun_variance` (着順バラツキ)
- **計算方法:** 過去5走の着順の分散
- **範囲:** 0～100程度
- **解釈:**
  - 高値: 1着と15着が混在 → 展開次第で激走
  - 低値: 5-8着が続く → 力通り
- **SQL実装:**
```sql
VARIANCE(cast(seum.kakutei_chakujun as float)) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
) AS past_chakujun_variance
```

**期待効果:**
- 穴馬的中率: +15～20%
- 誤検知: 調子の悪い人気馬を穴馬と誤認する可能性（過去成績との組み合わせで回避）

---

### 2. 前走の上がり3F順位

**目的:** 「上がり最速なのに負けた馬」= 展開が向かなかっただけ

**現状の問題:**
- 上がり3Fタイムはあるが、レース内での相対順位を見ていない
- 「上がり1位で5着」の馬を見逃している

**ランキング学習での効果:**
- 🎯 **レース内差別化**: 「上がり1位だった馬」vs「上がり10位だった馬」で明確な差
- 🎯 **隠れた実力の発見**: 着順は悪いが上がりが速い → 他馬との相対比較で浮上
- 🎯 **NDCGへの貢献**: 前走展開不利の馬を正しく上位に引き上げる

**追加特徴量:**

#### `zenso_agari_rank` (前走上がり3F順位)
- **計算方法:** 前走レース内での上がり3F順位
- **範囲:** 1～18程度
- **解釈:**
  - 1-3位: 脚が使えている → 展開次第で激走
  - 15位以降: 脚が使えていない → 要注意
- **SQL実装:**
```sql
LAG(
    RANK() OVER (
        PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango
        ORDER BY CASE 
            WHEN seum.kohan_3f = '000' OR seum.kohan_3f = '999' THEN 9999 
            ELSE cast(seum.kohan_3f as integer) 
        END
    )
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
) AS zenso_agari_rank
```

#### `zenso_agari_gap` (前走上がり順位 vs 着順ギャップ)
- **計算方法:** 前走の(最終着順 - 上がり3F順位)
- **範囲:** -15～+15程度
- **解釈:**
  - 正の値: 上がりは良いのに着順が悪い → 展開不利で負けた
  - 負の値: 上がりは悪いのに着順が良い → まぐれ？
- **Python計算:**
```python
df['zenso_agari_gap'] = df.groupby('ketto_toroku_bango').apply(
    lambda x: x['kakutei_chakujun_numeric'].shift(1) - x['agari_rank'].shift(1)
).values
```

**期待効果:**
- 穴馬的中率: +10～15%
- 特に差し・追い込み型の穴馬を発見

---

### 3. 前走4コーナー位置 vs 最終着順（追い込み力）

**目的:** 「後方から追い込んで来た馬」= 展開次第で激走

**現状の問題:**
- corner_4はあるが、最終着順との比較をしていない
- 追い込み型の馬を評価できていない

**ランキング学習での効果:**
- 🎯 **脚質による差別化**: 同一レース内で「逃げ型」vs「追込型」を分離
- 🎯 **展開依存性の学習**: 追込力が高い馬はペース次第で順位が変動 → ペア比較で学習
- 🎯 **隠れたポテンシャル発見**: 着順は悪いが追込力がある馬を上位に引き上げ

**追加特徴量:**

#### `zenso_oikomi_power` (前走追い込み力)
- **計算方法:** 前走の(4コーナー位置 - 最終着順)
- **範囲:** -15～+15程度
- **解釈:**
  - 正の値(+5以上): 強い追い込み → 展開が向けば激走
  - 0付近: 位置取り変わらず → 力通り
  - 負の値: 逃げ・先行型 or 脚が止まった
- **SQL実装:**
```sql
LAG(
    cast(seum.corner_4 as float) - cast(seum.kakutei_chakujun as float)
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
) AS zenso_oikomi_power
```

#### `avg_oikomi_power` (平均追い込み力)
- **計算方法:** 過去5走の平均追い込み力
- **範囲:** -5～+5程度
- **解釈:**
  - 正の値: 追い込み馬 → スローペースで激走
  - 負の値: 逃げ・先行馬 → ハイペースで沈む
- **SQL実装:**
```sql
AVG(
    cast(seum.corner_4 as float) - cast(seum.kakutei_chakujun as float)
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
) AS avg_oikomi_power
```

**期待効果:**
- 穴馬的中率: +10～15%
- ペース展開が合う馬を事前に発見

---

### 4. 条件限定型スコア（極端な適性）

**目的:** 「この条件だけ異常に強い馬」= 条件が合えば激走

**現状の問題:**
- 平均的な適性しか見ていない
- 「芝は弱いがダートは強い」のようなギャップを捉えられない

**ランキング学習での効果:**
- 🎯 **条件マッチング**: 今日の条件（芝/ダート、距離、馬場）で「この馬だけ得意」を発見
- 🎯 **レース内差別化**: 同一条件でgapが大きい馬を他馬と分離
- 🎯 **ペア比較最適化**: オールラウンダー vs 条件限定型のパターン学習
- 💡 **実装のコツ**: 現在のレース条件との相性スコアを計算し、レース内での相対値を活用

**追加特徴量:**

#### `turf_vs_dirt_gap` (芝ダートスコア差)
- **計算方法:** |芝での平均スコア - ダートでの平均スコア|
- **範囲:** 0.0～0.5程度
- **解釈:**
  - 高値(0.3+): 路面による差が大きい → 条件が合えば激走
  - 低値(0.1-): オールラウンダー
- **Python計算:**
```python
turf_score = df[df['track_code'].between('10', '22')].groupby('ketto_toroku_bango')['chakujun_score'].mean()
dirt_score = df[df['track_code'].between('23', '29')].groupby('ketto_toroku_bango')['chakujun_score'].mean()
df['turf_vs_dirt_gap'] = abs(turf_score - dirt_score)
```

#### `short_vs_long_gap` (短距離vs長距離スコア差)
- **計算方法:** |短距離平均スコア - 長距離平均スコア|
- **範囲:** 0.0～0.5程度
- **解釈:**
  - 高値: 距離による差が大きい → 得意距離では激走
  - 低値: 距離対応力が高い

#### `heavy_baba_specialist` (重馬場スペシャリスト)
- **計算方法:** 重馬場での勝率 - 良馬場での勝率
- **範囲:** -0.3～+0.3程度
- **解釈:**
  - 正の値: 重馬場で強い → 雨の日は狙い目
  - 負の値: 良馬場専門

**期待効果:**
- 穴馬的中率: +8～12%
- 条件が合った時の取りこぼしを減少

---

## 🟡 重要（優先的に実装）

### 5. 騎手変更の効果

**目的:** 「無名騎手→有力騎手」で激走パターン

**ランキング学習での効果:**
- 🎯 **レース内差別化**: 騎手変更した馬vs継続騎乗の馬で明確に分離
- 🎯 **組み合わせ学習**: 騎手変更×前走着順×厩舎力で複合的なパターンを捉える
- 💡 **0/1フラグの注意**: バイナリ特徴量はスプリット点が明確 → ランキングに貢献しやすい

**追加特徴量:**

#### `kishu_changed` (騎手変更フラグ)
- **計算方法:** 前走と騎手が変わったか（0 or 1）
- **SQL実装:**
```sql
CASE 
    WHEN seum.kishu_code != LAG(seum.kishu_code) OVER (
        PARTITION BY seum.ketto_toroku_bango
        ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    ) THEN 1 
    ELSE 0 
END AS kishu_changed
```

#### `kishu_upgrade` (騎手格上げフラグ)
- **計算方法:** 前走騎手より今回騎手の方がスコアが高いか
- **解釈:** 1 = 厩舎が本気、0 = 通常 or 格下げ

**期待効果:**
- 穴馬的中率: +5～10%
- 「厩舎の本気度」を間接的に評価

---

### 6. 調教師の穴馬仕上げ率

**目的:** 「穴をよく出す厩舎」の馬は狙い目

**ランキング学習での効果:**
- 🎯 **厩舎単位の差別化**: 同一レース内で「穴馬を出す厩舎」vs「堅実な厩舎」を分離
- 🎯 **過去実績による重み付け**: upset_rateが高い馬は相対的に上位配置されやすい
- 💡 **レース横断情報**: 厩舎の傾向は全レースで共通 → 強い一般化力を持つ特徴量

**追加特徴量:**

#### `chokyoshi_upset_rate` (調教師穴馬率)
- **計算方法:** 過去50走で「前走10着以下→今回3着以内」の割合
- **範囲:** 0.0～0.3程度
- **解釈:**
  - 高値(0.15+): 穴馬を仕上げるのが上手い厩舎
  - 低値(0.05-): 堅実な厩舎
- **SQL実装:**
```sql
COUNT(
    CASE 
        WHEN cast(seum.kakutei_chakujun as integer) <= 3 
        AND LAG(cast(seum.kakutei_chakujun as integer)) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
        ) >= 10
        THEN 1 
    END
) OVER (
    PARTITION BY seum.chokyoshi_code
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
) / 50.0 AS chokyoshi_upset_rate
```

**期待効果:**
- 穴馬的中率: +5～8%

---

### 7. 休養明け × 前走惨敗の組み合わせ

**目的:** 「前走大敗→休養→復帰」= 立て直しパターン

**前提条件:** 休養期間（kyuyo_kikan）の実装が必要

**ランキング学習での効果:**
- 🎯 **交互作用の学習**: 休養期間×前走着順の組み合わせでペア比較最適化
- 🎯 **レース内差別化**: 休養明け馬vs連戦馬で差をつける
- 💡 **2値フラグの効果**: 決定木が「この条件が揃えば上位」と明確に判断できる

**追加特徴量:**

#### `kyuyo_after_bad_race` (休養明け×前走惨敗フラグ)
- **計算方法:** 前走が10着以下 かつ 休養90日以上（0 or 1）
- **解釈:** 1 = 立て直して復帰の可能性
- **SQL実装:**
```sql
CASE 
    WHEN kyuyo_kikan >= 90 
    AND LAG(cast(seum.kakutei_chakujun as integer)) OVER (
        PARTITION BY seum.ketto_toroku_bango
        ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    ) >= 10
    THEN 1 
    ELSE 0 
END AS kyuyo_after_bad_race
```

**期待効果:**
- 穴馬的中率: +5～8%
- 「叩いて良化」パターンを捉える

---

### 8. 前走での位置取り不利（包まれ判定）

**目的:** 「道中で不利を受けた馬」= 次走で巻き返し

**ランキング学習での効果:**
- 🎯 **不利の可視化**: 前走で包まれた馬を明確に分離
- 🎯 **NDCGへの貢献**: 「実力はあるが前走不利」の馬を正しく上位に配置
- 💡 **レース再現性**: 同様の展開パターンでの順位変動を学習

**追加特徴量:**

#### `zenso_kakoi_komon` (前走包まれ度)
- **計算方法:** 前走の(2コーナー位置 - 4コーナー位置)
- **範囲:** -10～+10程度
- **解釈:**
  - 正の値(+3以上): 包まれた可能性 → 次走巻き返し
  - 0付近: スムーズ
  - 負の値: 先団キープ or 突っ込んだ
- **SQL実装:**
```sql
LAG(
    cast(seum.corner_2 as float) - cast(seum.corner_4 as float)
) OVER (
    PARTITION BY seum.ketto_toroku_bango
    ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
) AS zenso_kakoi_komon
```

**期待効果:**
- 穴馬的中率: +3～5%

---

### 9. クラス降級直後

**目的:** 「格下になった馬」= 人気にならないが実力上位

**前提条件:** クラス変動（class_change）の実装が必要

**ランキング学習での効果:**
- 🎯 **明確な分離**: 降級馬vs非降級馬で決定的な差
- 🎯 **相対的実力の発見**: 同一レース内で「実は格上経験あり」の馬を浮上させる
- 🎯 **NDCGへの直接貢献**: 降級馬が好走しやすい傾向を上位配置に反映
- 💡 **高いインパクト**: クラス差は他の特徴量より強い予測力を持つことが多い

**追加特徴量:**

#### `class_downgrade` (クラス降級フラグ)
- **計算方法:** 前走よりクラスが下がったか（1 or 0）
- **解釈:** 1 = 格下で有利
- **SQL実装:**
```sql
CASE 
    WHEN cast(ra.kyoso_joken_code as integer) < LAG(cast(ra.kyoso_joken_code as integer)) OVER (
        PARTITION BY seum.ketto_toroku_bango
        ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
    )
    THEN 1 
    ELSE 0 
END AS class_downgrade
```

**期待効果:**
- 穴馬的中率: +8～12%
- 降級馬は格上で激走しやすい

---

## 🟢 補助的（余裕があれば）

### 10. 同じ条件での最高着順

**ランキング学習での効果:**
- 🎯 **ポテンシャル発見**: 「過去に同条件で好走」の馬を相対的に引き上げ
- 💡 **補助的な役割**: メイン特徴量と組み合わせて効果を発揮

**追加特徴量:**

#### `best_chakujun_similar_condition` (同条件最高着順)
- **計算方法:** 同一距離帯・同一路面での最高着順
- **範囲:** 1～18
- **解釈:** 1-3着経験あり = ポテンシャルあり

**期待効果:** +2～5%

---

### 11. 馬体重の急激な変化

**前提条件:** 馬体重データの実装が必要

**ランキング学習での効果:**
- 🎯 **状態変化の検出**: 体重変動で「調整失敗or成長」を他馬と比較
- 💡 **ノイズに注意**: 単体では効果薄、他特徴量との組み合わせで活用

**追加特徴量:**

#### `bataiju_drastic_change` (馬体重急変フラグ)
- **計算方法:** 前走から±15kg以上変化（0 or 1）
- **解釈:**
  - -15kg以上: 調整失敗 or ダイエット成功
  - +15kg以上: 太め残り or 成長期

**期待効果:** +3～5%

---

## 📊 実装優先順位（SQL実装難易度順・2026/1/20改訂）

### フェーズ1: SQL実装容易 & 効果大（即実装推奨）

実装難易度: ⭐ / 期待効果: 🔥🔥🔥

| # | 特徴量名 | SQL実装方法 | 期待効果 | 実装状況 |
|---|---------|-----------|----------|----------|
| 1 | `past_score_std` | STDDEV() OVER (ROWS 5 PRECEDING) | 成績ムラで穴馬検出 +15% | ✅ 実装済み（NULL→-1に変更） |
| 2 | `past_chakujun_variance` | VARIANCE() OVER (ROWS 5 PRECEDING) | 着順ムラで穴馬検出 +15% | ✅ 実装済み（NULL→-1に変更） |
| 3 | `zenso_oikomi_power` | LAG(corner_4 - chakujun) | 追い込み力で展開依存検出 +10% | ✅ 実装済み（重要度169K, rank 5） |
| 4 | `kishu_changed` | LAG(kishu_code) != kishu_code | 騎手変更で厩舎本気度 +5% | ❌ **削除済み（重要度1,923, 効果なし）** |
| 5 | `class_downgrade` | LAG(kyoso_joken_code) > kyoso_joken_code | クラス降級で実力差検出 +10% | ❌ **削除済み（重要度112, 効果なし）** |
| 6 | `zenso_kakoi_komon` | LAG(corner_2 - corner_4) | 前走包まれで不利検出 +3% | ✅ 実装済み |

**実装理由**: 単純なWINDOW関数・LAG処理で完結し、訓練・速報の両方で利用可能

---

### フェーズ2: SQL実装やや複雑 & 効果中（優先実装）

実装難易度: ⭐⭐ / 期待効果: 🔥🔥

| # | 特徴量名 | SQL実装方法 | 期待効果 |
|---|---------|-----------|----------|
| 7 | `zenso_agari_rank` | RANK() OVER (ORDER BY kohan_3f) → LAG | 前走上がり最速検出 +10% |
| 8 | `zenso_agari_gap` | LAG(chakujun - agari_rank) | 上がり良いのに負けた馬 +10% |
| 9 | `avg_oikomi_power` | AVG(corner_4 - chakujun) OVER (ROWS 5 PRECEDING) | 平均追い込み力 +5% |
| 10 | `kyuyo_after_bad_race` | (kyuyo_kikan >= 90) AND (LAG(chakujun) >= 10) | 休養明けの立て直し +5% |

**実装理由**: 2段階の集計やサブクエリが必要だが、SQL内で完結可能

---

### フェーズ3: SQL実装困難 or 効果不明（後回し）

実装難易度: ⭐⭐⭐ / 期待効果: 🔥

| # | 特徴量名 | SQL実装方法 | 備考 |
|---|---------|-----------|------|
| 11 | `turf_vs_dirt_gap` | 条件別集計 → ABS(turf_avg - dirt_avg) | 条件分岐が複雑 |
| 12 | `short_vs_long_gap` | 距離帯別集計 | 条件分岐が複雑 |
| 13 | `kishu_upgrade` | 騎手スコアテーブル必要 | 騎手スコアDB化が前提 |
| 14 | `chokyoshi_upset_rate` | 調教師別穴馬集計 | 複雑な条件集計 |
| 15 | `heavy_baba_specialist` | 馬場状態別集計 | 条件分岐が複雑 |
| 16 | `best_chakujun_similar_condition` | 条件マッチング | 実装困難 |
| 17 | `bataiju_drastic_change` | ABS(LAG(bataiju) - bataiju) > 15 | 馬体重データ不足 |

**実装判断**: フェーズ1・2実装後のPrecision次第で追加検討

---

### フェーズ3.5: 次回実装予定（2026年1月20日）

実装難易度: ⭐⭐ / 期待効果: 🔥🔥

**背景:** Phase 1実装後の特徴量重要度分析により、kishu_changed (1,923) と class_downgrade (112) の効果が極めて低いことが判明。これらを削除し、より効果的な特徴量に置き換える。

| # | 特徴量名 | SQL実装方法 | 期待効果 | 実装予定 |
|---|---------|-----------|----------|----------|
| 1 | `zenso_ninki_gap` | LAG(popularity_rank - kakutei_chakujun) | 前走人気と着順のギャップで過小評価馬検出 +10% | 🔄 実装中 |
| 2 | `zenso_nigeba` | LAG(corner_1 == 1) | 前走逃げ成功で展開依存性検出 +5% | 🔄 実装中 |
| 3 | `zenso_taihai` | LAG(kakutei_chakujun > 10) | 前走大敗後の巻き返し検出 +5% | 🔄 実装中 |
| 4 | `zenso_agari_rank` | RANK() OVER (ORDER BY kohan_3f) → LAG | 前走上がり最速で隠れた実力検出 +10% | 🔄 実装中 |
| 5 | `saikin_kaikakuritsu` | COUNT(改善レース) / 3走 | 直近3走の成績改善傾向 +8% | 🔄 実装中 |

**削除予定の既存特徴量:**

| # | 特徴量名 | 削除理由 |
|---|---------|----------|
| 1 | `wakuban_inner` | 短距離専用特徴量、汎用モデルには不要 |
| 2 | `wakuban_outer` | 短距離専用特徴量、汎用モデルには不要 |
| 3 | `estimated_running_style` | 推定値でノイズが多い |
| 4 | `tenko_code` | 天候コードの効果不明 |
| 5 | `distance_change` | 距離変化は他の特徴量で捉えられる |

**特徴量数の変化:**
- 現在: 28個 (Phase 3: 6個)
- 削除: -7個 (Phase 3で2個、既存で5個)
- 追加: +5個 (Phase 3.5)
- **合計: 26個** (Phase 3関連: 4個 → 9個)

---

### 実装スケジュール（2026年1月〜3月）

**Week 1 (完了)**: フェーズ1（6特徴量）をSQL実装 → 再訓練 → Precision評価  
- ✅ past_score_std, past_chakujun_variance, zenso_oikomi_power, zenso_kakoi_komon 実装済み
- ❌ kishu_changed, class_downgrade は効果なしのため削除
- 📊 結果: Precision 1.78% max（目標8%未達）

**Week 2 (進行中)**: Phase 3.5（5特徴量）を実装  
- 🔄 zenso_ninki_gap, zenso_nigeba, zenso_taihai, zenso_agari_rank, saikin_kaikakuritsu
- 🗑️ 既存特徴量7個を削除（wakuban_inner/outer, estimated_running_style, tenko_code, distance_change など）
- 🎯 目標: Precision 8%達成

**Week 3-4**: Precision 8%未達ならフェーズ2（4特徴量）を追加実装  
**Month 2-3**: Precision達成後、フェーズ3は効果検証してから判断

---

## 🎯 穴馬予測モデルの設計戦略（ランキング学習視点）

### LambdaRankで穴馬を拾うアプローチ

現在のモデルは`lambdarank`（NDCG最適化）を使用しているため、
穴馬予測は「穴馬を上位に引き上げる」=「穴馬のランキングスコアを高くする」がゴール。

### 戦略A: 既存モデルに穴馬特徴量を追加（推奨）
- 本ドキュメントの特徴量を追加して、モデル自体が穴馬も拾えるように
- **ランキング学習との相性◎**: 同一レース内で穴馬を差別化できれば自動的に上位に
- メリット: 1つのモデルで完結、lambdarankの恩恵を最大化
- デメリット: 特徴量バランスの調整が必要

### 戦略B: 2段階ランキング（高度）
1. 第1段階: 本命予測（通常のlambdarank）
2. 第2段階: 穴馬リランキング（期待値ベースの調整）
- **ポスト処理で期待値を考慮**
- メリット: 着順予測と回収率最大化を分離
- デメリット: システムが複雑

### 戦略C: 専用穴馬検出モデル（分類）
- 目的ラベルを変更: `is_upset = (chakujun <= 3 AND relative_popularity低)`
- 穴馬を当てる**分類モデル**を別途作成
- **ランキング学習ではない**ので、本プロジェクトとは別ライン
- メリット: 穴馬検出に特化
- デメリット: ランキング情報を活用できない

**推奨:** 戦略A（既存ランキングモデル + 穴馬特徴量追加）

#### 戦略Aの実装ポイント
1. **穴馬特徴量を追加**: past_score_std, zenso_agari_rank, class_downgrade など
2. **NDCG評価で効果検証**: 穴馬が3着以内に来るレースでNDCGが改善するか
3. **SHAP分析**: 穴馬特徴量がどの程度予測に貢献しているか確認
4. **回収率で最終評価**: 最終的にはシミュレーションで回収率改善を確認

---

## 📈 期待効果

### 現状（ベースライン）
- 穴馬的中率: 5～10%程度（推定）
- 回収率: 70～85%

### フェーズ1実装後
- 穴馬的中率: 20～25%（+15%）
- 回収率: 85～100%（+15%）

### 全特徴量実装後
- 穴馬的中率: 35～45%（+30～35%）
- 回収率: 110～130%（+40～45%）

---

## 🔄 変更履歴

### 2026-01-20
- ✅ Phase 1実装完了と効果分析
  - past_score_std, past_chakujun_variance: NULL→-1に変更
  - zenso_oikomi_power: 重要度169K (rank 5)、効果確認
  - kishu_changed (重要度1,923), class_downgrade (重要度112): 削除決定
- 📋 Phase 3.5計画策定
  - 追加: zenso_ninki_gap, zenso_nigeba, zenso_taihai, zenso_agari_rank, saikin_kaikakuritsu
  - 削除: wakuban_inner/outer, estimated_running_style, tenko_code, distance_change (7個)
  - 特徴量数: 28個 → 26個
- 🎯 目標: Precision 1.78% → 8%達成

### 2026-01-19
- ランキング学習（LambdaRank）視点での説明を追加
- 各特徴量に「ランキング学習での効果」セクションを追加
- 設計戦略をランキング学習に適合する形に修正

### 2026-01-18
- 初版作成
- 穴馬予測特徴量11種を定義
- 実装優先順位を3フェーズに分類
- SQL実装例を追加

---

**最終更新:** 2026年1月20日  
**作成者:** GitHub Copilot  
**関連ドキュメント:** [FEATURE_LIST.md](FEATURE_LIST.md), [README.md](README.md)
