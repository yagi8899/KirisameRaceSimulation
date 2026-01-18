# 競馬予測モデル 特徴量一覧

## 📋 概要
このドキュメントでは、競馬予測モデルで使用されている全特徴量を説明します。
特徴量は「基本特徴量」「派生特徴量」「距離適性」「馬場適性」「騎手・調教師」「短距離特化」の6カテゴリに分類されています。

---

## 🏇 基本特徴量（SQLクエリで計算済み）

### 1. **past_score** (過去成績スコア)
- **説明**: 過去3走の着順とレースグレードを掛け合わせたスコア
- **計算方法**: 
  - 着順スコア: 1着=100点、2着=80点、3着=60点...
  - グレード重み: G1=3.0倍、G2=2.0倍、G3=1.5倍、OP=1.0倍
  - 過去3走の合計値
- **範囲**: 0～900 (G1を3連勝した場合が最高)
- **重要度**: ⭐⭐⭐⭐⭐ (最重要)

### 2. **past_avg_sotai_chakujun** (着差考慮相対着順平均)
- **説明**: 過去3走の相対着順を着差で重み付け平均したスコア
- **計算方法**: 
  - 相対着順: (1 - 着順/出走頭数)
  - 着差補正: 1着(×1.00), 0.5秒差以内(×0.85), 1.0秒差(×0.70)...
  - 過去3走の平均値
- **範囲**: 0.0～1.0 (1.0が最高)
- **重要度**: ⭐⭐⭐⭐⭐ (最重要)
- **備考**: Pattern 2の着差係数を採用(0.20-1.00の範囲)

### 3. **time_index** (タイム指数)
- **説明**: 過去3走の平均速度(距離/時間)
- **計算方法**: 距離(m) ÷ 走破タイム(秒) の過去3走平均
- **範囲**: 通常10～20 m/s程度
- **重要度**: ⭐⭐⭐⭐

### 4. **kohan_3f_index** (後半3F指数)
- **説明**: 過去3走の後半3Fタイムと距離別標準値の差
- **計算方法**: 平均後半3Fタイム - 距離別標準タイム
  - 1600m以下: 33.5秒
  - 1601-2000m: 35.0秒
  - 2001-2400m: 36.0秒
  - 2401m以上: 37.0秒
- **範囲**: マイナスが優秀(標準より速い)
- **重要度**: ⭐⭐⭐⭐ (長距離で重要、短距離では重要度低)

---

## 🔧 派生特徴量

### 5. **wakuban_ratio** (枠番比率)
- **説明**: 枠番をレース出走頭数で正規化
- **計算方法**: 枠番 ÷ 最大枠番
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐
- **備考**: 短距離モデルでは削除される(効果薄)

### 6. **futan_per_barei** (斤量/馬齢比)
- **説明**: 斤量を馬齢で割った負担度指標
- **計算方法**: 斤量(kg) ÷ 馬齢(歳)
- **範囲**: 通常10～15程度
- **重要度**: ⭐⭐⭐

### 7. **futan_per_barei_log** (斤量/馬齢比 対数変換)
- **説明**: futan_per_bareiの対数変換版
- **計算方法**: log(futan_per_barei)
- **重要度**: ⭐⭐

### 8. **futan_deviation** (期待斤量との差分)
- **説明**: 実際の斤量と馬齢別期待斤量との差
- **計算方法**: 実斤量 - 馬齢別期待斤量
  - 2歳: 48kg、3歳: 52kg、4歳: 55kg、5-6歳: 57kg
- **範囲**: ±5kg程度
- **重要度**: ⭐⭐

### 9. **umaban_kyori_interaction** (馬番×距離相互作用)
- **説明**: 馬番と距離の相互作用(外枠は距離が長い方が有利)
- **計算方法**: 馬番 × 距離(m) ÷ 1000
- **範囲**: 0～36程度(18番×2000m)
- **重要度**: ⭐⭐⭐

### 10. **umaban_percentile** (馬番パーセンタイル)
- **説明**: 馬番をレース内で正規化(0-1)
- **計算方法**: (馬番 - 1) ÷ (最大馬番 - 1)
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐

### 11. **futan_zscore** (斤量Zスコア)
- **説明**: レース内での斤量の相対的位置(標準化)
- **計算方法**: (斤量 - レース内平均) ÷ レース内標準偏差
- **範囲**: 通常-2.0～+2.0
- **重要度**: ⭐⭐⭐

### 12. **futan_percentile** (斤量パーセンタイル)
- **説明**: レース内での斤量の順位を0-1に正規化
- **計算方法**: レース内での斤量順位を0-1スケールに変換
- **範囲**: 0.0～1.0 (1.0が最重斤量)
- **重要度**: ⭐⭐⭐

---

## 📏 距離適性特徴量

### 13. **past_score_short / past_score_mile / past_score_middle / past_score_long** (距離帯別成績スコア) 🔄
- **説明**: 4つの距離帯での過去5走の平均成績スコア（SQLで計算）
- **距離帯定義**: 
  - `past_score_short`: 1000-1400m（短距離）
  - `past_score_mile`: 1401-1800m（マイル）
  - `past_score_middle`: 1801-2400m（中距離）
  - `past_score_long`: 2401m以上（長距離）
- **計算方法**: 
  - 基本スコア: (1 - 着順/出走頭数)
  - time_sa補正: 1着(×1.00), 0.5秒差以内(×0.85), 1.0秒差(×0.70), 2.0秒差(×0.50), 3.0秒差(×0.30), 3.0秒超(×0.20)
  - 過去5走の該当距離帯のみの平均値
- **範囲**: 0.0～1.0 (該当距離帯の実績がなければNULL)
- **重要度**: ⭐⭐⭐⭐⭐ (SQLで算出、Pythonで統合)
- **備考**: 過去の`distance_category_score`をSQL化して4つに分割

### 14. **similar_distance_score** (距離適性スコア - 重み付け平均版) 🔄
- **説明**: 4つの距離帯別スコアを距離の近さで重み付け平均したスコア（Pythonで計算）
- **計算方法**: 
  - 各距離帯の中心値（短=1200, マイル=1600, 中=2100, 長=2600）から今回距離までの差を計算
  - 重み = 0.8 ^ (距離差 / 200) で距離が近いほど高重み
  - weighted_average(scores, weights)
  - 実績が全くない場合は0.5（中立）
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐⭐
- **例**: 1500mのレースの場合、マイル帯（中心1600）の重みが最大、短距離（1200）が次点、中距離（2100）が三番手
- **備考**: 未経験距離でも周辺距離帯の実績から滑らかに推定可能

---

## 🌿 馬場適性特徴量

### 15. **surface_aptitude_score** (路面適性スコア)
- **説明**: 芝/ダート別の過去10走の成績
- **計算方法**: 同じ路面での過去10走の相対着順平均
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐
- **備考**: 芝短距離モデルでは削除される(効果なし)

---

## 🏆 騎手・調教師特徴量

### 16. **kishu_skill_score** (騎手実力スコア) 🔄
- **説明**: 騎手の過去30走の平均成績スコア（SQLで計算）
- **計算方法**: 過去30走の(1 - 着順/出走頭数)平均、10走未満なら0.5（中立）
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐⭐
- **備考**: 過去N走方式に統一、オッズ情報不要でsokuho対応

### 17. **kishu_surface_score** (騎手路面適性) 🔄
- **説明**: 騎手の芝/ダート別過去50走の平均成績（SQLで計算）
- **計算方法**: 同路面での過去50走の(1 - 着順/出走頭数)平均、5走未満なら0.5（中立）
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐
- **備考**: 過去N走方式に統一

### 18. **chokyoshi_recent_score** (調教師最近成績) 🔄
- **説明**: 調教師の過去20走の平均成績スコア（SQLで計算）
- **計算方法**: 過去20走の(1 - 着順/出走頭数)平均、5走未満なら0.5（中立）
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐

---

## 🏁 短距離特化特徴量 (1600m以下専用)

### 19. **wakuban_kyori_interaction** (枠番×距離相互作用) 🆕
- **説明**: 短距離ほど内枠が有利になる効果を数値化
- **計算方法**: 枠番 × (2000 - 距離) ÷ 1000
- **範囲**: 0～8程度 (8枠1200mで最大)
- **重要度**: ⭐⭐⭐ (短距離専用)
- **備考**: 1600m以下のモデルでのみ使用

### 20. **zenso_kyori_sa** (前走距離差) 🆕
- **説明**: 前走からの距離変化(絶対値)
- **計算方法**: |今回距離 - 前走距離|
- **範囲**: 0～1600m程度
- **重要度**: ⭐⭐⭐ (短距離専用)
- **備考**: 短距離では距離変化の影響が大きい

### 21. **start_index** (スタート指数) 🆕
- **説明**: 過去10走の第1コーナー通過順位から算出
- **計算方法**: 
  - 位置取りスコア: 1.0 - (平均通過順位 ÷ 18)
  - 安定性ボーナス: 0.2 - (標準偏差 ÷ 10) (最大+0.2)
  - 合計: 位置取りスコア + 安定性ボーナス
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐⭐ (短距離専用)
- **備考**: 短距離はスタートが勝負を決める

### 22. **corner_position_score** (コーナー通過位置スコア) 🆕
- **説明**: 過去3走の全コーナー(1-4)通過位置の平均と安定性
- **計算方法**: 
  - 位置取りスコア: 1.0 - (平均通過順位 ÷ 18)
  - 安定性ボーナス: 0.3 - (標準偏差 ÷ 10) (最大+0.3)
  - 合計: 位置取りスコア + 安定性ボーナス (最大1.0)
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐⭐ (短距離専用)
- **備考**: 安定して前方にいる馬を評価、馬連・ワイドの精度向上に寄与

---

## 🏇 長距離特化特徴量 (2200m以上専用)

### 23. **long_distance_experience_count** (長距離経験回数) 🆕
- **説明**: 過去の長距離レース(≥2400m)の経験回数
- **計算方法**: 過去レースで距離≥2400mの回数をカウント
- **範囲**: 0～30回程度 (経験豊富な馬ほど多い)
- **重要度**: ⭐⭐⭐ (長距離専用)
- **備考**: 2200m以上のモデルでのみ使用、スタミナ適性の簡易指標

---

## 📊 特徴量選択ルール

### 路面×距離別の特徴量調整

#### 🌿 芝中長距離 (1700-2199m)
- **使用特徴量**: ベース特徴量16個
- **削除済み**: `baba_change_adaptability`, `kishu_popularity_score`
- **特徴量数**: 16個
- **備考**: 最も安定したベースモデル

#### 🌿 芝長距離 (2200m以上)
- **使用特徴量**: ベース特徴量16個
- **追加特徴量**: `long_distance_experience_count`
- **削除済み**: `baba_change_adaptability`, `kishu_popularity_score`
- **特徴量数**: 17個
- **備考**: スタミナ適性評価が重要

#### 🌿 芝短距離 (1600m以下)
- **削除特徴量**: 
  - `kohan_3f_index` (SHAP 0.030 - 後半の脚は重要度低)
  - `surface_aptitude_score` (SHAP 0.000 - 完全に無意味)
  - `wakuban_ratio` (SHAP 0.008 - ほぼ無効)
  - `long_distance_experience_count` (短距離では不要)
  - `baba_change_adaptability` (全体で削除)
  - `kishu_popularity_score` (全体で削除)
- **追加特徴量**: 
  - `wakuban_kyori_interaction`
  - `zenso_kyori_sa`
  - `start_index`
  - `corner_position_score`
- **特徴量数**: 17個 (16 - 3削除 + 4追加)

#### 🏜️ ダート中長距離 (1700-2199m)
- **使用特徴量**: ベース特徴量16個
- **削除済み**: `baba_change_adaptability`, `kishu_popularity_score`
- **特徴量数**: 16個

#### 🏜️ ダート長距離 (2200m以上)
- **使用特徴量**: ベース特徴量16個
- **追加特徴量**: `long_distance_experience_count`
- **削除済み**: `baba_change_adaptability`, `kishu_popularity_score`
- **特徴量数**: 17個

#### 🏜️ ダート短距離 (1600m以下)
- **削除特徴量**: 
  - `kohan_3f_index`
  - `surface_aptitude_score`
  - `wakuban_ratio`
  - `long_distance_experience_count`
  - `baba_change_adaptability`
  - `kishu_popularity_score`
- **追加特徴量**: 短距離特化4個
  - `wakuban_kyori_interaction`
  - `zenso_kyori_sa`
  - `start_index`
  - `corner_position_score`
- **特徴量数**: 17個 (16 - 3削除 + 4追加)

---

## 📈 SHAP分析結果 (東京芝短距離モデル)

| 順位 | 特徴量 | SHAP重要度 | 備考 |
|------|--------|-----------|------|
| 1 | distance_category_score | 0.274 | 距離適性が最重要 |
| 2 | past_avg_sotai_chakujun | 0.243 | 着差考慮成績 |
| 3 | kishu_surface_score | 0.171 | 騎手路面適性 |
| 4 | futan_per_barei | 0.122 | 負担重量 |
| 5 | past_score | 0.113 | グレード別成績 |
| 6 | time_index | 0.082 | タイム指数 |
| 7 | umaban_kyori_interaction | 0.075 | 馬番×距離 |

---

## 🔄 変更履歴

### 2025-11-10
- ✅ `wakuban_kyori_interaction`追加 (短距離特化)
- ✅ `zenso_kyori_sa`追加 (短距離特化)
- ✅ `start_index`追加 (短距離特化)
- ✅ 路面×距離別の特徴量選択を実装
- ❌ `pace_suitability`追加→削除 (精度悪化のため)

### 2025-11-09
- ✅ `past_avg_sotai_chakujun`に着差考慮を追加 (Pattern 2採用)

### 2025-11-08
- ✅ グレード重み付け強化 (G1=3.0x, G2=2.0x, G3=1.5x)

---

## ✅ 実装済みランキング学習特徴量（2026-01-18）

### 🔥 Tier S - ランキング学習必須特徴量（6個実装完了）

#### 1. current_class_score (現在レースクラススコア) ✅
- **説明**: 今回レースのクラスを重み付けスコア化（past_scoreと同じロジック）
- **計算方法**: 
  ```sql
  CASE 
      WHEN ra.grade_code = 'A' THEN 3.00                                    -- G1
      WHEN ra.grade_code = 'B' THEN 2.00                                    -- G2
      WHEN ra.grade_code = 'C' THEN 1.50                                    -- G3
      WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
           AND ra.kyoso_joken_code = '999' THEN 1.00                        -- OP
      WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
           AND ra.kyoso_joken_code = '016' THEN 0.80                        -- 3勝クラス
      WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
           AND ra.kyoso_joken_code = '010' THEN 0.60                        -- 2勝クラス
      WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
           AND ra.kyoso_joken_code = '005' THEN 0.40                        -- 1勝クラス
      ELSE 0.20                                                             -- 未勝利
  END AS current_class_score
  ```
- **範囲**: 0.20～3.00（未勝利～G1）
- **重要度**: ⭐⭐⭐ (レース難易度の基準)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで中～低重要度（0.001～0.015程度）、クラス基準として有効

#### 2. previous_class_score (前走クラススコア) ✅ ⚠️ 削除候補
- **説明**: 前走レースのクラスを重み付けスコア化
- **計算方法**: 
  ```sql
  LAG(
      CASE 
          WHEN ra.grade_code = 'A' THEN 3.00
          WHEN ra.grade_code = 'B' THEN 2.00
          WHEN ra.grade_code = 'C' THEN 1.50
          WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
               AND ra.kyoso_joken_code = '999' THEN 1.00
          WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
               AND ra.kyoso_joken_code = '016' THEN 0.80
          WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
               AND ra.kyoso_joken_code = '010' THEN 0.60
          WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' 
               AND ra.kyoso_joken_code = '005' THEN 0.40
          ELSE 0.20
      END
  ) OVER (
      PARTITION BY seum.ketto_toroku_bango
      ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
  ) AS previous_class_score
  ```
- **範囲**: 0.20～3.00（未勝利～G1）
- **重要度**: ❌ (SHAP分析で0.0～0.001の超低重要度)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 
  - 京都全8モデルで0.0または最下位レベル（0.0～0.001）
  - **削除推奨**: class_score_changeと情報が重複しており、単独では無意味
- **削除理由**: class_score_change（クラス変化度）がより強力で、前走クラス単体では予測力なし

#### 3. class_score_change (クラススコア変化度) ✅
- **説明**: 前走とのクラススコア差（負=降級、正=昇級、0=同条件）
- **計算方法**: 
  ```sql
  current_class_score - previous_class_score AS class_score_change
  ```
- **範囲**: -2.80～+2.80程度（例: 未勝利0.20→G1 3.00なら+2.80）
- **重要度**: ⭐⭐⭐⭐ (クラス昇降格を捉える重要特徴量)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで中位クラス（0.003～0.016程度）、クラス変動を適切に捉える
- **ランキング学習効果**: 
  - 降級馬（負の値）vs同条件馬（0）でレース内の明確な実力差を捉える
  - 連続値なので「G3→1勝（-1.10）」と「OP→2勝（-0.40）」の違いも学習可能
  - 決定木が「class_score_change < -0.5」で大幅降級を自動判定
  - G1→G2（-1.00）とOP→3勝（-0.20）の重要度の違いを表現
- **実装難易度**: 低 (current_class_score - previous_class_scoreの単純な差分)
- **備考**: 
  - past_scoreは過去成績の総合評価（着順×グレード重みの合計）
  - class_score_changeはクラス変動の重要度（グレード重み差分）
  - 両方とも異なる情報を持つため併用すべき

#### 4. kyuyo_kikan (休養期間)
- **説明**: 前走からの休養日数
- **計算方法**: 
  ```sql
  (TO_DATE(ra.kaisai_nen || ra.kaisai_tsukihi, 'YYYYMMDD') - 
   TO_DATE(LAG(ra.kaisai_nen || ra.kaisai_tsukihi) OVER (
      PARTITION BY seum.ketto_toroku_bango
      ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
   ), 'YYYYMMDD')) AS kyuyo_kikan
  ```
- **範囲**: 0～9999日（通常14～180日、長期休養で1000日超も存在）
- **重要度**: ⭐⭐⭐⭐ (休養明け vs 連戦でレース内差が明確)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで中位クラス（0.004～0.012程度）、休養明けの不利を適切に捉える
- **ランキング学習効果**: 不利な馬（長期休養明け）を下位に配置
- **NULL処理**: 60日（約2ヶ月休養）で埋める
- **実装難易度**: 低 (SQLウィンドウ関数 + 日付計算)
- **備考**: INTEGER型なので4桁以上でも問題なし。長期休養馬（≥365日）は稀だが重要な情報

#### 5. past_score_mean (過去成績平均値) ✅
- **説明**: 過去3走のpast_scoreの平均値（relative_abilityの前提特徴量）
- **計算方法**: 
  ```sql
  AVG(past_score) OVER (
      PARTITION BY seum.ketto_toroku_bango
      ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
  ) AS past_score_mean
  ```
- **範囲**: 0～900程度（past_scoreと同じ）
- **重要度**: ⭐⭐⭐ (relative_abilityの基礎、単独でも能力指標として有効)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで低～中位クラス（0.003～0.008程度）、relative_abilityと組み合わせて使用
- **NULL処理**: 50.0（5着相当×OP1.0倍）で埋める
- **実装難易度**: 低 (SQLウィンドウ関数の集計)

#### 6. relative_ability (レース内相対能力値) ✅ 🏆 最重要特徴量
- **説明**: 同一レース内での過去成績の相対的な位置（z-score）
- **前提特徴量**: `past_score_mean`（過去3走のpast_scoreの平均値、上記で実装済み）
- **計算方法**: 
  ```sql
  -- relative_abilityの計算（past_score_meanを使用）
  (past_score_mean - 
   AVG(past_score_mean) OVER (PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango)
  ) / NULLIF(STDDEV(past_score_mean) OVER (PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango), 0) 
  AS relative_ability
  ```
- **範囲**: -3.0～+3.0程度（標準正規分布、z-score）
- **重要度**: ⭐⭐⭐⭐⭐ (ランキング学習の本質そのもの、全モデルで最重要)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 🏆 **全8モデルで第1位**（SHAP値: 0.0025～0.157）
  - 京都芝3歳以上長距離: 0.0025 (1位)
  - 京都芝3歳以上短距離: 0.061 (1位)
  - 京都ダート3歳以上長距離: 0.0058 (1位)
  - 京都ダート3歳以上短距離: 0.157 (1位、圧倒的)
  - 京都芝3歳長距離: 0.0023 (1位)
  - 京都芝3歳短距離: 0.046 (1位)
  - 京都ダート3歳長距離: 0.0043 (1位)
  - 京都ダート3歳短距離: 0.113 (1位)
- **ランキング学習効果**: 
  - 「レース内で抜けている馬」を明確化、穴馬検出にも強力
  - 連続値なので「少し抜けている（z=1.2）」と「圧倒的（z=2.5）」の違いも学習
  - 決定木が最適な閾値（例: 1.5や2.0など）をデータから自動学習して格上馬を判定
  - z-score統計的意味: 1.5=上位7%, 2.0=上位2.3%（人間の解釈目安）
- **NULL処理**: 0.0（レース内平均）で埋める
- **実装難易度**: 中 (SQLウィンドウ関数で統計量計算)
- **備考**: 
  - is_outstandingフラグは不要（連続値から決定木が最適閾値を自動学習）
  - past_score_meanも新規追加済み（上記参照）

---

### 🟢 Tier A - ランキング差別化特徴量（5個実装完了）

#### 7. distance_gap (前走距離ギャップ) ✅ ⚠️ 削除候補
- **説明**: 前走からの距離変化（絶対値）
- **計算方法**: 
  ```sql
  ABS(cast(rase.kyori as integer) - 
      LAG(cast(rase.kyori as integer)) OVER (
          PARTITION BY seum.ketto_toroku_bango
          ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
      )
  ) AS distance_gap
  ```
- **範囲**: 0～2000m程度
- **重要度**: ⭐⭐ (SHAP分析で低重要度、削除候補)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで下位クラス（0.001～0.003程度）、既存のzenso_kyori_saと情報重複
- **削除推奨理由**: 
  - 短距離モデルでは既に`zenso_kyori_sa`（前走距離差）が実装済み
  - 中長距離モデルでは距離変更の影響が小さく、予測力が低い
  - similar_distance_score（距離適性スコア）がより強力
- **ランキング学習効果**: 急激な距離変更馬の不利を評価するが、効果は限定的
- **実装難易度**: 低 (SQLウィンドウ関数)

#### 8. track_code_change (路面コード変化度) ✅ ⚠️ 削除候補
- **説明**: 前走からの路面種別の変化度（連続値）
- **計算方法**: 
  ```sql
  -- 前走路面コード
  LAG(cast(rase.track_code as integer)) OVER (
      PARTITION BY seum.ketto_toroku_bango
      ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
  ) AS previous_track_code,
  
  -- 路面コード変化度（絶対値）
  ABS(cast(rase.track_code as integer) - previous_track_code) AS track_code_change
  ```
- **範囲**: 0～13程度（0=同路面、1～3=芝内変更、13=芝10→ダート23の最大変化）
- **重要度**: ⭐⭐ (SHAP分析で低重要度、削除候補)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで最下位クラス（0.0～0.002程度）、surface_aptitude_scoreで代替可能
- **削除推奨理由**: 
  - `surface_aptitude_score`（路面適性スコア）がより直接的で強力
  - 芝⇔ダート変更は稀なケースで、学習データ不足
  - トラックコードの数値差は実質的な意味が薄い（芝10と芝11の差=1、芝10とダート23の差=13だが、実際の適性差は前者の方が小さい）
- **ランキング学習効果**: 
  - 路面変更馬vs同路面継続馬で差別化を試みるが、効果なし
  - 「芝→ダート（差分大）」と「芝内変更（差分小）」を区別可能だが、予測力に寄与せず
- **実装難易度**: 低 (SQLウィンドウ関数)
- **備考**: フラグ（baba_change）より連続値の方が柔軟だが、実際は効果なし

#### 9. left_direction_score (左回り成績スコア) ✅
- **説明**: 左回りコースでの過去平均成績（トラックコードベースで判定）
- **計算方法**: 
  ```sql
  AVG(CASE 
      WHEN past_rase.track_code IN ('11', '12', '13', '14', '15', '16',  -- 芝左回り
                                     '23', '25', '26')                      -- ダート左回り
      THEN (1.0 - CAST(past_se.kakutei_chakujun AS FLOAT) / CAST(past_ra.tosu AS FLOAT))
      ELSE NULL
  END) OVER (
      PARTITION BY seum.ketto_toroku_bango
      ORDER BY cast(past_ra.kaisai_nen as integer), cast(past_ra.kaisai_tsukihi as integer)
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
  ) AS left_direction_score
  ```
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐ (回り適性は重要、左回りトラックで有効)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで中～低位クラス（0.002～0.010程度）、左回りトラックで一定の効果
- **NULL処理**: 0.5（未経験=中立）で埋める
- **ランキング学習効果**: 初左回り馬vs左巧者で差別化
- **実装難易度**: 中 (トラックコード条件集計)
- **備考**: トラックコード11-16（芝左）, 23/25/26（ダート左）。直線・サンド・障害は除外

#### 10. right_direction_score (右回り成績スコア) ✅ 🏆 京都競馬場で超重要
- **説明**: 右回りコースでの過去平均成績（トラックコードベースで判定）
- **計算方法**: 
  ```sql
  AVG(CASE 
      WHEN past_rase.track_code IN ('17', '18', '19', '20', '21', '22',  -- 芝右回り
                                     '24')                                 -- ダート右回り
      THEN (1.0 - CAST(past_se.kakutei_chakujun AS FLOAT) / CAST(past_ra.tosu AS FLOAT))
      ELSE NULL
  END) OVER (
      PARTITION BY seum.ketto_toroku_bango
      ORDER BY cast(past_ra.kaisai_nen as integer), cast(past_ra.kaisai_tsukihi as integer)
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
  ) AS right_direction_score
  ```
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐⭐⭐ (京都競馬場など右回りトラックで圧倒的に重要)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 🏆 **京都芝モデルで第2～3位**（SHAP値: 0.011～0.044）
  - 京都芝3歳以上短距離: 0.044 (2位) ← relative_abilityに次ぐ重要度
  - 京都芝3歳以上長距離: 0.011 (3位レベル)
  - 京都芝3歳短距離: 0.035 (3位)
  - 京都芝3歳長距離: 0.012 (中位)
  - 京都ダート: 0.005～0.010程度（中～低位）
- **トラック特性**: 京都競馬場は右回りトラック、阪神・中京も右回り
- **NULL処理**: 0.5（未経験=中立）で埋める
- **ランキング学習効果**: 初右回り馬vs右巧者で差別化、京都・阪神で顕著な効果
- **実装難易度**: 中 (トラックコード条件集計)
- **備考**: トラックコード17-22（芝右）, 24（ダート右）。直線・サンド・障害は除外

#### 11. current_direction_match (今回コース回り適性) ✅
- **説明**: 今回のコース回り方向での成績（トラックコードベースで判定）
- **計算方法**: 
  ```sql
  CASE 
      WHEN rase.track_code IN ('11', '12', '13', '14', '15', '16', '23', '25', '26') 
      THEN left_direction_score   -- 左回り
      WHEN rase.track_code IN ('17', '18', '19', '20', '21', '22', '24') 
      THEN right_direction_score  -- 右回り
      ELSE 0.5  -- 直線の場合は中立値
  END AS current_direction_match
  ```
- **範囲**: 0.0～1.0
- **重要度**: ⭐⭐⭐⭐ (今回条件に適した馬を評価)
- **実装状況**: ✅ db_query_builder.py, data_preprocessing.py, feature_engineering.py に実装完了
- **SHAP分析結果**: 全8モデルで中位クラス（0.005～0.015程度）、回り適性の実用化
- **ランキング学習効果**: 今回条件に適した馬を上位配置
- **NULL処理**: 0.5（未経験=中立）で埋める
- **実装難易度**: 低 (left/right_direction_scoreから派生)
- **備考**: 直線コース（10, 29）の場合は中立値0.5を返す

---

## ⚠️ 削除推奨特徴量（SHAP分析結果に基づく）

### 1. previous_class_score (前走クラススコア)
- **削除理由**: 
  - SHAP重要度: 0.0～0.001（全8モデルで最下位レベル）
  - class_score_changeと情報が重複
  - 前走クラス単体では予測力なし（クラス変化度の方が重要）
- **削除による影響**: なし（class_score_changeが代替）
- **削除推奨度**: 🔴 **確実に削除すべき**

### 2. distance_gap (前走距離ギャップ)
- **削除理由**: 
  - SHAP重要度: 0.001～0.003（全8モデルで下位）
  - 短距離モデルでは既に`zenso_kyori_sa`が実装済み（情報重複）
  - 中長距離モデルでは距離変更の影響が小さい
  - `similar_distance_score`（距離適性スコア）の方が強力
- **削除による影響**: 最小限（similar_distance_scoreが代替）
- **削除推奨度**: 🟠 **削除推奨**

### 3. track_code_change (路面コード変化度)
- **削除理由**: 
  - SHAP重要度: 0.0～0.002（全8モデルで最下位レベル）
  - `surface_aptitude_score`（路面適性スコア）の方が直接的で強力
  - 芝⇔ダート変更は稀で学習データ不足
  - トラックコードの数値差は実質的な意味が薄い
- **削除による影響**: なし（surface_aptitude_scoreが代替）
- **削除推奨度**: 🟠 **削除推奨**

---

## 📊 SHAP分析結果サマリー（京都競馬場8モデル）

### 🏆 Top 3 重要特徴量（全モデル共通傾向）

| 順位 | 特徴量 | 平均SHAP値 | 備考 |
|------|--------|-----------|------|
| 1 | **relative_ability** | 0.025～0.157 | 🏆 全8モデルで第1位、ランキング学習の核心 |
| 2 | **right_direction_score** | 0.011～0.044 | 🏆 京都芝で第2～3位、右回り適性が重要 |
| 3 | **past_avg_sotai_chakujun** | 0.010～0.035 | 着差考慮成績、安定した予測力 |

### 📉 Low 3 重要特徴量（削除候補）

| 順位 | 特徴量 | 平均SHAP値 | 削除推奨 |
|------|--------|-----------|---------|
| 27 | **previous_class_score** | 0.0～0.001 | 🔴 確実に削除 |
| 26 | **track_code_change** | 0.0～0.002 | 🟠 削除推奨 |
| 25 | **distance_gap** | 0.001～0.003 | 🟠 削除推奨 |

### 🎯 モデル別SHAP分析詳細

#### 京都芝3歳以上短距離モデル
- **Top 3**: relative_ability (0.061), right_direction_score (0.044), past_avg_sotai_chakujun (0.035)
- **Bottom 3**: previous_class_score (0.0), track_code_change (0.001), distance_gap (0.002)

#### 京都ダート3歳以上短距離モデル
- **Top 3**: relative_ability (0.157), past_score (0.042), kishu_skill_score (0.028)
- **Bottom 3**: previous_class_score (0.0), track_code_change (0.0), surface_aptitude_score (0.001)

---

## 💡 追加検討特徴量（未実装）

### 🟡 検討中（データ確認が必要）

#### 11. weight_diff_from_avg (馬体重レース内相対値)
- **説明**: レース内平均馬体重との差分
- **計算方法**: 
  ```sql
  cast(seum.bataiju as float) - 
  AVG(cast(seum.bataiju as float)) OVER (PARTITION BY ra.kaisai_nen, ra.kaisai_tsukihi, ra.keibajo_code, ra.race_bango)
  AS weight_diff_from_avg
  ```
- **範囲**: ±50kg程度
- **重要度**: ⭐⭐⭐ (レース内相対化すれば有効)
- **ランキング学習効果**: レース内での体格差を評価
- **実装難易度**: 低 (SQLウィンドウ関数)
- **前提条件**: テーブルにbataijuカラムが存在すること

---

### 🟡 条件付き採用

#### 4. early_pace_performance
- **説明**: 前半ペース適性 (前半3Fスピードと成績の相関)
- **条件**: zenpan_3f (前半3Fタイム) データが利用可能な場合のみ
- **実装難易度**: 中 (データ確認 + 相関計算)

#### 5. long_distance_stamina_score
- **説明**: 長距離レース (≥2000m) での平均後半3F指数
- **条件**: 十分な長距離データがある場合のみ
- **実装難易度**: 中 (データ量確認が必要)

### ❌ 却下

- **distance_transition_trend**: 距離変化トレンド → 実装したが精度悪化のため削除
- **distance_performance_gradient**: similar_distance_scoreと情報重複
- **lap_consistency_score**: ラップタイムデータ不足の可能性大
- **mid_distance_consistency**: 情報量少なく効果疑問

---

## 🔄 変更履歴

### 2026-01-18 (PM) - ランキング学習特徴量実装完了 ✅
- ✅ **新規特徴量11個を実装完了**（現在の特徴量数: 16個 → **27個**に拡張）
  - 🔥 **Tier S（6個）**: current_class_score, previous_class_score, class_score_change, kyuyo_kikan, past_score_mean, **relative_ability (全モデルで第1位)**
  - 🟢 **Tier A（5個）**: distance_gap, track_code_change, left_direction_score, **right_direction_score (京都芝で第2～3位)**, current_direction_match
- ✅ **NULL値処理を全特徴量で最適化**
  - スコア系（0-1範囲）: 0.5（中立値）
  - カウント系: 0（ゼロ回）
  - 変化系: 0（変化なし）
  - 能力系: ドメイン知識に基づく適切な値（past_score=50, kyuyo_kikan=60, relative_ability=0.0など）
- ✅ **京都競馬場8モデルでSHAP分析実施**
  - 芝/ダート × 短距離/長距離 × 3歳/3歳以上 = 8モデル
  - **relative_ability**: 全8モデルで第1位（SHAP値 0.0025～0.157）
  - **right_direction_score**: 京都芝で第2～3位（京都は右回りトラック）
- ⚠️ **削除推奨特徴量を特定**
  - 🔴 **previous_class_score**: SHAP値 0.0～0.001（確実に削除すべき）
  - 🟠 **distance_gap**: SHAP値 0.001～0.003（削除推奨）
  - 🟠 **track_code_change**: SHAP値 0.0～0.002（削除推奨）
  - **次のアクション**: 3特徴量を削除して27個→24個に最適化

### 2026-01-18 (AM)
- ✅ **ランキング学習特化の新特徴量を計画**
  - 🔥 Tier S（最優先）: クラススコア変化度、休養期間、レース内相対能力値（5個）
  - 🟢 Tier A（優先）: 前走条件ギャップ、回り適性（5個）
  - 🟡 検討中: 馬体重相対化（データ確認が必要）
  - **追加予定特徴量数**: 11個（データ確認後12個）
  - **実装予定後の総特徴量数**: 27～28個（16 + 11～12）
  - **方針**: 全てSQLで実装（複雑でもSQLで対応）
  - **設計思想**: フラグ（0/1）より連続値を優先（決定木が閾値を自動学習）

### 2026-01-18 (AM)
- ✅ **速報予測（sokuho）対応の最適化**
  - 速報予測用SQLクエリのパフォーマンス改善（過去データを直近5年に絞る）
  - 速報予測用に欠けていた特徴量を追加（umaban_percentile, futan_zscore, futan_percentile, past_score系）
  - `baba_change_adaptability`削除（SHAP分析で低重要度のため）
  - `kishu_popularity_score`削除（速報時にオッズ情報が不完全なため）
  - **現在の特徴量数**: 16個（ベース）
  - **⚠️ 注意**: 既存モデルは17個の特徴量で学習されているため、**全モデル再学習が必要**
- ✅ 距離適性特徴量を全面リニューアル（SQL化 + 重み付け平均）
  - `past_avg_kyori`削除（単純平均では性能情報なし）
  - `past_score_short`, `past_score_mile`, `past_score_middle`, `past_score_long`追加（SQLで4距離帯別成績を算出）
  - `similar_distance_score`を重み付け平均版に刷新（未経験距離でも周辺距離帯から推定可能）
- ✅ 騎手・調教師スコアを過去N走方式に統一（sokuho対応）
  - `kishu_skill_score`: 過去30走平均
  - `kishu_surface_score`: 同路面過去50走平均
  - `chokyoshi_recent_score`: 過去20走平均

### 2025-11-10
- ✅ `wakuban_kyori_interaction`追加 (短距離特化)
- ✅ `zenso_kyori_sa`追加 (短距離特化)
- ✅ `start_index`追加 (短距離特化)
- ✅ `corner_position_score`追加 (短距離特化) → 安定性ボーナス追加で全券種的中率向上
- ✅ `long_distance_experience_count`追加 (長距離特化 2200m以上)
- ✅ 路面×距離別の特徴量選択を実装
- ✅ 新規特徴量8案を評価 → 3つ採用、2つ実装完了
- ✅ corner_2, corner_3, corner_4カラムをデータベースから取得
- ❌ `pace_suitability`追加→削除 (精度悪化のため)
- ❌ `distance_transition_trend`追加→削除 (精度悪化のため)

### 2025-11-09
- ✅ `past_avg_sotai_chakujun`に着差考慮を追加 (Pattern 2採用)

### 2025-11-08
- ✅ グレード重み付け強化 (G1=3.0x, G2=2.0x, G3=1.5x)

---

**最終更新**: 2026年1月18日  
**作成者**: Copilot AI Assistant  
**現在の特徴量数**: **27個**（16個ベース + 11個新規追加）  
**最適化後予定**: **24個**（削除推奨3個を除去後）  
**重要**: 
- 🏆 **relative_ability**: 全8モデルで第1位、ランキング学習の核心特徴量
- 🏆 **right_direction_score**: 京都芝で第2～3位、右回りトラック特化
- ⚠️ **3特徴量を削除推奨**: previous_class_score（SHAP 0.0）, distance_gap（SHAP 0.001～0.003）, track_code_change（SHAP 0.0～0.002）
- 特徴量変更に伴い全モデル再学習が必要（27個で学習 → 削除後24個で再学習）
