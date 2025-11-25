# 🏇 競馬予測モデル 実践ワークフローガイド

このドキュメントは、本プロジェクトでモデルを作成・評価・改善するための完全な実践ガイドです。

---

## 📋 目次

1. [ワークフロー全体像](#ワークフロー全体像)
2. [フェーズ1: モデル作成](#フェーズ1-モデル作成)
3. [フェーズ2: モデル評価](#フェーズ2-モデル評価)
4. [フェーズ3: SHAP分析によるモデル理解](#フェーズ3-shap分析によるモデル理解)
5. [フェーズ4: モデル改善](#フェーズ4-モデル改善)
6. [フェーズ5: 改善効果の検証](#フェーズ5-改善効果の検証)
7. [トラブルシューティング](#トラブルシューティング)
8. [ベストプラクティス](#ベストプラクティス)

---

## 🎯 ワークフロー全体像

```
┌─────────────────────────────────────────────────────────────┐
│                   モデル改善サイクル                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │ 1. モデル作成 (batch_model_creator.py)       │
    │    - 学習データ: 2016~2022年                  │
    │    - Optunaで自動ハイパーパラメータ最適化     │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │ 2. モデル評価 (universal_test.py)            │
    │    - テストデータ: 2023~2025年                │
    │    - 的中率・回収率を計算                     │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │ 3. SHAP分析 (model_explainer.py)             │
    │    - 特徴量重要度を可視化                     │
    │    - 予測理由を解析                           │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │ 4. 詳細分析 (analyze_shap_results.py)        │
    │    - 削除推奨特徴量の特定                     │
    │    - 改善提案の生成                           │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │ 5. 特徴量改善 (コード修正)                    │
    │    - 不要特徴量の削除                         │
    │    - 重要特徴量の強化                         │
    │    - 新規特徴量の追加                         │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │ 6. 改善効果検証 (再度モデル作成→評価)        │
    │    - 的中率が向上したか確認                   │
    │    - 向上した → 成功！                        │
    │    - 向上しない → 分析に戻る                  │
    └──────────────────────────────────────────────┘
                           ↓
                    (改善サイクル継続)
```

---

## 🔧 フェーズ1: モデル作成

### 1-1. 基本的なモデル作成

**標準モデルの一括作成:**

```bash
# デフォルト学習期間: 2016~2022年
python batch_model_creator.py

# 学習期間を指定（例: 2018~2023年）
python batch_model_creator.py 2018-2023

# 単一年で学習（例: 2022年のみ）
python batch_model_creator.py 2022
```

**カスタムモデルの作成:**

```bash
# model_configs.jsonのcustom_modelsセクションで定義したモデルを作成
python batch_model_creator.py custom

# 学習期間を指定してカスタムモデル作成
python batch_model_creator.py custom 2020-2023
```

### 1-2. モデル作成の設定

**`model_configs.json`の構造:**

```json
{
  "standard_models": [
    {
      "track_code": "05",              // 競馬場コード(05=東京)
      "kyoso_shubetsu_code": "13",     // 競走種別(13=3歳以上)
      "surface_type": "turf",          // 路面タイプ(turf/dirt)
      "min_distance": 1700,            // 最小距離(m)
      "max_distance": 9999,            // 最大距離(9999=上限なし)
      "model_filename": "tokyo_turf_3ageup_long_baseline.sav",
      "description": "東京芝中長距離3歳以上(Baseline)"
    }
  ],
  "custom_models": [
    {
      // カスタムモデルの設定...
    }
  ]
}
```

### 1-3. モデル作成の内部処理

1. **データ取得**: PostgreSQLから条件に合うレースデータを取得
2. **特徴量計算**: 20個の特徴量を自動生成（詳細は`FEATURE_LIST.md`参照）
3. **時系列分割**: 75%訓練データ、25%検証データ（時系列順）
4. **ハイパーパラメータ最適化**: Optunaで50試行
   - 評価指標: NDCG@1,3,5（順位予測の精度）
   - 最適化対象: num_leaves, max_depth, learning_rate など
5. **モデル学習**: 最適パラメータでLightGBM LambdaRankモデル訓練
6. **モデル保存**: `models/`ディレクトリに`.sav`形式で保存

### 1-4. 出力ファイル

```
models/
├── tokyo_turf_3ageup_long_baseline.sav  # モデルファイル
├── tokyo_turf_3ageup_short_baseline.sav
└── ...

sql_log.txt  # 実行されたSQLクエリ（デバッグ用）
```

### 1-5. モデル作成の所要時間

- 1モデル: 約30秒~2分
- 標準モデル24個: 約15~40分
- カスタムモデル: 設定次第

---

## 📊 フェーズ2: モデル評価

### 2-1. バックテスト実行

**全モデル一括テスト:**

```bash
# デフォルトテスト期間: 2023年
python universal_test.py multi

# テスト期間を指定（例: 2023~2025年）
python universal_test.py multi 2023-2025

# 単一年でテスト（例: 2024年のみ）
python universal_test.py multi 2024
```

**単一モデルテスト:**

```bash
# デフォルトモデル(tokyo_turf_3ageup_long_baseline)で2023年テスト
python universal_test.py

# テスト期間を指定
python universal_test.py 2023-2025

# 単一年でテスト
python universal_test.py 2024
```

### 2-2. 評価指標の見方

**主要指標:**

```
[STATS] 購入推奨馬数: 80       # AIが購入推奨した馬の総数
[STATS] 実購入馬数: 80         # 実際に購入した馬の総数
[STATS] 的中数: 18             # 1着的中した回数
[STATS] 的中率: 22.50%         # 的中数/実購入馬数
[STATS] 総投資額: 80,000円     # 投資した総額
[STATS] 最終資金: 971,600円    # 払戻金の合計
[STATS] 損益: -28,400円        # 最終資金 - 総投資額
[STATS] 回収率: 64.50%         # 最終資金/総投資額
```

**評価基準:**

| 指標 | 優秀 | 良好 | 改善必要 | 備考 |
|------|------|------|----------|------|
| 的中率 | ≥30% | 25-30% | <25% | ベースライン: 23.52% |
| 回収率 | ≥100% | 80-100% | <80% | 100%超で利益確定 |
| 損益 | プラス | ±10% | 大幅マイナス | 長期的な収支 |

### 2-3. 出力ファイル

```
results/
├── predicted_results.tsv        # 通常レースの全馬（38列、分析用列なし）
├── predicted_results_skipped.tsv # スキップレースの全馬（43列、分析用列あり）
├── betting_summary.tsv          # 的中率・回収率サマリー
├── model_comparison.tsv         # 複数モデル比較(multi実行時)
└── betting_summary_*.tsv        # モデル別サマリー
```

**ファイル分割の仕様:**
- **`predicted_results.tsv`**: スコア差が基準値以上のレース（通常レース）の全馬を記録。分析用列（スコア差、スキップ理由、購入推奨、購入額、現在資金）は含まない。
- **`predicted_results_skipped.tsv`**: スコア差が基準値未満でスキップされたレースの全馬を記録。分析用列を含む。

**`predicted_results.tsv`の見方:**

| 列名 | 意味 | 備考 |
|------|------|------|
| tansho_ninkijun_numeric | 人気順位 | 1=1番人気 |
| kakutei_chakujun | 実際の着順 | 1=1着 |
| predicted_rank | AI予測順位 | モデルの予測 |
| tansho_odds | 単勝オッズ | 配当の目安 |

**`predicted_results_skipped.tsv`の追加列:**

| 列名 | 意味 | 備考 |
|------|------|------|
| スコア差 | 1位と2位の予測スコア差 | この値が小さいとスキップ |
| スキップ理由 | スキップされた理由 | low_score_diff等 |
| 購入推奨 | 購入推奨フラグ | 常にFalse |
| 購入額 | 購入額 | 常に0 |
| 現在資金 | 現在の資金残高 | 参考値 |

### 2-4. 複数モデル比較

**`model_comparison.tsv`の見方:**

```bash
python universal_test.py multi
```

実行後、各モデルの的中率・回収率を横並びで比較できます。

**比較のポイント:**
- 最も的中率が高いモデル → 精度重視
- 最も回収率が高いモデル → 利益重視
- 的中率と回収率のバランス → 実用性重視

---

## 🔬 フェーズ3: SHAP分析によるモデル理解

### 3-1. SHAP分析とは？

**SHAP (SHapley Additive exPlanations)** は、機械学習モデルの「予測理由」を数値化する技術です。

**できること:**
- ✅ どの特徴量が予測に最も貢献しているか定量評価
- ✅ 各特徴量の影響度を可視化
- ✅ 不要な特徴量（ノイズ）を特定
- ✅ 個別レースの予測理由を詳細表示

### 3-2. SHAP分析の実行

```bash
# 基本的な実行（デフォルト: 東京芝中長距離モデル、2022年データ）
python model_explainer.py

# コマンドライン引数で指定（モデルファイル名、テスト年）
python model_explainer.py tokyo_turf_3ageup_short_baseline.sav 2023
```

### 3-3. SHAP分析の出力ファイル

```
shap_analysis/
├── tokyo_turf_3ageup_long_importance.csv    # 特徴量重要度CSV
├── tokyo_turf_3ageup_long_summary.png       # Summary Plot
├── tokyo_turf_3ageup_long_bar.png           # Bar Plot
├── tokyo_turf_3ageup_long_dependence.png    # Dependence Plot
├── tokyo_turf_3ageup_long_force_1.png       # Force Plot (個別レース1)
├── tokyo_turf_3ageup_long_force_2.png       # Force Plot (個別レース2)
└── ...
```

### 3-4. SHAP分析結果の見方

#### 📈 Summary Plot（最重要）

**見るべきポイント:**
1. **縦軸**: 特徴量名（上から重要度順）
2. **横軸**: SHAP値（予測への影響度）
3. **色**: 特徴量の値（赤=高、青=低）

**解釈例:**
```
past_score:  ●●●●●●●●●●●●●●● ← 最重要特徴量
            |  0.0  |  0.2  |
            赤=過去成績が高い馬は着順が良くなる
```

#### 📊 Bar Plot（重要度ランキング）

**見るべきポイント:**
- 平均絶対SHAP値の大きさ = 重要度
- 上位10個が全体の70~80%を占めることが多い

**活用方法:**
- Top 3特徴量 → 最優先で強化
- SHAP < 0.005 → 削除候補

#### 🔗 Dependence Plot（相関関係）

**見るべきポイント:**
- X軸: 特徴量の値
- Y軸: SHAP値
- 色: 別の特徴量との相互作用

**解釈例:**
```
past_score (X軸) vs SHAP値 (Y軸)
- 右上がり → 値が大きいほど好影響
- 右下がり → 値が大きいほど悪影響
- 横ばい → 影響なし（削除候補）
```

#### ⚡ Force Plot（個別レース分析）

**見るべきポイント:**
- 赤いバー: プラス影響（着順を上げる）
- 青いバー: マイナス影響（着順を下げる）

**活用方法:**
- 的中したレース → どの特徴量が決め手だったか
- 外れたレース → 何が予測を誤らせたか

### 3-5. `importance.csv`の見方

```csv
feature,mean_abs_shap,lgb_gain
past_avg_sotai_chakujun,0.2624,1854.23
umaban_kyori_interaction,0.1297,844.40
past_score,0.0761,515.88
...
```

**列の意味:**
- `feature`: 特徴量名
- `mean_abs_shap`: SHAP値の平均絶対値（**重要度の指標**）
- `lgb_gain`: LightGBMの内部重要度（参考値）

**重要度の基準:**
- **超重要** (SHAP ≥ 0.1): モデルの主力特徴量
- **重要** (0.05 ≤ SHAP < 0.1): 補助的に貢献
- **やや重要** (0.01 ≤ SHAP < 0.05): 微小な貢献
- **不要** (SHAP < 0.005): 削除推奨

---

## 📈 フェーズ4: モデル改善

### 4-1. SHAP結果の詳細分析

```bash
# SHAP分析結果を統計的に分析
python analyze_shap_results.py
```

**出力内容:**
1. 基本統計量（SHAP値の合計、平均、標準偏差）
2. 累積寄与率分析（Top N で何%カバーできるか）
3. カテゴリ別分析（過去成績系、斤量系など）
4. 削除推奨特徴量（SHAP < 0.005）
5. LightGBM GainとSHAPの相関分析
6. 改善提案の自動生成

### 4-2. 分析結果の読み方

#### 累積寄与率分析

```
累積寄与率 50% に必要な特徴量数: 3個
  → Top3: past_avg_sotai_chakujun, umaban_kyori_interaction, past_score
累積寄与率 70% に必要な特徴量数: 6個
累積寄与率 80% に必要な特徴量数: 9個
累積寄与率 90% に必要な特徴量数: 13個
```

**意味:**
- **Top 3で50%** → 少数の特徴量が支配的
- **Top 13で90%** → 残り7個はほぼ無意味

**活用方法:**
- Top 3を最優先で強化
- 90%に寄与しない特徴量は削除候補

#### カテゴリ別分析

```
カテゴリ   特徴量数  SHAP合計  SHAP平均  寄与率(%)
過去成績系     3      0.4125    0.1375    45.2
馬番・枠番系   4      0.1852    0.0463    20.3
斤量系         6      0.1205    0.0201    13.2
...
```

**活用方法:**
- 寄与率が高いカテゴリ → 特徴量を追加
- 寄与率が低いカテゴリ → 特徴量を削減

#### 削除推奨特徴量

```
【削除推奨特徴量(SHAP < 0.005)】
削除候補: 5個

  [ERROR] wakuban_ratio                  SHAP=0.001700
  [ERROR] baba_change_adaptability       SHAP=0.004700
  [ERROR] chokyoshi_recent_score         SHAP=0.004900
  ...

削除することで:
  - 特徴量数: 24個 → 19個
  - 削減率: 20.8%
  - 失われる情報量: 1.42%
```

**判断基準:**
- 失われる情報量 < 5% → **削除OK**
- 失われる情報量 ≥ 5% → 慎重に判断

#### Gain vs SHAP 乖離分析

```
Gainが高いのにSHAPが低い特徴量(モデルが過剰に使用):
  futan_percentile          Gain= 234.56 SHAP=0.0123 比率=19.07
  
SHAPが高いのにGainが低い特徴量(効率的な特徴量):
  past_avg_sotai_chakujun   Gain=1854.23 SHAP=0.2624 比率= 7.07
```

**意味:**
- **Gain高・SHAP低** → モデルが無駄に分岐している（過学習の兆候）
- **SHAP高・Gain低** → 少ない分岐で大きな効果（優秀な特徴量）

### 4-3. 改善戦略

#### 戦略1: 不要特徴量の削除

**対象:**
- SHAP < 0.005 の特徴量
- 失われる情報量 < 5%

**手順:**
1. `model_creator.py`を開く
2. 該当特徴量の計算コードをコメントアウト
3. `universal_test.py`も同様に修正
4. モデル再作成・評価

**例:**
```python
# model_creator.py
# 削除前
df['wakuban_ratio'] = df['wakuban'] / df['total_horses']

# 削除後
# df['wakuban_ratio'] = df['wakuban'] / df['total_horses']  # SHAP=0.0017 削除
```

#### 戦略2: 重要特徴量の強化

**対象:**
- SHAP上位3特徴量

**手順:**
1. 特徴量の定義を確認
2. より精度の高い計算方法に変更
3. 関連する新規特徴量を追加

**例:**
```python
# 強化前: 単純平均
past_score = past_races['score'].mean()

# 強化後: 近い過去ほど高重み
weights = [1.5, 1.2, 1.0, 0.8]  # 直近4走
past_score = np.average(past_races['score'][-4:], weights=weights)
```

#### 戦略3: カテゴリバランスの調整

**対象:**
- 寄与率が偏っているカテゴリ

**例:**
- 過去成績系が45% → 他カテゴリを強化
- 騎手系が3% → 騎手特徴量を追加

#### 戦略4: 相互作用特徴量の追加

**対象:**
- Dependence Plotで相関が見られる特徴量ペア

**例:**
```python
# umaban_kyori_interaction (既存)
df['umaban_kyori_interaction'] = df['umaban'] * df['kyori_category']

# 新規: 斤量×距離の相互作用
df['futan_kyori_interaction'] = df['futan_weight'] * df['kyori_category']
```

---

## ✅ フェーズ5: 改善効果の検証

### 5-1. 改善サイクル

```
1. 特徴量修正
   ↓
2. モデル再作成
   python batch_model_creator.py custom 2016-2022
   ↓
3. バックテスト実行
   python universal_test.py 2023-2025
   ↓
4. 結果比較
   的中率が向上したか？
   ├─ YES → 成功！ → Git commit
   └─ NO  → SHAP再分析 → 別の改善策
```

### 5-2. 比較すべき指標

**Before/After比較表:**

| 指標 | Before | After | 差分 | 判定 |
|------|--------|-------|------|------|
| 的中率 | 23.52% | 25.30% | +1.78pt | ✅ 改善 |
| 回収率 | 87.0% | 92.5% | +5.5pt | ✅ 改善 |
| 特徴量数 | 24個 | 20個 | -4個 | ✅ シンプル化 |
| 学習時間 | 45秒 | 38秒 | -7秒 | ✅ 高速化 |

**判定基準:**
- **大成功**: 的中率+3pt以上、回収率+10pt以上
- **成功**: 的中率+1pt以上、回収率+5pt以上
- **微改善**: 的中率+0.5pt以上、回収率変化なし
- **失敗**: 的中率低下、回収率低下

### 5-3. A/Bテストの実施

**複数モデルの同時比較:**

```bash
# モデルA: ベースライン
python batch_model_creator.py custom 2016-2022
mv models/tokyo_turf_3ageup_long.sav models/tokyo_turf_3ageup_long_modelA.sav

# モデルB: 改善版
# (コード修正後)
python batch_model_creator.py custom 2016-2022
mv models/tokyo_turf_3ageup_long.sav models/tokyo_turf_3ageup_long_modelB.sav

# 両方テスト
python universal_test.py multi 2023-2025
```

### 5-4. Git管理のベストプラクティス

**成功時:**
```bash
git add model_creator.py universal_test.py
git commit -m "特徴量改善: wakuban_ratio削除で的中率+1.8pt向上"
git push
```

**失敗時:**
```bash
git restore model_creator.py universal_test.py
# または
git reset --hard HEAD
```

---

## 🛠️ トラブルシューティング

### エラー1: モデルファイルが見つかりません

**症状:**
```
[ERROR] モデルファイル tokyo_turf_3ageup_long.sav が見つかりません。
```

**原因:**
- モデル未作成
- ファイル名の不一致

**解決策:**
```bash
# モデル作成
python batch_model_creator.py

# modelsディレクトリ確認
ls -lh models/

# ファイル名をuniversal_test.pyと一致させる
```

### エラー2: 特徴量数の不一致

**症状:**
```
LightGBMError: The number of features in data (22) is not the same as it was in training data (24).
```

**原因:**
- model_creator.pyとuniversal_test.pyの特徴量が不一致

**解決策:**
1. 両ファイルで同じ特徴量を計算しているか確認
2. 削除した特徴量をuniversal_test.pyでも削除
3. モデル再作成

### エラー3: SHAP分析がエラー

**症状:**
```
KeyError: 'tansho_ninkijun_numeric'
```

**原因:**
- データ取得SQLと特徴量計算の不一致

**解決策:**
1. `sql_log.txt`でSQLを確認
2. 必要なカラムがSELECT句にあるか確認
3. model_explainer.pyのSQL修正

### エラー4: 的中率が極端に低い

**症状:**
```
的中率: 5.0% (ベースライン: 23.52%)
```

**原因:**
- データリーク（未来情報の混入）
- 過学習
- 特徴量計算バグ

**解決策:**
1. SHAP分析で異常な特徴量を特定
2. 時系列分割が正しいか確認
3. 特徴量計算ロジックをレビュー

---

## 🏆 ベストプラクティス

### 実験管理

**推奨フロー:**
1. ベースライン測定（改善前の的中率記録）
2. 1つの変更ごとにコミット
3. 実験ログを記録（Markdown推奨）
4. 成功した改善のみマージ

**実験ログ例:**
```markdown
## Phase 2-1: 人気系特徴量追加

**日付:** 2025-11-13
**変更内容:** popularity_percentile, popularity_score_gap_normalized追加
**学習期間:** 2016-2022
**テスト期間:** 2023-2025

### 結果
- 的中率: 17.39% (ベースライン: 23.52%, **-6.13pt**)
- 回収率: 68.5%
- **判定:** ❌ 失敗（人気依存度スコア8/10で超高依存）

### SHAP分析
- popularity_percentile: SHAP=0.462 (1位)
- 人気1番→予測1番: 89.47%
- **結論:** 人気のコピーになっており、独自予測ができていない

### 対応
- Phase 2-1をロールバック
- 人気系特徴量を完全削除
```

### 特徴量設計の原則

1. **時系列厳守**: 未来情報を絶対に使わない
2. **シンプル優先**: 複雑な特徴量より単純な特徴量
3. **解釈可能性**: 「なぜこの値になるか」説明できる
4. **ドメイン知識**: 競馬の常識に基づく
5. **SHAP検証**: 追加後は必ずSHAP分析

### モデル改善の優先順位

```
1. 不要特徴量の削除 (SHAP < 0.005)
   ├─ リスク: 低
   ├─ 効果: 過学習防止、高速化
   └─ 工数: 小

2. Top3特徴量の強化
   ├─ リスク: 中
   ├─ 効果: 精度向上（大）
   └─ 工数: 中

3. カテゴリバランス調整
   ├─ リスク: 中
   ├─ 効果: 精度向上（中）
   └─ 工数: 中

4. 新規特徴量の追加
   ├─ リスク: 高（過学習リスク）
   ├─ 効果: 不確定
   └─ 工数: 大
```

### SHAP分析のタイミング

**必須:**
- ✅ モデル新規作成時
- ✅ 特徴量追加・削除時
- ✅ 的中率が大幅変動時

**推奨:**
- 月次レビュー
- 大会前の最終確認

**不要:**
- ハイパーパラメータのみ変更時
- 学習期間変更のみ

---

## 📚 補足資料

### 関連ドキュメント

- `FEATURE_LIST.md` - 全特徴量の詳細説明
- `README.md` - プロジェクト概要・コマンドリファレンス
- `shap_analysis_report.md` - SHAP分析レポート例
- `model_configs.json` - モデル設定ファイル

### 重要なファイル一覧

| ファイル | 役割 | 編集頻度 |
|---------|------|---------|
| `model_creator.py` | モデル作成・特徴量定義 | 高 |
| `universal_test.py` | モデル評価・バックテスト | 高 |
| `batch_model_creator.py` | 一括モデル作成 | 低 |
| `model_explainer.py` | SHAP分析 | 中 |
| `analyze_shap_results.py` | SHAP統計分析 | 低 |
| `model_configs.json` | モデル設定 | 中 |
| `keiba_constants.py` | 定数定義 | 低 |

### コマンド早見表

```bash
# モデル作成
python batch_model_creator.py [year_start-year_end]
python batch_model_creator.py custom [year_start-year_end]

# モデル評価
python universal_test.py [year_start-year_end]
python universal_test.py multi [year_start-year_end]

# SHAP分析
python model_explainer.py [model_file] [test_year]
python analyze_shap_results.py

# Git操作
git status
git add .
git commit -m "メッセージ"
git push
git restore <file>
git reset --hard HEAD
```

---

## 🎓 学習リソース

### SHAP理解を深める

- [SHAP公式ドキュメント](https://shap.readthedocs.io/)
- [SHAP解説記事（日本語）](https://qiita.com/tags/shap)
- Summary Plot, Force Plot, Dependence Plotの読み方

### LightGBM LambdaRank理解

- ランキング学習の基礎
- NDCG指標の意味
- ハイパーパラメータの調整

### 競馬ドメイン知識

- 斤量の影響
- 馬場状態の読み方
- 騎手・調教師の実力評価

---

**最終更新:** 2025-11-13  
**バージョン:** 1.0  
**作成者:** AI Assistant (GitHub Copilot)
