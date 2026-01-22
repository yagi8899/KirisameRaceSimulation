# 要件定義書: 競馬賭けシミュレーションシステム

## 1. システム概要

### 1.1 目的
競馬予測システムの出力（予測結果ファイル）を入力として、様々な賭け戦略・資金管理手法をシミュレーションし、期待収益・リスク指標を算出する**汎用シミュレーションシステム**を構築する。

### 1.2 設計思想
- **汎用性**: 特定の目標金額に縛られず、任意のパラメータでシミュレーション可能
- **独立性**: 予測システムとは別システムとして新規構築
- **入力統一**: 予測結果ファイル（TSV形式）を唯一のデータソースとする
- **拡張性**: 新しい賭け戦略・評価指標を容易に追加可能

### 1.3 システム構成図
```
┌─────────────────────────────────────────────────────────────────┐
│                 競馬賭けシミュレーションシステム                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  予測結果     │    │  戦略エンジン  │    │  資金管理     │      │
│  │  ローダー     │───▶│              │───▶│  エンジン     │      │
│  │              │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              シミュレーションエンジン                   │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐      │      │
│  │  │ 単純シミュ   │  │ モンテカルロ │  │ Walk-Forward│      │      │
│  │  │ レーション  │  │ シミュレーション│  │ シミュレーション│      │      │
│  │  └────────────┘  └────────────┘  └────────────┘      │      │
│  └──────────────────────────────────────────────────────┘      │
│                            │                                   │
│                            ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                   評価・レポートエンジン                 │      │
│  │  ・統計サマリー ・グラフ生成 ・リスク指標 ・Go/No-Go判定 │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          ▲
          │
┌─────────────────────┐
│   予測結果ファイル     │
│   (TSV形式)          │
│   ・predicted_results│
│   ・穴馬確率          │
│   ・オッズ情報        │
└─────────────────────┘
```

---

## 2. 入力データ仕様

### 2.1 予測結果ファイル形式
**ファイル形式**: TSV（タブ区切り）

**必須カラム**:
| カラム名 | 型 | 説明 |
|---------|-----|------|
| 競馬場 | string | 競馬場名（東京、京都など） |
| 開催年 | int | 開催年（2023など） |
| 開催日 | int | 開催日（MMDD形式：128 = 1月28日） |
| レース番号 | int | レース番号（1-12） |
| 芝ダ区分 | string | 芝 or ダート |
| 距離 | int | 距離（メートル） |
| 馬番 | int | 馬番 |
| 馬名 | string | 馬名 |
| 単勝オッズ | float | 単勝オッズ |
| 人気順 | int | 人気順位 |
| 確定着順 | int | 実際の着順 |
| 予測順位 | int | モデルによる予測順位 |
| 予測スコア | float | ランキング学習のスコア |

**オプションカラム（穴馬予測）**:
| カラム名 | 型 | 説明 |
|---------|-----|------|
| 穴馬確率 | float | 穴馬である確率（0-1） |
| 穴馬候補 | int | 穴馬候補フラグ（0/1） |
| 実際の穴馬 | int | 実際に穴馬だったか（0/1） |

**オプションカラム（馬券情報）**:
| カラム名 | 型 | 説明 |
|---------|-----|------|
| 複勝1着馬番, 複勝1着オッズ, ... | - | 複勝情報 |
| 馬連馬番1, 馬連馬番2, 馬連オッズ | - | 馬連情報 |
| ワイド1_2馬番1, ..., ワイド1_2オッズ, ... | - | ワイド情報 |
| 馬単馬番1, 馬単馬番2, 馬単オッズ | - | 馬単情報 |
| ３連複オッズ | float | 三連複オッズ |

### 2.2 レース単位への集約
予測結果ファイルは馬単位のデータだが、シミュレーションではレース単位に集約して処理する。

**レース識別キー**: `(競馬場, 開催年, 開催日, レース番号)`

---

## 3. シミュレーション設定

### 3.1 基本パラメータ
```yaml
simulation:
  # 資金設定
  initial_fund: 250000        # 初期資金（円）
  target_fund: null           # 目標資金（null=無制限）
  bankruptcy_threshold: 0.1   # 破産閾値（初期資金比、0.1=10%以下で破産）
  
  # シミュレーション期間
  start_date: "2023-01-01"    # 開始日
  end_date: "2025-12-31"      # 終了日
  
  # モンテカルロ設定
  monte_carlo:
    enabled: true
    iterations: 10000         # 試行回数
    random_seed: 42           # 再現性のためのシード
```

### 3.2 賭け金管理パラメータ
```yaml
betting:
  # 賭け金計算方式
  method: "kelly"             # fixed / percentage / kelly
  
  # 固定賭け金方式（method: fixed）
  fixed:
    amount: 1000              # 1点あたり固定金額
  
  # 資金比率方式（method: percentage）
  percentage:
    rate: 0.02                # 資金の2%を1レースに投入
  
  # ケリー基準方式（method: kelly）
  kelly:
    fraction: 0.25            # クォーターケリー（25%）
    max_bet_ratio: 0.05       # 1点あたり最大資金比率
  
  # 共通制約
  constraints:
    min_bet: 100              # 最小賭け金
    max_bet_per_ticket: 5000  # 1点あたり上限
    max_bet_per_race: 20000   # 1レースあたり上限
    max_bet_per_day: 50000    # 1日あたり上限
```

### 3.2.1 オッズインパクトの現実的制約
競馬のオッズは賭け金によって変動する。特に穴馬（7-12番人気）は元々の売上が少ないため影響を受けやすい。

**オッズ変動の目安**（JRA中央競馬の場合）:
| 馬券種 | 売上規模 | 1点あたり安全圏 | 影響が出始める金額 |
|--------|----------|-----------------|-------------------|
| 単勝 | 数百万〜数千万円 | 〜5,000円 | 10,000円以上 |
| 複勝 | 数百万〜数千万円 | 〜10,000円 | 20,000円以上 |
| 馬連 | 数千万円 | 〜5,000円 | 10,000円以上 |
| ワイド | 数百万円 | 〜3,000円 | 5,000円以上 |
| 三連複 | 数千万円 | 〜3,000円 | 5,000円以上 |
| 三連単 | 数億円 | 〜5,000円 | 10,000円以上 |

**穴馬単勝の特殊性**:
- 7-12番人気の単勝オッズは通常30倍〜100倍
- 売上が少ないため、1万円賭けるだけでオッズが数倍下がる可能性あり
- **対策**: 単勝よりも馬連・三連複で穴馬を「相手」として活用

### 3.2.2 馬券種別の賭け金上限（推奨値）
| 馬券種 | 1点あたり上限 | 1レースあたり上限 | 理由 |
|--------|--------------|------------------|------|
| 単勝（本命） | 5,000円 | 5,000円 | オッズ変動を最小化 |
| 単勝（穴馬） | 1,000円 | 1,000円 | 穴馬はオッズ影響大 |
| 複勝 | 5,000円 | 10,000円 | 比較的影響小 |
| 馬連 | 3,000円 | 15,000円 | メイン馬券として活用 |
| ワイド | 3,000円 | 12,000円 | サブ・保険として活用、穴馬予測との相性◎ |
| 三連複 | 2,000円 | 20,000円 | 高配当狙い |

### 3.2.3 資金規模別の投資上限（参考値）
| フェーズ | 資金規模 | 1日上限 | 1レース上限 |
|----------|----------|---------|------------|
| 初期 | 〜50万円 | 25,000円 | 5,000円 |
| 成長期 | 50-100万円 | 50,000円 | 10,000円 |
| 拡大期 | 100万円〜 | 100,000円 | 20,000円 |

### 3.3 馬券種別パラメータ
```yaml
ticket_types:
  win:                        # 単勝
    enabled: true
    allocation: 0.10          # 資金配分比率
  place:                      # 複勝
    enabled: true
    allocation: 0.10
  quinella:                   # 馬連
    enabled: true
    allocation: 0.35
  wide:                       # ワイド
    enabled: true
    allocation: 0.30
  trio:                       # 三連複
    enabled: true
    allocation: 0.15
  exacta:                     # 馬単
    enabled: false
    allocation: 0.00
  trifecta:                   # 三連単
    enabled: false
    allocation: 0.00
```

### 3.4 戦略パラメータ
```yaml
strategy:
  # 本命馬選択
  favorite:
    method: "predicted_rank"  # predicted_rank / popularity / score
    top_n: 2                  # 上位N頭を軸馬として使用
  
  # 穴馬選択
  longshot:
    enabled: true
    method: "upset_prob"      # upset_prob / popularity_range
    threshold: 0.20           # 穴馬確率の閾値
    popularity_range: [7, 12] # 人気順の範囲
    max_count: 3              # 最大選択頭数
  
  # レース選択フィルタ
  race_filter:
    min_runners: 12           # 最小出走頭数
    track: null               # 競馬場フィルタ（null=全て）
    surface: null             # 芝/ダートフィルタ
    distance_range: null      # 距離範囲フィルタ
    confidence_threshold: 0.0 # 信頼度閾値
```

### 3.5 条件別設定（オプション）
```yaml
conditional_settings:
  # 競馬場別の設定上書き
  by_track:
    "函館":
      allocation_boost: 1.5   # 資金配分を1.5倍
      kelly_fraction: 0.30    # ケリー係数を上げる
    "東京":
      allocation_boost: 1.2
  
  # 馬券種×条件の設定
  by_ticket_track:
    quinella:
      "函館": { allocation: 0.50 }
```

### 3.5.1 条件別ROI実績（参考値: 2024年穴馬予測）
高パフォーマンス条件への資金傾斜配分の根拠となる実績データ。

| 条件 | ROI | 優先度 |
|------|-----|--------|
| 函館芝中長距離 | 492% | ★★★★★ |
| 東京芝中長距離 | 378% | ★★★★☆ |
| 阪神芝中長距離 | 267% | ★★★☆☆ |
| 阪神芝短距離 | 205% | ★★★☆☆ |

### 3.5.2 資金配分Tier（推奨）
| 条件カテゴリ | ROI | 資金配分比率 |
|--------------|-----|-------------|
| Tier1（ROI 400%+） | 400%+ | 40% |
| Tier2（ROI 250-400%） | 250-400% | 35% |
| Tier3（ROI 150-250%） | 150-250% | 20% |
| Tier4（ROI 150%未満） | <150% | 5%（様子見） |

### 3.6 レース選択フィルタ（詳細）

#### 3.6.1 参加条件
| 条件 | 基準 | 理由 |
|------|------|------|
| 出走頭数 | 12頭以上 | 穴馬の出現確率確保 |
| レース信頼度 | 0.6以上 | race_confidence_scorer.pyの出力 |
| 穴馬検出 | 1頭以上 | 戦略の前提条件 |
| モデル対応 | 対応モデルあり | 未対応条件は除外 |

#### 3.6.2 見送り条件
- 新馬戦・未勝利戦（データ不足）
- 荒天時（予測精度低下）
- 穴馬検出なし（戦略適用不可）

#### 3.6.3 条件別フィルタ（競馬場・距離・芝ダート）

過去の実績データに基づき、ROIが低い条件を除外または資金を絞る。

**競馬場フィルタ**:
```yaml
track_filter:
  mode: "whitelist"           # whitelist / blacklist / tier
  
  # ホワイトリスト方式（指定競馬場のみ参加）
  whitelist:
    - "函館"    # ROI 492%
    - "東京"    # ROI 378%
    - "阪神"    # ROI 267%
  
  # ブラックリスト方式（指定競馬場を除外）
  blacklist:
    - "小倉"    # ROIが低い場合
    - "新潟"
  
  # Tier方式（ROI実績に基づく段階的資金配分）
  tier_allocation:
    "函館": 1.5    # 資金配分1.5倍
    "東京": 1.2
    "阪神": 1.0
    "京都": 1.0
    "中山": 0.8
    "中京": 0.5
    "小倉": 0.3    # 資金配分0.3倍（様子見）
```

**芝/ダートフィルタ**:
```yaml
surface_filter:
  mode: "whitelist"
  whitelist:
    - "芝"                    # 芝のみ参加（穴馬予測は芝に強い傾向）
  # または両方参加
  # whitelist: ["芝", "ダート"]
```

**距離フィルタ**:
```yaml
distance_filter:
  mode: "range"               # range / category
  
  # 距離範囲指定
  range:
    min: 1600                 # 1600m以上
    max: 2400                 # 2400m以下
  
  # カテゴリ指定
  categories:
    - "中距離"                 # 1600-2200m
    - "長距離"                 # 2200m以上
  # 除外: 短距離（1600m未満）は予測精度が低い傾向
```

**複合条件フィルタ（推奨設定例）**:
```yaml
# 高ROI条件に絞った攻めの設定
aggressive_filter:
  track_whitelist: ["函館", "東京", "阪神"]
  surface: "芝"
  distance_min: 1600
  distance_max: 2400

# 安定重視の設定
conservative_filter:
  track_blacklist: ["小倉", "新潟"]  # 低ROI競馬場を除外
  surface: null                      # 芝/ダート両方
  distance_min: 1400
  distance_max: 3000
```

**条件別ROI実績の活用**:
| 条件 | ROI | フィルタ推奨 |
|------|-----|-------------|
| 函館・芝・中長距離 | 492% | ✅ 積極参加 |
| 東京・芝・中長距離 | 378% | ✅ 積極参加 |
| 阪神・芝・全距離 | 236% | ✅ 参加 |
| 中山・芝・中長距離 | 180% | ⚠️ 控えめに |
| 小倉・芝 | 95% | ❌ 見送り推奨 |
| ダート全般 | 120% | ⚠️ 条件次第 |

### 3.7 期待値ベースのレース参加判定

**最重要**: 収支をプラスにするためには、「期待値（Expected Value）」が1.0を超えるレースのみに参加することが必須。

#### 3.7.1 期待値の計算
```
期待値（EV） = 勝率 × オッズ

例1: 穴馬確率=0.25, オッズ=15倍
  → EV = 0.25 × 15 = 3.75 ✅ 超高期待値

例2: 穴馬確率=0.10, オッズ=8倍
  → EV = 0.10 × 8 = 0.80 ❌ 賭けるほど損

例3: 穴馬確率=0.20, オッズ=4倍
  → EV = 0.20 × 4 = 0.80 ❌ オッズが低すぎ
```

#### 3.7.2 期待値フィルタ設定
```yaml
expected_value_filter:
  enabled: true
  
  # 最小期待値（これ未満は見送り）
  min_ev: 1.0
  
  # 安全マージン（推奨: 1.1〜1.2）
  # EVは推定値なので、マージンを持たせる
  min_ev_with_margin: 1.1
  
  # 馬券種別の期待値閾値
  by_ticket_type:
    win:      1.2    # 単勝は高めに設定（分散が大きい）
    place:    1.1    # 複勝
    quinella: 1.1    # 馬連
    wide:     1.0    # ワイドは的中率重視で低めOK
    trio:     1.2    # 三連複は高めに設定
```

#### 3.7.3 オッズバリュー判定
予測確率から算出した「適正オッズ」と実際のオッズを比較。

```yaml
odds_value_filter:
  enabled: true
  
  # バリュー比率 = 適正オッズ ÷ 実際オッズ
  # 1.0超 = オッズに対して過小評価されている（お買い得）
  min_value_ratio: 1.2    # 20%以上のバリューがあれば参加
  
  # 計算例
  # 予測勝率: 20% → 適正オッズ = 1/0.20 = 5.0倍
  # 実際オッズ: 8.0倍
  # バリュー比率 = 8.0 / 5.0 = 1.6 ✅ 60%のバリュー
```

#### 3.7.4 統合判定ロジック
```python
def should_enter_race(race, config) -> bool:
    """レース参加判定（全条件をANDで評価）"""
    
    # 1. 基本フィルタ
    if race.runners < config.min_runners:
        return False  # 出走頭数不足
    
    if race.confidence < config.confidence_threshold:
        return False  # 信頼度不足
    
    # 2. 条件別フィルタ
    if config.track_filter.mode == "whitelist":
        if race.track not in config.track_filter.whitelist:
            return False  # 対象外競馬場
    
    if config.surface_filter.enabled:
        if race.surface not in config.surface_filter.whitelist:
            return False  # 対象外路面
    
    if config.distance_filter.enabled:
        if not (config.distance_filter.min <= race.distance <= config.distance_filter.max):
            return False  # 対象外距離
    
    # 3. 穴馬検出チェック
    upset_candidates = [h for h in race.horses if h.upset_prob >= config.upset_threshold]
    if len(upset_candidates) == 0:
        return False  # 穴馬なし
    
    # 4. 期待値チェック（最重要）
    best_ev = max(h.upset_prob * h.odds for h in upset_candidates)
    if best_ev < config.min_ev_with_margin:
        return False  # 期待値不足
    
    # 5. オッズバリューチェック
    if config.odds_value_filter.enabled:
        best_value = max(h.odds / (1/h.upset_prob) for h in upset_candidates)
        if best_value < config.min_value_ratio:
            return False  # バリュー不足
    
    return True  # 全条件クリア → 参加
```

#### 3.7.5 判定結果のログ出力例
```
=== レース参加判定 ===
東京11R 天皇賞（秋）

[基本条件]
✅ 出走頭数: 16頭 (≥12)
✅ 信頼度: 0.78 (≥0.6)
✅ モデル対応: あり

[条件フィルタ]
✅ 競馬場: 東京 (ホワイトリスト内)
✅ 路面: 芝
✅ 距離: 2000m (1600-2400m内)

[穴馬検出]
✅ 穴馬候補: 2頭
   - 8番 ダークホース (確率: 0.28, オッズ: 18.5)
   - 12番 ミラクルラン (確率: 0.22, オッズ: 24.0)

[期待値判定]
✅ 最高EV: 5.28 (8番: 0.28 × 18.5)
✅ EVマージン: OK (5.28 > 1.1)

[オッズバリュー判定]
✅ バリュー比率: 1.48 (適正5.56倍 vs 実際8.2倍)

【判定】✅ 参加
```

---

## 4. 戦略エンジン

### 4.1 サポートする賭け戦略

#### 4.1.1 単勝戦略
| 戦略名 | 説明 |
|--------|------|
| `favorite_win` | 予測1位の単勝を購入 |
| `longshot_win` | 穴馬候補の単勝を購入 |
| `value_win` | 期待値が閾値以上の単勝を購入 |

#### 4.1.2 複勝戦略
| 戦略名 | 説明 |
|--------|------|
| `favorite_place` | 予測上位の複勝を購入 |
| `longshot_place` | 穴馬候補の複勝を購入 |

#### 4.1.3 馬連戦略
| 戦略名 | 説明 |
|--------|------|
| `favorite_quinella` | 予測上位2頭の馬連 |
| `favorite_longshot_quinella` | 本命軸-穴馬相手の馬連 |
| `box_quinella` | 予測上位N頭のボックス |

#### 4.1.4 ワイド戦略
| 戦略名 | 説明 |
|--------|------|
| `favorite_wide` | 予測上位2頭のワイド |
| `favorite_longshot_wide` | 本命軸-穴馬相手のワイド（保険） |
| `box_wide` | 予測上位N頭のボックス |

**ワイド戦略が有効な理由:**
1. **モデルとの完全一致**: 穴馬予測は「3着以内」を予測 → ワイドの的中条件と同じ
2. **保険効果**: 馬連（1-2着限定）が外れても、穴馬が3着ならワイドで回収
3. **複数的中の可能性**: ワイドは3組まで的中するため、本命2頭+穴馬で最大2点的中
4. **連敗リスク軽減**: 的中率が高いため、資金の安定性が向上

**ワイド活用の具体例:**
```
レース結果: 1着=本命A、2着=人気馬、3着=穴馬X

【馬連】本命A - 穴馬X → ハズレ（穴馬が3着のため）
【ワイド】本命A - 穴馬X → 的中！（3着以内同士）

→ 馬連は外れたが、ワイドで回収できた
```

#### 4.1.5 三連複戦略
| 戦略名 | 説明 |
|--------|------|
| `favorite_trio` | 予測上位3頭の三連複 |
| `favorite2_longshot_trio` | 本命2頭軸-穴馬流し |
| `formation_trio` | フォーメーション購入 |

### 4.2 複合戦略
複数の戦略を組み合わせて使用可能。

```yaml
combined_strategy:
  name: "balanced_longshot"
  description: "本命軸×穴馬のバランス型"
  components:
    - strategy: "favorite_longshot_quinella"
      weight: 0.40
    - strategy: "favorite_longshot_wide"
      weight: 0.35
    - strategy: "favorite2_longshot_trio"
      weight: 0.25
```

---

## 5. 資金管理エンジン

### 5.1 賭け金計算方式

#### 5.1.1 固定賭け金方式（Fixed）
```
賭け金 = 固定金額（例: 1000円）
```
- シンプルで分かりやすい
- 資金増減に関係なく一定

#### 5.1.2 資金比率方式（Percentage）
```
賭け金 = 現在資金 × 比率（例: 2%）
```
- 資金が増えると賭け金も増加
- 資金が減ると賭け金も減少（自動的にリスク軽減）

#### 5.1.3 ケリー基準方式（Kelly）
```
ケリー比率 = (勝率 × オッズ - 1) / (オッズ - 1)
賭け金 = 現在資金 × ケリー比率 × フラクション
```
- 理論的に最適な賭け金
- フラクション（1/4, 1/2）で保守的に調整

### 5.2 制約チェック
賭け金計算後、以下の制約をチェック:
1. 最小賭け金（100円）未満 → 見送り
2. 1点上限超過 → 上限に丸め
3. 1レース上限超過 → 比例配分で調整
4. 1日上限超過 → 以降のレースは見送り
5. 残資金不足 → 購入可能な範囲に調整

---

## 6. シミュレーションエンジン

### 6.1 単純シミュレーション
過去の予測結果を時系列順に処理し、資金推移を計算。

**処理フロー**:
```
1. 予測結果を日付順にソート
2. 各レースについて:
   a. レース選択フィルタを適用
   b. 戦略に基づき購入馬券を決定
   c. 賭け金を計算（資金管理エンジン）
   d. 的中判定・払戻計算
   e. 資金を更新
   f. 破産チェック
3. 最終結果を集計
```

### 6.2 モンテカルロシミュレーション
過去の的中/不的中パターンをもとに、将来の資金推移を確率的にシミュレーション。

**手法1: ブートストラップ法**
```
1. 過去のレース結果をプールとして保持
2. 各試行で:
   a. プールからランダムにレースをサンプリング（復元抽出）
   b. 時系列に沿って資金推移を計算
   c. 最終資金を記録
3. 統計量を算出
```

**手法2: 確率分布ベース法**
```
1. 過去の結果から的中率・配当分布を推定
2. 各試行で:
   a. 的中/不的中を確率的に決定
   b. 的中時は配当分布からサンプリング
   c. 資金推移を計算
3. 統計量を算出
```

### 6.3 Walk-Forwardシミュレーション
学習期間と検証期間をスライドさせながらシミュレーション。

```
期間1: 学習[2014-2020] → 検証[2021]
期間2: 学習[2015-2021] → 検証[2022]
期間3: 学習[2016-2022] → 検証[2023]
期間4: 学習[2017-2023] → 検証[2024]
期間5: 学習[2018-2024] → 検証[2025]
```

**出力**: 各期間の成績を比較し、安定性を評価

---

## 7. 評価指標

### 7.1 収益指標
| 指標 | 説明 | 計算式 |
|------|------|--------|
| ROI | 投資収益率 | 回収額 / 投資額 × 100% |
| 純利益 | 最終利益額 | 最終資金 - 初期資金 |
| 年率リターン | 年換算リターン | (最終資金/初期資金)^(1/年数) - 1 |
| 複利成長率 | CAGR | 同上 |

### 7.2 リスク指標
| 指標 | 説明 | 計算式 |
|------|------|--------|
| 最大ドローダウン | ピークからの最大下落率 | max((peak - trough) / peak) |
| 破産確率 | 破産に至る確率 | 破産シナリオ数 / 総シナリオ数 |
| シャープレシオ | リスク調整後リターン | (平均リターン - 無リスク金利) / 標準偏差 |
| ソルティノレシオ | 下方リスク調整後リターン | (平均リターン - 目標) / 下方偏差 |
| VaR (95%) | 95%信頼区間での最大損失 | 5%分位点 |
| CVaR (95%) | VaRを超える損失の平均 | VaR以下の平均損失 |

### 7.3 的中指標
| 指標 | 説明 |
|------|------|
| 的中率 | 的中レース数 / 購入レース数 |
| 回収率 | 払戻金額 / 購入金額 |
| 連敗数 | 最大連続不的中数 |
| 的中間隔 | 的中から次の的中までの平均レース数 |

### 7.4 分布指標（モンテカルロ）
| 指標 | 説明 |
|------|------|
| 中央値到達資金 | 50%分位点の最終資金 |
| 5%分位点 | 下位5%シナリオの最終資金 |
| 95%分位点 | 上位5%シナリオの最終資金 |
| 目標達成確率 | 目標資金に到達する確率 |
| 期待到達日数 | 目標達成までの期待日数 |

---

## 8. 出力・レポート

### 8.1 出力ファイル
| ファイル | 形式 | 内容 |
|----------|------|------|
| `simulation_result.json` | JSON | シミュレーション結果の全データ |
| `simulation_summary.txt` | Text | 統計サマリー |
| `fund_history.csv` | CSV | 資金推移データ |
| `bet_history.csv` | CSV | 全購入履歴 |
| `charts/` | PNG/HTML | 各種グラフ（詳細は8.4参照） |

### 8.2 サマリーレポート例
```
========================================
競馬賭けシミュレーション結果
========================================

【設定】
初期資金: 250,000円
シミュレーション期間: 2023-01-01 〜 2025-12-31
賭け金方式: ケリー基準（25%）
戦略: 本命軸×穴馬（馬連40%/ワイド35%/三連複25%）

【収益結果】
最終資金: 1,234,567円
純利益: +984,567円
ROI: 493.8%
年率リターン: 71.2%

【リスク指標】
最大ドローダウン: 32.1%
最大連敗: 18連敗
シャープレシオ: 1.85

【的中成績】
購入レース数: 1,234
的中レース数: 287
的中率: 23.3%
回収率: 198.5%

【モンテカルロ分析】(10,000試行)
破産確率: 2.3%
中央値到達資金: 892,345円
5%分位点: 187,234円
95%分位点: 2,345,678円
目標達成確率（500万円）: 18.7%

【判定】
✅ Go条件クリア
- 破産確率 2.3% < 5%
- 期待ROI 493.8% > 150%
- 最大ドローダウン 32.1% < 50%

========================================
```

### 8.3 Go/No-Go判定機能
設定した基準に基づき、実運用可否を自動判定。

```yaml
go_nogo_criteria:
  go_conditions:  # 全て満たす必要あり
    max_bankruptcy_prob: 0.05      # 破産確率5%以下
    min_expected_roi: 1.50         # 期待ROI 150%以上
    max_drawdown_95percentile: 0.50  # 最大ドローダウン50%以下
    min_stable_years: 2            # 3年中2年以上でROI>100%
  
  nogo_conditions:  # 1つでも該当したらNo-Go
    bankruptcy_prob_limit: 0.10    # 破産確率10%以上
    min_roi_limit: 1.20            # 期待ROI 120%未満
    max_consecutive_losses: 30     # 30連敗以上でNo-Go
```

### 8.4 グラフ・可視化一覧

シミュレーション結果を多角的に分析するため、以下のグラフを生成する。

#### 8.4.1 資金推移系グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **資金推移（単一）** | `fund_history.png` | 時系列での資金推移 | 基本的な成績確認 |
| **資金推移（複数シナリオ）** | `fund_scenarios.png` | モンテカルロの複数シナリオを重ねて表示 | リスク範囲の視覚化 |
| **資金推移（信頼区間）** | `fund_confidence_band.png` | 5-95%信頼区間を帯で表示 | 期待される資金レンジ |
| **対数スケール資金推移** | `fund_log_scale.png` | 対数軸で表示（大きな変動を見やすく） | 長期トレンド分析 |
| **累積リターン** | `cumulative_return.png` | 初期資金からの累積リターン率 | パフォーマンス比較 |

#### 8.4.2 リスク分析系グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **ドローダウン推移** | `drawdown_history.png` | 時系列でのドローダウン推移 | リスク発生タイミング把握 |
| **最大ドローダウン分布** | `max_drawdown_dist.png` | モンテカルロでの最大DD分布 | リスク許容度の判断 |
| **水中グラフ（Underwater）** | `underwater.png` | ピークからの下落を水面下として表示 | 回復期間の可視化 |
| **VaR/CVaRグラフ** | `var_cvar.png` | 信頼水準別のVaR/CVaR | リスク定量化 |
| **破産確率推移** | `bankruptcy_prob_over_time.png` | 時間経過での破産確率変化 | いつ破産リスクが高いか |

#### 8.4.3 収益分析系グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **最終資金分布（ヒストグラム）** | `final_fund_histogram.png` | モンテカルロの最終資金分布 | 期待値とばらつき |
| **最終資金分布（箱ひげ図）** | `final_fund_boxplot.png` | 四分位で表示 | 外れ値の確認 |
| **ROI分布** | `roi_distribution.png` | ROIのヒストグラム | 戦略の安定性 |
| **月次リターン** | `monthly_return.png` | 月ごとのリターン棒グラフ | 季節性の確認 |
| **年次リターン比較** | `yearly_return.png` | 年ごとのリターン比較 | 安定性の確認 |
| **リターン分布（正規QQ）** | `return_qq_plot.png` | リターンの正規性チェック | 統計的検証 |

#### 8.4.4 的中・馬券分析系グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **的中率推移** | `hit_rate_history.png` | 移動平均での的中率推移 | モデル劣化の検知 |
| **回収率推移** | `roi_history.png` | 移動平均での回収率推移 | トレンド把握 |
| **連敗ヒストグラム** | `losing_streak_hist.png` | 連敗回数の分布 | 連敗耐性の確認 |
| **馬券種別収支** | `ticket_type_pnl.png` | 馬連/ワイド/三連複別の収支 | 馬券種の有効性 |
| **馬券種別的中率** | `ticket_type_hit_rate.png` | 馬券種別の的中率比較 | 戦略調整の参考 |
| **オッズ別的中率** | `odds_vs_hit_rate.png` | オッズ帯別の的中率 | 期待値分析 |
| **配当分布** | `payout_distribution.png` | 的中時の配当分布 | 高配当の頻度 |

#### 8.4.5 条件別分析グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **競馬場別ROI** | `roi_by_track.png` | 競馬場ごとのROI棒グラフ | 得意競馬場の特定 |
| **競馬場別収支** | `pnl_by_track.png` | 競馬場ごとの収支 | 資金配分の最適化 |
| **距離別ROI** | `roi_by_distance.png` | 距離帯別のROI | 距離適性の確認 |
| **芝/ダート別ROI** | `roi_by_surface.png` | 路面別のROI | 路面適性の確認 |
| **人気別的中率** | `hit_rate_by_popularity.png` | 人気帯別の的中率 | 穴馬予測の検証 |
| **穴馬確率別的中率** | `hit_rate_by_upset_prob.png` | 穴馬確率帯別の的中率 | 閾値最適化の参考 |
| **ヒートマップ（競馬場×馬券種）** | `heatmap_track_ticket.png` | 条件の組み合わせ別ROI | 最適条件の発見 |

#### 8.4.6 戦略比較グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **戦略別資金推移** | `strategy_comparison.png` | 複数戦略の資金推移を重ねて表示 | 戦略選択の参考 |
| **戦略別リスクリターン** | `risk_return_scatter.png` | 横軸:リスク、縦軸:リターンの散布図 | 効率的フロンティア |
| **戦略別シャープレシオ** | `sharpe_ratio_comparison.png` | 戦略別のシャープレシオ比較 | リスク調整後比較 |
| **パラメータ感度分析** | `parameter_sensitivity.png` | パラメータ変化による成績変化 | 最適パラメータ探索 |

#### 8.4.7 Walk-Forward分析グラフ

| グラフ名 | ファイル名 | 説明 | 用途 |
|---------|-----------|------|------|
| **期間別ROI推移** | `walk_forward_roi.png` | 各検証期間のROI推移 | 安定性の確認 |
| **期間別ドローダウン** | `walk_forward_dd.png` | 各期間の最大DD | リスクの安定性 |
| **学習期間感度** | `train_period_sensitivity.png` | 学習期間長による成績変化 | 最適学習期間 |

#### 8.4.8 インタラクティブグラフ（HTML）

静的PNGに加え、インタラクティブなHTMLグラフも生成。

| グラフ名 | ファイル名 | 機能 |
|---------|-----------|------|
| **資金推移（インタラクティブ）** | `fund_interactive.html` | ズーム、ホバーで詳細表示 |
| **シナリオ探索** | `scenario_explorer.html` | 個別シナリオの選択・表示 |
| **条件別ダッシュボード** | `condition_dashboard.html` | フィルタで条件絞り込み |

---

## 9. GUI/インターフェース設計

### 9.1 採用方針

**個人利用を目的とし、サービス化は予定していない。**

以下の段階的アプローチで実装する：

```
Phase 1: CLI + 静的グラフ（PNG）  ← 基本機能
    ↓
Phase 2: Jupyter Notebook        ← 分析・検証フェーズ
    ↓
Phase 3: Streamlit              ← 最終形（個人利用ダッシュボード）
```

### 9.2 インターフェース方式の比較

| 方式 | メリット | デメリット | 本プロジェクトでの採用 |
|------|---------|-----------|---------------------|
| **CLI（コマンドライン）** | シンプル、自動化しやすい、軽量 | 視覚的な操作が難しい | ✅ 採用（Phase 1） |
| **Jupyter Notebook** | インタラクティブ、グラフ埋め込み、試行錯誤向き | 本番運用には不向き | ✅ 採用（Phase 2） |
| **Streamlit** | 簡単にWebアプリ化、Pythonのみ、リアルタイム更新 | 大規模アプリには不向き | ✅ 採用（Phase 3・最終形） |
| **Dash/Plotly** | 高度なダッシュボード、カスタマイズ性高 | 学習コスト高め | ❌ 不採用 |
| **Flask/FastAPI + React** | 完全なカスタマイズ、スケーラブル | 開発コスト大 | ❌ 不採用（サービス化しないため） |

### 9.3 Streamlit ダッシュボード設計案（最終形）

#### 9.3.1 画面構成
```
┌─────────────────────────────────────────────────────────────────┐
│  競馬賭けシミュレーター                           [設定] [実行]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ サイドバー ─────┐  ┌─ メインエリア ──────────────────────┐ │
│  │                  │  │                                      │ │
│  │ 📁 データ選択    │  │  [サマリー] [グラフ] [詳細] [比較]   │ │
│  │  └ ファイル選択  │  │  ─────────────────────────────────   │ │
│  │                  │  │                                      │ │
│  │ 💰 資金設定      │  │  ┌─────────────────────────────────┐ │ │
│  │  └ 初期資金      │  │  │                                 │ │ │
│  │  └ 目標資金      │  │  │       資金推移グラフ             │ │ │
│  │  └ 破産閾値      │  │  │                                 │ │ │
│  │                  │  │  └─────────────────────────────────┘ │ │
│  │ 🎯 戦略設定      │  │                                      │ │
│  │  └ 馬券種選択    │  │  ┌──────────┐ ┌──────────┐          │ │
│  │  └ 資金配分      │  │  │ ROI      │ │ 最大DD   │          │ │
│  │  └ 穴馬閾値      │  │  │ 285.3%   │ │ 28.5%    │          │ │
│  │                  │  │  └──────────┘ └──────────┘          │ │
│  │ 🎲 シミュレーション│  │                                      │ │
│  │  └ モンテカルロ  │  │  ┌─────────────────────────────────┐ │ │
│  │  └ 試行回数      │  │  │     条件別ROIヒートマップ        │ │ │
│  │                  │  │  │                                 │ │ │
│  │ [▶ 実行]        │  │  └─────────────────────────────────┘ │ │
│  │                  │  │                                      │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 9.3.2 主要画面

| 画面名 | 機能 | 主なグラフ |
|--------|------|-----------|
| **サマリー** | 主要指標の一覧表示 | KPIカード、ゲージ |
| **資金推移** | 資金の時系列変化 | 資金推移、信頼区間、ドローダウン |
| **リスク分析** | リスク指標の詳細 | DD分布、VaR、破産確率 |
| **的中分析** | 的中率・回収率の詳細 | 的中率推移、連敗分布、配当分布 |
| **条件別分析** | 競馬場・馬券種別の成績 | ヒートマップ、棒グラフ |
| **戦略比較** | 複数戦略の比較 | 比較グラフ、リスクリターン散布図 |
| **モンテカルロ** | シミュレーション結果詳細 | シナリオ一覧、分布グラフ |
| **設定** | パラメータ設定画面 | - |

#### 9.3.3 インタラクティブ機能

| 機能 | 説明 |
|------|------|
| **リアルタイム計算** | パラメータ変更時に即座に再計算・グラフ更新 |
| **フィルタリング** | 期間・競馬場・馬券種で絞り込み |
| **ドリルダウン** | グラフクリックで詳細表示 |
| **エクスポート** | CSV/PNG/PDFでダウンロード |
| **設定保存/読込** | YAML形式で設定を保存・読込 |
| **比較モード** | 2つの戦略を並べて比較 |

### 9.4 Jupyter Notebook テンプレート

分析フェーズ用のNotebookテンプレートも用意。

```
notebooks/
├── 01_data_exploration.ipynb     # データ探索
├── 02_backtest_analysis.ipynb    # バックテスト分析
├── 03_monte_carlo_analysis.ipynb # モンテカルロ分析
├── 04_strategy_comparison.ipynb  # 戦略比較
├── 05_parameter_tuning.ipynb     # パラメータチューニング
└── 06_report_generation.ipynb    # レポート生成
```

### 9.5 GUI実装の工数見積もり

| 方式 | 追加工数 | 本プロジェクトでの実装 |
|------|---------|---------------------|
| CLI + 静的グラフ | 0日 | ✅ 基本実装に含む |
| Jupyter Notebook | +2日 | ✅ テンプレート6種作成 |
| Streamlit（基本） | +5日 | ✅ 主要画面のみ |
| Streamlit（フル） | +10日 | ✅ 全画面・全機能（最終形） |
| Dash/Plotly | - | ❌ 不採用 |

---

## 10. システム実装設計

### 10.1 ディレクトリ構成
```
betting_simulator/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default_config.yaml       # デフォルト設定
│   ├── strategies/               # 戦略定義
│   │   ├── balanced_longshot.yaml
│   │   ├── conservative.yaml
│   │   └── aggressive.yaml
│   └── examples/                 # サンプル設定
│       ├── 50x_target.yaml       # 50万→500万設定
│       └── low_risk.yaml         # 低リスク設定
├── src/
│   ├── __init__.py
│   ├── loader/                   # データローダー
│   │   ├── __init__.py
│   │   ├── prediction_loader.py  # 予測結果ファイル読み込み
│   │   └── race_aggregator.py    # レース単位への集約
│   ├── filter/                   # レースフィルタ（参加判定）
│   │   ├── __init__.py
│   │   ├── race_filter.py        # 基本フィルタ（頭数・信頼度）
│   │   ├── condition_filter.py   # 条件フィルタ（競馬場・距離・芝ダート）
│   │   ├── ev_filter.py          # 期待値フィルタ
│   │   └── value_filter.py       # オッズバリューフィルタ
│   ├── strategy/                 # 戦略エンジン
│   │   ├── __init__.py
│   │   ├── base_strategy.py      # 戦略基底クラス
│   │   ├── ticket_strategies.py  # 馬券種別戦略
│   │   └── combined_strategy.py  # 複合戦略
│   ├── fund_manager/             # 資金管理エンジン
│   │   ├── __init__.py
│   │   ├── base_manager.py       # 基底クラス
│   │   ├── fixed_manager.py      # 固定賭け金
│   │   ├── percentage_manager.py # 資金比率
│   │   └── kelly_manager.py      # ケリー基準
│   ├── simulator/                # シミュレーションエンジン
│   │   ├── __init__.py
│   │   ├── simple_simulator.py   # 単純シミュレーション
│   │   ├── monte_carlo.py        # モンテカルロ
│   │   └── walk_forward.py       # Walk-Forward
│   ├── evaluator/                # 評価エンジン
│   │   ├── __init__.py
│   │   ├── metrics.py            # 評価指標計算
│   │   ├── go_nogo.py            # Go/No-Go判定
│   │   └── reporter.py           # レポート生成
│   └── visualizer/               # 可視化
│       ├── __init__.py
│       ├── base_chart.py         # グラフ基底クラス
│       ├── fund_charts.py        # 資金推移系グラフ
│       ├── risk_charts.py        # リスク分析系グラフ
│       ├── profit_charts.py      # 収益分析系グラフ
│       ├── ticket_charts.py      # 馬券分析系グラフ
│       ├── condition_charts.py   # 条件別分析グラフ
│       ├── comparison_charts.py  # 戦略比較グラフ
│       ├── walkforward_charts.py # Walk-Forward分析グラフ
│       ├── interactive.py        # インタラクティブHTML生成
│       └── chart_config.py       # グラフ設定・スタイル
├── dashboard/                    # Streamlitダッシュボード
│   ├── __init__.py
│   ├── app.py                    # メインアプリ
│   ├── pages/
│   │   ├── summary.py            # サマリー画面
│   │   ├── fund_analysis.py      # 資金推移画面
│   │   ├── risk_analysis.py      # リスク分析画面
│   │   ├── hit_analysis.py       # 的中分析画面
│   │   ├── condition_analysis.py # 条件別分析画面
│   │   ├── strategy_compare.py   # 戦略比較画面
│   │   ├── monte_carlo.py        # モンテカルロ画面
│   │   └── settings.py           # 設定画面
│   └── components/
│       ├── sidebar.py            # サイドバー
│       ├── kpi_cards.py          # KPIカード
│       └── filters.py            # フィルターコンポーネント
├── notebooks/                    # Jupyter Notebook
│   ├── 01_data_exploration.ipynb
│   ├── 02_backtest_analysis.ipynb
│   ├── 03_monte_carlo_analysis.ipynb
│   ├── 04_strategy_comparison.ipynb
│   ├── 05_parameter_tuning.ipynb
│   └── 06_report_generation.ipynb
├── tests/                        # テスト
│   ├── test_loader.py
│   ├── test_strategy.py
│   ├── test_simulator.py
│   └── test_evaluator.py
├── scripts/                      # 実行スクリプト
│   ├── run_simulation.py         # メイン実行
│   ├── compare_strategies.py     # 戦略比較
│   └── optimize_params.py        # パラメータ最適化
└── output/                       # 出力ディレクトリ
    └── (シミュレーション結果)
```

### 10.2 主要クラス設計

#### 10.2.1 PredictionLoader
```python
class PredictionLoader:
    """予測結果ファイルを読み込み、レース単位に集約"""
    
    def load(self, file_path: str) -> pd.DataFrame:
        """TSVファイルを読み込み"""
        pass
    
    def aggregate_to_races(self, df: pd.DataFrame) -> List[Race]:
        """レース単位に集約"""
        pass
    
    def filter_by_date(self, races: List[Race], 
                       start: date, end: date) -> List[Race]:
        """日付でフィルタ"""
        pass
```

#### 10.2.2 BaseStrategy
```python
class BaseStrategy(ABC):
    """戦略の基底クラス"""
    
    @abstractmethod
    def select_tickets(self, race: Race) -> List[Ticket]:
        """レースから購入馬券を選択"""
        pass
    
    @abstractmethod
    def estimate_win_probability(self, ticket: Ticket) -> float:
        """勝率を推定（ケリー基準用）"""
        pass
```

#### 10.2.3 FundManager
```python
class FundManager(ABC):
    """資金管理の基底クラス"""
    
    @abstractmethod
    def calculate_bet_amount(self, 
                             current_fund: float,
                             ticket: Ticket,
                             win_prob: float) -> float:
        """賭け金を計算"""
        pass
    
    def apply_constraints(self, amount: float, 
                          constraints: dict) -> float:
        """制約を適用"""
        pass
```

#### 10.2.4 Simulator
```python
class SimpleSimulator:
    """単純シミュレーション"""
    
    def run(self, 
            races: List[Race],
            strategy: BaseStrategy,
            fund_manager: FundManager,
            initial_fund: float) -> SimulationResult:
        """シミュレーション実行"""
        pass

class MonteCarloSimulator:
    """モンテカルロシミュレーション"""
    
    def run(self,
            races: List[Race],
            strategy: BaseStrategy,
            fund_manager: FundManager,
            initial_fund: float,
            iterations: int) -> MonteCarloResult:
        """モンテカルロ実行"""
        pass
```

### 10.3 データモデル
```python
@dataclass
class Horse:
    """馬データ"""
    number: int              # 馬番
    name: str                # 馬名
    odds: float              # 単勝オッズ
    popularity: int          # 人気順
    actual_rank: int         # 確定着順
    predicted_rank: int      # 予測順位
    predicted_score: float   # 予測スコア
    upset_prob: float = 0.0  # 穴馬確率
    is_upset_candidate: bool = False  # 穴馬候補

@dataclass
class Race:
    """レースデータ"""
    track: str               # 競馬場
    year: int                # 開催年
    date: int                # 開催日
    race_number: int         # レース番号
    surface: str             # 芝/ダート
    distance: int            # 距離
    horses: List[Horse]      # 出走馬リスト
    payouts: dict            # 払戻情報

@dataclass
class Ticket:
    """馬券データ"""
    ticket_type: str         # 馬券種（win/place/quinella/wide/trio）
    horse_numbers: List[int] # 馬番リスト
    odds: float              # オッズ
    amount: int = 0          # 賭け金

@dataclass
class SimulationResult:
    """シミュレーション結果"""
    fund_history: List[float]  # 資金推移
    bet_history: List[dict]    # 購入履歴
    metrics: dict              # 評価指標
```

---

## 11. 流用可能な既存コンポーネント

本システムの構築にあたり、以下の既存コードを参考・流用可能。

| 既存ファイル | 流用内容 | 備考 |
|-------------|---------|------|
| `bet_evaluator.py` | 的中判定ロジック、払戻計算 | 一部リファクタリング必要 |
| `kelly_criterion.py` | ケリー基準計算 | ほぼそのまま使用可能 |
| `expected_value_calculator.py` | 期待値計算 | 参考として使用 |
| `race_confidence_scorer.py` | レース信頼度スコア | フィルタに使用可能 |
| `upset_threshold_config.json` | 穴馬閾値設定 | 設定ファイルとして参照 |

---

## 12. 開発ロードマップ

### Phase 1: コア機能実装（2週間）
| タスク | 優先度 | 工数 |
|--------|--------|------|
| PredictionLoader実装 | ★★★★★ | 1日 |
| Race/Horse/Ticketモデル実装 | ★★★★★ | 1日 |
| BaseStrategy + 基本戦略実装 | ★★★★★ | 3日 |
| FundManager実装（3方式） | ★★★★★ | 2日 |
| SimpleSimulator実装 | ★★★★★ | 2日 |
| 基本評価指標実装 | ★★★★☆ | 1日 |

### Phase 2: 拡張機能実装（1週間）
| タスク | 優先度 | 工数 |
|--------|--------|------|
| MonteCarloSimulator実装 | ★★★★★ | 3日 |
| WalkForwardSimulator実装 | ★★★★☆ | 2日 |
| リスク指標追加 | ★★★★☆ | 1日 |
| Go/No-Go判定機能 | ★★★★☆ | 1日 |

### Phase 3: 可視化・レポート（1.5週間）
| タスク | 優先度 | 工数 |
|--------|--------|------|
| 資金推移系グラフ（5種） | ★★★★★ | 1日 |
| リスク分析系グラフ（5種） | ★★★★☆ | 1日 |
| 収益分析系グラフ（6種） | ★★★★☆ | 1日 |
| 馬券・条件別グラフ（13種） | ★★★☆☆ | 2日 |
| 戦略比較・WF分析グラフ（7種） | ★★★☆☆ | 1日 |
| インタラクティブHTML（3種） | ★★★☆☆ | 1日 |
| サマリーレポート生成 | ★★★★☆ | 1日 |

### Phase 4: ダッシュボード構築（オプション、1.5週間）
| タスク | 優先度 | 工数 |
|--------|--------|------|
| Streamlit基本構造 | ★★★★☆ | 1日 |
| サマリー・資金推移画面 | ★★★★☆ | 2日 |
| リスク・的中分析画面 | ★★★☆☆ | 2日 |
| 条件別・戦略比較画面 | ★★★☆☆ | 2日 |
| 設定画面・エクスポート | ★★★☆☆ | 1日 |

### Phase 5: テスト・最適化（1週間）
| タスク | 優先度 | 工数 |
|--------|--------|------|
| ユニットテスト | ★★★★★ | 2日 |
| 統合テスト | ★★★★★ | 2日 |
| パフォーマンス最適化 | ★★★☆☆ | 1日 |
| ドキュメント整備 | ★★★☆☆ | 1日 |

**合計: 約6.5週間**（ダッシュボード含む場合: 約8週間）

---

## 13. リスク管理

### 13.1 想定リスクと対策
| リスク | 影響度 | 対策 |
|--------|--------|------|
| モデル精度の低下 | 高 | 定期的な再学習、Walk-Forward継続 |
| オッズ変動 | 中 | 賭け金上限の厳守、馬連メイン |
| 連敗による資金減少 | 高 | ケリー基準、最大ドローダウン監視 |
| データ品質の劣化 | 中 | データパイプラインの監視 |

### 13.2 撤退ルール（参考値）
| 条件 | アクション |
|------|-----------|
| 資金が初期の50%以下 | 一時停止、戦略見直し |
| 3ヶ月連続でROI 100%未満 | モデル再検証 |
| 破産ライン到達 | 完全撤退、ゼロベースで再構築 |

---

## 14. 使用例

### 14.1 基本的な使用方法
```bash
# デフォルト設定でシミュレーション
python scripts/run_simulation.py \
  --input predicted_results.tsv \
  --config config/default_config.yaml

# カスタム設定でシミュレーション
python scripts/run_simulation.py \
  --input predicted_results.tsv \
  --config config/examples/50x_target.yaml \
  --output output/50x_simulation/
```

### 14.2 戦略比較
```bash
# 複数戦略を比較
python scripts/compare_strategies.py \
  --input predicted_results.tsv \
  --strategies conservative balanced aggressive \
  --output output/strategy_comparison/
```

### 14.3 パラメータ最適化
```bash
# ケリー係数の最適化
python scripts/optimize_params.py \
  --input predicted_results.tsv \
  --param kelly_fraction \
  --range 0.1 0.5 0.05 \
  --output output/kelly_optimization/
```

---

## 付録A: 設定ファイル例（50万→500万目標）

```yaml
# config/examples/50x_target.yaml
simulation:
  initial_fund: 250000      # 初期25万円（予備25万円は別管理）
  target_fund: 5000000      # 目標500万円
  bankruptcy_threshold: 0.1
  
  monte_carlo:
    enabled: true
    iterations: 10000

betting:
  method: "kelly"
  kelly:
    fraction: 0.25
    max_bet_ratio: 0.05
  
  constraints:
    min_bet: 100
    max_bet_per_ticket: 5000
    max_bet_per_race: 20000
    max_bet_per_day: 50000

ticket_types:
  quinella:
    enabled: true
    allocation: 0.40
  wide:
    enabled: true
    allocation: 0.35
  trio:
    enabled: true
    allocation: 0.25

strategy:
  favorite:
    method: "predicted_rank"
    top_n: 2
  longshot:
    enabled: true
    method: "upset_prob"
    threshold: 0.20
    max_count: 3

go_nogo_criteria:
  go_conditions:
    max_bankruptcy_prob: 0.05
    min_expected_roi: 1.50
    max_drawdown_95percentile: 0.50
```

---

## 付録B: 用語定義

| 用語 | 定義 |
|------|------|
| ROI | Return on Investment。回収額 ÷ 投資額 × 100% |
| ドローダウン | ピーク資金からの下落率 |
| ケリー基準 | 最適な賭け金比率を算出する数学的手法 |
| Walk-Forward | 時系列データで学習期間と検証期間をスライドさせる検証手法 |
| モンテカルロシミュレーション | 乱数を用いて多数のシナリオを生成する手法 |
| VaR | Value at Risk。特定の信頼水準での最大損失額 |
| CVaR | Conditional VaR。VaRを超える損失の期待値 |

## 付録C: 流用元ファイル（予測システム）

| ファイル | 役割 |
|----------|------|
| bet_evaluator.py | 賭け評価 |
| expected_value_calculator.py | 期待値計算 |
| kelly_criterion.py | ケリー基準 |
| upset_classifier_creator.py | 穴馬分類モデル |
| universal_test.py | 汎用テスト |
| race_confidence_scorer.py | レース信頼度スコア |

---

**文書管理**
- 作成日: 2026年1月21日
- バージョン: 1.0
- 作成者: GitHub Copilot
- 最終更新: 2026年1月21日
