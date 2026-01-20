# Walk-Forward検証 効率化プラン

## 📊 現状の問題分析

### 現在の処理時間
**単一期間モード（10モデル、1年テスト）**: **約12-19時間/試行**

- モデル作成: 10モデル × (30分 + 40-80分UPSET) = **11.6-18.3時間**
- テスト実行: 10モデル × 5分 = **50分**

## ✅ 実装済み最適化

### Phase 1: UPSET分類器共通化（完了）✅

**優先度**: ⭐⭐⭐⭐⭐  
**難易度**: 低（1-2時間）  
**期待効果**: **50-70%削減** (12-19h → 5.6-6.3h)  
**実装日**: 2025-01-XX  
**実装状態**: ✅ 完了

#### 実装内容
- `walk_forward_validation.py` 826-860行および995-1030行:
  - `run_single_period_mode()`と`run_compare_periods_mode()`でUPSET分類器を独立チェック
  - 全モデルループ前にUPSET分類器を1回だけ作成
  - progress.jsonに記録してスキップ制御
- Universal Rankerも記録 (`universal_ranker_{train_start}-{train_end}`)
- 541-615行: `_create_upset_classifier()`バグ修正
  - Universal Ranker存在時にUPSET未存在なら作成続行
  - `universal_config`のUnboundLocalError修正

**効果**: UPSET分類器作成 40-80分 × (モデル数-1) の削減

---

### Phase 2: モデル作成並列化（完了）✅

**優先度**: ⭐⭐⭐⭐  
**難易度**: 中（3-5時間）  
**期待効果**: **追加で60-75%削減** (5.6-6.3h → 2.1-2.6h)  
**実装日**: 2025-01-XX  
**実装状態**: ✅ 完了

#### 実装内容
- `walk_forward_validation.py`:
  - 1-30行: multiprocessing, threading, ProcessPoolExecutor追加
  - 58行: progress.json排他制御用`threading.Lock`追加
  - 214-217行: `_save_progress()`にロック機構追加
  - 451-527行: `_create_model_worker()`静的メソッド作成
    - 各プロセスで独立したDB接続とロガー
    - モデルファイル名生成とcreate_universal_model呼び出し
  - 912-971行: `run_single_period_mode()`モデル作成フェーズ並列化
    - max_workers=4でProcessPoolExecutor使用
    - as_completed()で完了順に結果取得
    - progress.json排他ロック付き記録
  - 1185-1244行: `run_compare_periods_mode()`も同様に並列化

**効果**: 
- 4並列実行でモデル作成を4倍高速化
- メモリ使用: ~4GB × 4 = 16GB (ピーク)
- DB接続はプロセスごとに独立

---

## 🚀 残りの効率化プラン（Phase 3-5）
                test_year,
                training_years,
                period,
                False  # create_upset_classifier=False
            ): model_name
            for model_name in target_models
        }
        
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                success, model_path = future.result()
                if success:
                    self.log(f"✅ {model_name} 完了")
                else:
                    self.log(f"❌ {model_name} 失敗")
            except Exception as e:
                self.log(f"❌ {model_name} エラー: {e}")

def _create_single_model(self, model_name, test_year, training_years, period, create_upset):
    """単一モデル作成（並列実行用）"""
    # DB接続はプロセスごとに独立させる
    return self.create_model_for_year(
        model_name=model_name,
        test_year=test_year,
        training_years=training_years,
        period=period,
        create_upset_classifier=create_upset
    )
```

**注意点**:
- `progress.json` 更新時の排他制御が必要（`threading.Lock` または `fasteners` ライブラリのファイルロック）
- DB接続は各プロセスで独立して確立
- メモリ使用量に注意（4並列 × 1モデル分のメモリ）

---

### Phase 3: DBクエリ最適化（中期改善）

**優先度**: ⭐⭐⭐  
**難易度**: 高（8-12時間）  
**期待効果**: **追加で30-50%削減** (2.1-2.6h → 1.3-1.6h)

#### 実装内容

**ファイル**: `db_query_builder.py`  
**対象関数**: `build_race_data_query()` (129-320行)

#### 3-1. サブクエリをCTEに変換

**現在のコード**（2段階ネスト）:
```sql
SELECT 
    base_features.*,
    (base_features.past_score_mean - AVG(base_features.past_score_mean) OVER race_window) 
        / NULLIF(STDDEV(base_features.past_score_mean) OVER race_window, 0) 
        AS relative_ability
FROM (
    -- 内側クエリ: base_features
    SELECT ...
    FROM jvd_sed_uma
    ...
) base_features
```

**改善後のコード**（CTE使用）:
```sql
WITH base_features AS (
    SELECT 
        uma.*,
        AVG(score) OVER (PARTITION BY ketto_toroku_bango ...) AS past_score_mean,
        ...
    FROM jvd_sed_uma uma
    WHERE ...
),
race_stats AS (
    SELECT 
        kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango,
        AVG(past_score_mean) AS avg_past_score,
        STDDEV(past_score_mean) AS std_past_score
    FROM base_features
    GROUP BY kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango
)
SELECT 
    bf.*,
    (bf.past_score_mean - rs.avg_past_score) 
        / NULLIF(rs.std_past_score, 0) AS relative_ability
FROM base_features bf
LEFT JOIN race_stats rs
    ON bf.kaisai_nen = rs.kaisai_nen
    AND bf.kaisai_tsukihi = rs.kaisai_tsukihi
    AND bf.keibajo_code = rs.keibajo_code
    AND bf.race_bango = rs.race_bango
```

**期待効果**: クエリプランナーがCTEを最適化しやすく、実行時間20-40%削減

---

#### 3-2. インデックス追加

**実装**: PostgreSQL側で実行

```sql
-- 馬の過去レース検索用
CREATE INDEX IF NOT EXISTS idx_uma_ketto_kaisai 
    ON jvd_sed_uma (ketto_toroku_bango, kaisai_nen, kaisai_tsukihi);

-- 騎手の過去成績検索用
CREATE INDEX IF NOT EXISTS idx_uma_kishu_kaisai
    ON jvd_sed_uma (kishu_code, kaisai_nen, kaisai_tsukihi);

-- 調教師の過去成績検索用
CREATE INDEX IF NOT EXISTS idx_uma_chokyoshi_kaisai
    ON jvd_sed_uma (chokyoshi_code, kaisai_nen, kaisai_tsukihi);

-- レース情報検索用
CREATE INDEX IF NOT EXISTS idx_uma_race
    ON jvd_sed_uma (kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango);

-- 着順検索用（集計に使用）
CREATE INDEX IF NOT EXISTS idx_uma_chakujun
    ON jvd_sed_uma (kakutei_chakujun);
```

**期待効果**: ウィンドウ関数の実行速度30-50%向上

**確認方法**:
```sql
-- 既存インデックス確認
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'jvd_sed_uma';

-- クエリプラン確認
EXPLAIN ANALYZE <your query>;
```

---

#### 3-3. マテリアライズドビュー（オプション）

**対象**: 頻繁に参照される集計データ

```sql
-- 過去レーススコアの事前計算
CREATE MATERIALIZED VIEW past_score_cache AS
SELECT 
    ketto_toroku_bango,
    kaisai_nen,
    kaisai_tsukihi,
    keibajo_code,
    race_bango,
    AVG(score) OVER (
        PARTITION BY ketto_toroku_bango 
        ORDER BY kaisai_nen, kaisai_tsukihi 
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS past_score_mean,
    STDDEV(score) OVER (
        PARTITION BY ketto_toroku_bango 
        ORDER BY kaisai_nen, kaisai_tsukihi 
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS past_score_std
FROM jvd_sed_uma;

-- インデックス追加
CREATE INDEX idx_past_score_cache_uma ON past_score_cache (ketto_toroku_bango, kaisai_nen, kaisai_tsukihi);

-- 定期更新（日次バッチなど）
REFRESH MATERIALIZED VIEW past_score_cache;
```

**クエリでの使用**:
```sql
-- db_query_builder.py で past_score_cache を JOIN
SELECT 
    uma.*,
    psc.past_score_mean,
    psc.past_score_std,
    ...
FROM jvd_sed_uma uma
LEFT JOIN past_score_cache psc
    ON uma.ketto_toroku_bango = psc.ketto_toroku_bango
    AND uma.kaisai_nen = psc.kaisai_nen
    AND uma.kaisai_tsukihi = psc.kaisai_tsukihi
    AND uma.keibajo_code = psc.keibajo_code
    AND uma.race_bango = psc.race_bango
```

**期待効果**: ウィンドウ関数の計算不要 → 50-70%高速化

**注意点**:
- データ更新時に `REFRESH MATERIALIZED VIEW` が必要
- ストレージ容量増加
- walk-forward validation では過去データのみ使用するため、事前計算と相性良い

---

### Phase 4: 特徴量キャッシュ導入（2回目以降高速化）

**優先度**: ⭐⭐  
**難易度**: 中（4-6時間）  
**期待効果**: **2回目以降80-90%削減** (1.3-1.6h → 10-20m)

#### 実装内容

**新規ファイル**: `feature_cache.py`

```python
"""
特徴量キャッシング機構

特徴量計算結果をParquet形式でディスクにキャッシュし、
2回目以降の実行を高速化する。
"""

import hashlib
import json
from pathlib import Path
import pandas as pd


class FeatureCache:
    """特徴量キャッシュマネージャー"""
    
    def __init__(self, cache_dir='feature_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = self._compute_code_version()
    
    def _compute_code_version(self):
        """コードバージョンをハッシュで計算"""
        # db_query_builder.py と feature_engineering.py の内容からハッシュ生成
        files_to_hash = [
            'db_query_builder.py',
            'feature_engineering.py'
        ]
        
        hasher = hashlib.md5()
        for file_path in files_to_hash:
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()[:8]
    
    def get_cache_key(self, year, track_code, surface_type):
        """キャッシュキーを生成"""
        return f"{self.version}_{year}_{track_code}_{surface_type}"
    
    def get_cache_path(self, cache_key):
        """キャッシュファイルパスを取得"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def exists(self, year, track_code, surface_type):
        """キャッシュが存在するか確認"""
        cache_key = self.get_cache_key(year, track_code, surface_type)
        cache_path = self.get_cache_path(cache_key)
        return cache_path.exists()
    
    def load(self, year, track_code, surface_type):
        """キャッシュから読み込み"""
        cache_key = self.get_cache_key(year, track_code, surface_type)
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            print(f"✅ キャッシュヒット: {cache_key}")
            return df
        except Exception as e:
            print(f"⚠️ キャッシュ読み込み失敗: {e}")
            return None
    
    def save(self, df, year, track_code, surface_type):
        """キャッシュに保存"""
        cache_key = self.get_cache_key(year, track_code, surface_type)
        cache_path = self.get_cache_path(cache_key)
        
        try:
            df.to_parquet(cache_path, compression='snappy')
            print(f"💾 キャッシュ保存: {cache_key}")
        except Exception as e:
            print(f"⚠️ キャッシュ保存失敗: {e}")
    
    def clear_old_versions(self):
        """古いバージョンのキャッシュを削除"""
        current_prefix = f"{self.version}_"
        
        deleted_count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            if not cache_file.name.startswith(current_prefix):
                cache_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"🗑️  古いキャッシュ削除: {deleted_count}ファイル")


def get_data_with_features_cached(year, track_code, surface_type, compute_fn):
    """
    キャッシュ付きでデータ+特徴量を取得
    
    Args:
        year: 年
        track_code: 競馬場コード
        surface_type: 路面種別
        compute_fn: 計算関数（キャッシュミス時に実行）
    
    Returns:
        DataFrame (特徴量付き)
    """
    cache = FeatureCache()
    
    # キャッシュチェック
    df = cache.load(year, track_code, surface_type)
    if df is not None:
        return df
    
    # キャッシュミス → 計算
    print(f"⏳ 特徴量計算中: {year}年 競馬場{track_code} {surface_type}")
    df = compute_fn()
    
    # キャッシュ保存
    cache.save(df, year, track_code, surface_type)
    
    return df
```

#### 統合方法

**ファイル**: `model_creator.py`

```python
from feature_cache import get_data_with_features_cached

def create_model(track_code, year_start, year_end, ...):
    """モデル作成（キャッシュ対応版）"""
    
    # キャッシュ付きでデータ取得
    def compute_features():
        # 既存のロジック
        query = build_race_data_query(...)
        df = pd.read_sql_query(query, conn)
        df = preprocess_race_data(df)
        df = create_features(df)
        df = add_advanced_features(df, ...)
        return df
    
    df = get_data_with_features_cached(
        year=year_start,
        track_code=track_code,
        surface_type=surface_type,
        compute_fn=compute_features
    )
    
    # モデル訓練（以降は既存ロジック）
    ...
```

**ファイル**: `universal_test.py`

同様の変更を適用

---

### Phase 5: 競馬場別並列化（UPSET分類器作成高速化）

**優先度**: ⭐⭐  
**難易度**: 中（2-3時間）  
**期待効果**: **UPSET作成70-80%削減** (40-80分 → 4-8分)

#### 実装内容

**ファイル**: `analyze_upset_patterns.py`  
**対象関数**: `get_data_with_predictions()` (88-148行)

**現在のコード**（シリアル処理）:
```python
# analyze_upset_patterns.py 88-148行
all_data = []

for year in years:
    for track_code in track_codes:
        for surface in surfaces:
            # 1つずつ順次処理
            df = get_single_track_data(...)
            all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
```

**改善後のコード**（並列処理）:
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def get_data_with_predictions(model_path, years, track_codes, surfaces, ...):
    """競馬場別にデータ取得を並列化"""
    
    # タスクリスト作成
    tasks = [
        (year, track_code, surface)
        for year in years
        for track_code in track_codes
        for surface in surfaces
    ]
    
    max_workers = min(10, multiprocessing.cpu_count())
    print(f"🚀 データ取得並列化: {len(tasks)}タスク × {max_workers}並列")
    
    # 並列実行
    all_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _get_single_track_data_worker,
                model_path, year, track_code, surface, ...
            ): (year, track_code, surface)
            for year, track_code, surface in tasks
        }
        
        for future in as_completed(futures):
            year, track_code, surface = futures[future]
            try:
                df = future.result()
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    print(f"✅ {year}年 {track_code} {surface}: {len(df):,}頭")
            except Exception as e:
                print(f"❌ {year}年 {track_code} {surface}: {e}")
    
    # 結合
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ データ結合完了: {len(df):,}頭")
    
    return df


def _get_single_track_data_worker(model_path, year, track_code, surface, ...):
    """単一競馬場のデータ取得（ワーカープロセス用）"""
    # DB接続はプロセスごとに独立
    conn = get_db_connection()
    
    try:
        query = build_race_data_query(
            track_code=track_code,
            year_start=year,
            year_end=year,
            surface_type=surface,
            ...
        )
        df = pd.read_sql_query(query, conn)
        
        # 特徴量計算
        df = preprocess_race_data(df)
        df = create_features(df)
        df = add_advanced_features(df, ...)
        
        # 予測
        model = load_model(model_path)
        df['predicted_score'] = model.predict(df[feature_cols])
        
        return df
    finally:
        conn.close()
```

---

## 📈 効率化効果まとめ

| Phase | 改善内容 | 難易度 | 時間削減 | 累計削減率 | 処理時間 |
|-------|---------|--------|---------|-----------|---------|
| **現状** | - | - | - | - | **12-19h** |
| **Phase 1** | UPSET分類器共通化 | 低 | 50-70% | 50-70% | **5.6-6.3h** |
| **Phase 2** | モデル作成並列化 | 中 | 60-75% | 82-86% | **2.1-2.6h** |
| **Phase 3** | DBクエリ最適化 | 高 | 30-50% | 89-92% | **1.3-1.6h** |
| **Phase 4** | 特徴量キャッシュ | 中 | 80-90% (2回目以降) | 97-98% | **10-20m** |
| **Phase 5** | 競馬場別並列化 | 中 | UPSET作成70-80% | Phase 1を強化 | - |

---

## 🎯 実装の推奨順序

### 優先度1（即実装推奨）
1. **Phase 1: UPSET分類器共通化**
   - 理由: 最高効果・最低リスク・最短実装時間
   - 実装時間: 1-2時間
   - リスク: 低（既存ロジックの小修正のみ）

### 優先度2（短期実装推奨）
2. **Phase 2: モデル作成並列化**
   - 理由: 高効果・中程度のリスク
   - 実装時間: 3-5時間
   - リスク: 中（並列処理のデバッグが必要）
   - 前提: Phase 1完了後

3. **Phase 5: 競馬場別並列化**
   - 理由: Phase 1の効果をさらに強化
   - 実装時間: 2-3時間
   - リスク: 中（Phase 2と同様の並列処理）

### 優先度3（中期実装）
4. **Phase 4: 特徴量キャッシュ**
   - 理由: 2回目以降の実行で効果発揮
   - 実装時間: 4-6時間
   - リスク: 低（キャッシュ無効化の仕組みが必要）

### 優先度4（長期実装・要注意）
5. **Phase 3: DBクエリ最適化**
   - 理由: 影響範囲が広い・慎重なテストが必要
   - 実装時間: 8-12時間
   - リスク: 高（SQL変更により結果が変わる可能性）
   - 注意: 
     - 既存結果との整合性確認必須
     - インデックス追加は比較的安全（Phase 3-2から実施可能）

---

## ⚠️ Further Considerations（重要な考慮事項）

### 1. 実装の優先順位とリスク管理

#### Phase 1（UPSET共通化）
- **最優先で実装すべき**: 効果が最も高く、実装も簡単（1-2時間）
- **リスク**: 低 - 既存ロジックの小修正のみ
- **テスト方法**: 1モデルと10モデルで実行時間を比較
- **注意点**: `_create_upset_classifier()` 594-597行のバグ修正を忘れずに

#### Phase 2（並列化）
- **中程度の難易度**: 並列処理のデバッグに時間がかかる可能性（3-5時間）
- **リスク**: 中 - `progress.json`の排他制御が必須
- **テスト方法**: 
  - 2モデルで並列化を試し、progress.json が正しく更新されるか確認
  - メモリ使用量をモニタリング（`max_workers`調整が必要な場合あり）
- **注意点**: 
  - DB接続はプロセスごとに独立させる（psycopg2接続はプロセス間で共有不可）
  - Windows環境では `if __name__ == '__main__':` ガードが必須

#### Phase 3（DB最適化）
- **影響範囲が広い**: 慎重なテストが必要
- **リスク**: 高 - SQL変更により特徴量の値が変わる可能性
- **段階的実装**:
  1. **Phase 3-2（インデックス追加）から開始**: 比較的安全で効果も高い
  2. インデックス効果を確認後、Phase 3-1（CTE変換）を実施
  3. Phase 3-3（マテリアライズドビュー）は最後（運用負荷が増加）
- **テスト方法**:
  - SQL変更前後でサンプルデータの特徴量を比較（差異がないことを確認）
  - `EXPLAIN ANALYZE` でクエリプランを比較
  - 既存の訓練済みモデルと新規モデルで予測精度を比較
- **注意点**: 
  - インデックス追加でディスク容量増加（数GB〜）
  - マテリアライズドビューは定期更新が必要（cron等）

### 2. GPU利用の検討

#### LightGBM GPU版
```python
# LightGBM GPU版の設定例
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'gpu_use_dp': False,  # 単精度浮動小数点（高速）
    ...
}
```

**期待効果**: モデル訓練が **2-5倍高速化**

**前提条件**:
- CUDA対応GPU（NVIDIA）が必要
- LightGBM GPU版のインストール: `pip install lightgbm --install-option=--gpu`
- CUDAツールキットのインストール（10.0以降推奨）

**推奨判断**:
- GPUがある場合: Phase 1-2完了後に試す価値あり
- GPUがない場合: Phase 1-3の効果で十分（追加投資不要）

**注意点**:
- GPU版は `categorical_feature` の扱いが異なる場合あり
- メモリ不足エラーに注意（GPU RAMサイズ確認）

### 3. キャッシュ無効化戦略

#### Phase 4 特徴量キャッシュの課題
- **問題**: `db_query_builder.py` や `feature_engineering.py` を変更した場合、古いキャッシュが使われる
- **解決策**: コードバージョンハッシュをキャッシュキーに含める（実装済み）

#### キャッシュ管理のベストプラクティス
```python
# feature_cache.py に追加
class FeatureCache:
    def invalidate_all(self):
        """全キャッシュを削除"""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
        print(f"🗑️  全キャッシュ削除完了")
    
    def get_cache_stats(self):
        """キャッシュ統計を表示"""
        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        print(f"\n📊 キャッシュ統計")
        print(f"  ファイル数: {len(cache_files)}")
        print(f"  総容量: {total_size / 1024**3:.2f} GB")
        print(f"  現在バージョン: {self.version}")
```

#### コマンドライン引数でキャッシュ制御
```python
# walk_forward_validation.py に追加
parser.add_argument('--clear-cache', action='store_true', 
                    help='実行前にキャッシュをクリア')
parser.add_argument('--no-cache', action='store_true',
                    help='キャッシュを使用しない')

if args.clear_cache:
    FeatureCache().invalidate_all()
```

### 4. メモリ管理

#### 並列化時のメモリ使用量
- **Phase 2**: `max_workers=4` の場合、4モデル分のメモリが必要
- **推定**: 1モデル = 2-4GB → 4並列 = **8-16GB**
- **対策**: 
  ```python
  # メモリ使用量に応じて並列数を調整
  import psutil
  available_memory_gb = psutil.virtual_memory().available / 1024**3
  max_workers = min(4, int(available_memory_gb / 4))  # 1モデル=4GB想定
  ```

#### データフレームのメモリ削減
```python
# model_creator.py, universal_test.py に追加
def reduce_memory_usage(df):
    """データ型を最適化してメモリ削減"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

df = reduce_memory_usage(df)  # 特徴量作成後に実行
```

**期待効果**: メモリ使用量 **30-50%削減**

### 5. 進捗モニタリングの改善

#### Phase 2 並列化時の進捗表示
```python
# walk_forward_validation.py に追加
from tqdm import tqdm

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {...}
    
    # プログレスバー付き
    with tqdm(total=len(target_models), desc="モデル作成") as pbar:
        for future in as_completed(futures):
            model_name = futures[future]
            success, model_path = future.result()
            pbar.set_description(f"完了: {model_name}")
            pbar.update(1)
```

#### リアルタイム進捗ログ
```python
# 各ワーカーからログを収集
import logging
from logging.handlers import QueueHandler

# メインプロセス
log_queue = multiprocessing.Queue()
listener = logging.handlers.QueueListener(log_queue, *handlers)
listener.start()

# ワーカープロセス
logger = logging.getLogger()
logger.addHandler(QueueHandler(log_queue))
```

### 6. エラーハンドリングとリトライ

#### Phase 2 並列化時のエラーハンドリング
```python
def _create_single_model_with_retry(self, model_name, max_retries=3, ...):
    """リトライ付きモデル作成"""
    for attempt in range(max_retries):
        try:
            return self._create_single_model(model_name, ...)
        except Exception as e:
            if attempt < max_retries - 1:
                self.log(f"⚠️ {model_name} 失敗（リトライ {attempt+1}/{max_retries}）: {e}")
                time.sleep(5)  # 5秒待機
            else:
                self.log(f"❌ {model_name} 最終失敗: {e}")
                return False, None
```

### 7. データベース接続プール

#### psycopg2 接続プールの導入
```python
# db_connector.py（新規）
from psycopg2 import pool

class DatabasePool:
    _pool = None
    
    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            config = load_db_config()
            cls._pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=config['host'],
                port=config['port'],
                user=config['user'],
                password=config['password'],
                dbname=config['dbname']
            )
        return cls._pool
    
    @classmethod
    def get_connection(cls):
        return cls.get_pool().getconn()
    
    @classmethod
    def return_connection(cls, conn):
        cls.get_pool().putconn(conn)

# 使用例
conn = DatabasePool.get_connection()
try:
    df = pd.read_sql_query(query, conn)
finally:
    DatabasePool.return_connection(conn)
```

**期待効果**: DB接続のオーバーヘッド削減、並列処理時の接続エラー防止

### 8. バージョン管理とロールバック

#### 効率化実装のブランチ戦略
```bash
# 各Phaseごとにブランチ作成
git checkout -b optimize/phase1-upset-sharing
# Phase 1 実装 + テスト
git commit -m "Phase 1: UPSET分類器共通化"

git checkout -b optimize/phase2-parallel
# Phase 2 実装 + テスト
git commit -m "Phase 2: モデル作成並列化"
```

#### 性能測定の記録
```python
# performance_tracker.py（新規）
import time
import json
from pathlib import Path

class PerformanceTracker:
    def __init__(self, log_file='performance_log.json'):
        self.log_file = Path(log_file)
        self.metrics = []
    
    def record(self, phase, duration, models_count, test_year):
        self.metrics.append({
            'timestamp': time.time(),
            'phase': phase,
            'duration_seconds': duration,
            'duration_hours': duration / 3600,
            'models_count': models_count,
            'test_year': test_year
        })
        
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# walk_forward_validation.py で使用
tracker = PerformanceTracker()
start_time = time.time()

# ... モデル作成 ...

duration = time.time() - start_time
tracker.record('baseline', duration, len(target_models), test_year)
```

### 9. 実装後の検証項目

#### Phase 1 実装後
- [ ] 1モデル実行時: UPSET分類器が1回だけ作成されることを確認
- [ ] 10モデル実行時: 処理時間が 12-19h → 5.6-6.3h に削減されることを確認
- [ ] UPSET分類器ファイルが正しく保存されることを確認
- [ ] 各モデルがUPSET分類器を正しく参照できることを確認

#### Phase 2 実装後
- [ ] 並列実行時にエラーが発生しないことを確認
- [ ] progress.json が正しく更新されることを確認
- [ ] メモリ使用量が許容範囲内であることを確認
- [ ] 処理時間が 5.6-6.3h → 2.1-2.6h に削減されることを確認

#### Phase 3 実装後
- [ ] SQL変更前後で特徴量の値が一致することを確認（サンプルデータ）
- [ ] クエリ実行時間が削減されることを確認（EXPLAIN ANALYZE）
- [ ] 予測精度に変化がないことを確認（既存モデルと比較）

#### Phase 4 実装後
- [ ] 初回実行でキャッシュが作成されることを確認
- [ ] 2回目実行でキャッシュが使用されることを確認
- [ ] コード変更時にキャッシュが無効化されることを確認
- [ ] 処理時間が 1.3-1.6h → 10-20m に削減されることを確認（2回目以降）

---

## 📝 実装チェックリスト

### Phase 1: UPSET分類器共通化
- [ ] `walk_forward_validation.py` 826-852行を修正
- [ ] `upset_created` フラグ追加
- [ ] `_create_upset_classifier()` 594-597行のバグ修正
- [ ] 1モデルと10モデルでテスト実行
- [ ] 処理時間を記録・比較

### Phase 2: モデル作成並列化
- [ ] `ProcessPoolExecutor` 導入
- [ ] `_create_single_model()` メソッド実装
- [ ] progress.json 排他制御実装
- [ ] メモリ使用量モニタリング
- [ ] エラーハンドリング・リトライ実装
- [ ] 2モデル、4モデル、10モデルでテスト

### Phase 3: DBクエリ最適化
- [ ] Phase 3-2: PostgreSQLインデックス追加
- [ ] インデックス効果を `EXPLAIN ANALYZE` で確認
- [ ] Phase 3-1: CTE変換実装
- [ ] SQL変更前後で特徴量比較
- [ ] Phase 3-3: マテリアライズドビュー検討（オプション）

### Phase 4: 特徴量キャッシュ
- [ ] `feature_cache.py` 実装
- [ ] `model_creator.py` に統合
- [ ] `universal_test.py` に統合
- [ ] キャッシュ無効化テスト
- [ ] 2回目実行で高速化を確認

### Phase 5: 競馬場別並列化
- [ ] `analyze_upset_patterns.py` 並列化実装
- [ ] `_get_single_track_data_worker()` 実装
- [ ] 10競馬場でテスト
- [ ] UPSET分類器作成時間を記録・比較

---

## 🎓 参考情報

### 並列処理のベストプラクティス
- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
- [concurrent.futures documentation](https://docs.python.org/3/library/concurrent.futures.html)

### PostgreSQL最適化
- [PostgreSQL Performance Tips](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [PostgreSQL Indexes](https://www.postgresql.org/docs/current/indexes.html)
- [Materialized Views](https://www.postgresql.org/docs/current/rules-materializedviews.html)

### LightGBM
- [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [LightGBM Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

---

## 📊 UPSET分類器の特徴量一覧（Phase 3.5.1完了時点）

### 全体サマリー
- **総特徴量数**: 35個
- **特徴量カテゴリ**: 8つ
- **最終更新日**: 2026年1月20日（Phase 3.5.1完了）

---

### 1. ランキングモデルの出力（2個）
Universal Rankerの予測結果を活用した特徴量

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `predicted_rank` | Universal Rankerによる予測順位（1-18位） | 18位 |
| `predicted_score` | Universal Rankerの予測スコア（高いほど上位予測） | 22位 |

---

### 2. 人気・オッズ情報（3個）
馬券市場の評価と予測モデルとの乖離を示す特徴量

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `popularity_rank` | 単勝人気順位（1-18位） | **2位** ⭐ |
| `tansho_odds` | 単勝オッズ（倍） | 11位 |
| `value_gap` | 予測順位 - 人気順位（負の値＝過小評価） | 17位 |

---

### 3. 既存の重要特徴量（8個）
Universal Rankerでも使用されている基本特徴量

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `past_score` | 過去3走のグレード別スコア合計（1着100点、G1は3倍） | 10位 |
| `past_avg_sotai_chakujun` | 過去3走の相対着順平均（時計差考慮） | 9位 |
| `kohan_3f_index` | 過去3走の後半3F平均 - 距離別基準値 | 21位 |
| `time_index` | 過去3走の走破時計指数（距離/秒） | 12位 |
| `relative_ability` | レース内での相対能力値（past_score_meanのz-score） | 24位 |
| `current_class_score` | 今回レースのクラススコア（G1=3.0、未勝利=0.2） | 34位 |
| `class_score_change` | クラススコア変化（前走比、正=昇級、負=降級） | 7位 |
| `past_score_mean` | 過去3走のpast_scoreの平均値 | **5位** ⭐ |

---

### 4. 展開要因（2個）
レース展開を予測するための特徴量（Phase 2.5追加）

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `avg_4corner_position` | 過去の4コーナー平均位置（1-18） | **4位** ⭐ |
| `prev_rank_change` | 前走着順変化（前走着順 - 今回着順） | **1位** 🔥 |

**Note**: `prev_rank_change`は**圧倒的1位**（重要度43.8%）で、穴馬予測の最重要特徴量

---

### 5. Phase 3: 穴馬特化特徴量（4個）
成績の不安定さや展開的有利さを示す特徴量

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `past_score_std` | 過去5走の成績スコア標準偏差（高いほど波がある） | 31位 |
| `past_chakujun_variance` | 過去5走の着順分散（高いほど不安定） | 30位 |
| `zenso_oikomi_power` | 前走追い込み力（4コーナー位置 - 着順、正=追込） | 27位 |
| `zenso_kakoi_komon` | 前走包まれ度（2コーナー位置 - 4コーナー位置、正=外へ） | 34位 |

---

### 6. Phase 3.5: 前走パターン特徴量（5個）
前走のレース内容や成績変化パターンを示す特徴量

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `zenso_ninki_gap` | 前走人気着順ギャップ（人気 - 着順、正=過小評価だった） | 20位 |
| `zenso_nigeba` | 前走逃げ成功フラグ（1コーナー1位=1） | 29位 |
| `zenso_taihai` | 前走大敗フラグ（10着以下=1） | **3位** 🔥 |
| `zenso_agari_rank` | 前走上がり3F順位（レース内ランキング） | 14位 |
| `saikin_kaikakuritsu` | 直近3走改善率（前走より着順改善した割合0-1.0） | 15位 |

**Note**: `zenso_taihai`は**3位**（重要度8.4%）で、前走大敗後の巻き返しパターンを捉える

---

### 7. Phase 3.5.1: 騎手・調教師・馬統計（9個）
関係者の成績統計による穴馬の兆候を示す特徴量（2026-01-20追加）

| 特徴量名 | 説明 | 重要度順位 | 効果 |
|---------|------|-----------|-----|
| `jockey_win_rate` | 騎手勝率（過去50走で1着の割合） | 13位 | ✅ 有効 |
| `jockey_place_rate` | 騎手連対率（過去50走で3着以内の割合） | 23位 | △ 微妙 |
| `jockey_recent_form` | 騎手最近成績（過去10走平均着順スコア） | 19位 | ✅ 有効 |
| `trainer_win_rate` | 調教師勝率（過去50走で1着の割合） | 28位 | △ 微妙 |
| `trainer_place_rate` | 調教師連対率（過去50走で3着以内の割合） | **8位** 🔥 | ✅ 有効 |
| `trainer_recent_form` | 調教師最近成績（過去20走平均着順スコア） | 16位 | ✅ 有効 |
| `horse_career_win_rate` | 馬通算勝率（全レースで1着の割合） | 35位（最下位） | ❌ 無効 |
| `horse_career_place_rate` | 馬通算連対率（全レースで3着以内の割合） | 33位 | ❌ 無効 |
| `rest_weeks` | 休養週数（前走から今回までの週数） | 26位 | △ 微妙 |

**総合評価**: 
- ✅ **trainer_place_rate**が8位で**最も有効**
- ✅ 騎手・調教師の統計は一定の効果あり
- ❌ 馬の通算成績は**ほぼ無効**（穴馬予測に不適）
- 📊 **Precision改善効果**: 5.29% → 6.20%（+0.91%）

---

### 8. レース条件（3個）
レースの基本条件

| 特徴量名 | 説明 | 重要度順位 |
|---------|------|-----------|
| `kyori` | レース距離（m） | **6位** ⭐ |
| `baba_jotai_code_numeric` | 馬場状態コード（1=良、2=稍重、3=重、4=不良） | - |
| `keibajo_code_numeric` | 競馬場コード（1=札幌、9=阪神など） | 25位 |

---

### 📈 特徴量重要度トップ10

| 順位 | 特徴量名 | 重要度 | 累積割合 | カテゴリ |
|-----|---------|-------|---------|---------|
| 1 | `prev_rank_change` | 43.8% | 43.8% | 展開要因 |
| 2 | `popularity_rank` | 27.7% | 71.5% | 人気・オッズ |
| 3 | `zenso_taihai` | 8.4% | 79.9% | Phase 3.5 |
| 4 | `avg_4corner_position` | 2.6% | 82.5% | 展開要因 |
| 5 | `past_score_mean` | 2.5% | 85.0% | 基本特徴量 |
| 6 | `kyori` | 2.1% | 87.1% | レース条件 |
| 7 | `class_score_change` | 1.7% | 88.8% | 基本特徴量 |
| 8 | `trainer_place_rate` | 1.6% | 90.4% | Phase 3.5.1 |
| 9 | `past_avg_sotai_chakujun` | 1.4% | 91.8% | 基本特徴量 |
| 10 | `past_score` | 1.3% | 93.1% | 基本特徴量 |

**分析結果**:
- トップ3特徴量で**80%を説明**
- 上位9特徴量で**95%を達成**
- **展開要因（prev_rank_change）が圧倒的**

---

### 🗑️ 削除済み特徴量

以下の特徴量はPhase 3.5で削除されました：

| 特徴量名 | 削除理由 |
|---------|---------|
| `wakuban_inner` | 短距離専用で汎用性に欠ける |
| `wakuban_outer` | 短距離専用で汎用性に欠ける |
| `estimated_running_style` | 推定値でノイズが多い |
| `tenko_code` | 効果不明瞭 |
| `distance_change` | 距離適性スコアで吸収済み |
| `weight_change` | 速報データで利用不可（馬体重未確定） |

---

### 📊 今後の改善方向性

#### ✅ 有効な方向
1. **展開要因の強化**: prev_rank_changeが圧倒的1位
   - ペース予測、逃げ馬数、脚質分布など
2. **前走パターン**: zenso_taihaiが3位
   - 前走敗因分析、レース質の変化など
3. **調教師統計**: trainer_place_rateが8位
   - 調教師×騎手コンビ、厩舎別傾向など

#### ❌ 効果薄い方向
1. **馬の通算成績**: 最下位グループ
   - 穴馬は「波がある馬」なので通算成績は無意味
2. **基本特徴量の追加**: すでに飽和気味
   - これ以上追加しても効果は限定的

---

**作成日**: 2026年1月20日  
**対象**: walk_forward_validation.py, analyze_upset_patterns.py, db_query_builder.py, feature_engineering.py, model_creator.py  
**目標**: Walk-Forward検証の処理時間を 12-19時間 → 10-20分（2回目以降）に短縮
