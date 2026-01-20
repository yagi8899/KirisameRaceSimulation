#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汎用競馬予測テストスクリプト

このスクリプトは、複数のモデルファイルに対応した競馬予測テストを実行します。
model_creator.pyで作成したモデルを使用して予測を行い、結果を保存します。
"""

import psycopg2
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from keiba_constants import get_track_name, format_model_description
from model_config_loader import get_all_models, get_legacy_model
from db_query_builder import build_race_data_query

# Phase 1: 期待値・ケリー基準・信頼度スコアの統合
from expected_value_calculator import ExpectedValueCalculator
from kelly_criterion import KellyCriterion
from race_confidence_scorer import RaceConfidenceScorer

# Phase 2.5: 穴馬予測システム（全10競馬場統合）
import pickle
import json


def load_upset_threshold(track_code: str = None, surface: str = None, distance_category: str = None) -> float:
    """
    設定ファイルから穴馬候補判定の閾値を読み込む
    
    優先順位:
    1. by_track_surface_distance（最も具体的）
    2. by_track_surface
    3. by_track
    4. by_surface
    5. by_distance
    6. default_threshold（フォールバック）
    
    Args:
        track_code: 競馬場コード（01-10）
        surface: 芝ダ区分（'turf' or 'dirt'）
        distance_category: 距離区分（'short' or 'long'）
    
    Returns:
        float: 閾値
    """
    config_path = Path(__file__).parent / 'upset_threshold_config.json'
    default_threshold = 0.20
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[UPSET-THRESHOLD] 設定ファイル読み込みエラー: {e}")
        print(f"[UPSET-THRESHOLD] デフォルト閾値 {default_threshold} を使用")
        return default_threshold
    
    default_threshold = config.get('default_threshold', default_threshold)
    thresholds = config.get('thresholds_by_condition', {})
    
    # 1. 最も具体的: 競馬場_芝ダ_距離区分
    if track_code and surface and distance_category:
        key = f"{track_code}_{surface}_{distance_category}"
        if key in thresholds.get('by_track_surface_distance', {}):
            threshold = thresholds['by_track_surface_distance'][key]
            print(f"[UPSET-THRESHOLD] {key} の閾値を使用: {threshold}")
            return threshold
    
    # 2. 競馬場_芝ダ
    if track_code and surface:
        key = f"{track_code}_{surface}"
        if key in thresholds.get('by_track_surface', {}):
            threshold = thresholds['by_track_surface'][key]
            print(f"[UPSET-THRESHOLD] {key} の閾値を使用: {threshold}")
            return threshold
    
    # 3. 競馬場
    if track_code and track_code in thresholds.get('by_track', {}):
        threshold = thresholds['by_track'][track_code]
        print(f"[UPSET-THRESHOLD] track={track_code} の閾値を使用: {threshold}")
        return threshold
    
    # 4. 芝ダ区分
    if surface and surface in thresholds.get('by_surface', {}):
        threshold = thresholds['by_surface'][surface]
        print(f"[UPSET-THRESHOLD] surface={surface} の閾値を使用: {threshold}")
        return threshold
    
    # 5. 距離区分
    if distance_category and distance_category in thresholds.get('by_distance', {}):
        threshold = thresholds['by_distance'][distance_category]
        print(f"[UPSET-THRESHOLD] distance={distance_category} の閾値を使用: {threshold}")
        return threshold
    
    # 6. デフォルト
    print(f"[UPSET-THRESHOLD] デフォルト閾値を使用: {default_threshold}")
    return default_threshold


def add_purchase_logic(
    output_df: pd.DataFrame,
    prediction_rank_max: int = 3,
    popularity_rank_max: int = 3,
    min_odds: float = 1.5,
    max_odds: float = 20.0,
    min_score_diff: float = 0.05,
    initial_bankroll: float = 1000000,
    bet_unit: int = 1000
) -> pd.DataFrame:
    """
    予測結果に購入判断・購入額を追加 (新戦略: 本命×予測上位フィルター)
    
    Phase 1新戦略:
    - 予測順位1-3位 AND 人気順1-3位 のみ対象
    - オッズ範囲でフィルタリング (1.5倍～20倍)
    - 予測スコア差が一定以上のレースのみ (本命が明確)
    - 一律ベット (シンプル&確実)
    
    Args:
        output_df (DataFrame): 予測結果データフレーム
        prediction_rank_max (int): 予測順位の上限 (デフォルト: 3)
        popularity_rank_max (int): 人気順の上限 (デフォルト: 3)
        min_odds (float): 最低オッズ (デフォルト: 1.5倍)
        max_odds (float): 最高オッズ (デフォルト: 20倍)
        min_score_diff (float): 予測1位と2位のスコア差の最小値 (デフォルト: 0.05)
        initial_bankroll (float): 初期資金 (デフォルト: 100万円)
        bet_unit (int): 1頭あたりの購入額 (デフォルト: 1000円)
        
    Returns:
        DataFrame: 購入ロジックが追加されたデータフレーム
    """
    df = output_df.copy()
    
    # カラム名をマッピング (日本語 → 英語)
    df_work = df.rename(columns={
        '開催年': 'kaisai_year',
        '開催日': 'kaisai_date',
        '競馬場': 'keibajo_code',
        'レース番号': 'race_number',
        '馬番': 'umaban_numeric',
        '予測順位': 'predicted_rank',
        '予測スコア': 'predicted_score',
        '人気順': 'popularity_rank',
        '単勝オッズ': 'tansho_odds',
        '確定着順': 'chakujun_numeric'
    })
    
    # レースごとにグループ化して処理
    race_groups = df_work.groupby(['kaisai_year', 'kaisai_date', 'keibajo_code', 'race_number'])
    
    all_races = []
    current_bankroll = initial_bankroll
    total_purchased = 0
    total_wins = 0
    
    for race_id, race_df in race_groups:
        race_df = race_df.copy()
        
        # 予測スコアでソート(降順)
        race_df_sorted = race_df.sort_values('predicted_score', ascending=False).reset_index(drop=True)
        
        # 予測1位と2位のスコア差を計算
        if len(race_df_sorted) >= 2:
            score_diff = race_df_sorted.iloc[0]['predicted_score'] - race_df_sorted.iloc[1]['predicted_score']
        else:
            score_diff = 0
        
        # 全馬にレース情報を追加
        race_df['score_diff'] = score_diff
        race_df['skip_reason'] = None
        
        # フィルター1: 予測スコア差が小さいレースはスキップ
        if score_diff < min_score_diff:
            race_df['購入推奨'] = False
            race_df['購入額'] = 0
            race_df['skip_reason'] = 'low_score_diff'
            all_races.append(race_df)
            continue
        
        # フィルター2: 予測順位 AND 人気順 AND オッズ範囲
        race_df['購入推奨'] = (
            (race_df['predicted_rank'] <= prediction_rank_max) &
            (race_df['popularity_rank'] <= popularity_rank_max) &
            (race_df['tansho_odds'] >= min_odds) &
            (race_df['tansho_odds'] <= max_odds)
        )
        
        # スキップ理由を記録（優先順位順に判定）
        race_df.loc[~race_df['購入推奨'] & (race_df['predicted_rank'] > prediction_rank_max), 'skip_reason'] = 'low_predicted_rank'
        race_df.loc[~race_df['購入推奨'] & (race_df['popularity_rank'] > popularity_rank_max), 'skip_reason'] = 'low_popularity'
        race_df.loc[~race_df['購入推奨'] & (race_df['tansho_odds'] < min_odds), 'skip_reason'] = 'odds_too_low'
        race_df.loc[~race_df['購入推奨'] & (race_df['tansho_odds'] > max_odds), 'skip_reason'] = 'odds_too_high'
        
        # 購入推奨がFalseでskip_reasonがまだNoneの場合は「複合条件」として記録
        race_df.loc[~race_df['購入推奨'] & race_df['skip_reason'].isna(), 'skip_reason'] = 'multiple_conditions'
        
        # 購入推奨馬を抽出
        buy_horses = race_df[race_df['購入推奨']].copy()
        
        # 購入額列を初期化
        race_df['購入額'] = 0
        
        if len(buy_horses) > 0:
            # 一律ベット
            total_purchased += len(buy_horses)
            
            # 資金を更新
            total_bet = bet_unit * len(buy_horses)
            total_return = 0
            
            for idx in buy_horses.index:
                race_df.loc[idx, '購入額'] = bet_unit
                if race_df.loc[idx, 'chakujun_numeric'] == 1:
                    total_return += bet_unit * race_df.loc[idx, 'tansho_odds']
                    total_wins += 1
            
            current_bankroll = current_bankroll - total_bet + total_return
        
        # 現在の資金残高を記録
        race_df['現在資金'] = current_bankroll
        
        all_races.append(race_df)
    
    # 全レースを統合
    df_integrated = pd.concat(all_races, ignore_index=True)
    
    # カラム名を日本語に戻す(英語から日本語へ)
    df_integrated = df_integrated.rename(columns={
        'kaisai_year': '開催年',
        'kaisai_date': '開催日',
        'keibajo_code': '競馬場',
        'race_number': 'レース番号',
        'umaban_numeric': '馬番',
        'predicted_rank': '予測順位',
        'predicted_score': '予測スコア',
        'popularity_rank': '人気順',
        'tansho_odds': '単勝オッズ',
        'chakujun_numeric': '確定着順',
        'score_diff': 'スコア差',
        'skip_reason': 'スキップ理由'
    })
    
    return df_integrated


def save_results_with_append(df, filename, append_mode=True, output_dir='results'):
    """
    結果をTSVファイルに保存（追記モード対応）
    通常レースとスキップレースを別ファイルに分けて保存
    
    Args:
        df (DataFrame): 保存するデータフレーム
        filename (str): 保存先ファイル名
        append_mode (bool): True=追記モード、False=上書きモード
        output_dir (str): 出力先ディレクトリ（デフォルト: 'results'）
    """
    # 出力ディレクトリを作成（存在しない場合）
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # skip_reason列が存在する場合、データを分割
    if 'skip_reason' in df.columns or 'スキップ理由' in df.columns:
        skip_col = 'skip_reason' if 'skip_reason' in df.columns else 'スキップ理由'
        
        # レース単位で分析用列の有無を判定（レース内の最初のレコードでチェック）
        # レースIDを特定する列（競馬場、開催年、開催日、レース番号）
        race_id_cols = []
        for col in ['競馬場', 'keibajo_code', '開催年', 'kaisai_year', '開催日', 'kaisai_date', 'レース番号', 'race_number']:
            if col in df.columns:
                race_id_cols.append(col)
        
        if len(race_id_cols) >= 4:  # 最低4列（競馬場、年、日、レース番号）必要
            # 各レースの最初のレコードでskip_reasonの有無をチェック
            race_groups = df.groupby(race_id_cols[:4])
            skipped_races = []
            normal_races = []
            
            for race_key, race_df in race_groups:
                # レース内のいずれかのレコードにskip_reasonがあればスキップレース
                if race_df[skip_col].notna().any():
                    skipped_races.append(race_df)
                else:
                    normal_races.append(race_df)
            
            # スキップレース（分析用列を含む）
            if len(skipped_races) > 0:
                df_skipped = pd.concat(skipped_races, ignore_index=True)
            else:
                df_skipped = pd.DataFrame()
            
            # 通常レース（分析用列を削除）
            if len(normal_races) > 0:
                df_normal = pd.concat(normal_races, ignore_index=True)
                cols_to_drop = []
                for col in ['score_diff', 'スコア差', 'skip_reason', 'スキップ理由', '購入推奨', '購入額', '現在資金']:
                    if col in df_normal.columns:
                        cols_to_drop.append(col)
                df_normal_clean = df_normal.drop(columns=cols_to_drop)
            else:
                df_normal_clean = pd.DataFrame()
        else:
            # レースIDが特定できない場合は従来の方法（レコード単位）
            df_skipped = df[df[skip_col].notna()].copy()
            df_normal = df[df[skip_col].isna()].copy()
            cols_to_drop = []
            for col in ['score_diff', 'スコア差', 'skip_reason', 'スキップ理由', '購入推奨', '購入額', '現在資金']:
                if col in df_normal.columns:
                    cols_to_drop.append(col)
            df_normal_clean = df_normal.drop(columns=cols_to_drop)
        
        # 通常レース用ファイル（分析用列なし）
        if len(df_normal_clean) > 0:
            filepath_normal = output_path / filename
            if append_mode and filepath_normal.exists():
                print(f"[NOTE] 既存ファイル（通常レース）に追記: {filepath_normal}")
                df_normal_clean.to_csv(filepath_normal, mode='a', header=False, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
            else:
                print(f"[LIST] 新規ファイル作成（通常レース）: {filepath_normal}")
                df_normal_clean.to_csv(filepath_normal, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
        
        # スキップレース用ファイル（_skippedサフィックス）
        if len(df_skipped) > 0:
            skipped_filename = filename.replace('.tsv', '_skipped.tsv')
            filepath_skipped = output_path / skipped_filename
            if append_mode and filepath_skipped.exists():
                print(f"[NOTE] 既存ファイル（スキップレース）に追記: {filepath_skipped}")
                df_skipped.to_csv(filepath_skipped, mode='a', header=False, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
            else:
                print(f"[LIST] 新規ファイル作成（スキップレース）: {filepath_skipped}")
                df_skipped.to_csv(filepath_skipped, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
        
        # 全レース統合ファイル（通常+スキップ、分析用列なし）
        if len(df_normal_clean) > 0 or len(df_skipped) > 0:
            # スキップレースからも分析用列を削除
            df_skipped_clean = df_skipped.copy()
            cols_to_drop = []
            for col in ['score_diff', 'スコア差', 'skip_reason', 'スキップ理由', '購入推奨', '購入額', '現在資金']:
                if col in df_skipped_clean.columns:
                    cols_to_drop.append(col)
            if len(cols_to_drop) > 0:
                df_skipped_clean = df_skipped_clean.drop(columns=cols_to_drop)
            
            # 通常レースとスキップレースを結合
            all_races_list = []
            if len(df_normal_clean) > 0:
                all_races_list.append(df_normal_clean)
            if len(df_skipped_clean) > 0:
                all_races_list.append(df_skipped_clean)
            
            df_all = pd.concat(all_races_list, ignore_index=True)
            
            # 全レース統合ファイルを保存（_allサフィックス）
            all_filename = filename.replace('.tsv', '_all.tsv')
            filepath_all = output_path / all_filename
            if append_mode and filepath_all.exists():
                print(f"[NOTE] 既存ファイル（全レース統合）に追記: {filepath_all}")
                df_all.to_csv(filepath_all, mode='a', header=False, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
            else:
                print(f"[LIST] 新規ファイル作成（全レース統合）: {filepath_all}")
                df_all.to_csv(filepath_all, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
    else:
        # skip_reason列がない場合は従来通り
        filepath = output_path / filename
        if append_mode and filepath.exists():
            print(f"[NOTE] 既存ファイルに追記: {filepath}")
            df.to_csv(filepath, mode='a', header=False, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
        else:
            print(f"[LIST] 新規ファイル作成: {filepath}")
            df.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')


def predict_with_model(model_filename, track_code, kyoso_shubetsu_code, surface_type, 
                      min_distance, max_distance, test_year_start=2023, test_year_end=2023,
                      upset_classifier_path=None):
    """
    指定したモデルで予測を実行する汎用関数
    
    Args:
        model_filename (str): 使用するモデルファイル名
        track_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競争種別コード
        surface_type (str): 'turf' or 'dirt'
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        test_year_start (int): テスト対象開始年 (デフォルト: 2023)
        test_year_end (int): テスト対象終了年 (デフォルト: 2023)
        upset_classifier_path (str): 穴馬分類器のパス（Noneの場合は自動検索）
        
    Returns:
        tuple: (予測結果DataFrame, サマリーDataFrame, レース数)
    """
    
    # PostgreSQL コネクションの作成
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )
    
    # SQLクエリを共通化モジュールで生成
    # 注意: universal_test.pyでは払い戻し情報が必要なのでinclude_payout=True
    # また、year_start/year_endの範囲を広げて過去3年分も取得（past_avg_sotai_chakujun計算のため）
    sql = build_race_data_query(
        track_code=track_code,
        year_start=test_year_start - 3,  # 過去3年分も取得
        year_end=test_year_end,
        surface_type=surface_type.lower(),
        distance_min=min_distance,
        distance_max=max_distance,
        kyoso_shubetsu_code=kyoso_shubetsu_code,
        include_payout=True  # universal_test.pyでは払い戻し情報が必要
    )
    
    # テスト年範囲でフィルタリング（SQL生成後にPython側で追加フィルタリング）
    # build_race_data_queryで生成されたSQLにはyear_start-3～year_endの範囲が含まれるため、
    # テスト期間のみに絞り込むための追加WHERE条件を付与
    sql = f"""
    select * from (
        {sql}
    ) filtered_data
    where cast(filtered_data.kaisai_nen as integer) between {test_year_start} and {test_year_end}
    """
    
    # テスト用のSQLをログファイルに出力（常に上書き）
    log_filepath = Path('sql_log_test.txt')
    with open(log_filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== テスト用SQL ===\n")
        f.write(f"モデル: {model_filename}\n")
        f.write(f"テスト期間: {test_year_start}年〜{test_year_end}年\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{sql}\n")
    print(f"[NOTE] テスト用SQLをログファイルに出力: {log_filepath}")

    # データを取得
    df = pd.read_sql_query(sql=sql, con=conn)
    conn.close()
    
    if len(df) == 0:
        print(f"[ERROR] {model_filename} に対応するテストデータが見つかりませんでした。")
        return None, None, 0

    print(f"[+] テストデータ件数: {len(df)}件")

    # データ前処理（共通化モジュール使用）
    from data_preprocessing import preprocess_race_data
    df = preprocess_race_data(df, verbose=True)

    # 特徴量作成（共通化モジュール使用）
    from feature_engineering import create_features, add_advanced_features
    
    # 基本特徴量を作成
    X = create_features(df)
    
    # 高度な特徴量を追加（feature_engineering.pyで共通化）
    print("[START] 高度な特徴量生成...")
    X = add_advanced_features(
        df=df,
        X=X,
        surface_type=surface_type,
        min_distance=min_distance,
        max_distance=max_distance,
        logger=None,
        inverse_rank=False  # universal_test.pyでは着順を反転しない
    )
    print(f"[OK] 特徴量生成完了: {len(X.columns)}個")

    # 距離別特徴量選択はadd_advanced_features()内で実施済み
    print(f"\n[INFO] 特徴量リスト: {list(X.columns)}")

    # Phase 2.5: 展開要因特徴量を追加（穴馬予測用）
    from feature_engineering import add_upset_features
    df = add_upset_features(df)
    print("[UPSET] 展開要因特徴量を追加しました")

    # モデルをロード
    try:
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        print(f"[ERROR] モデルファイル {model_filename} が見つかりません。")
        return None, None, 0

    # シグモイド関数を定義
    def sigmoid(x):
        """値を0-1の範囲に収めるよ～"""
        return 1 / (1 + np.exp(-x))

    # 予測を実行して、シグモイド関数で変換
    raw_scores = model.predict(X)
    df['predicted_chakujun_score'] = sigmoid(raw_scores)

    # データをソート
    df = df.sort_values(by=['kaisai_nen', 'kaisai_tsukihi', 'race_bango', 'umaban'], ascending=True)

    # グループ内でのスコア順位を計算
    df['score_rank'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['predicted_chakujun_score'].rank(method='min', ascending=False)

    # kakutei_chakujun_numericを元の着順（1=1着）に戻す
    # db_query_builder.pyで「18 - 着順 + 1」で反転されてるので、元に戻す
    df['actual_chakujun'] = 19 - df['kakutei_chakujun_numeric']
    
    # kakutei_chakujun_numeric と score_rank を整数に変換
    df['kakutei_chakujun_numeric'] = df['kakutei_chakujun_numeric'].fillna(0).astype(int)
    df['actual_chakujun'] = df['actual_chakujun'].fillna(0).astype(int)
    df['tansho_ninkijun_numeric'] = df['tansho_ninkijun_numeric'].fillna(0).astype(int)
    df['score_rank'] = df['score_rank'].fillna(0).astype(int)

    # Phase 2.5: 穴馬予測を実行
    print("\n[UPSET] 穴馬分類モデルをロード中...")
    
    # 明示的にパスが指定されている場合はそれを使用
    upset_model_path = None
    if upset_classifier_path:
        upset_model_path = Path(upset_classifier_path)
        if upset_model_path.exists():
            print(f"[UPSET] 指定された穴馬分類器を使用: {upset_model_path}")
        else:
            print(f"[UPSET] 指定パスに穴馬分類器なし: {upset_model_path}")
            upset_model_path = None
    
    # パス指定がない、または見つからない場合は自動検索
    if upset_model_path is None:
        # まず、Rankerモデルのファイル名から学習期間を抽出
        model_filename_stem = Path(model_filename).stem
        train_period = None
        
        # ファイル名から期間を抽出（例: "tokyo_turf_3ageup_long_2013-2022"）
        import re
        period_match = re.search(r'(\d{4})-(\d{4})', model_filename_stem)
        if period_match:
            train_period = f"{period_match.group(1)}-{period_match.group(2)}"
            print(f"[UPSET] Rankerの学習期間検出: {train_period}")
        
        # 対応する期間の穴馬分類器を探す
        if train_period:
            # 期間ごとの穴馬分類器を優先
            period_upset_path = Path('models') / f'upset_classifier_{train_period}.sav'
            if period_upset_path.exists():
                upset_model_path = period_upset_path
                print(f"[UPSET] 期間対応穴馬分類器を使用: {period_upset_path.name}")
            else:
                print(f"[UPSET] 期間対応穴馬分類器なし: {period_upset_path.name}")
        
        # 期間対応がなければ汎用版を使用
        if upset_model_path is None:
            universal_upset_path = Path('models') / 'upset_classifier_universal.sav'
            if universal_upset_path.exists():
                upset_model_path = universal_upset_path
                print(f"[UPSET] 汎用穴馬分類器を使用: {universal_upset_path.name}")
            else:
                print(f"[UPSET] 穴馬分類器が見つかりません（Phase 1のみで継続）")
    
    if upset_model_path and upset_model_path.exists():
        with open(upset_model_path, 'rb') as f:
            upset_model_data = pickle.load(f)
        
        upset_models = upset_model_data['models']
        upset_feature_cols = upset_model_data['feature_cols']
        
        print(f"[UPSET] モデル数: {len(upset_models)}個（アンサンブル）")
        print(f"[UPSET] 特徴量数: {len(upset_feature_cols)}個")
        
        # 穴馬予測用の特徴量を準備（predicted_rankとpredicted_scoreを追加）
        df['predicted_rank'] = df['score_rank']
        # 重要: 学習時と同じく生スコア（シグモイド変換前）を使用
        df['predicted_score'] = raw_scores
        df['popularity_rank'] = df['tansho_ninkijun_numeric']
        df['value_gap'] = df['predicted_rank'] - df['popularity_rank']
        
        # keibajo_code_numericを追加（Phase 2.5で追加された特徴量）
        df['keibajo_code_numeric'] = df['keibajo_code'].astype(int)
        
        # 特徴量を抽出（欠損値・無限大を処理）
        X_upset = df[upset_feature_cols].copy()
        
        # デバッグ: 展開要因特徴量の値を確認
        print("[UPSET-DEBUG] 特徴量の分布:")
        for col in upset_feature_cols:
            if col in X_upset.columns:
                non_null = X_upset[col].notna().sum()
                print(f"  {col}: 非NULL={non_null}/{len(X_upset)}, mean={X_upset[col].mean():.4f}, min={X_upset[col].min():.4f}, max={X_upset[col].max():.4f}")
        
        X_upset = X_upset.fillna(0)
        X_upset = X_upset.replace([np.inf, -np.inf], 0)
        
        # アンサンブル予測（全モデルの平均） - 確率取得
        upset_proba_list = []
        for upset_model in upset_models:
            # LGBMClassifierかBoosterかで分岐
            if hasattr(upset_model, 'predict_proba'):
                # LGBMClassifier (scikit-learn API)
                proba = upset_model.predict_proba(X_upset, num_iteration=upset_model.best_iteration)[:, 1]
            else:
                # Booster (native API) - predict()がデフォルトで確率を返す
                proba = upset_model.predict(X_upset, num_iteration=upset_model.best_iteration)
            upset_proba_list.append(proba)
        
        df['upset_probability'] = np.mean(upset_proba_list, axis=0)
        
        # 穴馬候補判定（設定ファイルから閾値を読み込む）
        # 距離区分を判定（1800m以下=short, 1801m以上=long）
        distance_category = 'short' if max_distance <= 1800 else 'long'
        upset_threshold = load_upset_threshold(
            track_code=track_code,
            surface=surface_type.lower(),
            distance_category=distance_category
        )
        df['is_upset_candidate'] = (df['upset_probability'] > upset_threshold).astype(int)
        
        # 実際の穴馬判定（7-12番人気で3着以内）
        df['is_actual_upset'] = (
            (df['tansho_ninkijun_numeric'] >= 7) & 
            (df['tansho_ninkijun_numeric'] <= 12) & 
            (df['actual_chakujun'].isin([1, 2, 3]))
        ).astype(int)
        
        print(f"[UPSET] 穴馬候補数: {df['is_upset_candidate'].sum()}頭")
        print(f"[UPSET] 実際の穴馬数: {df['is_actual_upset'].sum()}頭")
        
        # 穴馬予測の的中率・ROI計算
        upset_candidates = df[df['is_upset_candidate'] == 1]
        if len(upset_candidates) > 0:
            upset_hits = upset_candidates[upset_candidates['is_actual_upset'] == 1]
            upset_precision = len(upset_hits) / len(upset_candidates) * 100
            
            # DEBUG: 候補と的中の詳細
            print(f"[UPSET-DEBUG] 候補詳細:")
            print(f"  候補数: {len(upset_candidates)}頭")
            print(f"  候補の人気範囲: {upset_candidates['tansho_ninkijun_numeric'].min():.0f}〜{upset_candidates['tansho_ninkijun_numeric'].max():.0f}番人気")
            print(f"  候補の確率範囲: {upset_candidates['upset_probability'].min():.4f}〜{upset_candidates['upset_probability'].max():.4f}")
            print(f"  的中数: {len(upset_hits)}頭")
            if len(upset_hits) == 0:
                # 候補の中で7-12番人気・3着以内を確認
                candidates_in_range = upset_candidates[
                    (upset_candidates['tansho_ninkijun_numeric'] >= 7) & 
                    (upset_candidates['tansho_ninkijun_numeric'] <= 12)
                ]
                print(f"  候補の中で7-12番人気: {len(candidates_in_range)}頭")
                candidates_hit_any = upset_candidates[upset_candidates['actual_chakujun'].isin([1, 2, 3])]
                print(f"  候補の中で3着以内: {len(candidates_hit_any)}頭")
            
            # ROI計算（単勝購入想定）
            total_bet = len(upset_candidates) * 100  # 1頭100円
            total_return = (upset_hits['tansho_odds'] * 100).sum()
            upset_roi = (total_return / total_bet) * 100 if total_bet > 0 else 0
            
            print(f"[UPSET] 適合率: {upset_precision:.2f}%")
            print(f"[UPSET] ROI: {upset_roi:.1f}%")
        else:
            print("[UPSET] 穴馬候補なし")
    else:
        print(f"[WARNING] 穴馬分類モデルが見つかりません: {upset_model_path}")
        df['upset_probability'] = 0.0
        df['is_upset_candidate'] = 0
        df['is_actual_upset'] = 0
    
    # surface_type列を追加（芝・ダート区分）
    from keiba_constants import get_surface_name
    df['surface_type_name'] = get_surface_name(surface_type)

    # 必要な列を選択
    output_columns = ['keibajo_name',
                      'kaisai_nen', 
                      'kaisai_tsukihi', 
                      'race_bango',
                      'surface_type_name',
                      'kyori',
                      'umaban', 
                      'bamei', 
                      'tansho_odds', 
                      'tansho_ninkijun_numeric', 
                      'actual_chakujun',  # 元の着順（1=1着）
                      'score_rank', 
                      'predicted_chakujun_score',
                      'upset_probability',  # Phase 2.5: 穴馬確率
                      'is_upset_candidate',  # Phase 2.5: 穴馬候補フラグ
                      'is_actual_upset',  # Phase 2.5: 実際の穴馬フラグ
                      '複勝1着馬番',
                      '複勝1着オッズ',
                      '複勝1着人気',
                      '複勝2着馬番',
                      '複勝2着オッズ',
                      '複勝2着人気',
                      '複勝3着馬番',
                      '複勝3着オッズ',
                      '複勝3着人気',
                      '馬連馬番1',
                      '馬連馬番2',
                      '馬連オッズ',
                      'ワイド1_2馬番1',
                      'ワイド1_2馬番2',
                      'ワイド2_3着馬番1',
                      'ワイド2_3着馬番2',
                      'ワイド1_3着馬番1',
                      'ワイド1_3着馬番2',
                      'ワイド1_2オッズ',
                      'ワイド2_3オッズ',
                      'ワイド1_3オッズ',
                      '馬単馬番1',
                      '馬単馬番2',
                      '馬単オッズ',
                      '３連複オッズ',]
    
    # DEBUG: upset列の存在確認
    upset_cols = ['upset_probability', 'is_upset_candidate', 'is_actual_upset']
    missing_cols = [c for c in upset_cols if c not in df.columns]
    if missing_cols:
        print(f"[DEBUG] 以下のupset列がdf.columnsに存在しません: {missing_cols}")
    
    # output_columns内の存在しない列をフィルタ
    available_output_columns = [c for c in output_columns if c in df.columns]
    if len(available_output_columns) < len(output_columns):
        missing_output = [c for c in output_columns if c not in df.columns]
        print(f"[DEBUG] 以下の列がdf.columnsに存在しないため除外: {missing_output}")
    
    output_df = df[available_output_columns]

    # 列名を変更
    output_df = output_df.rename(columns={
        'keibajo_name': '競馬場',
        'kaisai_nen': '開催年',
        'kaisai_tsukihi': '開催日',
        'race_bango': 'レース番号',
        'surface_type_name': '芝ダ区分',
        'kyori': '距離',
        'umaban': '馬番',
        'bamei': '馬名',
        'tansho_odds': '単勝オッズ',
        'tansho_ninkijun_numeric': '人気順',
        'actual_chakujun': '確定着順',  # 元の着順（1=1着）に戻したもの
        'score_rank': '予測順位',
        'predicted_chakujun_score': '予測スコア',
        'upset_probability': '穴馬確率',  # Phase 2.5
        'is_upset_candidate': '穴馬候補',  # Phase 2.5
        'is_actual_upset': '実際の穴馬'  # Phase 2.5
    })

    # 正しいレース数の計算方法はこれ～！
    race_count = len(output_df.groupby(['開催年', '開催日', 'レース番号']))

    # 的中率・回収率計算（元のtest.pyから移植）
    # 単勝の的中率と回収率
    tansho_hit = (output_df['確定着順'] == 1) & (output_df['予測順位'] == 1)
    tansho_hitrate = 100 * tansho_hit.sum() / race_count
    tansho_recoveryrate = 100 * (tansho_hit * output_df['単勝オッズ']).sum() / race_count

    # 複勝の的中率と回収率
    fukusho_hit = (output_df['確定着順'].isin([1, 2, 3])) & (output_df['予測順位'].isin([1, 2, 3]))
    fukusho_hitrate = fukusho_hit.sum() / (race_count * 3) * 100

    # 的中馬だけ取り出す
    hit_rows = output_df[fukusho_hit].copy()

    def extract_odds(row):
        if row['確定着順'] == 1:
            return row['複勝1着オッズ']
        elif row['確定着順'] == 2:
            return row['複勝2着オッズ']
        elif row['確定着順'] == 3:
            return row['複勝3着オッズ']
        else:
            return 0

    # 的中馬に対応する払戻を計算（100円賭けたとして）
    hit_rows['的中オッズ'] = hit_rows.apply(extract_odds, axis=1)
    total_payout = (hit_rows['的中オッズ'] * 100).sum()

    # 総購入額（毎レースで3頭に100円ずつ）
    total_bet = race_count * 3 * 100
    fukusho_recoveryrate = total_payout / total_bet * 100

    # 馬連の的中率と回収率
    umaren_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: set([1, 2]).issubset(set(x.sort_values('予測スコア', ascending=False).head(2)['確定着順'].values))
    )
    umaren_hitrate = 100 * umaren_hit.sum() / race_count
    umaren_recoveryrate = 100 * (umaren_hit * output_df.groupby(['開催年', '開催日', 'レース番号'])['馬連オッズ'].first()).sum() / race_count

    # ワイド的中率・回収率も計算（省略して簡略化）
    wide_hitrate = 0  # 計算が複雑なので省略
    wide_recoveryrate = 0

    # 馬単の的中率と回収率
    umatan_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: list(x.sort_values('予測スコア', ascending=False).head(2)['確定着順'].values) == [1, 2]
    )
    umatan_hitrate = 100 * umatan_hit.sum() / race_count
    
    umatan_odds_sum = 0
    for name, race_group in output_df.groupby(['開催年', '開催日', 'レース番号']):
        top_horses = race_group.sort_values('予測スコア', ascending=False).head(2)
        if list(top_horses['確定着順'].values) == [1, 2]:
            umatan_odds_sum += race_group['馬単オッズ'].iloc[0]

    umatan_recoveryrate = 100 * umatan_odds_sum / race_count

    # 三連複の的中率と回収率
    sanrenpuku_hit = output_df.groupby(['開催年', '開催日', 'レース番号']).apply(
        lambda x: set([1, 2, 3]).issubset(set(x.sort_values('予測スコア', ascending=False).head(3)['確定着順'].values))
    )
    sanrenpuku_hitrate = 100 * sanrenpuku_hit.sum() / len(sanrenpuku_hit)
    sanrenpuku_recoveryrate = 100 * (sanrenpuku_hit * output_df.groupby(['開催年', '開催日', 'レース番号'])['３連複オッズ'].first()).sum() / len(sanrenpuku_hit)

    # 結果をデータフレームにまとめる
    summary_df = pd.DataFrame({
        '的中数': [tansho_hit.sum(), fukusho_hit.sum(), umaren_hit.sum(), 0, umatan_hit.sum(), sanrenpuku_hit.sum()],
        '的中率(%)': [tansho_hitrate, fukusho_hitrate, umaren_hitrate, wide_hitrate, umatan_hitrate, sanrenpuku_hitrate],
        '回収率(%)': [tansho_recoveryrate, fukusho_recoveryrate, umaren_recoveryrate, wide_recoveryrate, umatan_recoveryrate, sanrenpuku_recoveryrate]
    }, index=['単勝', '複勝', '馬連', 'ワイド', '馬単', '３連複'])

    # Phase 1統合: 期待値・ケリー基準・信頼度スコアを追加
    print("[PHASE1] 新購入ロジック(本命×予測上位フィルター)を実行中...")
    try:
        output_df_with_logic = add_purchase_logic(
            output_df,
            prediction_rank_max=3,  # 予測順位1-3位
            popularity_rank_max=3,  # 人気順1-3位
            min_odds=1.5,  # 最低オッズ1.5倍
            max_odds=20.0,  # 最高オッズ20倍
            min_score_diff=0.0,  # 予測スコア差フィルタ無効化（全レース対象）
            initial_bankroll=1000000,
            bet_unit=1000  # 一律1000円ベット
        )
        print("[PHASE1] 購入ロジック統合完了!")
        
        # 購入推奨馬の統計
        buy_count = output_df_with_logic['購入推奨'].sum()
        total_bet = output_df_with_logic['購入額'].sum()
        final_bankroll = output_df_with_logic['現在資金'].iloc[-1]
        
        # 的中数を計算
        purchased = output_df_with_logic[output_df_with_logic['購入額'] > 0]
        wins = len(purchased[purchased['確定着順'] == 1])
        hit_rate = (wins / len(purchased) * 100) if len(purchased) > 0 else 0
        
        print(f"[STATS] 購入推奨馬数: {buy_count}")
        print(f"[STATS] 実購入馬数: {len(purchased)}")
        print(f"[STATS] 的中数: {wins}")
        print(f"[STATS] 的中率: {hit_rate:.2f}%")
        print(f"[STATS] 総投資額: {total_bet:,.0f}円")
        print(f"[STATS] 最終資金: {final_bankroll:,.0f}円 (初期: 1,000,000円)")
        print(f"[STATS] 損益: {final_bankroll - 1000000:+,.0f}円")
        
        # 回収率を計算
        if total_bet > 0:
            recovery_rate = (final_bankroll - 1000000 + total_bet) / total_bet * 100
            print(f"[STATS] 回収率: {recovery_rate:.2f}%")
        
        output_df = output_df_with_logic
    except Exception as e:
        print(f"[WARNING] Phase 1統合でエラー発生: {e}")
        print("[WARNING] 従来の予測結果のみ返します")
        import traceback
        traceback.print_exc()

    return output_df, summary_df, race_count


def test_multiple_models(test_year_start=2023, test_year_end=2023, model_type='all'):
    """
    複数のモデルを一括でテストする
    
    Args:
        test_year_start (int): テスト対象開始年 (デフォルト: 2023)
        test_year_end (int): テスト対象終了年 (デフォルト: 2023)
        model_type (str): テストするモデルタイプ ('all', 'standard', 'custom')
    """
    
    # JSONファイルからモデル設定を読み込み
    try:
        if model_type == 'custom':
            # customモデルのみ読み込み
            import json
            with open('model_configs.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            model_configs = config_data.get('custom_models', [])
            print(f"[INFO] customモデルを読み込み: {len(model_configs)}個")
        elif model_type == 'standard':
            # standardモデルのみ読み込み
            import json
            with open('model_configs.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            model_configs = config_data.get('standard_models', [])
            print(f"[INFO] standardモデルを読み込み: {len(model_configs)}個")
        else:
            # 全モデル読み込み
            model_configs = get_all_models()
            print(f"[INFO] 全モデルを読み込み: {len(model_configs)}個")
    except Exception as e:
        print(f"[ERROR] 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    if not model_configs:
        print("[!] テスト対象のモデル設定が見つかりませんでした。")
        return
    
    print("[RACE] 複数モデルテストを開始します！")
    print("=" * 60)
    
    all_results = {}
    # 統合ファイルの初回書き込みフラグ
    first_unified_write = True
    
    for i, config in enumerate(model_configs, 1):
        base_model_filename = config['model_filename']
        description = config.get('description', f"モデル{i}")
        
        print(f"\n【{i}/{len(model_configs)}】 {description} モデルをテスト中...")
        
        # 年範囲が指定されているモデルファイルを探す
        # 例: tokyo_turf_3ageup_long_2020-2022.sav
        import glob
        base_name = base_model_filename.replace('.sav', '')
        model_pattern = f"models/{base_name}_*-*.sav"
        matching_models = glob.glob(model_pattern)
        
        # マッチするモデルがなければ元のファイル名を使用
        if not matching_models:
            model_filename = base_model_filename
            train_year_range = "unknown"
        else:
            # 最新のモデルを使用（ファイル名でソート）
            model_filename = sorted(matching_models)[-1]
            # ファイル名から学習期間を抽出
            import re
            match = re.search(r'_(\d{4})-(\d{4})\.sav$', model_filename)
            if match:
                train_year_range = f"{match.group(1)}-{match.group(2)}"
            else:
                train_year_range = "unknown"
        
        print(f"[FILE] モデルファイル: {model_filename}")
        if train_year_range != "unknown":
            print(f"[RUN] 学習期間: {train_year_range}")
        
        # モデルファイルの存在確認
        model_path = model_filename
        if not os.path.exists(model_path):
            # modelsフォルダも確認
            models_path = f"models/{base_model_filename}"
            if os.path.exists(models_path):
                model_path = models_path
                train_year_range = "unknown"
                print(f"[DIR] modelsフォルダ内のファイルを使用: {models_path}")
            else:
                print(f"[!] モデルファイルが見つかりません。スキップします。")
                continue
        
        try:
            output_df, summary_df, race_count = predict_with_model(
                model_filename=model_path,  # 存在確認済みのパスを使用
                track_code=config['track_code'],
                kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
                surface_type=config['surface_type'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                test_year_start=test_year_start,
                test_year_end=test_year_end
            )
            
            if output_df is not None:
                # 結果ファイル名を生成（学習期間とテスト年を含める）
                base_filename = base_model_filename.replace('.sav', '')
                test_year_str = f"{test_year_start}-{test_year_end}" if test_year_start != test_year_end else str(test_year_start)
                individual_output_file = f"predicted_results_{base_filename}_train{train_year_range}_test{test_year_str}.tsv"
                summary_file = f"betting_summary_{base_filename}_train{train_year_range}_test{test_year_str}.tsv"
                
                # 個別モデル結果を上書き保存（追記ではなく上書き）
                save_results_with_append(output_df, individual_output_file, append_mode=False)
                
                # 全モデル統合ファイルに保存（初回は上書き、以降は追記）
                unified_output_file = "predicted_results.tsv"
                save_results_with_append(output_df, unified_output_file, append_mode=not first_unified_write)
                first_unified_write = False  # 初回書き込み完了
                
                # サマリーは個別ファイルに保存
                results_dir = Path('results')
                results_dir.mkdir(exist_ok=True)
                summary_filepath = results_dir / summary_file
                summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig', float_format='%.8f')
                
                print(f"[OK] 完了！レース数: {race_count}")
                print(f"  - 個別結果: {individual_output_file}")
                print(f"  - 統合結果（通常レース）: {unified_output_file}")
                print(f"  - 統合結果（スキップレース）: {unified_output_file.replace('.tsv', '_skipped.tsv')}")
                print(f"  - サマリー: {summary_file}")
                
                # 結果を保存（後で比較用）
                all_results[description] = {
                    'summary': summary_df,
                    'race_count': race_count,
                    'model_filename': model_filename
                }
                
                # 主要な結果を表示
                print(f"  - 単勝的中率: {summary_df.loc['単勝', '的中率(%)']:.2f}%")
                print(f"  - 単勝回収率: {summary_df.loc['単勝', '回収率(%)']:.2f}%")
                print(f"  - 複勝的中率: {summary_df.loc['複勝', '的中率(%)']:.2f}%")
                print(f"  - 複勝回収率: {summary_df.loc['複勝', '回収率(%)']:.2f}%")
                
            else:
                print(f"[ERROR] テストデータが見つかりませんでした。")
                
        except Exception as e:
            print(f"[ERROR] エラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # 複数モデルの比較結果を作成
    if len(all_results) > 1:
        print("\n[+] モデル比較結果")
        print("=" * 60)
        
        comparison_data = []
        for description, result in all_results.items():
            summary = result['summary']
            comparison_data.append({
                'モデル': description,
                'レース数': result['race_count'],
                '単勝的中率': f"{summary.loc['単勝', '的中率(%)']:.2f}%",
                '単勝回収率': f"{summary.loc['単勝', '回収率(%)']:.2f}%",
                '複勝的中率': f"{summary.loc['複勝', '的中率(%)']:.2f}%",
                '複勝回収率': f"{summary.loc['複勝', '回収率(%)']:.2f}%",
                '三連複的中率': f"{summary.loc['３連複', '的中率(%)']:.2f}%",
                '三連複回収率': f"{summary.loc['３連複', '回収率(%)']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 比較結果を保存
        comparison_file = 'model_comparison.tsv'
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        comparison_filepath = results_dir / comparison_file
        
        comparison_df.to_csv(comparison_filepath, index=False, sep='\t', encoding='utf-8-sig', float_format='%.8f')
        
        print(comparison_df.to_string(index=False))
        print(f"\n[LIST] 比較結果を {comparison_filepath} に保存しました！")
    
    print("\n[DONE] すべてのテストが完了しました！")


def predict_and_save_results():
    """
    旧バージョン互換性のための関数
    阪神競馬場の３歳以上芝中長距離モデルでテスト
    """
    output_df, summary_df, race_count = predict_with_model(
        model_filename='hanshin_shiba_3ageup_model.sav',
        track_code='09',  # 阪神
        kyoso_shubetsu_code='13',  # 3歳以上
        surface_type='turf',  # 芝
        min_distance=1700,  # 中長距離
        max_distance=9999  # 上限なし
    )
    
    if output_df is not None:
        # resultsディレクトリを作成
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 結果をTSVに保存（追記モード）
        output_file = 'predicted_results.tsv'
        save_results_with_append(output_df, output_file, append_mode=True)
        print(f"予測結果を results/{output_file} に保存しました！")

        # 的中率と回収率を別ファイルに保存
        summary_file = 'betting_summary.tsv'
        summary_filepath = results_dir / summary_file
        summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig', float_format='%.8f')
        print(f"的中率・回収率・的中数を results/{summary_file} に保存しました！")


if __name__ == '__main__':
    # 実行方法を選択できるように
    import sys
    
    # デフォルトのテスト年範囲
    test_year_start = 2023
    test_year_end = 2023
    
    # コマンドライン引数を解析
    mode = 'single'  # デフォルトは単一モデルテスト
    model_type = 'all'  # デフォルトは全モデル
    
    for arg in sys.argv[1:]:
        if arg == 'multi':
            mode = 'multi'
        elif arg in ['--custom', '--model-type=custom']:
            model_type = 'custom'
            mode = 'multi'
        elif arg in ['--standard', '--model-type=standard']:
            model_type = 'standard'
            mode = 'multi'
        elif '-' in arg and arg[0].isdigit():
            # "2020-2023" 形式の年範囲指定
            try:
                years = arg.split('-')
                if len(years) == 2:
                    test_year_start = int(years[0])
                    test_year_end = int(years[1])
                    print(f"[DATE] テスト年範囲指定: {test_year_start}年~{test_year_end}年")
            except ValueError:
                print(f"[!] 無効な年範囲フォーマット: {arg} (例: 2020-2023)")
        elif arg.isdigit() and len(arg) == 4:
            # "2023" 形式の単一年指定
            test_year_start = test_year_end = int(arg)
            print(f"[DATE] テスト年指定: {test_year_start}年")
    
    if mode == 'multi':
        # python universal_test.py multi [年範囲]
        # python universal_test.py --custom 2023 (customモデルのみ)
        test_multiple_models(test_year_start=test_year_start, test_year_end=test_year_end, model_type=model_type)
    else:
        # python universal_test.py [年範囲] (デフォルト)
        # 単一モデルテストで年範囲を使用
        output_df, summary_df, race_count = predict_with_model(
            model_filename='models/tokyo_turf_3ageup_long_baseline.sav',
            track_code='05',  # 東京
            kyoso_shubetsu_code='13',  # 3歳以上
            surface_type='turf',  # 芝
            min_distance=1700,  # 中長距離
            max_distance=9999,  # 上限なし
            test_year_start=test_year_start,
            test_year_end=test_year_end
        )
        
        if output_df is not None:
            # resultsディレクトリを作成
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # 結果をTSVに保存（追記モード）
            output_file = 'predicted_results.tsv'
            save_results_with_append(output_df, output_file, append_mode=True)
            print(f"予測結果を results/{output_file} に保存しました!")

            # 的中率と回収率を別ファイルに保存
            summary_file = 'betting_summary.tsv'
            summary_filepath = results_dir / summary_file
            summary_df.to_csv(summary_filepath, index=True, sep='\t', encoding='utf-8-sig', float_format='%.8f')
            print(f"的中率・回収率・的中数を results/{summary_file} に保存しました!")