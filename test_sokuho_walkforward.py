#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
速報予測ウォークフォワードテストスクリプト

このスクリプトは、sokuho_prediction.pyを使って過去データに対する
ウォークフォワードテストを実行します。

【テスト方法】
1. 2025年のレース日付リストを取得
2. 各日付に対して sokuho_prediction.py の target_date オプションを使用
3. 予測結果と実際の着順を比較
4. 結果を統合して評価

【重要】
- target_date 指定時、過去データは target_date より前の日付のみ使用
- 速報データは target_date の日付のみ使用
- これにより、実運用時と同じロジックでテスト可能
"""

import psycopg2
import pandas as pd
import json
import argparse
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 既存のsokuho_prediction.pyから必要な関数をインポート
from sokuho_prediction import predict_sokuho_model
from model_config_loader import get_all_models, get_custom_models

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """DB接続を取得"""
    with open('db_config.json', 'r', encoding='utf-8') as f:
        db_config = json.load(f)['database']
    
    return psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )


def get_race_dates_from_jvd(year: int, limit: int = None) -> list:
    """
    jvd_se（確定データ）から指定年のレース日付リストを取得
    ※中央競馬（keibajo_code 01-10）のみ
    
    Args:
        year: 対象年
        limit: 取得件数上限（デバッグ用）
    
    Returns:
        list: 'YYYYMMDD'形式の日付リスト
    """
    conn = get_db_connection()
    
    # jvd_seから日付リストを取得（中央競馬のみ: 01-10）
    sql = f"""
        SELECT DISTINCT kaisai_nen || kaisai_tsukihi as race_date
        FROM jvd_se
        WHERE kaisai_nen = '{year}'
          AND keibajo_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
        ORDER BY race_date
    """
    
    if limit:
        sql = sql.rstrip() + f" LIMIT {limit}"
    
    df = pd.read_sql_query(sql, conn)
    conn.close()
    
    return df['race_date'].tolist()


def get_actual_results(race_date: str) -> pd.DataFrame:
    """
    指定日付の実際の着順結果を取得
    
    Args:
        race_date: 'YYYYMMDD'形式の日付
    
    Returns:
        DataFrame: 実際の着順データ
    """
    conn = get_db_connection()
    
    kaisai_nen = race_date[:4]
    kaisai_tsukihi = race_date[4:]
    
    sql = f"""
        SELECT 
            ra.keibajo_code,
            se.kaisai_nen,
            se.kaisai_tsukihi,
            se.race_bango,
            se.umaban,
            se.kakutei_chakujun
        FROM jvd_se se
        JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen 
                       AND se.kaisai_tsukihi = ra.kaisai_tsukihi 
                       AND se.keibajo_code = ra.keibajo_code 
                       AND se.race_bango = ra.race_bango
        WHERE se.kaisai_nen = '{kaisai_nen}'
          AND se.kaisai_tsukihi = '{kaisai_tsukihi}'
          AND se.kakutei_chakujun IS NOT NULL
          AND se.keibajo_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    """
    
    df = pd.read_sql_query(sql, conn)
    conn.close()
    
    return df


def run_walkforward_test(
    year: int = 2025,
    model_type: str = 'custom',
    limit_dates: int = None,
    output_file: str = None
):
    """
    ウォークフォワードテストを実行
    
    Args:
        year: テスト対象年
        model_type: 'standard' または 'custom'
        limit_dates: テスト日数上限（デバッグ用）
        output_file: 結果出力ファイルパス
    """
    logger.info("=" * 60)
    logger.info("速報予測ウォークフォワードテストを開始")
    logger.info("=" * 60)
    
    # モデル設定の取得
    if model_type.lower() == 'standard':
        model_configs = get_all_models()
    else:
        model_configs = get_custom_models()
    
    # 存在するモデルのみフィルタリング
    valid_configs = []
    for config in model_configs:
        model_path = Path('models') / config['model_filename']
        if model_path.exists():
            valid_configs.append(config)
    
    logger.info(f"有効なモデル数: {len(valid_configs)}")
    
    # レース日付リストを取得
    race_dates = get_race_dates_from_jvd(year, limit_dates)
    logger.info(f"テスト対象日数: {len(race_dates)}")
    
    if len(race_dates) == 0:
        logger.error(f"{year}年のレースデータが見つかりません")
        return
    
    # 結果を格納するリスト
    all_predictions = []
    
    # 進捗追跡用
    import time
    start_time = time.time()
    processed_dates = 0
    total_predictions = 0
    
    # 並列処理用のワーカー関数
    def process_model_for_date(args):
        """1つのモデル×日付の組み合わせを処理"""
        config, race_date = args
        try:
            result = predict_sokuho_model(
                track_code=config['track_code'],
                surface_type=config['surface_type'],
                distance_min=config['min_distance'],
                distance_max=config['max_distance'],
                kyoso_shubetsu_code=config.get('kyoso_shubetsu_code'),
                model_filename=config['model_filename'],
                model_description=config['description'],
                target_date=race_date
            )
            
            if result is not None and len(result) > 0:
                result['test_date'] = race_date
                return (config['description'], result)
        except Exception as e:
            return (config['description'], f"ERROR: {e}")
        return (config['description'], None)
    
    # 並列ワーカー数（CPU数またはDB接続数を考慮）
    max_workers = min(8, os.cpu_count() or 4)
    logger.info(f"並列ワーカー数: {max_workers}")
    
    # 各日付に対してテスト実行（モデルを並列処理）
    for i, race_date in enumerate(race_dates, 1):
        date_start = time.time()
        
        # 経過時間と残り時間を計算
        elapsed = time.time() - start_time
        if processed_dates > 0:
            avg_time_per_date = elapsed / processed_dates
            remaining_dates = len(race_dates) - i + 1
            eta_seconds = avg_time_per_date * remaining_dates
            eta_str = f"残り約{int(eta_seconds // 60)}分{int(eta_seconds % 60)}秒"
        else:
            eta_str = "計算中..."
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(race_dates)}] {race_date} の予測を実行中...")
        logger.info(f"  経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒 | {eta_str}")
        logger.info(f"  モデル数: {len(valid_configs)}個を並列処理中...")
        
        # 各モデルを並列で予測
        tasks = [(config, race_date) for config in valid_configs]
        date_results = []
        hit_models = 0
        skip_models = 0
        error_models = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_model_for_date, task) for task in tasks]
            for future in as_completed(futures):
                desc, result = future.result()
                if result is None:
                    skip_models += 1
                elif isinstance(result, str) and result.startswith("ERROR"):
                    error_models += 1
                else:
                    hit_models += 1
                    date_results.append(result)
        
        date_elapsed = time.time() - date_start
        
        if len(date_results) > 0:
            df_date = pd.concat(date_results, ignore_index=True)
            all_predictions.append(df_date)
            total_predictions += len(df_date)
            logger.info(f"  ✓ 完了 ({date_elapsed:.1f}秒)")
            logger.info(f"    ヒット: {hit_models}モデル | スキップ: {skip_models}モデル | エラー: {error_models}モデル")
            logger.info(f"    予測件数: {len(df_date)}頭 (累計: {total_predictions}頭)")
        else:
            logger.info(f"  - 対象データなし ({date_elapsed:.1f}秒)")
            logger.info(f"    スキップ: {skip_models}モデル | エラー: {error_models}モデル")
        
        processed_dates += 1
    
    # 最終サマリー
    total_elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"全日付処理完了！")
    logger.info(f"  総処理時間: {int(total_elapsed // 60)}分{int(total_elapsed % 60)}秒")
    logger.info(f"  処理日数: {processed_dates}日")
    logger.info(f"  累計予測数: {total_predictions}頭")
    
    # 結果を統合
    if len(all_predictions) == 0:
        logger.warning("予測結果が0件でした")
        return
    
    df_all = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"\n総予測件数: {len(df_all)}")
    
    # 実際の着順と結合
    logger.info("実際の着順データを取得中...")
    actual_results_list = []
    for race_date in race_dates:
        actual = get_actual_results(race_date)
        if len(actual) > 0:
            actual_results_list.append(actual)
    
    if len(actual_results_list) > 0:
        df_actual = pd.concat(actual_results_list, ignore_index=True)
        
        # 予測結果と実際の着順を結合
        # キー: kaisai_nen, kaisai_tsukihi, race_bango, umaban
        # 注意: df_allの列名を確認して適切に結合
        
        # 列名の正規化（日本語→英語の可能性を考慮）
        # sokuho_prediction.pyの出力は日本語カラム名
        if '馬番' in df_all.columns:
            df_all = df_all.rename(columns={'馬番': 'umaban_pred'})
        elif 'umaban' in df_all.columns:
            df_all = df_all.rename(columns={'umaban': 'umaban_pred'})
        
        # 開催年/開催日/レース番号 → 英語に戻す
        rename_map = {
            '開催年': 'kaisai_nen',
            '開催日': 'kaisai_tsukihi',
            'レース番号': 'race_bango'
        }
        for jp_name, en_name in rename_map.items():
            if jp_name in df_all.columns and en_name not in df_all.columns:
                df_all = df_all.rename(columns={jp_name: en_name})
        
        # 結合キーの準備
        # 注意: kaisai_tsukihiが数値変換されている場合があるので、test_date列を使う
        df_all['join_key'] = (
            df_all['test_date'].astype(str) + '_' +
            df_all['race_bango'].astype(str).str.zfill(2) + '_' +
            df_all['umaban_pred'].astype(str).str.zfill(2)
        )
        
        df_actual['join_key'] = (
            df_actual['kaisai_nen'].astype(str) + 
            df_actual['kaisai_tsukihi'].astype(str).str.zfill(4) + '_' +
            df_actual['race_bango'].astype(str).str.zfill(2) + '_' +
            df_actual['umaban'].astype(str).str.zfill(2)
        )
        
        # 結合
        df_merged = df_all.merge(
            df_actual[['join_key', 'kakutei_chakujun']],
            on='join_key',
            how='left'
        )
        
        logger.info(f"着順データと結合完了: {len(df_merged)}件")
        
        # 評価指標を計算
        calculate_metrics(df_merged)
        
        # 結果を保存
        if output_file is None:
            output_file = f'walkforward_results_{year}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tsv'
        
        output_path = Path('walkforward_results')
        output_path.mkdir(exist_ok=True)
        filepath = output_path / output_file
        
        df_merged.to_csv(filepath, index=False, sep='\t', encoding='utf-8-sig')
        logger.info(f"\n結果を保存しました: {filepath}")
    
    logger.info("\nウォークフォワードテスト完了！")


def calculate_metrics(df: pd.DataFrame):
    """
    予測精度の評価指標を計算
    """
    logger.info("\n" + "=" * 40)
    logger.info("予測精度評価")
    logger.info("=" * 40)
    
    # 着順データがある行のみ
    df_valid = df[df['kakutei_chakujun'].notna()].copy()
    
    if len(df_valid) == 0:
        logger.warning("着順データが結合できませんでした")
        return
    
    # 着順を数値に変換
    df_valid['kakutei_chakujun'] = pd.to_numeric(df_valid['kakutei_chakujun'], errors='coerce')
    df_valid = df_valid[df_valid['kakutei_chakujun'].notna()]
    
    total_horses = len(df_valid)
    logger.info(f"評価対象: {total_horses}頭")
    
    # 予測スコアランク（score_rank列があれば使用）
    if 'score_rank' in df_valid.columns or '予測順位' in df_valid.columns:
        rank_col = 'score_rank' if 'score_rank' in df_valid.columns else '予測順位'
        
        # 予測1位の勝率
        pred_1st = df_valid[df_valid[rank_col] == 1]
        if len(pred_1st) > 0:
            win_rate = (pred_1st['kakutei_chakujun'] == 1).mean() * 100
            place_rate = (pred_1st['kakutei_chakujun'] <= 3).mean() * 100
            logger.info(f"予測1位 勝率: {win_rate:.1f}% ({(pred_1st['kakutei_chakujun'] == 1).sum()}/{len(pred_1st)})")
            logger.info(f"予測1位 複勝率: {place_rate:.1f}%")
        
        # 予測TOP3の複勝率
        pred_top3 = df_valid[df_valid[rank_col] <= 3]
        if len(pred_top3) > 0:
            place_rate_top3 = (pred_top3['kakutei_chakujun'] <= 3).mean() * 100
            logger.info(f"予測TOP3 複勝率: {place_rate_top3:.1f}%")
    
    # 穴馬候補の的中率（upset_candidate列があれば）
    if 'upset_candidate' in df_valid.columns or '穴馬候補' in df_valid.columns:
        upset_col = 'upset_candidate' if 'upset_candidate' in df_valid.columns else '穴馬候補'
        
        upset_horses = df_valid[df_valid[upset_col] == 1]
        if len(upset_horses) > 0:
            upset_win = (upset_horses['kakutei_chakujun'] == 1).sum()
            upset_place = (upset_horses['kakutei_chakujun'] <= 3).sum()
            logger.info(f"\n穴馬候補数: {len(upset_horses)}頭")
            logger.info(f"穴馬候補 勝率: {upset_win/len(upset_horses)*100:.1f}% ({upset_win}/{len(upset_horses)})")
            logger.info(f"穴馬候補 複勝率: {upset_place/len(upset_horses)*100:.1f}% ({upset_place}/{len(upset_horses)})")


def main():
    parser = argparse.ArgumentParser(description='速報予測ウォークフォワードテスト')
    parser.add_argument('--year', type=int, default=2025,
                        help='テスト対象年（デフォルト: 2025）')
    parser.add_argument('--model', type=str, default='custom',
                        help='モデルタイプ: standard または custom（デフォルト: custom）')
    parser.add_argument('--limit', type=int, default=None,
                        help='テスト日数上限（デバッグ用）')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ファイル名')
    
    args = parser.parse_args()
    
    run_walkforward_test(
        year=args.year,
        model_type=args.model,
        limit_dates=args.limit,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
