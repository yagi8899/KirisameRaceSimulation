"""
Walk-Forward Validation システム

競馬予測モデルのWalk-Forward Validationを自動化し、
最適な学習期間の決定とモデルの汎化性能評価を実現する。

Phase 2.5対応: Rankerと穴馬分類器を同時に学習・評価

使用方法:
    python walk_forward_validation.py
    python walk_forward_validation.py --config my_config.json
    python walk_forward_validation.py --resume
    python walk_forward_validation.py --dry-run
"""

import json
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import traceback
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

# 既存モジュールのインポート
from model_creator import create_universal_model
from model_config_loader import load_model_configs
import universal_test

# Phase 2.5: 穴馬分類器関連のインポート
from upset_classifier_creator import (
    prepare_features,
    train_with_class_weights
)
from analyze_upset_patterns import (
    get_data_with_predictions,
    create_training_dataset
)


class WalkForwardValidator:
    """Walk-Forward Validationを実行するメインクラス"""
    
    def __init__(self, config_path: str = "walk_forward_config.json"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.wfv_config = self.config['walk_forward_validation']
        self.model_configs = self._load_model_configs()
        self.progress_file = None
        self.progress_data = {}
        self.logger = None
        
        # Phase 2: progress.json排他制御用ロック
        self.progress_lock = threading.Lock()
        
        # 出力ディレクトリの設定
        self.output_dir = Path(self.wfv_config['output_dir'])
        
        # ログ設定
        self._setup_logging()
        
    def _load_config(self) -> Dict:
        """設定ファイルを読み込む"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model_configs(self) -> Dict:
        """model_configs.jsonを読み込む"""
        return load_model_configs()
    
    def _setup_logging(self):
        """ロギングを設定"""
        log_config = self.wfv_config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # ロガーの作成
        self.logger = logging.getLogger('WalkForwardValidation')
        self.logger.setLevel(log_level)
        
        # ハンドラーをクリア
        self.logger.handlers.clear()
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラー
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # ファイルハンドラー
        if 'file' in log_config:
            log_file = Path(log_config['file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _get_model_filename(self, base_name: str, train_start: int, train_end: int) -> str:
        """
        モデルファイル名を統一的に生成
        
        Args:
            base_name: モデルのベース名
            train_start: 学習開始年
            train_end: 学習終了年
            
        Returns:
            モデルファイル名
        """
        return f"{base_name}_{train_start}-{train_end}.sav"
    
    def _filter_models(self, models_setting: Any) -> List[str]:
        """
        model設定に基づいてモデルリストをフィルタリング
        
        Args:
            models_setting: "all", "standard", "custom", またはモデル名のリスト
            
        Returns:
            対象モデル名のリスト
        """
        if models_setting == "all":
            # 標準モデル
            standard_list = self.model_configs.get('standard_models', [])
            standard_names = [m['model_filename'].replace('.sav', '') for m in standard_list]
            # カスタムモデル
            custom_list = self.model_configs.get('custom_models', [])
            custom_names = [m['model_filename'].replace('.sav', '') for m in custom_list]
            return standard_names + custom_names
        
        elif models_setting == "standard":
            standard_list = self.model_configs.get('standard_models', [])
            return [m['model_filename'].replace('.sav', '') for m in standard_list]
        
        elif models_setting == "custom":
            custom_list = self.model_configs.get('custom_models', [])
            return [m['model_filename'].replace('.sav', '') for m in custom_list]
        
        elif isinstance(models_setting, list):
            return models_setting
        
        else:
            self.logger.warning(f"不明なmodels設定: {models_setting}。標準モデルを使用します。")
            standard_list = self.model_configs.get('standard_models', [])
            return [m['model_filename'].replace('.sav', '') for m in standard_list]
    
    def _get_model_config(self, model_name: str) -> Optional[Dict]:
        """
        モデル名から設定を取得
        
        Args:
            model_name: モデル名（拡張子なし）
            
        Returns:
            モデル設定辞書、見つからない場合はNone
        """
        # 標準モデルから検索
        for model in self.model_configs.get('standard_models', []):
            if model['model_filename'].replace('.sav', '') == model_name:
                return model
        
        # カスタムモデルから検索
        for model in self.model_configs.get('custom_models', []):
            if model['model_filename'].replace('.sav', '') == model_name:
                return model
        
        self.logger.error(f"モデル設定が見つかりません: {model_name}")
        return None
    
    def _get_universal_model_config(self) -> Optional[Dict]:
        """
        Universal Ranker（穴馬分類器用）の設定を取得
        
        Returns:
            モデル設定辞書、見つからない場合はNone
        """
        upset_models = self.model_configs.get('upset_classifier_models', [])
        if len(upset_models) > 0:
            return upset_models[0]  # 最初の1つを使用
        
        self.logger.error("upset_classifier_models設定が見つかりません")
        return None
        return None
    
    def _load_progress(self, progress_file: Path) -> Dict:
        """進捗ファイルを読み込む"""
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_progress(self):
        """進捗を保存 (Phase 2: 並列処理対応でロック付き)"""
        if self.progress_file:
            with self.progress_lock:
                self.progress_data['last_updated'] = datetime.now().isoformat()
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
    
    def _calculate_betting_results(self, buy_horses: pd.DataFrame, full_df: pd.DataFrame) -> Dict:
        """
        馬券種別ごとの的中率・回収率を計算（Phase 2.5対応: 穴馬予測含む）
        
        Args:
            buy_horses: 購入推奨馬のDataFrame
            full_df: 全馬のDataFrame
            
        Returns:
            集計結果の辞書
        """
        results = {}
        buy_count = len(buy_horses)
        
        if buy_count == 0:
            return {
                'tansho_hit': 0, 'tansho_rate': 0, 'tansho_return': 0,
                'fukusho_hit': 0, 'fukusho_rate': 0, 'fukusho_return': 0,
                'upset_candidates': 0, 'upset_hits': 0, 'upset_precision': 0, 'upset_roi': 0
            }
        
        # 単勝
        tansho_hit = len(buy_horses[buy_horses['確定着順'] == 1])
        tansho_rate = tansho_hit / buy_count
        tansho_return = (buy_horses[buy_horses['確定着順'] == 1]['単勝オッズ'].sum()) / buy_count
        
        results['tansho_hit'] = tansho_hit
        results['tansho_rate'] = tansho_rate
        results['tansho_return'] = tansho_return
        
        # 複勝（1-3着）
        fukusho_hit = len(buy_horses[buy_horses['確定着順'] <= 3])
        fukusho_rate = fukusho_hit / buy_count
        
        # 複勝オッズの計算（複勝1着～3着のオッズから該当するものを取得）
        fukusho_return_total = 0
        for _, horse in buy_horses.iterrows():
            chakujun = horse['確定着順']
            if chakujun <= 3:
                # 複勝オッズを取得
                if chakujun == 1 and '複勝1着オッズ' in horse and pd.notna(horse['複勝1着オッズ']):
                    fukusho_return_total += horse['複勝1着オッズ']
                elif chakujun == 2 and '複勝2着オッズ' in horse and pd.notna(horse['複勝2着オッズ']):
                    fukusho_return_total += horse['複勝2着オッズ']
                elif chakujun == 3 and '複勝3着オッズ' in horse and pd.notna(horse['複勝3着オッズ']):
                    fukusho_return_total += horse['複勝3着オッズ']
        
        fukusho_return = fukusho_return_total / buy_count
        
        results['fukusho_hit'] = fukusho_hit
        results['fukusho_rate'] = fukusho_rate
        results['fukusho_return'] = fukusho_return
        
        # 馬連・ワイド（レースごとに購入推奨馬が2頭以上いる場合のみ）
        race_groups = buy_horses.groupby(['開催年', '開催日', '競馬場', 'レース番号'])
        
        umaren_hit = 0
        umaren_bets = 0
        umaren_return_total = 0
        
        wide_hit = 0
        wide_bets = 0
        wide_return_total = 0
        
        for race_id, race_buy_horses in race_groups:
            if len(race_buy_horses) >= 2:
                # 購入推奨馬の組み合わせ数
                from itertools import combinations
                combos = list(combinations(race_buy_horses['馬番'].tolist(), 2))
                umaren_bets += len(combos)
                wide_bets += len(combos)
                
                # このレースの全馬情報を取得
                race_full = full_df[
                    (full_df['開催年'] == race_id[0]) &
                    (full_df['開催日'] == race_id[1]) &
                    (full_df['競馬場'] == race_id[2]) &
                    (full_df['レース番号'] == race_id[3])
                ]
                
                if len(race_full) == 0:
                    continue
                
                # 馬連・ワイドの的中判定
                race_sample = race_full.iloc[0]
                
                # 馬連
                if '馬連馬番1' in race_sample and pd.notna(race_sample['馬連馬番1']):
                    umaren_winning = (int(race_sample['馬連馬番1']), int(race_sample['馬連馬番2']))
                    for combo in combos:
                        if set(combo) == set(umaren_winning):
                            umaren_hit += 1
                            if '馬連オッズ' in race_sample and pd.notna(race_sample['馬連オッズ']):
                                umaren_return_total += race_sample['馬連オッズ']
                            break
                
                # ワイド（1-2着、2-3着、1-3着の3通り）
                wide_winning_pairs = []
                if 'ワイド1_2馬番1' in race_sample and pd.notna(race_sample['ワイド1_2馬番1']):
                    wide_winning_pairs.append((
                        (int(race_sample['ワイド1_2馬番1']), int(race_sample['ワイド1_2馬番2'])),
                        race_sample.get('ワイド1_2オッズ', 0)
                    ))
                if 'ワイド2_3着馬番1' in race_sample and pd.notna(race_sample['ワイド2_3着馬番1']):
                    wide_winning_pairs.append((
                        (int(race_sample['ワイド2_3着馬番1']), int(race_sample['ワイド2_3着馬番2'])),
                        race_sample.get('ワイド2_3オッズ', 0)
                    ))
                if 'ワイド1_3着馬番1' in race_sample and pd.notna(race_sample['ワイド1_3着馬番1']):
                    wide_winning_pairs.append((
                        (int(race_sample['ワイド1_3着馬番1']), int(race_sample['ワイド1_3着馬番2'])),
                        race_sample.get('ワイド1_3オッズ', 0)
                    ))
                
                for combo in combos:
                    for winning_pair, odds in wide_winning_pairs:
                        if set(combo) == set(winning_pair):
                            wide_hit += 1
                            if pd.notna(odds):
                                wide_return_total += odds
                            break
        
        # 馬連
        results['umaren_hit'] = umaren_hit
        results['umaren_rate'] = umaren_hit / umaren_bets if umaren_bets > 0 else 0
        results['umaren_return'] = umaren_return_total / umaren_bets if umaren_bets > 0 else 0
        
        # ワイド
        results['wide_hit'] = wide_hit
        results['wide_rate'] = wide_hit / wide_bets if wide_bets > 0 else 0
        results['wide_return'] = wide_return_total / wide_bets if wide_bets > 0 else 0
        
        # Phase 2.5: 穴馬予測の分析
        if '穴馬候補' in full_df.columns and '実際の穴馬' in full_df.columns:
            upset_candidates = full_df[full_df['穴馬候補'] == 1]
            upset_actual = full_df[full_df['実際の穴馬'] == 1]
            upset_hits = upset_candidates[upset_candidates['実際の穴馬'] == 1]
            
            upset_count = len(upset_candidates)
            upset_hit_count = len(upset_hits)
            upset_precision = (upset_hit_count / upset_count * 100) if upset_count > 0 else 0
            upset_recall = (upset_hit_count / len(upset_actual) * 100) if len(upset_actual) > 0 else 0
            
            # 穴馬ROI計算（単勝購入想定）
            total_bet = upset_count * 100
            total_return = (upset_hits['単勝オッズ'].sum() * 100) if len(upset_hits) > 0 else 0
            upset_roi = (total_return / total_bet * 100) if total_bet > 0 else 0
            
            results['upset_candidates'] = upset_count
            results['upset_hits'] = upset_hit_count
            results['upset_precision'] = upset_precision
            results['upset_recall'] = upset_recall
            results['upset_roi'] = upset_roi
        else:
            results['upset_candidates'] = 0
            results['upset_hits'] = 0
            results['upset_precision'] = 0
            results['upset_recall'] = 0
            results['upset_roi'] = 0
        
        return results
    
    def _initialize_progress(self, execution_mode: str, periods: List[int], test_years: List[int], models: List[str]):
        """進捗データを初期化"""
        if not self.progress_data:
            self.progress_data = {
                'execution_mode': execution_mode,
                'test_years': test_years,
                'started_at': datetime.now().isoformat(),
                'progress': {}
            }
            
            if execution_mode == 'single_period':
                self.progress_data['training_period'] = periods[0]
                period_key = f"period_{periods[0]}"
                self.progress_data['progress'][period_key] = {}
                for year in test_years:
                    self.progress_data['progress'][period_key][str(year)] = {}
            else:  # compare_periods
                self.progress_data['training_periods'] = periods
                for period in periods:
                    period_key = f"period_{period}"
                    self.progress_data['progress'][period_key] = {}
                    for year in test_years:
                        self.progress_data['progress'][period_key][str(year)] = {}
    
    def _is_model_created(self, period_key: str, year: int, model_name: str) -> bool:
        """モデルが既に作成済みか確認"""
        year_str = str(year)
        if period_key in self.progress_data.get('progress', {}):
            if year_str in self.progress_data['progress'][period_key]:
                if model_name in self.progress_data['progress'][period_key][year_str]:
                    return self.progress_data['progress'][period_key][year_str][model_name].get('model_created', False)
        return False
    
    def _is_model_tested(self, period_key: str, year: int, model_name: str) -> bool:
        """モデルが既にテスト済みか確認"""
        year_str = str(year)
        if period_key in self.progress_data.get('progress', {}):
            if year_str in self.progress_data['progress'][period_key]:
                if model_name in self.progress_data['progress'][period_key][year_str]:
                    return self.progress_data['progress'][period_key][year_str][model_name].get('model_tested', False)
        return False
    
    def _mark_model_created(self, period_key: str, year: int, model_name: str, model_path: str, success: bool = True):
        """モデル作成完了をマーク"""
        year_str = str(year)
        if period_key not in self.progress_data['progress']:
            self.progress_data['progress'][period_key] = {}
        if year_str not in self.progress_data['progress'][period_key]:
            self.progress_data['progress'][period_key][year_str] = {}
        if model_name not in self.progress_data['progress'][period_key][year_str]:
            self.progress_data['progress'][period_key][year_str][model_name] = {}
        
        self.progress_data['progress'][period_key][year_str][model_name]['model_created'] = success
        self.progress_data['progress'][period_key][year_str][model_name]['model_path'] = model_path
        self._save_progress()
    
    def _mark_model_tested(self, period_key: str, year: int, model_name: str, success: bool = True):
        """モデルテスト完了をマーク"""
        year_str = str(year)
        if period_key not in self.progress_data['progress']:
            self.progress_data['progress'][period_key] = {}
        if year_str not in self.progress_data['progress'][period_key]:
            self.progress_data['progress'][period_key][year_str] = {}
        if model_name not in self.progress_data['progress'][period_key][year_str]:
            self.progress_data['progress'][period_key][year_str][model_name] = {}
        
        self.progress_data['progress'][period_key][year_str][model_name]['model_tested'] = success
        self._save_progress()
    
    @staticmethod
    def _create_model_worker(args: Tuple) -> Tuple[str, bool, Optional[str]]:
        """
        Phase 2: 並列モデル作成用ワーカー関数
        
        各プロセスで独立してモデル作成を実行する。
        DB接続は各プロセスで独立に作成される。
        
        Args:
            args: (model_name, model_config, train_start, train_end, output_dir_str)
            
        Returns:
            (model_name, success, model_path)
        """
        model_name, model_config, train_start, train_end, output_dir_str = args
        output_dir = Path(output_dir_str)
        
        # 各プロセスで独立したロガーを作成
        logger = logging.getLogger(f'Worker-{model_name}')
        logger.setLevel(logging.INFO)
        
        try:
            # モデル設定から必要な情報を取得（常に取得する）
            track_code = model_config.get('track_code')
            surface_type = model_config.get('surface_type')
            kyoso_shubetsu_code = model_config.get('kyoso_shubetsu_code')
            min_distance = model_config.get('min_distance')
            max_distance = model_config.get('max_distance')
            
            # モデルファイル名を生成（model_config の model_filename をベースに使用）
            base_filename = model_config.get('model_filename', '').replace('.sav', '')
            
            # フォールバック: model_filename が無い場合は自動生成
            if not base_filename:
                parts = []
                if track_code:
                    parts.append(str(track_code))
                if surface_type:
                    parts.append(surface_type)
                if kyoso_shubetsu_code:
                    parts.append(str(kyoso_shubetsu_code))
                if min_distance or max_distance:
                    dist_parts = []
                    if min_distance:
                        dist_parts.append(f"{min_distance}m")
                    if max_distance:
                        dist_parts.append(f"{max_distance}m")
                    parts.append('-'.join(dist_parts))
                base_filename = '_'.join(parts)
            
            model_filename = f'{base_filename}_{train_start}-{train_end}.sav'
            model_path = output_dir / model_filename
            
            logger.info(f"モデル作成開始: {model_name} (学習期間: {train_start}-{train_end})")
            
            # Rankerモデル作成（各プロセスでDB接続が作成される）
            create_universal_model(
                track_code=track_code,
                kyoso_shubetsu_code=kyoso_shubetsu_code,
                surface_type=surface_type,
                min_distance=min_distance,
                max_distance=max_distance,
                model_filename=model_filename,
                output_dir=str(output_dir),
                year_start=train_start,
                year_end=train_end
            )
            
            if not model_path.exists():
                logger.error(f"Rankerモデル作成失敗: {model_filename}")
                return model_name, False, None
            
            logger.info(f"Rankerモデル作成完了: {model_filename}")
            return model_name, True, str(model_path)
            
        except Exception as e:
            logger.error(f"モデル作成エラー: {model_name}")
            logger.error(f"エラー詳細: {str(e)}")
            logger.error(traceback.format_exc())
            return model_name, False, None
    
    @staticmethod
    def _test_model_worker(args: Tuple) -> Tuple[str, bool, Optional[str]]:
        """
        並列テスト実行用ワーカー関数
        
        各プロセスで独立してテストを実行する。
        DB接続は各プロセスで独立に作成される。
        
        Args:
            args: (model_name, model_config, model_path, test_year, output_dir_str, upset_classifier_path_str)
            
        Returns:
            (model_name, success, result_filename)
        """
        model_name, model_config, model_path, test_year, output_dir_str, upset_classifier_path_str = args
        output_dir = Path(output_dir_str)
        
        # 各プロセスで独立したロガーを作成
        logger = logging.getLogger(f'TestWorker-{model_name}')
        logger.setLevel(logging.INFO)
        
        try:
            logger.info(f"テスト実行開始: {model_name} (テスト年: {test_year})")
            
            # テスト結果ファイル名
            train_period = Path(model_path).stem.split('_')[-1]  # 例: "2018-2022"
            result_filename = f"predicted_results_{model_name}_{train_period}_test{test_year}.tsv"
            
            # 穴馬分類器パスの処理（芝/ダート分離対応）
            upset_classifier_path = None
            if upset_classifier_path_str:
                # パイプ区切りの場合（芝/ダート分離モデル）
                if '|' in upset_classifier_path_str:
                    turf_path, dirt_path = upset_classifier_path_str.split('|', 1)
                    if os.path.exists(turf_path) and os.path.exists(dirt_path):
                        upset_classifier_path = upset_classifier_path_str
                        logger.info(f"[UPSET] 芝/ダート分離モデルを使用: turf={Path(turf_path).name}, dirt={Path(dirt_path).name}")
                    else:
                        logger.warning(f"[UPSET] 芝/ダート分離モデルが見つかりません: turf={turf_path} ({os.path.exists(turf_path)}), dirt={dirt_path} ({os.path.exists(dirt_path)})")
                # 従来の単一モデルの場合
                elif os.path.exists(upset_classifier_path_str):
                    upset_classifier_path = upset_classifier_path_str
                    logger.info(f"[UPSET] 穴馬分類器を使用: {Path(upset_classifier_path).name}")
            
            # universal_testのpredict_with_model関数を呼び出し
            result_df, summary_df, race_count = universal_test.predict_with_model(
                model_filename=model_path,
                track_code=model_config.get('track_code'),
                kyoso_shubetsu_code=model_config.get('kyoso_shubetsu_code'),
                surface_type=model_config.get('surface_type'),
                min_distance=model_config.get('min_distance'),
                max_distance=model_config.get('max_distance'),
                test_year_start=test_year,
                test_year_end=test_year,
                upset_classifier_path=upset_classifier_path
            )
            
            if result_df is None or len(result_df) == 0:
                logger.warning(f"テストデータなし: {model_name} (テスト年: {test_year})")
                return model_name, True, None  # データがないのはエラーではない
            
            # 結果を保存
            universal_test.save_results_with_append(
                df=result_df,
                filename=result_filename,
                append_mode=False,  # WFVでは年ごとに独立ファイルなので上書き
                output_dir=str(output_dir)
            )
            
            logger.info(f"テスト実行完了: {result_filename} (レース数: {race_count})")
            return model_name, True, result_filename
            
        except Exception as e:
            logger.error(f"テスト実行エラー: {model_name}")
            logger.error(f"エラー詳細: {str(e)}")
            logger.error(traceback.format_exc())
            return model_name, False, None

    def create_model_for_year(
        self, 
        model_name: str, 
        model_config: Dict, 
        train_start: int, 
        train_end: int, 
        output_dir: Path
    ) -> Tuple[bool, Optional[str]]:
        """
        1年分のモデルを作成（Phase 2.5: Ranker + 穴馬分類器）
        
        Args:
            model_name: モデル名
            model_config: モデル設定
            train_start: 学習開始年
            train_end: 学習終了年
            output_dir: 出力ディレクトリ
            
        Returns:
            (成功フラグ, モデルファイルパス)
        """
        try:
            # モデルファイル名を生成
            model_filename = self._get_model_filename(model_name, train_start, train_end)
            model_path = output_dir / model_filename
            
            self.logger.info(f"モデル作成開始: {model_name} (学習期間: {train_start}-{train_end})")
            
            # Phase 1: Rankerモデル作成
            create_universal_model(
                track_code=model_config.get('track_code'),
                kyoso_shubetsu_code=model_config.get('kyoso_shubetsu_code'),
                surface_type=model_config.get('surface_type'),
                min_distance=model_config.get('min_distance'),
                max_distance=model_config.get('max_distance'),
                model_filename=model_filename,
                output_dir=str(output_dir),
                year_start=train_start,
                year_end=train_end
            )
            
            if not model_path.exists():
                self.logger.error(f"Rankerモデル作成失敗: {model_filename}")
                return False, None
            
            self.logger.info(f"Rankerモデル作成完了: {model_filename}")
            
            # Phase 2.5: 穴馬分類器作成は run_single_period_mode() で独立して実行されるため、
            # ここでは実行しない（重複防止）
            
            return True, str(model_path)
            
        except Exception as e:
            self.logger.error(f"モデル作成エラー: {model_name}")
            self.logger.error(f"エラー詳細: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False, None
    
    def _create_upset_classifier(
        self,
        model_name: str,
        train_start: int,
        train_end: int,
        output_dir: Path
    ) -> bool:
        """
        Phase 2.5: 穴馬分類器を作成（Walk-Forward対応版）
        芝/ダート分離戦略: 芝モデルとダートモデルを別々に作成
        
        Universal Rankerで全競馬場予測 → その結果で穴馬学習データ生成 → 芝/ダート別に穴馬分類器学習
        
        Args:
            model_name: モデル名（実際には使用せず、universalモデルを使用）
            train_start: 学習開始年
            train_end: 学習終了年
            output_dir: 出力ディレクトリ
            
        Returns:
            成功フラグ
        """
        try:
            # 芝/ダート分離: 2つのファイルを作成
            upset_filename_turf = f"upset_classifier_turf_{train_start}-{train_end}.sav"
            upset_filename_dirt = f"upset_classifier_dirt_{train_start}-{train_end}.sav"
            upset_path_turf = output_dir / upset_filename_turf
            upset_path_dirt = output_dir / upset_filename_dirt
            
            # 両方存在する場合はスキップ
            if upset_path_turf.exists() and upset_path_dirt.exists():
                self.logger.info(f"[UPSET] 芝/ダート穴馬分類器は既に存在（スキップ）")
                return True
            
            self.logger.info(f"[UPSET] 芝/ダート分離 穴馬分類器作成開始 (期間: {train_start}-{train_end})")
            
            # Universal Rankerモデルのパスを取得
            universal_model_name = "all_tracks_all_surfaces_all_ages"
            universal_filename = self._get_model_filename(universal_model_name, train_start, train_end)
            universal_path = output_dir / universal_filename
            
            # upset_classifier_modelsセクションから設定を取得（先に取得）
            universal_config = self._get_universal_model_config()
            if universal_config is None:
                self.logger.error(f"[UPSET] Universal Ranker設定が見つかりません")
                return False
            
            # Universal Rankerが存在しない場合は作成
            universal_already_exists = universal_path.exists()
            
            if not universal_already_exists:
                self.logger.info(f"[UPSET] Universal Ranker作成: {universal_filename}")
                
                # Universal Ranker作成
                create_universal_model(
                    track_code=None,  # 全競馬場
                    kyoso_shubetsu_code=universal_config.get('kyoso_shubetsu_code'),
                    surface_type=None,  # 芝・ダート両方
                    min_distance=universal_config.get('min_distance'),
                    max_distance=universal_config.get('max_distance'),
                    model_filename=universal_filename,
                    output_dir=str(output_dir),
                    year_start=train_start,
                    year_end=train_end
                )
                
                if not universal_path.exists():
                    self.logger.error(f"[UPSET] Universal Ranker作成失敗: {universal_filename}")
                    return False
                
                self.logger.info(f"[UPSET] Universal Ranker作成完了")
            else:
                self.logger.info(f"[UPSET] Universal Rankerは既に存在: {universal_filename}")
            
            # 対象期間の年リスト（2020除く）
            years = [y for y in range(train_start, train_end + 1) if y != 2020]
            
            self.logger.info(f"[UPSET] Universal Rankerで全競馬場予測実行: {years}")
            
            # Universal Rankerモデルで対象期間のデータに予測を実行
            from analyze_upset_patterns import get_data_with_predictions
            
            # 全競馬場・全距離・芝ダ統合・全年齢で予測
            df_predicted = get_data_with_predictions(
                model_path=str(universal_path),
                years=years,
                track_codes=None,  # 全競馬場対象
                surface_type=None,  # 芝・ダート両方
                distance_min=1000,  # 最小距離
                distance_max=9999,  # 最大距離
                kyoso_shubetsu_code=universal_config.get('kyoso_shubetsu_code')  # 全年齢
            )
            
            if df_predicted is None or len(df_predicted) == 0:
                self.logger.error(f"[UPSET] 予測データが0件")
                return False
            
            self.logger.info(f"[UPSET] 予測完了: {len(df_predicted)}頭")
            
            # 芝/ダート分離: track_codeから芝/ダートを判定
            from keiba_constants import get_surface_type_from_track_cd
            df_predicted['surface_type'] = df_predicted['track_code'].apply(get_surface_type_from_track_cd)
            
            # 芝とダートでデータを分割
            df_turf = df_predicted[df_predicted['surface_type'] == 'turf'].copy()
            df_dirt = df_predicted[df_predicted['surface_type'] == 'dirt'].copy()
            
            self.logger.info(f"[UPSET] 芝データ: {len(df_turf)}頭")
            self.logger.info(f"[UPSET] ダートデータ: {len(df_dirt)}頭")
            
            # 穴馬学習データセットを作成（7-12番人気）
            from analyze_upset_patterns import create_training_dataset
            
            success_turf = True
            success_dirt = True
            
            # ===== 芝モデルの学習 =====
            if not upset_path_turf.exists() and len(df_turf) > 0:
                self.logger.info(f"[UPSET-TURF] 芝モデル学習開始")
                
                df_training_turf, feature_cols = create_training_dataset(
                    df_turf,
                    popularity_min=7,
                    popularity_max=12
                )
                
                if len(df_training_turf) > 0:
                    self.logger.info(f"[UPSET-TURF] 学習データ: {len(df_training_turf)}頭")
                    self.logger.info(f"[UPSET-TURF] 穴馬: {df_training_turf['is_upset'].sum()}頭 ({df_training_turf['is_upset'].mean()*100:.2f}%)")
                    
                    # 特徴量準備
                    X_turf, y_turf, feature_cols = prepare_features(df_training_turf)
                    
                    # 学習（クラスウェイト方式）
                    models_turf, cv_results_turf = train_with_class_weights(
                        X_turf, y_turf, feature_cols,
                        n_splits=5,
                        random_state=42,
                        use_calibration=False
                    )
                    
                    # モデル保存
                    success_turf = self._save_upset_classifier(
                        models_turf, feature_cols, upset_path_turf, 
                        train_start, train_end, surface_type='turf'
                    )
                else:
                    self.logger.warning(f"[UPSET-TURF] 芝学習データが0件")
                    success_turf = False
            else:
                self.logger.info(f"[UPSET-TURF] 芝モデルは既に存在またはデータなし")
            
            # ===== ダートモデルの学習 =====
            if not upset_path_dirt.exists() and len(df_dirt) > 0:
                self.logger.info(f"[UPSET-DIRT] ダートモデル学習開始")
                
                df_training_dirt, feature_cols = create_training_dataset(
                    df_dirt,
                    popularity_min=7,
                    popularity_max=12
                )
                
                if len(df_training_dirt) > 0:
                    self.logger.info(f"[UPSET-DIRT] 学習データ: {len(df_training_dirt)}頭")
                    self.logger.info(f"[UPSET-DIRT] 穴馬: {df_training_dirt['is_upset'].sum()}頭 ({df_training_dirt['is_upset'].mean()*100:.2f}%)")
                    
                    # 特徴量準備
                    X_dirt, y_dirt, feature_cols = prepare_features(df_training_dirt)
                    
                    # 学習（クラスウェイト方式）
                    models_dirt, cv_results_dirt = train_with_class_weights(
                        X_dirt, y_dirt, feature_cols,
                        n_splits=5,
                        random_state=42,
                        use_calibration=False
                    )
                    
                    # モデル保存
                    success_dirt = self._save_upset_classifier(
                        models_dirt, feature_cols, upset_path_dirt,
                        train_start, train_end, surface_type='dirt'
                    )
                else:
                    self.logger.warning(f"[UPSET-DIRT] ダート学習データが0件")
                    success_dirt = False
            else:
                self.logger.info(f"[UPSET-DIRT] ダートモデルは既に存在またはデータなし")
            
            self.logger.info(f"[UPSET] 芝/ダート分離完了: 芝={success_turf}, ダート={success_dirt}")
            self.logger.info(f"[UPSET] 必要に応じて models/ ディレクトリに手動コピーしてください")
            
            return success_turf and success_dirt
            
        except Exception as e:
            self.logger.error(f"[UPSET] 穴馬分類器作成エラー: {str(e)}")
            self.logger.error(f"[UPSET] スタックトレース:\n{traceback.format_exc()}")
            import sys
            self.logger.error(f"[UPSET] エラー詳細: type={type(e).__name__}, args={e.args}")
            return False
    
    def _save_upset_classifier(
        self,
        models: list,
        feature_cols: list,
        save_path: Path,
        train_start: int,
        train_end: int,
        surface_type: str
    ) -> bool:
        """
        穴馬分類器をファイルに保存するヘルパー関数
        
        Args:
            models: 学習済みモデルリスト
            feature_cols: 特徴量名リスト
            save_path: 保存先パス
            train_start: 学習開始年
            train_end: 学習終了年
            surface_type: 'turf' or 'dirt'
            
        Returns:
            成功フラグ
        """
        try:
            import pickle
            
            # 新しい構造（dictリスト）からモデルと校正器を分離
            lgb_models = [m['model'] for m in models]
            calibrators = [m['calibrator'] for m in models]
            calibration_method = models[0].get('calibration_method', None)
            has_calibration = calibrators[0] is not None
            
            model_data = {
                'models': lgb_models,
                'calibrators': calibrators,
                'feature_cols': feature_cols,
                'n_models': len(models),
                'train_period': f"{train_start}-{train_end}",
                'surface_type': surface_type,  # 芝/ダート区分を保存
                'has_calibration': has_calibration,
                'calibration_method': calibration_method
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            surface_name = '芝' if surface_type == 'turf' else 'ダート'
            self.logger.info(f"[UPSET-{surface_type.upper()}] {surface_name}穴馬分類器保存: {save_path.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[UPSET] 保存エラー: {e}")
            return False
    
    def test_model_for_year(
        self,
        model_name: str,
        model_config: Dict,
        model_path: str,
        test_year: int,
        output_dir: Path
    ) -> bool:
        """
        1年分のモデルをテスト
        
        Args:
            model_name: モデル名
            model_config: モデル設定
            model_path: モデルファイルパス
            test_year: テスト年
            output_dir: 出力ディレクトリ
            
        Returns:
            成功フラグ
        """
        try:
            self.logger.info(f"テスト実行開始: {model_name} (テスト年: {test_year})")
            
            # テスト結果ファイル名
            train_period = Path(model_path).stem.split('_')[-1]  # 例: "2018-2022"
            result_filename = f"predicted_results_{model_name}_{train_period}_test{test_year}.tsv"
            
            # 穴馬分類器のパスを計算（芝/ダート分離対応）
            model_dir = Path(model_path).parent
            
            # 芝/ダート分離版を優先して探す
            upset_classifier_path_turf = model_dir / f"upset_classifier_turf_{train_period}.sav"
            upset_classifier_path_dirt = model_dir / f"upset_classifier_dirt_{train_period}.sav"
            
            # 芝/ダート分離版が両方存在するか確認
            if upset_classifier_path_turf.exists() and upset_classifier_path_dirt.exists():
                self.logger.info(f"[UPSET] 芝/ダート分離モデルを使用: turf={upset_classifier_path_turf.name}, dirt={upset_classifier_path_dirt.name}")
                upset_classifier_path = f"{upset_classifier_path_turf}|{upset_classifier_path_dirt}"
            else:
                # 旧版（統合モデル）を探す
                upset_classifier_path_legacy = model_dir / f"upset_classifier_{train_period}.sav"
                if upset_classifier_path_legacy.exists():
                    self.logger.info(f"[UPSET] 統合穴馬分類器を使用: {upset_classifier_path_legacy.name}")
                    upset_classifier_path = str(upset_classifier_path_legacy)
                else:
                    self.logger.warning(f"[UPSET] 穴馬分類器が見つかりません")
                    upset_classifier_path = None
            
            # universal_testのpredict_with_model関数を呼び出し
            result_df, summary_df, race_count = universal_test.predict_with_model(
                model_filename=model_path,
                track_code=model_config.get('track_code'),
                kyoso_shubetsu_code=model_config.get('kyoso_shubetsu_code'),
                surface_type=model_config.get('surface_type'),
                min_distance=model_config.get('min_distance'),
                max_distance=model_config.get('max_distance'),
                test_year_start=test_year,
                test_year_end=test_year,
                upset_classifier_path=upset_classifier_path
            )
            
            if result_df is None or len(result_df) == 0:
                self.logger.warning(f"テストデータなし: {model_name} (テスト年: {test_year})")
                return True  # データがないのはエラーではない
            
            # 結果を保存
            universal_test.save_results_with_append(
                df=result_df,
                filename=result_filename,
                append_mode=False,  # WFVでは年ごとに独立ファイルなので上書き
                output_dir=str(output_dir)
            )
            
            self.logger.info(f"テスト実行完了: {result_filename} (レース数: {race_count})")
            return True
            
        except Exception as e:
            self.logger.error(f"テスト実行エラー: {model_name}")
            self.logger.error(f"エラー詳細: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_single_period_mode(self, resume: bool = False, dry_run: bool = False) -> bool:
        """
        単一期間モードを実行
        
        Args:
            resume: 前回から再開するか
            dry_run: ドライラン（実行計画のみ表示）
            
        Returns:
            成功フラグ
        """
        settings = self.wfv_config['single_period_settings']
        training_period = settings['training_period']
        test_years = self.wfv_config['test_years']
        models_setting = settings['models']
        
        # モデルリストを取得
        target_models = self._filter_models(models_setting)
        
        self.logger.info("=" * 80)
        self.logger.info("Walk-Forward Validation - 単一期間モード")
        self.logger.info(f"学習期間: {training_period}年")
        self.logger.info(f"テスト年: {test_years}")
        self.logger.info(f"対象モデル数: {len(target_models)}")
        self.logger.info("=" * 80)
        
        if dry_run:
            self.logger.info("[DRY RUN] 実行計画:")
            for year in test_years:
                train_start = year - training_period
                train_end = year - 1
                self.logger.info(f"  テスト年 {year}: 学習期間 {train_start}-{train_end}")
                for model_name in target_models:
                    self.logger.info(f"    - {model_name}")
            return True
        
        # 出力ディレクトリ作成
        period_key = f"period_{training_period}"
        period_dir = self.output_dir / period_key
        models_dir = period_dir / "models"
        test_results_dir = period_dir / "test_results"
        
        # 進捗ファイルの設定
        self.progress_file = self.output_dir / "progress.json"
        
        if resume and self.progress_file.exists():
            self.progress_data = self._load_progress(self.progress_file)
            self.logger.info("前回の進捗を読み込みました")
        else:
            self._initialize_progress('single_period', [training_period], test_years, target_models)
        
        # 各テスト年でループ
        for test_year in test_years:
            train_start = test_year - training_period
            train_end = test_year - 1
            
            self.logger.info("-" * 80)
            self.logger.info(f"テスト年: {test_year} (学習期間: {train_start}-{train_end})")
            self.logger.info("-" * 80)
            
            # 年ごとのディレクトリ作成
            year_models_dir = models_dir / str(test_year)
            year_test_dir = test_results_dir / str(test_year)
            year_models_dir.mkdir(parents=True, exist_ok=True)
            year_test_dir.mkdir(parents=True, exist_ok=True)
            
            # ★ UPSET分類器チェック（全モデルより前に独立して実行）
            # 芝/ダート分離: 2つの穴馬分類器を管理
            upset_filename_turf = f"upset_classifier_turf_{train_start}-{train_end}.sav"
            upset_filename_dirt = f"upset_classifier_dirt_{train_start}-{train_end}.sav"
            upset_model_name_turf = f"upset_classifier_turf_{train_start}-{train_end}"
            upset_model_name_dirt = f"upset_classifier_dirt_{train_start}-{train_end}"
            
            # progress.jsonでスキップチェック（両方作成済みならスキップ）
            turf_created = self._is_model_created(period_key, test_year, upset_model_name_turf)
            dirt_created = self._is_model_created(period_key, test_year, upset_model_name_dirt)
            
            if turf_created and dirt_created:
                self.logger.info(f"[UPSET] 穴馬分類器（芝/ダート）: スキップ（作成済み）")
            else:
                # UPSET分類器作成（最初のモデル名を使用）
                first_model_name = target_models[0] if target_models else "default"
                upset_success = self._create_upset_classifier(
                    first_model_name, train_start, train_end, year_models_dir
                )
                
                upset_path_turf = year_models_dir / upset_filename_turf
                upset_path_dirt = year_models_dir / upset_filename_dirt
                
                if upset_success:
                    # 芝モデルの記録
                    if upset_path_turf.exists():
                        self._mark_model_created(period_key, test_year, upset_model_name_turf, str(upset_path_turf), True)
                        self.logger.info(f"[UPSET] 穴馬分類器（芝）作成完了: {upset_filename_turf}")
                    
                    # ダートモデルの記録
                    if upset_path_dirt.exists():
                        self._mark_model_created(period_key, test_year, upset_model_name_dirt, str(upset_path_dirt), True)
                        self.logger.info(f"[UPSET] 穴馬分類器（ダート）作成完了: {upset_filename_dirt}")
                    
                    # Universal Rankerも記録
                    universal_filename = self._get_model_filename("all_tracks_all_surfaces_all_ages", train_start, train_end)
                    universal_path = year_models_dir / universal_filename
                    if universal_path.exists():
                        universal_model_name = f"universal_ranker_{train_start}-{train_end}"
                        self._mark_model_created(period_key, test_year, universal_model_name, str(universal_path), True)
                        self.logger.info(f"[UPSET] Universal Ranker記録完了: {universal_filename}")
                else:
                    self.logger.warning(f"[UPSET] 穴馬分類器作成失敗")
            
            # モデル作成フェーズ (Phase 2: 並列実行)
            self.logger.info(f"[モデル作成フェーズ] {len(target_models)}モデル (並列実行)")
            
            # 未作成モデルをフィルタリング
            models_to_create = []
            for model_name in target_models:
                if self._is_model_created(period_key, test_year, model_name):
                    self.logger.info(f"  {model_name}: スキップ（作成済み）")
                else:
                    model_config = self._get_model_config(model_name)
                    if model_config:
                        models_to_create.append((model_name, model_config))
                    else:
                        self._mark_model_created(period_key, test_year, model_name, "", False)
            
            if models_to_create:
                # ProcessPoolExecutorで並列実行
                max_workers = min(4, multiprocessing.cpu_count())
                self.logger.info(f"  並列実行ワーカー数: {max_workers}")
                
                # 引数リストを作成
                worker_args = [
                    (name, config, train_start, train_end, str(year_models_dir))
                    for name, config in models_to_create
                ]
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 全タスクを投入
                    future_to_model = {
                        executor.submit(self._create_model_worker, args): args[0]
                        for args in worker_args
                    }
                    
                    # 完了したタスクから結果を取得
                    completed_count = 0
                    for future in as_completed(future_to_model):
                        model_name = future_to_model[future]
                        try:
                            result_name, success, model_path = future.result()
                            completed_count += 1
                            
                            self.logger.info(
                                f"  [{completed_count}/{len(models_to_create)}] {result_name}: "
                                f"{'完了' if success else '失敗'}"
                            )
                            
                            # progress.jsonに記録（ロック付き）
                            self._mark_model_created(period_key, test_year, result_name, model_path or "", success)
                            
                            if not success:
                                error_action = self.wfv_config['execution'].get('on_model_creation_error', 'skip')
                                if error_action == 'stop':
                                    self.logger.error("モデル作成エラーにより処理を中断します")
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    return False
                        
                        except Exception as e:
                            self.logger.error(f"  {model_name}: 並列実行エラー - {str(e)}")
                            self._mark_model_created(period_key, test_year, model_name, "", False)
            else:
                self.logger.info("  作成対象モデルなし（全て作成済み）")
            
            # テスト実行フェーズ (並列実行)
            self.logger.info(f"[テスト実行フェーズ] {len(target_models)}モデル (並列実行)")
            
            # 穴馬分類器のパスを取得（芝/ダート分離版を優先）
            train_period = f"{train_start}-{train_end}"
            upset_classifier_path_turf = year_models_dir / f"upset_classifier_turf_{train_period}.sav"
            upset_classifier_path_dirt = year_models_dir / f"upset_classifier_dirt_{train_period}.sav"
            
            if upset_classifier_path_turf.exists() and upset_classifier_path_dirt.exists():
                # 芝/ダート分離版（パイプ区切りで渡す）
                upset_classifier_path_str = f"{upset_classifier_path_turf}|{upset_classifier_path_dirt}"
                self.logger.info(f"[UPSET] 芝/ダート分離モデルを使用: turf={upset_classifier_path_turf.name}, dirt={upset_classifier_path_dirt.name}")
            else:
                # 従来の統合版にフォールバック
                upset_classifier_path_legacy = year_models_dir / f"upset_classifier_{train_period}.sav"
                upset_classifier_path_str = str(upset_classifier_path_legacy) if upset_classifier_path_legacy.exists() else None
                if upset_classifier_path_str:
                    self.logger.info(f"[UPSET] 統合モデルを使用: {upset_classifier_path_legacy.name}")
            
            # 未テストモデルをフィルタリング
            models_to_test = []
            for model_name in target_models:
                # スキップ判定
                if self._is_model_tested(period_key, test_year, model_name):
                    self.logger.info(f"  {model_name}: スキップ（テスト済み）")
                    continue
                
                # モデルが作成されているか確認
                year_str = str(test_year)
                model_path = None
                if period_key in self.progress_data.get('progress', {}):
                    if year_str in self.progress_data['progress'][period_key]:
                        if model_name in self.progress_data['progress'][period_key][year_str]:
                            model_info = self.progress_data['progress'][period_key][year_str][model_name]
                            if not model_info.get('model_created', False):
                                self.logger.warning(f"  {model_name}: スキップ（モデル未作成）")
                                continue
                            
                            model_path = model_info.get('model_path')
                            if not model_path or not os.path.exists(model_path):
                                self.logger.warning(f"  {model_name}: スキップ（モデルファイル不明）")
                                continue
                
                model_config = self._get_model_config(model_name)
                if not model_config:
                    self._mark_model_tested(period_key, test_year, model_name, False)
                    continue
                
                models_to_test.append((model_name, model_config, model_path))
            
            if models_to_test:
                # ProcessPoolExecutorで並列実行
                max_workers = min(4, multiprocessing.cpu_count())
                self.logger.info(f"  並列実行ワーカー数: {max_workers}")
                
                # 引数リストを作成
                test_worker_args = [
                    (name, config, path, test_year, str(year_test_dir), upset_classifier_path_str)
                    for name, config, path in models_to_test
                ]
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 全タスクを投入
                    future_to_model = {
                        executor.submit(self._test_model_worker, args): args[0]
                        for args in test_worker_args
                    }
                    
                    # 完了したタスクから結果を取得
                    completed_count = 0
                    for future in as_completed(future_to_model):
                        model_name = future_to_model[future]
                        try:
                            result_name, success, result_filename = future.result()
                            completed_count += 1
                            
                            self.logger.info(
                                f"  [{completed_count}/{len(models_to_test)}] {result_name}: "
                                f"{'完了' if success else '失敗'}"
                            )
                            
                            # progress.jsonに記録（ロック付き）
                            self._mark_model_tested(period_key, test_year, result_name, success)
                            
                            if not success:
                                error_action = self.wfv_config['execution'].get('on_test_error', 'skip')
                                if error_action == 'stop':
                                    self.logger.error("テスト実行エラーにより処理を中断します")
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    return False
                        
                        except Exception as e:
                            self.logger.error(f"  {model_name}: 並列実行エラー - {str(e)}")
                            self._mark_model_tested(period_key, test_year, model_name, False)
            else:
                self.logger.info("  テスト対象モデルなし（全てテスト済みまたはモデル未作成）")
        
        self.logger.info("=" * 80)
        self.logger.info("単一期間モード完了")
        self.logger.info("=" * 80)
        
        return True
    
    def run_compare_periods_mode(self, resume: bool = False, dry_run: bool = False) -> bool:
        """
        期間比較モードを実行
        
        Args:
            resume: 前回から再開するか
            dry_run: ドライラン（実行計画のみ表示）
            
        Returns:
            成功フラグ
        """
        settings = self.wfv_config['compare_periods_settings']
        training_periods = settings['training_periods']
        test_years = self.wfv_config['test_years']
        models_setting = settings['models']
        
        # モデルリストを取得
        target_models = self._filter_models(models_setting)
        
        self.logger.info("=" * 80)
        self.logger.info("Walk-Forward Validation - 期間比較モード")
        self.logger.info(f"比較期間: {training_periods}年")
        self.logger.info(f"テスト年: {test_years}")
        self.logger.info(f"対象モデル数: {len(target_models)}")
        self.logger.info("=" * 80)
        
        if dry_run:
            self.logger.info("[DRY RUN] 実行計画:")
            for period in training_periods:
                self.logger.info(f"  期間: {period}年")
                for year in test_years:
                    train_start = year - period
                    train_end = year - 1
                    self.logger.info(f"    テスト年 {year}: 学習期間 {train_start}-{train_end}")
                    for model_name in target_models:
                        self.logger.info(f"      - {model_name}")
            return True
        
        # 進捗ファイルの設定
        self.progress_file = self.output_dir / "progress.json"
        
        if resume and self.progress_file.exists():
            self.progress_data = self._load_progress(self.progress_file)
            self.logger.info("前回の進捗を読み込みました")
        else:
            self._initialize_progress('compare_periods', training_periods, test_years, target_models)
        
        # 各期間でループ
        for training_period in training_periods:
            period_key = f"period_{training_period}"
            
            self.logger.info("=" * 80)
            self.logger.info(f"学習期間: {training_period}年")
            self.logger.info("=" * 80)
            
            # 期間ごとのディレクトリ作成
            period_dir = self.output_dir / period_key
            models_dir = period_dir / "models"
            test_results_dir = period_dir / "test_results"
            
            # 各テスト年でループ
            for test_year in test_years:
                train_start = test_year - training_period
                train_end = test_year - 1
                
                self.logger.info("-" * 80)
                self.logger.info(f"テスト年: {test_year} (学習期間: {train_start}-{train_end})")
                self.logger.info("-" * 80)
                
                # 年ごとのディレクトリ作成
                year_models_dir = models_dir / str(test_year)
                year_test_dir = test_results_dir / str(test_year)
                year_models_dir.mkdir(parents=True, exist_ok=True)
                year_test_dir.mkdir(parents=True, exist_ok=True)
                
                # ★ UPSET分類器チェック（全モデルより前に独立して実行）
                # 芝/ダート分離: 2つの穴馬分類器を管理
                upset_filename_turf = f"upset_classifier_turf_{train_start}-{train_end}.sav"
                upset_filename_dirt = f"upset_classifier_dirt_{train_start}-{train_end}.sav"
                upset_model_name_turf = f"upset_classifier_turf_{train_start}-{train_end}"
                upset_model_name_dirt = f"upset_classifier_dirt_{train_start}-{train_end}"
                
                # progress.jsonでスキップチェック（両方作成済みならスキップ）
                turf_created = self._is_model_created(period_key, test_year, upset_model_name_turf)
                dirt_created = self._is_model_created(period_key, test_year, upset_model_name_dirt)
                
                if turf_created and dirt_created:
                    self.logger.info(f"[UPSET] 穴馬分類器（芝/ダート）: スキップ（作成済み）")
                else:
                    # UPSET分類器作成（最初のモデル名を使用）
                    first_model_name = target_models[0] if target_models else "default"
                    upset_success = self._create_upset_classifier(
                        first_model_name, train_start, train_end, year_models_dir
                    )
                    
                    upset_path_turf = year_models_dir / upset_filename_turf
                    upset_path_dirt = year_models_dir / upset_filename_dirt
                    
                    if upset_success:
                        # 芝モデルの記録
                        if upset_path_turf.exists():
                            self._mark_model_created(period_key, test_year, upset_model_name_turf, str(upset_path_turf), True)
                            self.logger.info(f"[UPSET] 穴馬分類器（芝）作成完了: {upset_filename_turf}")
                        
                        # ダートモデルの記録
                        if upset_path_dirt.exists():
                            self._mark_model_created(period_key, test_year, upset_model_name_dirt, str(upset_path_dirt), True)
                            self.logger.info(f"[UPSET] 穴馬分類器（ダート）作成完了: {upset_filename_dirt}")
                        
                        # Universal Rankerも記録
                        universal_filename = self._get_model_filename("all_tracks_all_surfaces_all_ages", train_start, train_end)
                        universal_path = year_models_dir / universal_filename
                        if universal_path.exists():
                            universal_model_name = f"universal_ranker_{train_start}-{train_end}"
                            self._mark_model_created(period_key, test_year, universal_model_name, str(universal_path), True)
                            self.logger.info(f"[UPSET] Universal Ranker記録完了: {universal_filename}")
                    else:
                        self.logger.warning(f"[UPSET] 穴馬分類器作成失敗")
                
                # モデル作成フェーズ (Phase 2: 並列実行)
                self.logger.info(f"[モデル作成フェーズ] {len(target_models)}モデル (並列実行)")
                
                # 未作成モデルをフィルタリング
                models_to_create = []
                for model_name in target_models:
                    if self._is_model_created(period_key, test_year, model_name):
                        self.logger.info(f"  {model_name}: スキップ（作成済み）")
                    else:
                        model_config = self._get_model_config(model_name)
                        if model_config:
                            models_to_create.append((model_name, model_config))
                        else:
                            self._mark_model_created(period_key, test_year, model_name, "", False)
                
                if models_to_create:
                    # ProcessPoolExecutorで並列実行
                    max_workers = min(4, multiprocessing.cpu_count())
                    self.logger.info(f"  並列実行ワーカー数: {max_workers}")
                    
                    # 引数リストを作成
                    worker_args = [
                        (name, config, train_start, train_end, str(year_models_dir))
                        for name, config in models_to_create
                    ]
                    
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # 全タスクを投入
                        future_to_model = {
                            executor.submit(self._create_model_worker, args): args[0]
                            for args in worker_args
                        }
                        
                        # 完了したタスクから結果を取得
                        completed_count = 0
                        for future in as_completed(future_to_model):
                            model_name = future_to_model[future]
                            try:
                                result_name, success, model_path = future.result()
                                completed_count += 1
                                
                                self.logger.info(
                                    f"  [{completed_count}/{len(models_to_create)}] {result_name}: "
                                    f"{'完了' if success else '失敗'}"
                                )
                                
                                # progress.jsonに記録（ロック付き）
                                self._mark_model_created(period_key, test_year, result_name, model_path or "", success)
                                
                                if not success:
                                    error_action = self.wfv_config['execution'].get('on_model_creation_error', 'skip')
                                    if error_action == 'stop':
                                        self.logger.error("モデル作成エラーにより処理を中断します")
                                        executor.shutdown(wait=False, cancel_futures=True)
                                        return False
                            
                            except Exception as e:
                                self.logger.error(f"  {model_name}: 並列実行エラー - {str(e)}")
                                self._mark_model_created(period_key, test_year, model_name, "", False)
                else:
                    self.logger.info("  作成対象モデルなし（全て作成済み）")
                
                # テスト実行フェーズ (並列実行)
                self.logger.info(f"[テスト実行フェーズ] {len(target_models)}モデル (並列実行)")
                
                # 穴馬分類器のパスを取得（芝/ダート分離版を優先）
                train_period = f"{train_start}-{train_end}"
                upset_classifier_path_turf = year_models_dir / f"upset_classifier_turf_{train_period}.sav"
                upset_classifier_path_dirt = year_models_dir / f"upset_classifier_dirt_{train_period}.sav"
                
                if upset_classifier_path_turf.exists() and upset_classifier_path_dirt.exists():
                    # 芝/ダート分離版（パイプ区切りで渡す）
                    upset_classifier_path_str = f"{upset_classifier_path_turf}|{upset_classifier_path_dirt}"
                    self.logger.info(f"[UPSET] 芝/ダート分離モデルを使用: turf={upset_classifier_path_turf.name}, dirt={upset_classifier_path_dirt.name}")
                else:
                    # 従来の統合版にフォールバック
                    upset_classifier_path_legacy = year_models_dir / f"upset_classifier_{train_period}.sav"
                    upset_classifier_path_str = str(upset_classifier_path_legacy) if upset_classifier_path_legacy.exists() else None
                    if upset_classifier_path_str:
                        self.logger.info(f"[UPSET] 統合モデルを使用: {upset_classifier_path_legacy.name}")
                
                # 未テストモデルをフィルタリング
                models_to_test = []
                for model_name in target_models:
                    # スキップ判定
                    if self._is_model_tested(period_key, test_year, model_name):
                        self.logger.info(f"  {model_name}: スキップ（テスト済み）")
                        continue
                    
                    # モデルが作成されているか確認
                    year_str = str(test_year)
                    model_path = None
                    if period_key in self.progress_data.get('progress', {}):
                        if year_str in self.progress_data['progress'][period_key]:
                            if model_name in self.progress_data['progress'][period_key][year_str]:
                                model_info = self.progress_data['progress'][period_key][year_str][model_name]
                                if not model_info.get('model_created', False):
                                    self.logger.warning(f"  {model_name}: スキップ（モデル未作成）")
                                    continue
                                
                                model_path = model_info.get('model_path')
                                if not model_path or not os.path.exists(model_path):
                                    self.logger.warning(f"  {model_name}: スキップ（モデルファイル不明）")
                                    continue
                    
                    model_config = self._get_model_config(model_name)
                    if not model_config:
                        self._mark_model_tested(period_key, test_year, model_name, False)
                        continue
                    
                    models_to_test.append((model_name, model_config, model_path))
                
                if models_to_test:
                    # ProcessPoolExecutorで並列実行
                    max_workers = min(4, multiprocessing.cpu_count())
                    self.logger.info(f"  並列実行ワーカー数: {max_workers}")
                    
                    # 引数リストを作成
                    test_worker_args = [
                        (name, config, path, test_year, str(year_test_dir), upset_classifier_path_str)
                        for name, config, path in models_to_test
                    ]
                    
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # 全タスクを投入
                        future_to_model = {
                            executor.submit(self._test_model_worker, args): args[0]
                            for args in test_worker_args
                        }
                        
                        # 完了したタスクから結果を取得
                        completed_count = 0
                        for future in as_completed(future_to_model):
                            model_name = future_to_model[future]
                            try:
                                result_name, success, result_filename = future.result()
                                completed_count += 1
                                
                                self.logger.info(
                                    f"  [{completed_count}/{len(models_to_test)}] {result_name}: "
                                    f"{'完了' if success else '失敗'}"
                                )
                                
                                # progress.jsonに記録（ロック付き）
                                self._mark_model_tested(period_key, test_year, result_name, success)
                                
                                if not success:
                                    error_action = self.wfv_config['execution'].get('on_test_error', 'skip')
                                    if error_action == 'stop':
                                        self.logger.error("テスト実行エラーにより処理を中断します")
                                        executor.shutdown(wait=False, cancel_futures=True)
                                        return False
                            
                            except Exception as e:
                                self.logger.error(f"  {model_name}: 並列実行エラー - {str(e)}")
                                self._mark_model_tested(period_key, test_year, model_name, False)
                else:
                    self.logger.info("  テスト対象モデルなし（全てテスト済みまたはモデル未作成）")
        
        self.logger.info("=" * 80)
        self.logger.info("期間比較モード完了")
        self.logger.info("=" * 80)
        
        return True
    
    def run(self, resume: bool = False, dry_run: bool = False) -> bool:
        """
        WFVを実行
        
        Args:
            resume: 前回から再開するか
            dry_run: ドライラン（実行計画のみ表示）
            
        Returns:
            成功フラグ
        """
        execution_mode = self.wfv_config['execution_mode']
        
        # 出力ディレクトリ作成
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if execution_mode == 'single_period':
            success = self.run_single_period_mode(resume, dry_run)
            if success and not dry_run:
                # サマリー生成
                self.generate_single_period_summary()
                # 全予測結果の統合
                self.generate_consolidated_predictions('single_period')
            return success
        elif execution_mode == 'compare_periods':
            success = self.run_compare_periods_mode(resume, dry_run)
            if success and not dry_run:
                # サマリー生成
                self.generate_compare_periods_summary()
                # 全予測結果の統合
                self.generate_consolidated_predictions('compare_periods')
            return success
        else:
            self.logger.error(f"不明な実行モード: {execution_mode}")
            return False
    
    def generate_consolidated_predictions(self, mode: str):
        """全予測結果を期間ごとに統合
        
        Args:
            mode: 'single_period' または 'compare_periods'
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("全予測結果の統合中...")
            self.logger.info("=" * 80)
            
            if mode == 'single_period':
                settings = self.wfv_config['single_period_settings']
                training_periods = [settings['training_period']]
            else:
                settings = self.wfv_config['compare_periods_settings']
                training_periods = settings['training_periods']
            
            test_years = self.wfv_config['test_years']
            
            for training_period in training_periods:
                period_key = f"period_{training_period}"
                period_dir = self.output_dir / period_key
                test_results_dir = period_dir / "test_results"
                
                if not test_results_dir.exists():
                    self.logger.warning(f"{period_key}: テスト結果ディレクトリが見つかりません")
                    continue
                
                self.logger.info(f"{period_key}: 予測結果を統合中...")
                
                all_predictions = []
                file_count = 0
                
                # 各年・各モデルの_all.tsvファイルを収集
                for test_year in test_years:
                    year_test_dir = test_results_dir / str(test_year)
                    if not year_test_dir.exists():
                        continue
                    
                    # _all.tsvファイルを探す
                    for tsv_file in year_test_dir.glob("predicted_results_*_all.tsv"):
                        try:
                            # ファイル名からモデル名と学習期間を抽出
                            # 例: predicted_results_tokyo_turf_3ageup_long_2013-2022_test2023_all.tsv
                            filename_parts = tsv_file.stem.split('_')
                            
                            # モデル名を抽出（predicted_results_の後から学習期間の前まで）
                            model_name_parts = []
                            for part in filename_parts[2:]:
                                if '-' in part and part.replace('-', '').isdigit():
                                    # 学習期間（例: 2013-2022）に到達
                                    training_period_str = part
                                    break
                                model_name_parts.append(part)
                            
                            model_name = '_'.join(model_name_parts)
                            
                            # テスト年を抽出
                            for part in filename_parts:
                                if part.startswith('test') and part[4:].isdigit():
                                    test_year_str = part[4:]
                                    break
                            
                            # TSVファイルを読み込み
                            df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
                            
                            all_predictions.append(df)
                            file_count += 1
                            
                        except Exception as e:
                            self.logger.warning(f"ファイル読み込みエラー: {tsv_file.name} - {e}")
                            continue
                
                if all_predictions:
                    # 全データを統合
                    consolidated_df = pd.concat(all_predictions, ignore_index=True)
                    
                    # 保存
                    output_file = period_dir / f"all_predictions_period_{training_period}.tsv"
                    consolidated_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8', float_format='%.8f')
                    
                    self.logger.info(f"{period_key}: 統合完了")
                    self.logger.info(f"  ファイル数: {file_count}")
                    self.logger.info(f"  総レコード数: {len(consolidated_df)}")
                    self.logger.info(f"  保存先: {output_file}")
                    
                    # check_resultsフォルダにもコピー
                    check_results_dir = Path(__file__).parent / "check_results"
                    check_results_dir.mkdir(parents=True, exist_ok=True)
                    check_results_file = check_results_dir / f"predicted_results_all.tsv"
                    consolidated_df.to_csv(check_results_file, sep='\t', index=False, encoding='utf-8', float_format='%.8f')
                    self.logger.info(f"  check_resultsにもコピー: {check_results_file}")
                else:
                    self.logger.warning(f"{period_key}: 統合対象ファイルが見つかりませんでした")
            
            self.logger.info("全予測結果の統合完了")
            
        except Exception as e:
            self.logger.error(f"全予測結果の統合中にエラーが発生しました: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def generate_single_period_summary(self):
        """単一期間モードのサマリーを生成"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("サマリー生成中...")
            self.logger.info("=" * 80)
            
            settings = self.wfv_config['single_period_settings']
            training_period = settings['training_period']
            test_years = self.wfv_config['test_years']
            
            period_key = f"period_{training_period}"
            period_dir = self.output_dir / period_key
            test_results_dir = period_dir / "test_results"
            
            # 各年のテスト結果を収集
            all_results = []
            
            for test_year in test_years:
                year_test_dir = test_results_dir / str(test_year)
                if not year_test_dir.exists():
                    continue
                
                # スキップファイル（分析列含む）を探す
                for tsv_file in year_test_dir.glob("predicted_results_*_skipped.tsv"):
                    self.logger.info(f"結果集計中: {tsv_file.name}")
                    
                    # ファイル名からモデル名と学習期間を抽出
                    # 例: predicted_results_tokyo_turf_3ageup_long_2013-2022_test2023_skipped.tsv
                    filename_parts = tsv_file.stem.replace("predicted_results_", "").replace("_skipped", "").split("_")
                    # 最後が "testYYYY" の形式
                    # その前が学習期間 "YYYY-YYYY"
                    train_period_str = filename_parts[-2]  # "2013-2022"
                    model_name = "_".join(filename_parts[:-2])  # "tokyo_turf_3ageup_long"
                    
                    # TSVファイル読み込み
                    try:
                        df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8-sig')
                        
                        if len(df) == 0:
                            continue
                        
                        # 全レースのDataFrame
                        df_full = df.copy()
                        
                        # 購入推奨馬を抽出
                        if '購入推奨' in df.columns:
                            buy_horses = df[df['購入推奨'] == True].copy()
                        else:
                            # 購入推奨列がない場合はスキップ
                            self.logger.warning(f"購入推奨列なし: {tsv_file.name}")
                            continue
                        
                        if len(buy_horses) == 0:
                            self.logger.warning(f"購入推奨馬0頭: {tsv_file.name}")
                            continue
                        
                        # レース数（全レース）
                        race_count = df_full.groupby(['開催年', '開催日', '競馬場', 'レース番号']).ngroups
                        
                        # 購入推奨馬数
                        buy_count = len(buy_horses)
                        
                        # 馬券種別ごとの集計
                        results = self._calculate_betting_results(buy_horses, df_full)
                        
                        all_results.append({
                            'モデル名': model_name,
                            '学習期間': train_period_str,
                            'テスト年': test_year,
                            'レース数': race_count,
                            '購入推奨馬数': buy_count,
                            '単勝的中数': results['tansho_hit'],
                            '単勝的中率': f"{results['tansho_rate']*100:.1f}%",
                            '単勝回収率': f"{results['tansho_return']*100:.1f}%",
                            '複勝的中数': results['fukusho_hit'],
                            '複勝的中率': f"{results['fukusho_rate']*100:.1f}%",
                            '複勝回収率': f"{results['fukusho_return']*100:.1f}%",
                            '馬連的中数': results.get('umaren_hit', 0),
                            '馬連的中率': f"{results.get('umaren_rate', 0)*100:.1f}%",
                            '馬連回収率': f"{results.get('umaren_return', 0)*100:.1f}%",
                            'ワイド的中数': results.get('wide_hit', 0),
                            'ワイド的中率': f"{results.get('wide_rate', 0)*100:.1f}%",
                            'ワイド回収率': f"{results.get('wide_return', 0)*100:.1f}%",
                            '穴馬候補': results.get('upset_candidates', 0),
                            '穴馬的中': results.get('upset_hits', 0),
                            '穴馬適合率': f"{results.get('upset_precision', 0):.1f}%",
                            '穴馬再現率': f"{results.get('upset_recall', 0):.1f}%",
                            '穴馬ROI': f"{results.get('upset_roi', 0):.1f}%"
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"ファイル読み込みエラー: {tsv_file.name} - {str(e)}")
                        continue
            
            if len(all_results) == 0:
                self.logger.warning("集計可能な結果がありません")
                return
            
            # DataFrameに変換
            summary_df = pd.DataFrame(all_results)
            
            # サマリーファイルに保存
            summary_file = period_dir / f"summary_period_{training_period}.tsv"
            summary_df.to_csv(summary_file, sep='\t', index=False, encoding='utf-8-sig', float_format='%.8f')
            
            self.logger.info(f"サマリー保存完了: {summary_file}")
            self.logger.info(f"集計結果: {len(all_results)}件")
            
            # コンソールにも表示
            self.logger.info("\n" + summary_df.to_string(index=False))
            
        except Exception as e:
            self.logger.error(f"サマリー生成エラー: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    def generate_compare_periods_summary(self):
        """期間比較モードのサマリーを生成"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("期間比較サマリー生成中...")
            self.logger.info("=" * 80)
            
            settings = self.wfv_config['compare_periods_settings']
            training_periods = settings['training_periods']
            
            # 各期間のサマリーを生成
            for period in training_periods:
                self.logger.info(f"\n期間 {period}年 のサマリー生成...")
                # 一時的に設定を変更して単一期間サマリーを生成
                original_mode = self.wfv_config['execution_mode']
                original_settings = self.wfv_config.get('single_period_settings', {})
                
                self.wfv_config['execution_mode'] = 'single_period'
                self.wfv_config['single_period_settings'] = {
                    'training_period': period,
                    'rolling_type': 'fixed',
                    'models': settings['models']
                }
                
                self.generate_single_period_summary()
                
                # 設定を戻す
                self.wfv_config['execution_mode'] = original_mode
                self.wfv_config['single_period_settings'] = original_settings
            
            self.logger.info("\n期間比較モード: 全期間のサマリー生成完了")
            
        except Exception as e:
            self.logger.error(f"期間比較サマリー生成エラー: {str(e)}")
            self.logger.debug(traceback.format_exc())


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Walk-Forward Validation for Horse Racing Prediction Models'
    )
    parser.add_argument(
        '--config',
        default='walk_forward_config.json',
        help='設定ファイルのパス (デフォルト: walk_forward_config.json)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='前回の実行を途中から再開'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実行計画のみ表示（実際には実行しない）'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='前回の実行結果を削除してクリーンスタート'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログを出力（DEBUGレベル）'
    )
    
    args = parser.parse_args()
    
    # Validatorを作成
    try:
        validator = WalkForwardValidator(args.config)
    except Exception as e:
        print(f"エラー: {str(e)}")
        return 1
    
    # verboseモード
    if args.verbose:
        validator.logger.setLevel(logging.DEBUG)
    
    # cleanモード
    if args.clean:
        progress_file = validator.output_dir / "progress.json"
        if progress_file.exists():
            progress_file.unlink()
            validator.logger.info("進捗ファイルを削除しました")
    
    # 実行
    try:
        success = validator.run(resume=args.resume, dry_run=args.dry_run)
        return 0 if success else 1
    except KeyboardInterrupt:
        validator.logger.info("ユーザーによって中断されました")
        return 130
    except Exception as e:
        validator.logger.error(f"予期しないエラー: {str(e)}")
        validator.logger.debug(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
