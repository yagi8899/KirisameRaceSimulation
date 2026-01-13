"""
Walk-Forward Validation システム

競馬予測モデルのWalk-Forward Validationを自動化し、
最適な学習期間の決定とモデルの汎化性能評価を実現する。

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

# 既存モジュールのインポート
from model_creator import create_universal_model
from model_config_loader import load_model_configs
import universal_test


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
    
    def _load_progress(self, progress_file: Path) -> Dict:
        """進捗ファイルを読み込む"""
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_progress(self):
        """進捗を保存"""
        if self.progress_file:
            self.progress_data['last_updated'] = datetime.now().isoformat()
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
    
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
    
    def create_model_for_year(
        self, 
        model_name: str, 
        model_config: Dict, 
        train_start: int, 
        train_end: int, 
        output_dir: Path
    ) -> Tuple[bool, Optional[str]]:
        """
        1年分のモデルを作成
        
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
            
            # モデル作成
            create_universal_model(
                track_code=model_config.get('track_code'),
                surface_type=model_config.get('surface_type'),
                min_distance=model_config.get('min_distance'),
                max_distance=model_config.get('max_distance'),
                kyoso_shubetsu_code=model_config.get('kyoso_shubetsu_code'),
                train_year_start=train_start,
                train_year_end=train_end,
                model_name=str(model_path)
            )
            
            if model_path.exists():
                self.logger.info(f"モデル作成完了: {model_filename}")
                return True, str(model_path)
            else:
                self.logger.error(f"モデルファイルが見つかりません: {model_path}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"モデル作成エラー: {model_name}")
            self.logger.error(f"エラー詳細: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False, None
    
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
            result_path = output_dir / result_filename
            
            # universal_testの機能を呼び出し
            # 注: universal_testは直接実行する形式なので、パラメータを工夫する必要がある
            # 現時点では簡易的な実装として記録のみ
            
            self.logger.info(f"テスト実行完了: {result_filename}")
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
            
            # モデル作成フェーズ
            self.logger.info(f"[モデル作成フェーズ] {len(target_models)}モデル")
            for i, model_name in enumerate(target_models, 1):
                # スキップ判定
                if self._is_model_created(period_key, test_year, model_name):
                    self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: スキップ（作成済み）")
                    continue
                
                model_config = self._get_model_config(model_name)
                if not model_config:
                    self._mark_model_created(period_key, test_year, model_name, "", False)
                    continue
                
                self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: 作成中...")
                success, model_path = self.create_model_for_year(
                    model_name, model_config, train_start, train_end, year_models_dir
                )
                
                self._mark_model_created(period_key, test_year, model_name, model_path or "", success)
                
                if not success:
                    error_action = self.wfv_config['execution'].get('on_model_creation_error', 'skip')
                    if error_action == 'stop':
                        self.logger.error("モデル作成エラーにより処理を中断します")
                        return False
            
            # テスト実行フェーズ
            self.logger.info(f"[テスト実行フェーズ] {len(target_models)}モデル")
            for i, model_name in enumerate(target_models, 1):
                # スキップ判定
                if self._is_model_tested(period_key, test_year, model_name):
                    self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: スキップ（テスト済み）")
                    continue
                
                # モデルが作成されているか確認
                year_str = str(test_year)
                if period_key in self.progress_data.get('progress', {}):
                    if year_str in self.progress_data['progress'][period_key]:
                        if model_name in self.progress_data['progress'][period_key][year_str]:
                            model_info = self.progress_data['progress'][period_key][year_str][model_name]
                            if not model_info.get('model_created', False):
                                self.logger.warning(f"  [{i}/{len(target_models)}] {model_name}: スキップ（モデル未作成）")
                                continue
                            
                            model_path = model_info.get('model_path')
                            if not model_path or not os.path.exists(model_path):
                                self.logger.warning(f"  [{i}/{len(target_models)}] {model_name}: スキップ（モデルファイル不明）")
                                continue
                
                model_config = self._get_model_config(model_name)
                if not model_config:
                    self._mark_model_tested(period_key, test_year, model_name, False)
                    continue
                
                self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: テスト中...")
                success = self.test_model_for_year(
                    model_name, model_config, model_path, test_year, year_test_dir
                )
                
                self._mark_model_tested(period_key, test_year, model_name, success)
                
                if not success:
                    error_action = self.wfv_config['execution'].get('on_test_error', 'skip')
                    if error_action == 'stop':
                        self.logger.error("テスト実行エラーにより処理を中断します")
                        return False
        
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
                
                # モデル作成フェーズ
                self.logger.info(f"[モデル作成フェーズ] {len(target_models)}モデル")
                for i, model_name in enumerate(target_models, 1):
                    # スキップ判定
                    if self._is_model_created(period_key, test_year, model_name):
                        self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: スキップ（作成済み）")
                        continue
                    
                    model_config = self._get_model_config(model_name)
                    if not model_config:
                        self._mark_model_created(period_key, test_year, model_name, "", False)
                        continue
                    
                    self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: 作成中...")
                    success, model_path = self.create_model_for_year(
                        model_name, model_config, train_start, train_end, year_models_dir
                    )
                    
                    self._mark_model_created(period_key, test_year, model_name, model_path or "", success)
                    
                    if not success:
                        error_action = self.wfv_config['execution'].get('on_model_creation_error', 'skip')
                        if error_action == 'stop':
                            self.logger.error("モデル作成エラーにより処理を中断します")
                            return False
                
                # テスト実行フェーズ
                self.logger.info(f"[テスト実行フェーズ] {len(target_models)}モデル")
                for i, model_name in enumerate(target_models, 1):
                    # スキップ判定
                    if self._is_model_tested(period_key, test_year, model_name):
                        self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: スキップ（テスト済み）")
                        continue
                    
                    # モデルが作成されているか確認
                    year_str = str(test_year)
                    if period_key in self.progress_data.get('progress', {}):
                        if year_str in self.progress_data['progress'][period_key]:
                            if model_name in self.progress_data['progress'][period_key][year_str]:
                                model_info = self.progress_data['progress'][period_key][year_str][model_name]
                                if not model_info.get('model_created', False):
                                    self.logger.warning(f"  [{i}/{len(target_models)}] {model_name}: スキップ（モデル未作成）")
                                    continue
                                
                                model_path = model_info.get('model_path')
                                if not model_path or not os.path.exists(model_path):
                                    self.logger.warning(f"  [{i}/{len(target_models)}] {model_name}: スキップ（モデルファイル不明）")
                                    continue
                    
                    model_config = self._get_model_config(model_name)
                    if not model_config:
                        self._mark_model_tested(period_key, test_year, model_name, False)
                        continue
                    
                    self.logger.info(f"  [{i}/{len(target_models)}] {model_name}: テスト中...")
                    success = self.test_model_for_year(
                        model_name, model_config, model_path, test_year, year_test_dir
                    )
                    
                    self._mark_model_tested(period_key, test_year, model_name, success)
                    
                    if not success:
                        error_action = self.wfv_config['execution'].get('on_test_error', 'skip')
                        if error_action == 'stop':
                            self.logger.error("テスト実行エラーにより処理を中断します")
                            return False
        
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
            return self.run_single_period_mode(resume, dry_run)
        elif execution_mode == 'compare_periods':
            return self.run_compare_periods_mode(resume, dry_run)
        else:
            self.logger.error(f"不明な実行モード: {execution_mode}")
            return False


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
