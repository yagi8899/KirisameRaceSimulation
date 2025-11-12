#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル設定の読み込みユーティリティ

JSONファイルからモデル設定を読み込んで、各スクリプトで使用できるようにします。
"""

import json
import os
from pathlib import Path

def load_model_configs(config_file='model_configs.json'):
    """
    JSONファイルからモデル設定を読み込む
    
    Args:
        config_file (str): 設定ファイル名（デフォルト: 'model_configs.json'）
        
    Returns:
        dict: モデル設定辞書
        
    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        json.JSONDecodeError: JSONの形式が正しくない場合
    """
    
    # スクリプトと同じディレクトリで設定ファイルを探す
    script_dir = Path(__file__).parent
    config_path = script_dir / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイル {config_path} が見つかりません。")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        print(f"[LOAD] 設定ファイル {config_file} を読み込みました")
        print(f"  - 標準モデル: {len(configs.get('standard_models', []))}個")
        print(f"  - カスタムモデル: {len(configs.get('custom_models', []))}個")
        
        return configs
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"設定ファイル {config_file} の形式が正しくありません: {e}")

def get_standard_models():
    """
    標準モデル設定を取得
    
    Returns:
        list: 標準モデル設定のリスト
    """
    configs = load_model_configs()
    return configs.get('standard_models', [])

def get_custom_models():
    """
    カスタムモデル設定を取得
    
    Returns:
        list: カスタムモデル設定のリスト
    """
    configs = load_model_configs()
    return configs.get('custom_models', [])

def get_legacy_model():
    """
    旧バージョン互換用モデル設定を取得
    
    Returns:
        dict: 旧バージョンモデル設定
    """
    configs = load_model_configs()
    return configs.get('legacy_model', {})

def get_all_models():
    """
    全てのモデル設定を取得（標準 + カスタム + 旧バージョン）
    
    Returns:
        list: 全モデル設定のリスト
    """
    configs = load_model_configs()
    all_models = []
    
    # 標準モデルを追加
    all_models.extend(configs.get('standard_models', []))
    
    # # カスタムモデルを追加
    # all_models.extend(configs.get('custom_models', []))
    
    # # 旧バージョンモデルを追加
    # legacy = configs.get('legacy_model')
    # if legacy:
    #     all_models.append(legacy)
    
    return all_models

def save_model_configs(configs, config_file='model_configs.json'):
    """
    モデル設定をJSONファイルに保存
    
    Args:
        configs (dict): 保存するモデル設定
        config_file (str): 設定ファイル名
    """
    script_dir = Path(__file__).parent
    config_path = script_dir / config_file
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 設定ファイル {config_file} に保存しました")

def add_custom_model(track_code, kyoso_shubetsu_code, surface_type, 
                    min_distance, max_distance, model_filename, description):
    """
    新しいカスタムモデル設定を追加
    
    Args:
        track_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競争種別コード
        surface_type (str): 路面種別
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        model_filename (str): モデルファイル名
        description (str): モデルの説明
    """
    configs = load_model_configs()
    
    new_model = {
        'track_code': track_code,
        'kyoso_shubetsu_code': kyoso_shubetsu_code,
        'surface_type': surface_type,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'model_filename': model_filename,
        'description': description
    }
    
    if 'custom_models' not in configs:
        configs['custom_models'] = []
    
    configs['custom_models'].append(new_model)
    save_model_configs(configs)
    
    print(f"[NOTE] 新しいカスタムモデルを追加しました: {description}")

def validate_model_config(config):
    """
    モデル設定の妥当性をチェック
    
    Args:
        config (dict): チェックするモデル設定
        
    Returns:
        bool: 妥当な場合True
        
    Raises:
        ValueError: 設定に問題がある場合
    """
    required_keys = [
        'track_code', 'kyoso_shubetsu_code', 'surface_type', 
        'min_distance', 'max_distance', 'model_filename'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"必須項目 '{key}' が設定されていません")
    
    # 値の妥当性チェック
    if config['surface_type'] not in ['turf', 'dirt']:
        raise ValueError(f"surface_type は 'turf' または 'dirt' である必要があります: {config['surface_type']}")
    
    if config['min_distance'] < 0 or config['max_distance'] < 0:
        raise ValueError("距離は0以上である必要があります")
    
    if config['min_distance'] > config['max_distance'] and config['max_distance'] != 9999:
        raise ValueError("min_distance は max_distance 以下である必要があります")
    
    return True

# テスト用の関数
if __name__ == '__main__':
    try:
        # 設定読み込みのテスト
        standard = get_standard_models()
        custom = get_custom_models()
        legacy = get_legacy_model()
        
        print("[RUN] 設定読み込みテスト結果:")
        print(f"標準モデル数: {len(standard)}")
        print(f"カスタムモデル数: {len(custom)}")
        print(f"旧バージョンモデル: {'あり' if legacy else 'なし'}")
        
        # 妥当性チェックのテスト
        for i, config in enumerate(standard[:3]):  # 最初の3つだけテスト
            try:
                validate_model_config(config)
                print(f"[OK] 標準モデル{i+1}: 設定OK")
            except ValueError as e:
                print(f"[ERROR] 標準モデル{i+1}: {e}")
        
    except Exception as e:
        print(f"[ERROR] テスト中にエラーが発生しました: {e}")