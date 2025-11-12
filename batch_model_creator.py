#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数競馬場・条件のモデル一括作成スクリプト

このスクリプトは複数の競馬場・競走条件の予測モデルを一度に作成します。
model_creator.pyの汎用関数を利用して効率的にモデルを生成できます。
"""

from model_creator import create_universal_model
from keiba_constants import get_track_name, get_surface_name, get_age_type_name
from model_config_loader import get_standard_models, get_custom_models
import time
import traceback


def create_all_models(output_dir='models', year_start=2013, year_end=2022):
    """
    標準モデルを一括作成する関数(設定はJSONファイルから読み込み)
    
    Args:
        output_dir (str): モデル保存先ディレクトリ (デフォルト: 'models')
        year_start (int): 学習データ開始年 (デフォルト: 2013)
        year_end (int): 学習データ終了年 (デフォルト: 2022)
    """
    
    # JSONファイルから標準モデル設定を読み込み
    try:
        model_configs = get_standard_models()
    except Exception as e:
        print(f"[ERROR] 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    print("[START] 複数モデル一括作成を開始します！")
    print(f"作成予定モデル数: {len(model_configs)}個")
    print("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for i, config in enumerate(model_configs, 1):
        # 設定に説明があればそれを使用、なければ従来通り生成
        if 'description' in config:
            description = config['description']
        else:
            track_name = get_track_name(config['track_code'])
            surface_jp = get_surface_name(config['surface_type'])
            age_type = get_age_type_name(config['kyoso_shubetsu_code'])
            description = f"{track_name}{surface_jp}{age_type}"
        
        print(f"\n【{i}/{len(model_configs)}】 {description} モデル作成中...")
        print(f"[FILE] ファイル名: {config['model_filename']}")
        print(f"[TRACK]  競馬場: {get_track_name(config['track_code'])}")
        print(f"[TURF] 路面: {get_surface_name(config['surface_type'])}")
        print(f"[AGE] 年齢区分: {get_age_type_name(config['kyoso_shubetsu_code'])}")
        distance_desc = f"{config['min_distance']}m以上" if config['max_distance'] == 9999 else f"{config['min_distance']}-{config['max_distance']}m"
        print(f"[DIST] 距離: {distance_desc}")
        print(f"[DATE] 学習期間: {year_start}年~{year_end}年")
        
        start_time = time.time()
        
        try:
            create_universal_model(
                track_code=config['track_code'],
                kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
                surface_type=config['surface_type'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                model_filename=config['model_filename'],
                output_dir=output_dir,
                year_start=year_start,
                year_end=year_end
            )
            
            elapsed_time = time.time() - start_time
            print(f"[OK] 完了！ (所要時間: {elapsed_time:.1f}秒)")
            successful_models.append(config['model_filename'])
            
        except Exception as e:
            elapsed_time = time.time() - start_time  
            print(f"[ERROR] エラーが発生しました (所要時間: {elapsed_time:.1f}秒)")
            print(f"エラー内容: {str(e)}")
            failed_models.append({
                'filename': config['model_filename'],
                'error': str(e)
            })
            
            # デバッグ用：詳細なエラー情報を表示
            print("詳細なエラー情報:")
            traceback.print_exc()
        
        print("-" * 60)
    
    # 結果サマリーを表示
    print("\n" + "=" * 60)
    print("[TARGET] モデル作成結果サマリー")
    print("=" * 60)
    print(f"[OK] 成功: {len(successful_models)}個")
    print(f"[ERROR] 失敗: {len(failed_models)}個")
    
    if successful_models:
        print("\n[LIST] 作成成功したモデル:")
        for model in successful_models:
            print(f"  - {model}")
    
    if failed_models:
        print("\n[!]  作成失敗したモデル:")
        for model in failed_models:
            print(f"  - {model['filename']}: {model['error']}")
    
    print("\n[DONE] すべての処理が完了しました！")


def create_custom_models(output_dir='models', year_start=2013, year_end=2022):
    """
    カスタムモデルを一括作成する関数(設定はJSONファイルから読み込み)
    
    Args:
        output_dir (str): モデル保存先ディレクトリ (デフォルト: 'models')
        year_start (int): ベース学習データ開始年 (デフォルト: 2013)
        year_end (int): ベース学習データ終了年 (デフォルト: 2022)
        latest_year (int, optional): 最新年の上限。アンサンブルのrecent/latestモデルで使用
        enable_ensemble (bool): アンサンブルモデル作成を有効化 (デフォルト: True)
    """
    
    # JSONファイルからカスタムモデル設定を読み込み
    try:
        custom_configs = get_custom_models()
    except Exception as e:
        print(f"[ERROR] 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    if not custom_configs:
        print("[TOOL] カスタムモデルの設定が見つかりませんでした。")
        return
    
    print("[START] カスタムモデル作成を開始します！")
    print(f"[DATE] 学習期間: {year_start}年~{year_end}年")
    print(f"作成予定モデル数: {len(custom_configs)}個")
    print("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for i, config in enumerate(custom_configs, 1):
        description = config.get('description', f"カスタムモデル{i}")
        base_filename = config['model_filename'].replace('.sav', '')
        # ファイル名に学習期間を追加
        model_filename = f"{base_filename}_{year_start}-{year_end}.sav"
        
        print(f"\n【{i}/{len(custom_configs)}】 {description} モデル作成中...")
        print(f"[FILE] ファイル名: {model_filename}")
        print(f"[DATE] 学習期間: {year_start}年~{year_end}年")
        
        start_time = time.time()
        
        try:
            create_universal_model(
                track_code=config['track_code'],
                kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
                surface_type=config['surface_type'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                model_filename=model_filename,
                output_dir=output_dir,
                year_start=year_start,
                year_end=year_end
            )
            
            elapsed_time = time.time() - start_time
            print(f"[OK] 完了！ (所要時間: {elapsed_time:.1f}秒)")
            successful_models.append(model_filename)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[ERROR] エラーが発生しました (所要時間: {elapsed_time:.1f}秒)")
            print(f"エラー内容: {str(e)}")
            failed_models.append({
                'filename': model_filename,
                'error': str(e)
            })
            traceback.print_exc()
        
        print("-" * 60)
    
    # 結果サマリーを表示
    print("\n" + "=" * 60)
    print("[START] カスタムモデル作成結果サマリー")
    print("=" * 60)
    print(f"[OK] 成功: {len(successful_models)}個")
    print(f"[ERROR] 失敗: {len(failed_models)}個")
    
    if successful_models:
        print("\n[LIST] 作成成功したモデル:")
        for model in successful_models:
            print(f"  - {model}")
    
    if failed_models:
        print("\n[!]  作成失敗したモデル:")
        for model in failed_models:
            print(f"  - {model['filename']}: {model['error']}")
    
    print("\n[DONE] すべての処理が完了しました！")


if __name__ == '__main__':
    # 実行方法を選択できるように
    import sys
    
    # デフォルトの年範囲
    year_start = 2013
    year_end = 2022
    
    # コマンドライン引数から年範囲を解析
    mode = 'standard'  # デフォルトは標準モデル
    
    for arg in sys.argv[1:]:
        if arg == 'custom':
            mode = 'custom'
        elif '-' in arg:
            # "2020-2023" 形式の年範囲指定
            try:
                years = arg.split('-')
                if len(years) == 2:
                    year_start = int(years[0])
                    year_end = int(years[1])
                    print(f"[DATE] 年範囲指定: {year_start}年~{year_end}年")
            except ValueError:
                print(f"[!] 無効な年範囲フォーマット: {arg} (例: 2020-2023)")
        elif arg.isdigit() and len(arg) == 4:
            # "2023" 形式の単一年指定
            year_start = year_end = int(arg)
            print(f"[DATE] 単一年指定: {year_start}年")
    
    if mode == 'custom':
        # python batch_model_creator.py custom [年範囲]
        create_custom_models(year_start=year_start, year_end=year_end)
    else:
        # python batch_model_creator.py [年範囲] (デフォルト)
        create_all_models(year_start=year_start, year_end=year_end)