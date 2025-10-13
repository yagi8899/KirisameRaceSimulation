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


def create_all_models(output_dir='models'):
    """
    標準モデルを一括作成する関数（設定はJSONファイルから読み込み）
    
    Args:
        output_dir (str): モデル保存先ディレクトリ (デフォルト: 'models')
    """
    
    # JSONファイルから標準モデル設定を読み込み
    try:
        model_configs = get_standard_models()
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    print("🚀 複数モデル一括作成を開始します！")
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
        print(f"📁 ファイル名: {config['model_filename']}")
        print(f"🏟️  競馬場: {get_track_name(config['track_code'])}")
        print(f"🌱 路面: {get_surface_name(config['surface_type'])}")
        print(f"🎯 年齢区分: {get_age_type_name(config['kyoso_shubetsu_code'])}")
        distance_desc = f"{config['min_distance']}m以上" if config['max_distance'] == 9999 else f"{config['min_distance']}-{config['max_distance']}m"
        print(f"📏 距離: {distance_desc}")
        
        start_time = time.time()
        
        try:
            create_universal_model(
                track_code=config['track_code'],
                kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
                surface_type=config['surface_type'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                model_filename=config['model_filename'],
                output_dir=output_dir
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ 完了！ (所要時間: {elapsed_time:.1f}秒)")
            successful_models.append(config['model_filename'])
            
        except Exception as e:
            elapsed_time = time.time() - start_time  
            print(f"❌ エラーが発生しました (所要時間: {elapsed_time:.1f}秒)")
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
    print("🎯 モデル作成結果サマリー")
    print("=" * 60)
    print(f"✅ 成功: {len(successful_models)}個")
    print(f"❌ 失敗: {len(failed_models)}個")
    
    if successful_models:
        print("\n📋 作成成功したモデル:")
        for model in successful_models:
            print(f"  - {model}")
    
    if failed_models:
        print("\n⚠️  作成失敗したモデル:")
        for model in failed_models:
            print(f"  - {model['filename']}: {model['error']}")
    
    print("\n🏁 すべての処理が完了しました！")


def create_custom_models(output_dir='models'):
    """
    カスタムモデルを一括作成する関数（設定はJSONファイルから読み込み）
    
    Args:
        output_dir (str): モデル保存先ディレクトリ (デフォルト: 'models')
    """
    
    # JSONファイルからカスタムモデル設定を読み込み
    try:
        custom_configs = get_custom_models()
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みに失敗しました: {e}")
        return
    
    if not custom_configs:
        print("🔧 カスタムモデルの設定が見つかりませんでした。")
        return
    
    print("🔧 カスタムモデル作成を開始します！")
    print(f"作成予定モデル数: {len(custom_configs)}個")
    print("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for i, config in enumerate(custom_configs, 1):
        description = config.get('description', f"カスタムモデル{i}")
        
        print(f"\n【{i}/{len(custom_configs)}】 {description} モデル作成中...")
        print(f"📁 ファイル名: {config['model_filename']}")
        
        start_time = time.time()
        
        try:
            create_universal_model(
                track_code=config['track_code'],
                kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
                surface_type=config['surface_type'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                model_filename=config['model_filename'],
                output_dir=output_dir
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ 完了！ (所要時間: {elapsed_time:.1f}秒)")
            successful_models.append(config['model_filename'])
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ エラーが発生しました (所要時間: {elapsed_time:.1f}秒)")
            print(f"エラー内容: {str(e)}")
            failed_models.append({
                'filename': config['model_filename'],
                'error': str(e)
            })
            traceback.print_exc()
        
        print("-" * 60)
    
    # 結果サマリーを表示
    print("\n" + "=" * 60)
    print("🚀 カスタムモデル作成結果サマリー")
    print("=" * 60)
    print(f"✅ 成功: {len(successful_models)}個")
    print(f"❌ 失敗: {len(failed_models)}個")
    
    if successful_models:
        print("\n📋 作成成功したモデル:")
        for model in successful_models:
            print(f"  - {model}")
    
    if failed_models:
        print("\n⚠️  作成失敗したモデル:")
        for model in failed_models:
            print(f"  - {model['filename']}: {model['error']}")
    
    print("\n🏁 すべての処理が完了しました！")


if __name__ == '__main__':
    # 実行方法を選択できるように
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'custom':
        # python batch_model_creator.py custom
        create_custom_models()
    else:
        # python batch_model_creator.py (デフォルト)
        create_all_models()