#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バッチSHAP分析ツール

複数のモデルに対して一括でSHAP値分析を実行します。

使用例:
    # 標準モデルすべてに対して2023年のSHAP分析を実行
    python batch_shap_analyzer.py --models standard --year 2023
    
    # カスタムモデルすべてに対して2024年のSHAP分析を実行
    python batch_shap_analyzer.py --models custom --year 2024
    
    # すべてのモデルに対して実行
    python batch_shap_analyzer.py --models all --year 2023
    
    # 特定のモデルのみ実行（モデルファイル名で指定）
    python batch_shap_analyzer.py --model-names tokyo_turf_3ageup_long,hanshin_turf_3ageup_short --year 2023
"""

import argparse
import subprocess
import sys
from pathlib import Path
from model_config_loader import get_standard_models, get_custom_models, get_all_models


def extract_model_name(model_filename):
    """
    モデルファイル名からモデル名を抽出
    例: tokyo_turf_3ageup_long.sav -> tokyo_turf_3ageup_long
    
    Args:
        model_filename (str): モデルファイル名
        
    Returns:
        str: モデル名（拡張子除去済み）
    """
    return Path(model_filename).stem


def run_shap_analysis(model_config, year, verbose=True):
    """
    1つのモデルに対してSHAP分析を実行
    
    Args:
        model_config (dict): モデル設定辞書
        year (int): 分析対象年
        verbose (bool): 詳細ログ表示フラグ
        
    Returns:
        tuple: (success: bool, model_name: str, error_message: str)
    """
    model_name = extract_model_name(model_config['model_filename'])
    model_path = Path('models') / model_config['model_filename']
    
    if not model_path.exists():
        error_msg = f"モデルファイルが見つかりません: {model_path}"
        print(f"❌ [{model_name}] {error_msg}")
        return False, model_name, error_msg
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 SHAP分析開始: {model_name} ({year}年)")
        print(f"{'='*60}")
    
    try:
        # model_explainer.py を実行
        cmd = [
            sys.executable,
            'model_explainer.py',
            '--model', str(model_path),
            '--test-year', str(year),
            '--track-code', model_config['track_code'],
            '--surface-type', model_config['surface_type'],
            '--min-distance', str(model_config['min_distance']),
            '--max-distance', str(model_config['max_distance']),
            '--kyoso-shubetsu-code', model_config['kyoso_shubetsu_code']
        ]
        
        if verbose:
            print(f"実行コマンド: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True
        )
        
        # SHAP CSV出力パスを推測
        shap_csv = Path('shap_analysis') / f"{model_name}_importance.csv"
        
        if not shap_csv.exists():
            error_msg = f"SHAP CSVファイルが生成されませんでした: {shap_csv}"
            print(f"⚠️ [{model_name}] {error_msg}")
            return False, model_name, error_msg
        
        # analyze_shap_results.py を実行
        output_dir = Path('shap_analysis') / model_name / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_analyze = [
            sys.executable,
            'analyze_shap_results.py',
            '--input', str(shap_csv),
            '--model-name', model_name,
            '--output-dir', str(output_dir)
        ]
        
        if verbose:
            print(f"\n詳細分析コマンド: {' '.join(cmd_analyze)}")
        
        result_analyze = subprocess.run(
            cmd_analyze,
            check=True,
            capture_output=not verbose,
            text=True
        )
        
        if verbose:
            print(f"\n✅ [{model_name}] SHAP分析完了!")
            print(f"   出力先: {output_dir}")
        
        return True, model_name, ""
        
    except subprocess.CalledProcessError as e:
        error_msg = f"実行エラー: {str(e)}"
        print(f"❌ [{model_name}] {error_msg}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"   エラー詳細: {e.stderr}")
        return False, model_name, error_msg
    
    except Exception as e:
        error_msg = f"予期しないエラー: {str(e)}"
        print(f"❌ [{model_name}] {error_msg}")
        return False, model_name, error_msg


def main():
    parser = argparse.ArgumentParser(
        description='複数モデルに対してバッチSHAP分析を実行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--models',
        type=str,
        choices=['standard', 'custom', 'all'],
        help='分析対象モデルグループ (standard/custom/all)'
    )
    
    parser.add_argument(
        '--model-names',
        type=str,
        help='分析対象モデル名（カンマ区切り、例: tokyo_turf_3ageup_long,hanshin_turf_3ageup_short）'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='分析対象年（例: 2023）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='詳細ログを非表示にする'
    )
    
    args = parser.parse_args()
    
    # モデルリスト取得
    if args.model_names:
        # 特定のモデル名が指定された場合
        specified_names = [name.strip() for name in args.model_names.split(',')]
        all_models = get_all_models()
        target_models = [
            m for m in all_models 
            if extract_model_name(m['model_filename']) in specified_names
        ]
        
        if not target_models:
            print(f"❌ エラー: 指定されたモデルが見つかりません: {args.model_names}")
            sys.exit(1)
            
    elif args.models == 'standard':
        target_models = get_standard_models()
    elif args.models == 'custom':
        target_models = get_custom_models()
    elif args.models == 'all':
        target_models = get_all_models()
    else:
        print("❌ エラー: --models または --model-names のいずれかを指定してください")
        parser.print_help()
        sys.exit(1)
    
    if not target_models:
        print(f"❌ エラー: 対象モデルが見つかりません")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # バッチ処理開始
    print(f"\n{'='*70}")
    print(f"🚀 バッチSHAP分析開始")
    print(f"{'='*70}")
    print(f"対象モデル数: {len(target_models)}")
    print(f"分析対象年: {args.year}")
    print(f"{'='*70}\n")
    
    results = []
    success_count = 0
    
    for i, model_config in enumerate(target_models, 1):
        model_name = extract_model_name(model_config['model_filename'])
        print(f"\n[{i}/{len(target_models)}] {model_name}")
        
        success, name, error_msg = run_shap_analysis(model_config, args.year, verbose)
        results.append({
            'model_name': name,
            'success': success,
            'error': error_msg
        })
        
        if success:
            success_count += 1
    
    # サマリー表示
    print(f"\n\n{'='*70}")
    print(f"📊 バッチSHAP分析完了")
    print(f"{'='*70}")
    print(f"成功: {success_count}/{len(target_models)}")
    print(f"失敗: {len(target_models) - success_count}/{len(target_models)}")
    print(f"{'='*70}\n")
    
    # 失敗したモデルがあれば詳細表示
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("⚠️  失敗したモデル:")
        for r in failed_results:
            print(f"   - {r['model_name']}: {r['error']}")
        print()
    
    # 成功したモデル一覧
    if success_count > 0:
        print("✅ 成功したモデル:")
        successful_models = [r['model_name'] for r in results if r['success']]
        for name in successful_models:
            print(f"   - {name}")
        print()
    
    # 終了コード
    sys.exit(0 if success_count == len(target_models) else 1)


if __name__ == '__main__':
    main()
