#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Validation自動実行スクリプト

このスクリプトは複数の学習期間で順次モデルを作成し、
各年の予測性能を評価することで、モデルの安定性を検証します。
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import re


def run_model_test(model_file, test_year):
    """
    単一モデルのテスト実行
    
    Args:
        model_file (Path): モデルファイルのパス
        test_year (int): テスト年
    
    Returns:
        bool: 成功時True
    """
    import json
    
    # モデルファイル名から設定を推測
    model_name = model_file.stem
    
    # model_configs.jsonから該当モデルの設定を取得
    try:
        with open('model_configs.json', 'r', encoding='utf-8') as f:
            configs_data = json.load(f)
        
        # standard_modelsとcustom_modelsを統合
        configs = configs_data.get('standard_models', []) + configs_data.get('custom_models', [])
        
        # モデル名のベース部分を取得（年号を除く）
        base_name = re.sub(r'_\d{4}-\d{4}$', '', model_name)
        
        # 該当する設定を探す
        config = None
        for c in configs:
            config_base = c['model_filename'].replace('.sav', '')
            if config_base == base_name:
                config = c
                break
        
        if not config:
            print(f"[ERROR] {base_name} の設定が model_configs.json に見つかりません")
            return False
        
        # universal_test.pyを直接呼び出す代わりに、
        # predict_with_model を直接インポートして実行
        from universal_test import predict_with_model
        
        output_df, summary_df, race_count = predict_with_model(
            model_filename=str(model_file),
            track_code=config['track_code'],
            kyoso_shubetsu_code=config['kyoso_shubetsu_code'],
            surface_type=config['surface_type'],
            min_distance=config['min_distance'],
            max_distance=config['max_distance'],
            test_year_start=test_year,
            test_year_end=test_year
        )
        
        if output_df is None:
            print(f"[ERROR] テストデータが見つかりませんでした")
            return False
        
        # 結果をファイルに保存
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # ファイル名を生成
        match = re.search(r'_(\d{4})-(\d{4})$', model_name)
        if match:
            train_range = f"{match.group(1)}-{match.group(2)}"
        else:
            train_range = "unknown"
        
        summary_file = f"betting_summary_{base_name}_train{train_range}_test{test_year}.tsv"
        summary_path = results_dir / summary_file
        
        summary_df.to_csv(summary_path, index=True, sep='\t', encoding='utf-8-sig')
        
        print(f"[OK] 完了 - レース数: {race_count}")
        print(f"   -> {summary_file}")
        print(f"   単勝: 的中率{summary_df.loc['単勝', '的中率(%)']:.1f}%, 回収率{summary_df.loc['単勝', '回収率(%)']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_walk_forward(train_window=3, test_years=[2023, 2024, 2025]):
    """
    Walk-Forward Validationを実行
    
    Args:
        train_window (int): 学習期間の年数 (デフォルト: 3年)
        test_years (list): テスト対象年のリスト
    
    Returns:
        list: 各年のテスト結果
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"[START] Walk-Forward Validation開始")
    print(f"   学習期間: {train_window}年")
    print(f"   テスト年: {test_years}")
    print(f"{'='*60}\n")
    
    for test_year in test_years:
        train_start = test_year - train_window
        train_end = test_year - 1
        
        print(f"\n{'='*60}")
        print(f"[TEST] {test_year}年本番想定のテスト")
        print(f"   学習期間: {train_start}-{train_end}")
        print(f"   テスト年: {test_year}")
        print(f"{'='*60}\n")
        
        # Step 1: モデル作成
        cmd_train = f"python batch_model_creator.py custom {train_start}-{train_end}"
        print(f"[RUN] 実行: {cmd_train}")
        result_train = subprocess.run(cmd_train, shell=True, capture_output=True, text=True)
        
        # 標準出力を表示（デバッグ用）
        if result_train.stdout:
            print(result_train.stdout)
        
        if result_train.returncode != 0:
            print(f"[ERROR] モデル作成失敗: {result_train.stderr}")
            if result_train.stderr:
                print(result_train.stderr)
            continue
        
        print(f"[OK] モデル作成完了")
        
        # Step 2: 作成されたモデルファイルを探す
        models_dir = Path('models')
        model_pattern = f"*_{train_start}-{train_end}.sav"
        created_models = list(models_dir.glob(model_pattern))
        
        if not created_models:
            print(f"[ERROR] モデルファイルが見つかりません: {model_pattern}")
            continue
        
        print(f"[FILE] 見つかったモデル: {len(created_models)}個")
        for model_file in created_models:
            print(f"   - {model_file.name}")
        
        # Step 3: 各モデルをテスト実行（model_configs.jsonを使わず直接実行）
        test_success = True
        for model_file in created_models:
            model_name = model_file.stem  # ファイル名から拡張子を除く
            
            print(f"\n[TEST] テスト実行: {model_file.name}")
            
            # universal_test.pyを直接呼び出さず、ここで直接テスト実行
            # （model_configs.jsonとの不整合を避けるため）
            test_result = run_model_test(model_file, test_year)
            
            if not test_result:
                print(f"[ERROR] {model_file.name} のテスト失敗")
                test_success = False
        
        if not test_success:
            print(f"[WARN] 一部のモデルでテストが失敗しましたが、続行します")
        else:
            print(f"\n[OK] 全モデルのテスト完了")
        
        # Step 3: 結果を読み込み
        result_data = load_test_results(train_start, train_end, test_year)
        if result_data:
            results.append(result_data)
            print_result_summary(result_data)
    
    # 統計分析
    if results:
        analyze_results(results, train_window)
    else:
        print("\n[!] 有効な結果が得られませんでした")
    
    return results


def load_test_results(train_start, train_end, test_year):
    """
    テスト結果ファイルを読み込む
    
    Args:
        train_start (int): 学習開始年
        train_end (int): 学習終了年
        test_year (int): テスト年
    
    Returns:
        dict: テスト結果データ
    """
    results_dir = Path('results')
    
    # betting_summary_*_train{train_start}-{train_end}_test{test_year}.tsvを探す
    pattern = f"betting_summary_*_train{train_start}-{train_end}_test{test_year}.tsv"
    matching_files = list(results_dir.glob(pattern))
    
    if not matching_files:
        print(f"[!] 結果ファイルが見つかりません: {pattern}")
        return None
    
    # 複数ファイルがあれば全部統合
    all_data = []
    for file in matching_files:
        try:
            df = pd.read_csv(file, sep='\t', index_col=0)
            all_data.append(df)
        except Exception as e:
            print(f"[!] {file} の読み込みエラー: {e}")
    
    if not all_data:
        return None
    
    # 統合（平均を取る）
    combined_df = pd.concat(all_data).groupby(level=0).mean()
    
    return {
        'train_start': train_start,
        'train_end': train_end,
        'test_year': test_year,
        'train_period': f"{train_start}-{train_end}",
        '単勝的中率': combined_df.loc['単勝', '的中率(%)'] / 100,
        '複勝的中率': combined_df.loc['複勝', '的中率(%)'] / 100,
        '馬連的中率': combined_df.loc['馬連', '的中率(%)'] / 100,
        'ワイド的中率': combined_df.loc['ワイド', '的中率(%)'] / 100,
        '三連複的中率': combined_df.loc['３連複', '的中率(%)'] / 100,
        '単勝回収率': combined_df.loc['単勝', '回収率(%)'] / 100,
        '複勝回収率': combined_df.loc['複勝', '回収率(%)'] / 100,
        '馬連回収率': combined_df.loc['馬連', '回収率(%)'] / 100,
        'ワイド回収率': combined_df.loc['ワイド', '回収率(%)'] / 100,
        '三連複回収率': combined_df.loc['３連複', '回収率(%)'] / 100,
    }


def print_result_summary(result_data):
    """結果のサマリーを表示"""
    print(f"\n[RESULT] {result_data['test_year']}年テスト結果:")
    print(f"   単勝: 的中率{result_data['単勝的中率']:.1%}, 回収率{result_data['単勝回収率']:.1%}")
    print(f"   複勝: 的中率{result_data['複勝的中率']:.1%}, 回収率{result_data['複勝回収率']:.1%}")
    print(f"   三連複: 的中率{result_data['三連複的中率']:.1%}, 回収率{result_data['三連複回収率']:.1%}")


def analyze_results(results, train_window):
    """
    Walk-Forward Validation結果の統計分析
    
    Args:
        results (list): 各年の結果データ
        train_window (int): 学習期間
    """
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"[STATS] Walk-Forward Validation統計分析")
    print(f"   学習期間: {train_window}年")
    print(f"   テスト回数: {len(results)}回")
    print(f"{'='*60}\n")
    
    # 主要指標の統計
    metrics = ['単勝的中率', '複勝的中率', '三連複的中率', '単勝回収率', '複勝回収率']
    
    for metric in metrics:
        values = df[metric]
        mean = values.mean()
        std = values.std()
        cv = std / mean if mean > 0 else 0
        min_val = values.min()
        max_val = values.max()
        
        print(f"【{metric}】")
        print(f"  平均:     {mean:.1%}")
        print(f"  標準偏差: {std:.1%}")
        print(f"  変動係数: {cv:.3f}")
        print(f"  範囲:     {min_val:.1%} ~ {max_val:.1%}")
        print(f"  95%信頼区間: [{mean-2*std:.1%}, {mean+2*std:.1%}]")
        print()
    
    # 安定性判定
    print(f"{'='*60}")
    print(f"[CHECK] 安定性判定")
    print(f"{'='*60}\n")
    
    tansho_cv = df['単勝的中率'].std() / df['単勝的中率'].mean()
    tansho_mean = df['単勝的中率'].mean()
    tansho_min = df['単勝的中率'].mean() - 2 * df['単勝的中率'].std()
    
    print(f"単勝的中率:")
    print(f"  平均: {tansho_mean:.1%}")
    print(f"  変動係数: {tansho_cv:.3f}")
    print(f"  下限(95%): {tansho_min:.1%}")
    print()
    
    # 判定基準
    stable = False
    if tansho_cv < 0.15 and tansho_mean > 0.15 and tansho_min > 0.10:
        print("[OK] 【安定性: 高】")
        print("   変動係数15%未満、平均15%超、下限10%超")
        print("   -> 本番投入推奨!")
        stable = True
    elif tansho_cv < 0.20 and tansho_mean > 0.12 and tansho_min > 0.05:
        print("[WARN] 【安定性: 中】")
        print("   一定の変動あり、慎重に検討")
        print("   -> 追加検証を推奨")
    else:
        print("[NG] 【安定性: 低】")
        print("   高い変動、または平均的中率が低い")
        print("   -> モデル改善が必要")
    
    print()
    
    # 結果をCSVで保存
    output_file = f"results/walk_forward_train{train_window}years.tsv"
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8-sig')
    print(f"[FILE] 詳細結果を {output_file} に保存しました")
    
    return stable


if __name__ == '__main__':
    import sys
    
    # コマンドライン引数でカスタマイズ可能
    train_window = 3  # デフォルト3年
    test_years = [2023, 2024, 2025]  # デフォルト
    
    if len(sys.argv) > 1:
        train_window = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        # カンマ区切りで複数年指定: python walk_forward_validation.py 3 2023,2024,2025
        test_years = [int(y) for y in sys.argv[2].split(',')]
    
    # Walk-Forward Validation実行
    results = run_walk_forward(train_window=train_window, test_years=test_years)
    
    print(f"\n{'='*60}")
    print("[DONE] Walk-Forward Validation完了!")
    print(f"{'='*60}")
