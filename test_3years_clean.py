"""
リークフリー3年間テストスクリプト
2023年→2016-2022モデル, 2024年→2017-2023モデル, 2025年→2018-2024モデル
"""

import re
import json
from pathlib import Path

def run_model_test(model_file, test_year):
    """
    単一モデルのテスト実行
    
    Args:
        model_file (Path): モデルファイルのパス
        test_year (int): テスト年
    
    Returns:
        bool: 成功時True
    """
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
        
        # betting_summaryを保存
        summary_file = f"betting_summary_{base_name}_train{train_range}_test{test_year}.tsv"
        summary_path = results_dir / summary_file
        summary_df.to_csv(summary_path, index=True, sep='\t', encoding='utf-8-sig')
        
        # predicted_resultsも保存
        predicted_file = f"predicted_results_{base_name}_train{train_range}_test{test_year}.tsv"
        predicted_path = results_dir / predicted_file
        output_df.to_csv(predicted_path, index=False, sep='\t', encoding='utf-8-sig')
        
        print(f"[OK] 完了 - レース数: {race_count}")
        print(f"   -> {summary_file}")
        print(f"   -> {predicted_file}")
        print(f"   単勝: 的中率{summary_df.loc['単勝', '的中率(%)']:.1f}%, 回収率{summary_df.loc['単勝', '回収率(%)']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    3年間リークフリーテストを実行
    """
    print("=" * 60)
    print("リークフリー3年間テスト開始")
    print("=" * 60)
    
    # テスト計画: {テスト年: 学習期間}
    test_plan = {
        2023: "2016-2022",
        2024: "2017-2023",
        2025: "2018-2024"
    }
    
    models_dir = Path('models')
    
    # 長距離と短距離のモデルをテスト
    base_names = [
        'tokyo_turf_3ageup_long',
        'tokyo_turf_3ageup_short'
    ]
    
    results = []
    
    for test_year, train_range in test_plan.items():
        print(f"\n{'='*60}")
        print(f"【{test_year}年テスト】 学習期間: {train_range}")
        print(f"{'='*60}")
        
        for base_name in base_names:
            model_filename = f"{base_name}_{train_range}.sav"
            model_path = models_dir / model_filename
            
            if not model_path.exists():
                print(f"[WARNING] {model_filename} が見つかりません。スキップ。")
                continue
            
            print(f"\n[RUN] {base_name} ({train_range} → {test_year}年)")
            success = run_model_test(model_path, test_year)
            
            if success:
                results.append({
                    'test_year': test_year,
                    'train_range': train_range,
                    'model': base_name
                })
    
    print("\n" + "=" * 60)
    print("3年間テスト完了")
    print("=" * 60)
    print(f"成功: {len(results)}/{len(test_plan) * len(base_names)}件")
    
    # 結果サマリー表示
    print("\n[完了したテスト]")
    for r in results:
        print(f"  {r['test_year']}年: {r['model']} (学習: {r['train_range']})")

if __name__ == '__main__':
    main()
