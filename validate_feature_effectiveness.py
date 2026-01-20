"""
Phase 3.5特徴量の有効性検証スクリプト

単一モデル(2015-2024)とwalk-forward(period_10)の結果を比較して、
新特徴量が本当に有効か、過学習していないかを判定する
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_single_model_vs_walkforward():
    """単一モデルとwalk-forwardの結果を比較"""
    print("=" * 80)
    print("Phase 3.5特徴量有効性検証")
    print("=" * 80)
    
    # 単一モデルの特徴量重要度
    single_model_path = Path("walk_forward_results_custom2/period_10/models/2025/upset_classifier_2015-2024.sav")
    
    if not single_model_path.exists():
        print(f"\n❌ 単一モデルが見つかりません: {single_model_path}")
        print("   まずStep 1を実行してください:")
        print("   python train_upset_classifier.py --years 2015-2024")
        return
    
    # モデル読み込み
    print(f"\n✅ 単一モデル読み込み: {single_model_path}")
    with open(single_model_path, 'rb') as f:
        ensemble_models = pickle.load(f)
    
    # アンサンブルの最初のモデルから特徴量重要度を取得
    if isinstance(ensemble_models, list) and len(ensemble_models) > 0:
        model = ensemble_models[0]
        feature_names = model.feature_name()
        importances = model.feature_importance(importance_type='gain')
        
        # DataFrame作成
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 特徴量数: {len(feature_names)}個")
        print(f"総重要度: {importances.sum():,.0f}")
        
        # Phase 3.5特徴量の確認
        phase35_features = [
            'zenso_ninki_gap', 'zenso_nigeba', 'zenso_taihai',
            'zenso_agari_rank', 'saikin_kaikakuritsu'
        ]
        
        print("\n" + "=" * 80)
        print("Phase 3.5新特徴量の重要度:")
        print("=" * 80)
        
        phase35_importance_sum = 0
        phase35_found = []
        
        for feat in phase35_features:
            if feat in df_importance['feature'].values:
                imp = df_importance[df_importance['feature'] == feat]['importance'].values[0]
                rank = df_importance[df_importance['feature'] == feat].index[0] + 1
                phase35_importance_sum += imp
                phase35_found.append(feat)
                print(f"  ✅ {feat:25s}: {imp:>10,.0f} (rank {rank:>2d})")
            else:
                print(f"  ❌ {feat:25s}: モデルに含まれていません")
        
        # Phase 3既存特徴量の確認
        phase3_features = [
            'past_score_std', 'past_chakujun_variance',
            'zenso_oikomi_power', 'zenso_kakoi_komon'
        ]
        
        print("\n" + "=" * 80)
        print("Phase 3既存特徴量の重要度:")
        print("=" * 80)
        
        phase3_importance_sum = 0
        
        for feat in phase3_features:
            if feat in df_importance['feature'].values:
                imp = df_importance[df_importance['feature'] == feat]['importance'].values[0]
                rank = df_importance[df_importance['feature'] == feat].index[0] + 1
                phase3_importance_sum += imp
                print(f"  ✅ {feat:25s}: {imp:>10,.0f} (rank {rank:>2d})")
        
        # 統計サマリー
        total_importance = importances.sum()
        phase35_ratio = (phase35_importance_sum / total_importance) * 100
        phase3_ratio = (phase3_importance_sum / total_importance) * 100
        upset_total_ratio = phase35_ratio + phase3_ratio
        
        print("\n" + "=" * 80)
        print("UPSET特徴量の統計:")
        print("=" * 80)
        print(f"  Phase 3.5特徴量の重要度合計: {phase35_importance_sum:>10,.0f} ({phase35_ratio:>5.2f}%)")
        print(f"  Phase 3特徴量の重要度合計:   {phase3_importance_sum:>10,.0f} ({phase3_ratio:>5.2f}%)")
        print(f"  UPSET特徴量の総合重要度:     {phase35_importance_sum + phase3_importance_sum:>10,.0f} ({upset_total_ratio:>5.2f}%)")
        
        # Top 10特徴量
        print("\n" + "=" * 80)
        print("Top 10重要特徴量:")
        print("=" * 80)
        for i, row in df_importance.head(10).iterrows():
            is_upset = '🔥' if row['feature'] in phase35_features + phase3_features else '  '
            print(f"  {is_upset} {i+1:2d}. {row['feature']:30s}: {row['importance']:>10,.0f}")
        
        # 判定基準
        print("\n" + "=" * 80)
        print("walk-forward成功予測:")
        print("=" * 80)
        
        success_score = 0
        max_score = 5
        
        # 判定1: Phase 3.5特徴量の重要度比率
        if phase35_ratio >= 10:
            print(f"  ✅ Phase 3.5重要度 {phase35_ratio:.1f}% ≥ 10% → 高い影響力")
            success_score += 1
        elif phase35_ratio >= 5:
            print(f"  ⚠️  Phase 3.5重要度 {phase35_ratio:.1f}% ≥ 5% → 中程度の影響力")
            success_score += 0.5
        else:
            print(f"  ❌ Phase 3.5重要度 {phase35_ratio:.1f}% < 5% → 影響力不足")
        
        # 判定2: Top 10に何個入っているか
        top10_count = len([f for f in df_importance.head(10)['feature'].values 
                           if f in phase35_features])
        if top10_count >= 2:
            print(f"  ✅ Phase 3.5特徴量がTop 10に{top10_count}個 → 高重要度")
            success_score += 1
        elif top10_count >= 1:
            print(f"  ⚠️  Phase 3.5特徴量がTop 10に{top10_count}個 → 中程度")
            success_score += 0.5
        else:
            print(f"  ❌ Phase 3.5特徴量がTop 10に0個 → 重要度不足")
        
        # 判定3: UPSET特徴量全体の重要度
        if upset_total_ratio >= 20:
            print(f"  ✅ UPSET総合重要度 {upset_total_ratio:.1f}% ≥ 20% → 強力な影響")
            success_score += 1
        elif upset_total_ratio >= 15:
            print(f"  ⚠️  UPSET総合重要度 {upset_total_ratio:.1f}% ≥ 15% → 中程度の影響")
            success_score += 0.5
        else:
            print(f"  ❌ UPSET総合重要度 {upset_total_ratio:.1f}% < 15% → 影響不足")
        
        # 判定4: バランスの良さ（既存重要特徴量も維持）
        top5_features = df_importance.head(5)['feature'].values
        important_base_features = ['popularity_rank', 'predicted_rank', 'value_gap']
        base_in_top5 = len([f for f in top5_features if f in important_base_features])
        
        if base_in_top5 >= 2:
            print(f"  ✅ 既存重要特徴量がTop 5に{base_in_top5}個 → バランス良好")
            success_score += 1
        elif base_in_top5 >= 1:
            print(f"  ⚠️  既存重要特徴量がTop 5に{base_in_top5}個 → やや偏り")
            success_score += 0.5
        else:
            print(f"  ❌ 既存重要特徴量がTop 5に0個 → 過学習の疑い")
        
        # 判定5: Phase 3.5特徴量の均等性
        if len(phase35_found) >= 4:
            phase35_importances = [df_importance[df_importance['feature'] == f]['importance'].values[0] 
                                   for f in phase35_found]
            cv = np.std(phase35_importances) / np.mean(phase35_importances) if np.mean(phase35_importances) > 0 else 0
            if cv < 1.5:
                print(f"  ✅ Phase 3.5特徴量のバラツキ CV={cv:.2f} < 1.5 → 均等に有効")
                success_score += 1
            else:
                print(f"  ⚠️  Phase 3.5特徴量のバラツキ CV={cv:.2f} ≥ 1.5 → 一部のみ有効")
                success_score += 0.5
        
        # 最終判定
        print("\n" + "=" * 80)
        print(f"総合スコア: {success_score:.1f} / {max_score}")
        print("=" * 80)
        
        if success_score >= 4.0:
            print("\n🎉 判定: walk-forwardでも良好な結果が期待できます!")
            print("   - 新特徴量が強力な影響力を持っている")
            print("   - 既存特徴量とのバランスも良好")
            print("   - 過学習のリスクは低い")
            confidence = "高い"
        elif success_score >= 3.0:
            print("\n⚠️  判定: walk-forwardでも改善は見込めますが、幅は小さい可能性")
            print("   - 新特徴量は有効だが、影響力は限定的")
            print("   - 単一モデルとwalk-forwardで差が出る可能性あり")
            confidence = "中程度"
        else:
            print("\n❌ 判定: walk-forwardでの改善は限定的な可能性が高い")
            print("   - 新特徴量の影響力が不足")
            print("   - 再度特徴量設計を見直すことを推奨")
            confidence = "低い"
        
        print(f"\nwalk-forward成功の確信度: {confidence}")
        
        # 推奨アクション
        print("\n" + "=" * 80)
        print("推奨アクション:")
        print("=" * 80)
        
        if success_score >= 3.0:
            print("  1. ✅ walk-forward実行を推奨")
            print("     python walk_forward_validation.py --with-upset")
            print("  2. 結果比較:")
            print("     - 単一モデルのPrecisionと比較")
            print("     - 改善幅が50%以上維持されればOK")
        else:
            print("  1. ⚠️ まず単一モデルで追加調整を推奨")
            print("     - Phase 2特徴量の追加を検討")
            print("     - ハイパーパラメータチューニング")
            print("  2. 調整後にwalk-forward実行")
        
        return {
            'success_score': success_score,
            'phase35_ratio': phase35_ratio,
            'phase3_ratio': phase3_ratio,
            'upset_total_ratio': upset_total_ratio,
            'confidence': confidence
        }
    
    else:
        print("\n❌ モデル形式が不正です")
        return None

if __name__ == "__main__":
    analyze_single_model_vs_walkforward()
