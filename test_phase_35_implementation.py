"""
Phase 3.5特徴量実装のテストスクリプト

SQLクエリ生成と新特徴量の存在確認を行う
"""
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_query_builder import build_race_data_query, build_sokuho_race_data_query

def test_phase_35_features():
    """Phase 3.5特徴量がSQLクエリに含まれているかテスト"""
    print("=" * 80)
    print("Phase 3.5特徴量実装テスト")
    print("=" * 80)
    
    # 新規追加の5特徴量
    new_features = [
        'zenso_ninki_gap',
        'zenso_nigeba',
        'zenso_taihai',
        'zenso_agari_rank',
        'saikin_kaikakuritsu'
    ]
    
    print("\n[1] 訓練用クエリ (build_race_data_query) のテスト")
    print("-" * 80)
    try:
        # 東京芝中長距離の例
        query = build_race_data_query(
            track_code='05',  # 東京
            year_start=2024,
            year_end=2024,
            surface_type='turf',
            distance_min=1700,
            distance_max=9999,
            include_payout=False
        )
        print("✅ クエリ生成成功")
        
        # 新特徴量の存在確認
        print("\n新特徴量の存在確認:")
        for feature in new_features:
            if feature in query:
                print(f"  ✅ {feature}: 含まれています")
            else:
                print(f"  ❌ {feature}: 含まれていません")
        
        # クエリ長確認
        print(f"\nクエリ長: {len(query):,} 文字")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("[2] 速報予測用クエリ (build_sokuho_race_data_query) のテスト")
    print("-" * 80)
    try:
        query = build_sokuho_race_data_query(
            track_code='05',  # 東京
            surface_type='turf',
            distance_min=1700,
            distance_max=9999
        )
        print("✅ クエリ生成成功")
        
        # 新特徴量の存在確認
        print("\n新特徴量の存在確認:")
        for feature in new_features:
            if feature in query:
                print(f"  ✅ {feature}: 含まれています")
            else:
                print(f"  ❌ {feature}: 含まれていません")
        
        # クエリ長確認
        print(f"\nクエリ長: {len(query):,} 文字")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("テスト完了!")
    print("=" * 80)
    
    print("\n📝 次のステップ:")
    print("  1. モデル再訓練: python train_upset_classifier.py --years 2015-2024")
    print("  2. 特徴量重要度分析: python analyze_upset_model_features.py <model_path>")
    print("  3. 閾値最適化: python analyze_upset_threshold.py <model_path>")
    print("  4. テスト実行: python universal_test.py 2025")

if __name__ == "__main__":
    test_phase_35_features()
