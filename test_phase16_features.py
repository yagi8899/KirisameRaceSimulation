"""
Phase 1.6特徴量実装のテストスクリプト

新しい特徴量（track_upset_score, num_runners, is_full_field）がSQLクエリに含まれているか確認
"""
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_query_builder import build_race_data_query, build_sokuho_race_data_query

def test_phase16_features():
    """Phase 1.6特徴量がSQLクエリに含まれているかテスト"""
    print("=" * 80)
    print("Phase 1.6特徴量実装テスト (track_upset_score, num_runners, is_full_field)")
    print("=" * 80)
    
    # 新規追加の特徴量
    new_features = [
        'track_upset_score',
        'num_runners',
        'is_full_field',
        'is_local_track'
    ]
    
    # CTEの確認用
    cte_features = [
        'track_upset_stats',
        'track_upset_rates'
    ]
    
    print("\n[1] 訓練用クエリ (build_race_data_query) のテスト")
    print("-" * 80)
    try:
        # 東京芝中長距離の例
        query = build_race_data_query(
            track_code='05',  # 東京
            year_start=2020,
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
        
        # CTE確認
        print("\nCTE (Common Table Expression) の存在確認:")
        for cte in cte_features:
            if cte in query:
                print(f"  ✅ {cte}: 含まれています")
            else:
                print(f"  ❌ {cte}: 含まれていません")
        
        print(f"\nクエリ長: {len(query):,} 文字")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("\n[2] 速報用クエリ (build_sokuho_race_data_query) のテスト")
    print("-" * 80)
    try:
        # 東京芝中長距離の例（速報クエリは年指定不要、自動で直近5年）
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
        
        # CTE確認
        print("\nCTE (Common Table Expression) の存在確認:")
        for cte in cte_features:
            if cte in query:
                print(f"  ✅ {cte}: 含まれています")
            else:
                print(f"  ❌ {cte}: 含まれていません")
        
        print(f"\nクエリ長: {len(query):,} 文字")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("テスト完了")

if __name__ == "__main__":
    test_phase16_features()
