"""
Phase 3 SQL特徴量のテストスクリプト（軽量版）
フェーズ1の6特徴量がカラムとして存在するか確認する
"""
import psycopg2
import pandas as pd
import json
from db_query_builder import build_race_data_query

print("=" * 60)
print("Phase 3 穴馬特化特徴量テスト（軽量版）")
print("=" * 60)

# DB接続情報読み込み
with open('db_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
db_config = config['database']

# Step 1: SQL生成確認
print("\n[Step 1] SQL生成確認...")
sql = build_race_data_query('09', 2023, 2023, 'turf', 1800, 2400, '13')
print("✅ SQL生成成功")

# Step 2: 新特徴量がSQLに含まれているか確認
new_features = [
    'past_score_std',
    'past_chakujun_variance',
    'zenso_oikomi_power',
    'kishu_changed',
    'class_downgrade',
    'zenso_kakoi_komon'
]

print("\n[Step 2] SQL内の特徴量存在確認...")
for feat in new_features:
    if feat in sql:
        print(f"  ✓ {feat} - SQL内に存在")
    else:
        print(f"  ✗ {feat} - SQL内に存在しない")

# Step 3: データベース接続とカラム名取得（データは取得しない）
print("\n[Step 3] PostgreSQL接続とカラム名取得（軽量）...")
print(f"  接続先: {db_config['host']}:{db_config['port']}/{db_config['dbname']}")
conn = psycopg2.connect(**db_config)

# LIMIT 1で1レコードだけ取得してカラム名確認
test_sql = sql + " LIMIT 1"
try:
    df_test = pd.read_sql_query(test_sql, conn)
    print(f"✅ データ取得成功（テスト用1レコード）")
    
    print(f"\n📊 総カラム数: {len(df_test.columns)}")
    
    # 新特徴量のカラム存在確認
    print("\n[Step 4] 新特徴量カラム存在確認...")
    available = []
    missing = []
    for feat in new_features:
        if feat in df_test.columns:
            available.append(feat)
            print(f"  ✓ {feat} - カラム存在")
        else:
            missing.append(feat)
            print(f"  ✗ {feat} - カラム存在しない")
    
    print(f"\n🎯 結果: {len(available)}/{len(new_features)} 特徴量が実装済み")
    
    if len(available) == len(new_features):
        print("\n✅ 全特徴量の実装成功！")
        print("\n次のステップ: upset_classifier_creator.py で再訓練を実行してください")
    else:
        print(f"\n⚠️  {len(missing)}個の特徴量が見つかりません:")
        for feat in missing:
            print(f"    - {feat}")
            
except Exception as e:
    print(f"\n❌ エラー発生: {e}")
    print("\nSQLの一部を表示（最初の500文字）:")
    print(sql[:500])
finally:
    conn.close()

print("\n" + "=" * 60)
